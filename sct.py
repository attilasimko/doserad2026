"""MR -> synthetic CT (sCT) generation for the MR arm of the challenge.

The MR task is handled by turning the MR into a CT and then running the
*exact* same dose pipeline as the CT task: HU -> density LUT, body mask,
pencil-beam baseline, and the same correction model. There is no
MR-specific branch anywhere downstream of this module.

The generator is a standalone nnUNet regression model exported to
TorchScript (``mrtoct_bundle/``: ``model.pt`` + ``metadata.json``). This
module is self-contained: torch + numpy + scipy only, no nnUNet import
needed at inference time.

This is deliberately kept as close to the shipped standalone inference code
as possible. There is exactly ONE behavioural deviation, plus one bug fix:

* DEVIATION - ``SCT_TRAINING_SPACING``. The model was trained on the *proton*
  cohort, whose grid is 1x1x3 mm, while the photon cohort is 2 mm isotropic,
  and metadata.json does not record the training grid. The MR is resampled
  onto it before inference and the sCT is resampled back afterwards. Measured
  on three validating patients this is worth 35 HU body MAE against the real
  CT, versus 67 HU running on the native grid. Set DOSERAD_SCT_SPACING=none
  to disable it and reproduce the shipped behaviour exactly.
* BUG FIX - volumes smaller than the patch, or not divisible by the network's
  downsampling factors (8 along z, 32 in-plane), are zero-padded and cropped
  back. The shipped code fed them to the model unpadded, which throws.

Normalisation follows metadata.json as shipped: nnUNet's own
``ZScoreNormalization`` is per-image, but measured on the same three patients
the two are equivalent (34.5 vs 35.0 HU), so there was no reason to deviate.

Typical use::

    from sct import mr_to_synthetic_ct
    sct_hu = mr_to_synthetic_ct(mr_array, spacing=(2.0, 2.0, 2.0))
"""

import json
import contextlib
import os
from functools import lru_cache
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter

# --------------------------------------------------------------------------
# Bundle / grid configuration
# --------------------------------------------------------------------------

#: Default location of the exported bundle (model.pt + metadata.json).
DEFAULT_BUNDLE_DIR = os.environ.get(
    "DOSERAD_SCT_BUNDLE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "mrtoct_bundle"),
)

#: The TorchScript weights are 390 MB, so they are not in git. They live here:
HF_REPO = "zimmeryWo/MRI-sCT_converter-DoseRAD2026"
HF_BUNDLE_URL = f"https://huggingface.co/{HF_REPO}/resolve/main/mrtoct_bundle"
MODEL_PT_BYTES = 408_644_663


def fetch_bundle(bundle_dir: Union[str, Path, None] = None,
                 force: bool = False) -> Path:
    """Download model.pt (and metadata.json) from Hugging Face if absent.

    Plain urllib rather than huggingface_hub: this is two files from a public
    repo, and the container already pins its dependencies tightly.

    The size check is not paranoia. An interrupted download leaves a truncated
    file that torch.jit.load accepts far enough to fail later with an opaque
    error, so a short file is deleted and reported here instead.
    """
    import urllib.request

    d = Path(bundle_dir or DEFAULT_BUNDLE_DIR)
    d.mkdir(parents=True, exist_ok=True)
    for name, expect in (("metadata.json", None), ("model.pt", MODEL_PT_BYTES)):
        dst = d / name
        if dst.exists() and not force:
            if expect is None or dst.stat().st_size == expect:
                continue
            print(f"[sct] {dst} is {dst.stat().st_size} bytes, expected {expect}"
                  f" -- re-downloading", flush=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        print(f"[sct] fetching {name} from {HF_REPO} ...", flush=True)
        urllib.request.urlretrieve(f"{HF_BUNDLE_URL}/{name}", tmp)
        if expect is not None and tmp.stat().st_size != expect:
            got = tmp.stat().st_size
            tmp.unlink()
            raise RuntimeError(
                f"{name} downloaded as {got} bytes, expected {expect}. "
                f"Truncated or the remote changed; not installing it.")
        tmp.replace(dst)
    return d

#: Voxel spacing (z, y, x in mm) the bundled model was trained on.
#:
#: metadata.json does not carry the plans' target spacing, but the traced
#: graph does: the stem convolution has kernel (1, 3, 3) and only the
#: in-plane axes are downsampled at the first stage, which is exactly what
#: nnUNet emits when axis 0 is >=3x coarser than the in-plane axes. Combined
#: with the patch size (48, 192, 192) and the dataset name
#: (Dataset038_DoseRAD2026_proton_...), that pins the training grid to the
#: proton cohort's 1x1x3 mm (sitk x,y,z) == (3, 1, 1) mm in array order.
#:
#: Set DOSERAD_SCT_SPACING="z,y,x" to override, or "none" to disable
#: resampling and run on the native grid.
SCT_TRAINING_SPACING: Optional[Tuple[float, float, float]] = (3.0, 1.0, 1.0)

#: Input size divisibility of the traced network, per axis (z, y, x).
#: Probed from the bundle: 3 downsamplings along z, 5 in-plane.
# Optional half precision for the sCT network. Off unless asked for, because
# the sCT feeds every downstream dose number on the MR track.
_SCT_AMP_DTYPE = {"fp16": torch.float16, "float16": torch.float16,
                  "bf16": torch.bfloat16, "": None, "off": None, "fp32": None,
                  }[os.environ.get("DOSERAD_SCT_AMP", "").lower()]

SIZE_DIVISOR = (8, 32, 32)

#: HU range the model was trained to produce (metadata output min/max).
SCT_HU_RANGE = (-1024.0, 3071.0)

_env_spacing = os.environ.get("DOSERAD_SCT_SPACING")
if _env_spacing:
    if _env_spacing.strip().lower() in ("none", "native", "off"):
        SCT_TRAINING_SPACING = None
    else:
        SCT_TRAINING_SPACING = tuple(float(v) for v in _env_spacing.split(","))  # type: ignore[assignment]


# ============================================================================
# Helper Functions
# ============================================================================

def compute_steps_for_sliding_window(
    image_size: Tuple[int, ...],
    tile_size: Tuple[int, ...],
    tile_step_size: float
) -> List[List[int]]:
    """Calculate sliding window step positions for tiled inference.

    Args:
        image_size: Size of the input image (e.g., (512, 512, 300))
        tile_size: Size of each tile/patch (e.g., (128, 128, 128))
        tile_step_size: Step size as fraction of tile_size (0 < value <= 1)
                       0.5 = 50% overlap, 1.0 = no overlap

    Returns:
        List of lists containing step positions for each dimension
    """
    assert all(i >= j for i, j in zip(image_size, tile_size)), \
        "Image size must be >= tile size in all dimensions"
    assert 0 < tile_step_size <= 1, "tile_step_size must be in range (0, 1]"

    target_step_sizes_in_voxels = [int(i * tile_step_size) for i in tile_size]

    num_steps = [
        int(np.ceil((i - k) / j)) + 1
        for i, j, k in zip(image_size, target_step_sizes_in_voxels, tile_size)
    ]

    steps = []
    for dim in range(len(tile_size)):
        max_step_value = image_size[dim] - tile_size[dim]
        if num_steps[dim] > 1:
            actual_step_size = max_step_value / (num_steps[dim] - 1)
        else:
            actual_step_size = 99999999999  # only one step, at 0
        steps.append([int(np.round(actual_step_size * i)) for i in range(num_steps[dim])])

    return steps


@lru_cache(maxsize=4)
def compute_gaussian_weight(
    tile_size: Tuple[int, ...],
    sigma_scale: float = 1.0 / 8,
    value_scaling_factor: float = 1.0,
    dtype: torch.dtype = torch.float32,
    device: Union[str, torch.device] = "cuda"
) -> torch.Tensor:
    """Gaussian importance map used to blend overlapping patch predictions."""
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1

    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = torch.from_numpy(gaussian_importance_map)
    gaussian_importance_map /= (torch.max(gaussian_importance_map) / value_scaling_factor)
    gaussian_importance_map = gaussian_importance_map.to(device=device, dtype=dtype)

    # No zeros: they would produce 0/0 in the weight normalisation.
    mask = gaussian_importance_map == 0
    if mask.any():
        gaussian_importance_map[mask] = torch.min(gaussian_importance_map[~mask])

    return gaussian_importance_map


def _pad_to_minimum(volume: torch.Tensor, minimum: Sequence[int],
                    divisor: Sequence[int]) -> Tuple[torch.Tensor, List[int]]:
    """Symmetrically pad the trailing 3 axes to at least ``minimum`` and to a
    multiple of ``divisor``. Returns the padded volume and the per-axis
    ``(before, after)`` amounts flattened as [z0, z1, y0, y1, x0, x1]."""
    shape = volume.shape[-3:]
    pads = []
    for size, mn, dv in zip(shape, minimum, divisor):
        target = max(size, mn)
        if target % dv:
            target += dv - (target % dv)
        total = target - size
        pads.append((total // 2, total - total // 2))
    if all(p == (0, 0) for p in pads):
        return volume, [0, 0, 0, 0, 0, 0]
    # F.pad takes the LAST axis first.
    pad_arg = [pads[2][0], pads[2][1], pads[1][0], pads[1][1], pads[0][0], pads[0][1]]
    padded = F.pad(volume, pad_arg, mode="constant", value=0.0)
    return padded, [pads[0][0], pads[0][1], pads[1][0], pads[1][1], pads[2][0], pads[2][1]]


def _resample_zyx(volume: torch.Tensor, out_shape: Sequence[int]) -> torch.Tensor:
    """Trilinear resample a [D, H, W] tensor to ``out_shape``."""
    if tuple(volume.shape) == tuple(out_shape):
        return volume
    return F.interpolate(
        volume[None, None].float(), size=tuple(int(s) for s in out_shape),
        mode="trilinear", align_corners=False,
    )[0, 0]


def _nonzero_bbox(arr: np.ndarray, margin: int = 2) -> Tuple[slice, slice, slice]:
    """nnUNet-style crop-to-nonzero bounding box, with a small safety margin."""
    nz = arr > 0
    if not nz.any():
        return (slice(None), slice(None), slice(None))
    out = []
    for ax in range(3):
        proj = nz.any(axis=tuple(a for a in range(3) if a != ax))
        idx = np.nonzero(proj)[0]
        lo = max(int(idx[0]) - margin, 0)
        hi = min(int(idx[-1]) + 1 + margin, arr.shape[ax])
        out.append(slice(lo, hi))
    return tuple(out)  # type: ignore[return-value]


# ============================================================================
# Main Inference Class
# ============================================================================

class StandaloneRegressionInference:
    """Standalone inference for a TorchScript-exported nnUNet regression model.

    Self-contained: TorchScript load, normalisation/denormalisation, and
    sliding-window inference with Gaussian blending. No nnUNet dependency.
    """

    def __init__(
        self,
        model_path: Union[str, Path],
        device: Union[str, torch.device] = "cuda",
        verbose: bool = True,
    ):
        self.model_path = Path(model_path)
        self.device = torch.device(device) if isinstance(device, str) else device
        self.verbose = verbose

        metadata_path = self.model_path / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {metadata_path}\n"
                f"Fetch the bundle:  python -c 'import sct; sct.fetch_bundle()'")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)

        model_file = self.model_path / "model.pt"
        if not model_file.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_file}\n"
                f"The TorchScript weights are 390 MB and are not in git. Fetch "
                f"them with:  python -c 'import sct; sct.fetch_bundle()'\n"
                f"or from https://huggingface.co/{HF_REPO}")

        self.model = torch.jit.load(str(model_file), map_location=self.device)
        self.model.eval()

        cfg = self.metadata['inference_config']
        self.patch_size = tuple(cfg['patch_size'])
        self.tile_step_size = cfg['tile_step_size']
        self.use_gaussian = cfg['use_gaussian']

        if self.use_gaussian:
            self._gaussian_cache = compute_gaussian_weight(
                self.patch_size, sigma_scale=1.0 / 8,
                value_scaling_factor=10.0,   # nnUNet's blending scale
                dtype=torch.float32, device=self.device,
            )
        else:
            self._gaussian_cache = None

    # -- normalisation ------------------------------------------------------
    def _normalize(self, data: np.ndarray, channel: str = 'input') -> np.ndarray:
        """Normalise the input the way nnUNet did during training.

        One subtlety: nnUNet's ``ZScoreNormalization`` is a *per-image* z-score
        (``(x - x.mean()) / x.std()``), whereas the mean/std in metadata.json
        are the dataset-wide foreground statistics collected by the planner.
        Measured against the real CT on three validating patients the two are
        indistinguishable (34.5 HU with the dataset statistics, 35.0 HU
        per-image), so the default is the dataset statistics -- what the
        exported bundle's own metadata carries. Set
        ``normalization.input.per_image = true`` for the nnUNet-faithful
        variant.

        (For contrast, the resampling in ``mr_to_synthetic_ct`` is NOT a wash:
        running on the native 2 mm grid instead of the 1x1x3 mm training grid
        costs 67 vs 35 HU.)
        """
        norm_config = self.metadata['normalization'][channel]
        scheme = norm_config['scheme']
        data = data.astype(np.float32, copy=False)

        if scheme == 'ZScoreNormalization':
            if norm_config.get('per_image', False):
                mean = float(data.mean())
                std = float(data.std())
            else:
                mean, std = norm_config['mean'], norm_config['std']
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'CTNormalization':
            mean, std = norm_config['mean'], norm_config['std']
            lower, upper = norm_config['percentile_00_5'], norm_config['percentile_99_5']
            data = np.clip(data, lower, upper)
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'GlobalNormalization':
            mean, std = norm_config['mean'], norm_config['std']
            data = (data - mean) / max(std, 1e-8)

        elif scheme == 'NoNormalization':
            pass

        elif scheme == 'RescaleTo01Normalization':
            data = data - data.min()
            data = data / np.clip(data.max(), a_min=1e-8, a_max=None)

        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

        return data

    def _denormalize(self, data: np.ndarray, channel: str = 'output') -> np.ndarray:
        """Map a normalised prediction back to the original intensity range."""
        norm_config = self.metadata['normalization'][channel]
        scheme = norm_config['scheme']

        if scheme in ('ZScoreNormalization', 'CTNormalization', 'GlobalNormalization'):
            data = data * norm_config['std'] + norm_config['mean']
        elif scheme in ('NoNormalization', 'RescaleTo01Normalization'):
            pass
        else:
            raise ValueError(f"Unknown normalization scheme: {scheme}")

        return data

    # -- sliding window -----------------------------------------------------
    @torch.no_grad()
    def _sliding_window_inference(
        self,
        data: torch.Tensor,
        tile_step_size: Optional[float] = None,
        mirror_axes: Sequence[int] = (),
    ) -> torch.Tensor:
        """Sliding-window inference with Gaussian blending on a [1, 1, D, H, W]
        tensor. Volumes smaller than the patch (or not divisible by the
        network's downsampling factors) are zero-padded and cropped back."""
        step = tile_step_size if tile_step_size is not None else self.tile_step_size

        data, pad = _pad_to_minimum(data, self.patch_size, SIZE_DIVISOR)
        data_shape = tuple(data.shape[2:])

        steps = compute_steps_for_sliding_window(data_shape, self.patch_size, step)

        accum = torch.zeros((1, 1) + data_shape, dtype=torch.float32, device=self.device)
        weights = torch.zeros(data_shape, dtype=torch.float32, device=self.device)

        if self.use_gaussian and self._gaussian_cache is not None:
            gaussian = self._gaussian_cache
        else:
            gaussian = torch.ones(self.patch_size, dtype=torch.float32, device=self.device)

        p0, p1, p2 = self.patch_size
        for x in steps[0]:
            for y in steps[1]:
                for z in steps[2]:
                    sl = (slice(None), slice(None),
                          slice(x, x + p0), slice(y, y + p1), slice(z, z + p2))
                    patch = data[sl]
                    prediction = self._predict_patch(patch, mirror_axes)
                    accum[sl] += prediction * gaussian
                    weights[sl[2:]] += gaussian

        accum = accum / weights

        # Undo the padding.
        z0, z1, y0, y1, x0, x1 = pad
        if any(pad):
            accum = accum[
                :, :,
                z0: accum.shape[2] - z1 if z1 else None,
                y0: accum.shape[3] - y1 if y1 else None,
                x0: accum.shape[4] - x1 if x1 else None,
            ]
        return accum

    def _predict_patch(self, patch: torch.Tensor, mirror_axes: Sequence[int]) -> torch.Tensor:
        """Forward one patch, optionally averaging over mirrored copies (TTA).

        DOSERAD_SCT_AMP=fp16 autocasts the network only. The accumulator and the
        Gaussian blend in _sliding_window_inference stay float32, and the patch
        is a per-image z-score (values O(1)), so nothing here approaches fp16's
        range limits."""
        with self._amp():
            prediction = self.model(patch)
            if not mirror_axes:
                return prediction.float()
            axes = [a + 2 for a in mirror_axes]
            n = 1
            for k in range(1, 1 << len(axes)):
                flip = [axes[i] for i in range(len(axes)) if k & (1 << i)]
                prediction = prediction + torch.flip(self.model(torch.flip(patch, flip)), flip)
                n += 1
        return prediction.float() / n

    def _amp(self):
        dt = _SCT_AMP_DTYPE
        if dt is None or getattr(self.device, "type", str(self.device)) != "cuda":
            return contextlib.nullcontext()
        return torch.amp.autocast(device_type="cuda", dtype=dt, enabled=True)

    # -- public API ---------------------------------------------------------
    def predict(
        self,
        input_array: np.ndarray,
        apply_normalization: bool = True,
        apply_denormalization: bool = True,
        tile_step_size: Optional[float] = None,
        mirror_axes: Sequence[int] = (),
    ) -> np.ndarray:
        """Run inference on a [D, H, W] (or [1, D, H, W]) NumPy array."""
        if input_array.ndim == 4 and input_array.shape[0] == 1:
            input_array = input_array[0]
        if input_array.ndim != 3:
            raise ValueError(f"Expected a 3D volume, got shape {input_array.shape}")

        if apply_normalization:
            input_array = self._normalize(input_array, channel='input')

        tensor = torch.from_numpy(np.ascontiguousarray(input_array)).float()
        tensor = tensor[None, None].to(self.device)

        prediction = self._sliding_window_inference(tensor, tile_step_size, mirror_axes)
        prediction_np = prediction[0, 0].cpu().numpy()

        if apply_denormalization:
            prediction_np = self._denormalize(prediction_np, channel='output')

        return prediction_np


# ============================================================================
# Repo-facing API
# ============================================================================

_PREDICTOR: dict = {"key": None, "obj": None}


def get_predictor(bundle_dir: Optional[str] = None,
                  device: Optional[Union[str, torch.device]] = None,
                  verbose: bool = True) -> StandaloneRegressionInference:
    """Process-level singleton predictor (the bundle is ~400 MB; load once)."""
    bundle_dir = bundle_dir or DEFAULT_BUNDLE_DIR
    if device is None:
        # torch.cuda.is_available() can still report True in a forked worker
        # (the answer is cached from the parent) while initialising CUDA there
        # raises. Probe for real.
        use_cuda = torch.cuda.is_available()
        if use_cuda:
            try:
                torch.cuda.current_device()
            except RuntimeError:
                use_cuda = False
        device = "cuda" if use_cuda else "cpu"
    key = (str(bundle_dir), str(device))
    if _PREDICTOR["key"] != key:
        _PREDICTOR["obj"] = StandaloneRegressionInference(bundle_dir, device=device,
                                                         verbose=verbose)
        _PREDICTOR["key"] = key
    return _PREDICTOR["obj"]


def mr_to_synthetic_ct(
    mr_volume,
    spacing: Sequence[float] = (2.0, 2.0, 2.0),
    bundle_dir: Optional[str] = None,
    device: Optional[Union[str, torch.device]] = None,
    target_spacing: Optional[Sequence[float]] = "default",  # type: ignore[assignment]
    crop_to_nonzero: bool = False,
    tile_step_size: Optional[float] = None,
    mirror_axes: Sequence[int] = (),
    verbose: bool = False,
) -> np.ndarray:
    """Convert an MR volume to a synthetic CT in HU, on the SAME grid.

    Args:
        mr_volume: [D, H, W] MR array (array order z, y, x), any real dtype.
        spacing:   voxel spacing of ``mr_volume`` in mm, array order (z, y, x).
        target_spacing: grid the model expects, array order (z, y, x). The
            default resamples to ``SCT_TRAINING_SPACING``; pass ``None`` to run
            on the native grid.
        crop_to_nonzero: mirror nnUNet's preprocessing crop (voxels outside the
            nonzero bounding box become air, -1024 HU). Off by default: the
            shipped inference code does not crop, and this module stays as
            close to it as possible.

    Returns:
        float32 array of HU, same shape as ``mr_volume``.
    """
    predictor = get_predictor(bundle_dir, device, verbose=verbose)
    if target_spacing == "default":
        # A bundle exported by export_nnunet_model_to_standalone() carries the
        # plans' spacing; older bundles fall back to the pinned constant.
        target_spacing = predictor.metadata['inference_config'].get(
            'target_spacing', SCT_TRAINING_SPACING)

    was_tensor = isinstance(mr_volume, torch.Tensor)
    src_device = mr_volume.device if was_tensor else None
    mr = (mr_volume.detach().cpu().numpy() if was_tensor
          else np.asarray(mr_volume))
    mr = mr.astype(np.float32, copy=False)
    if mr.ndim != 3:
        raise ValueError(f"mr_to_synthetic_ct expects a 3D volume, got {mr.shape}")

    full_shape = mr.shape
    bbox = _nonzero_bbox(mr) if crop_to_nonzero else (slice(None),) * 3
    cropped = mr[bbox]
    dev = predictor.device

    work = torch.from_numpy(np.ascontiguousarray(cropped)).to(dev)

    # Resample onto the training grid (physical extent preserved).
    if target_spacing is not None:
        scale = [float(s) / float(t) for s, t in zip(spacing, target_spacing)]
        resampled_shape = [max(int(round(n * f)), 1) for n, f in zip(cropped.shape, scale)]
        work = _resample_zyx(work, resampled_shape)

    sct = predictor.predict(
        work.cpu().numpy(), tile_step_size=tile_step_size, mirror_axes=mirror_axes)

    # Back onto the input grid.
    sct_t = torch.from_numpy(sct).to(dev)
    if tuple(sct_t.shape) != tuple(cropped.shape):
        sct_t = _resample_zyx(sct_t, cropped.shape)
    sct_t = sct_t.clamp(*SCT_HU_RANGE)

    out = torch.full(full_shape, SCT_HU_RANGE[0], dtype=torch.float32, device=dev)
    out[bbox] = sct_t

    if was_tensor:
        return out.to(src_device)
    return out.cpu().numpy()


# ============================================================================
# Model Export (dev-only; requires nnUNet)
# ============================================================================

def export_nnunet_model_to_standalone(
    checkpoint_path: Union[str, Path],
    output_dir: Union[str, Path],
    example_input_shape: Optional[Tuple[int, ...]] = None,
    device: Union[str, torch.device] = "cuda",
):
    """Export a trained nnUNet regression checkpoint to ``model.pt`` +
    ``metadata.json``. Only needed to (re-)create a bundle; inference above
    does not import nnUNet.
    """
    try:
        import nnunetv2  # type: ignore
        from nnunetv2.utilities.plans_handling.plans_handler import PlansManager  # type: ignore
        from nnunetv2.utilities.find_class_by_name import recursive_find_python_class  # type: ignore
        from batchgenerators.utilities.file_and_folder_operations import load_json, join  # type: ignore
    except ImportError as e:
        raise ImportError("nnUNet must be installed to export models: "
                          "pip install nnunetv2") from e

    checkpoint_path = Path(checkpoint_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(device) if isinstance(device, str) else device

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    trainer_name = checkpoint['trainer_name']
    init_args = checkpoint['init_args']
    network_weights = checkpoint['network_weights']

    model_dir = checkpoint_path.parent.parent
    plans = load_json(str(model_dir / 'plans.json'))
    dataset_json = load_json(str(model_dir / 'dataset.json'))

    plans_manager = PlansManager(plans)
    config_manager = plans_manager.get_configuration(init_args['configuration'])
    patch_size = config_manager.patch_size
    norm_stats = plans_manager.foreground_intensity_properties_per_channel
    norm_schemes = config_manager.normalization_schemes

    trainer_class = recursive_find_python_class(
        folder=join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
        class_name=trainer_name,
        current_module="nnunetv2.training.nnUNetTrainer",
    )
    if trainer_class is None:
        raise RuntimeError(f'Unable to locate trainer class {trainer_name}.')

    network = trainer_class.build_network_architecture(
        architecture_class_name=config_manager.network_arch_class_name,
        arch_init_kwargs=config_manager.network_arch_init_kwargs,
        arch_init_kwargs_req_import=config_manager.network_arch_init_kwargs_req_import,
        num_input_channels=1, num_output_channels=1, enable_deep_supervision=False,
    )
    network.load_state_dict(network_weights)
    network = network.to(device).eval()

    if example_input_shape is None:
        example_input_shape = tuple(patch_size)
    with torch.no_grad():
        traced_model = torch.jit.trace(
            network, torch.randn(1, 1, *example_input_shape, device=device))
    torch.jit.save(traced_model, str(output_dir / "model.pt"))

    metadata = {
        "model_info": {
            "trainer_name": trainer_name,
            "dataset_name": dataset_json.get('name', 'Unknown'),
            "configuration": init_args['configuration'],
            "fold": init_args['fold'],
        },
        "inference_config": {
            "patch_size": list(patch_size),
            "tile_step_size": 0.5,
            "use_gaussian": True,
            # Not consumed by nnUNet, but the sCT wrapper needs it: the grid
            # the network was trained on, in array (z, y, x) order.
            "target_spacing": list(config_manager.spacing),
        },
        "normalization": {},
    }
    for channel_idx, channel_name in [(0, 'input'), (1, 'output')]:
        if str(channel_idx) in norm_stats:
            stats = norm_stats[str(channel_idx)]
            scheme = (norm_schemes[channel_idx] if channel_idx < len(norm_schemes)
                      else 'ZScoreNormalization')
            metadata['normalization'][channel_name] = {
                'scheme': scheme,
                'mean': float(stats.get('mean', 0)),
                'std': float(stats.get('std', 1)),
                'percentile_00_5': float(stats.get('percentile_00_5', 0)),
                'percentile_99_5': float(stats.get('percentile_99_5', 0)),
                'median': float(stats.get('median', 0)),
                'min': float(stats.get('min', 0)),
                'max': float(stats.get('max', 0)),
            }

    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    StandaloneRegressionInference(output_dir, device=device)
    print(f"Model bundle written to {output_dir.absolute()}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Export an nnUNet regression model to a standalone bundle")
    parser.add_argument('-i', '--input', required=True,
                        help="nnUNet checkpoint (e.g. fold_0/checkpoint_best.pth)")
    parser.add_argument('-o', '--output', required=True, help="Output bundle directory")
    parser.add_argument('--input-shape', type=int, nargs='+',
                        help="Trace input shape (default: the config's patch size)")
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()

    export_nnunet_model_to_standalone(
        checkpoint_path=args.input, output_dir=args.output,
        example_input_shape=tuple(args.input_shape) if args.input_shape else None,
        device=args.device,
    )
