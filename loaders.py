import json
import pydosert as PDRT
import numpy as np
import torch
import SimpleITK as sitk
import matplotlib.pyplot as plt
from configs import machine_config, default_data_path
from materials_v2 import relative_zoa_from_hu
import logging
from natsort import natsorted
from torch.utils.data import Dataset
from scipy.ndimage import binary_fill_holes, label as cc_label
import os
import torch.nn.functional as F
import math
import torch
from torch.nn.functional import max_pool2d, max_pool3d
logging.getLogger().setLevel(logging.ERROR)
os.environ["TQDM_DISABLE"] = "1"

# Challenge HU -> density lookup table, hard-coded from the DoseRAD2026
# beam_parameters.json so we never depend on the file being present at
# runtime (a missing file used to silently fall back to pydosert's
# default conversion). HU is strictly increasing; density is
# intentionally non-monotonic at the segment boundaries (-200/-199,
# -10/-9, 120/121) — np.interp only needs monotonic x, which holds.
_CHALLENGE_HU = np.array(
    [-1024, -999, -200, -199, -10, -9, 120, 121, 3000, 4000],
    dtype=np.float64,
)
_CHALLENGE_DENSITY = np.array(
    [1.200000e-03, 1.210000e-03, 8.043754e-01, 8.183035e-01,
     1.006579e+00, 9.966749e-01, 1.126553e+00, 1.095097e+00,
     3.027294e+00, 3.698428e+00],
    dtype=np.float64,
)


def convert_HU_to_density_lut(hu_volume):
    """Piecewise-linear HU→density conversion using the challenge LUT.

    Uses the hard-coded challenge table; values outside [-1024, 4000]
    clamp to the endpoint densities (np.interp default).
    """
    if isinstance(hu_volume, torch.Tensor):
        np_vol = hu_volume.detach().cpu().numpy()
        out = np.interp(np_vol, _CHALLENGE_HU, _CHALLENGE_DENSITY).astype(np_vol.dtype, copy=False)
        return torch.from_numpy(out).to(hu_volume.device)
    return np.interp(np.asarray(hu_volume), _CHALLENGE_HU, _CHALLENGE_DENSITY)


def convert_HU_to_electron_density_lut(hu_volume):
    """Piecewise HU → water-relative ELECTRON density.

    Drop-in replacement for :func:`convert_HU_to_density_lut` wherever the
    quantity is used for photon TRANSPORT (radiological depth, TERMA,
    kernel scaling).

    At the challenge spectrum -- fluence-weighted mean 1.354 MeV, 2.1% of
    fluence below 200 keV -- Compton dominates, and the Compton mass
    attenuation coefficient is proportional to Z/A. So

        mu = (mu/rho) * rho  ~  rho * (Z/A)  =  electron density,

    not mass density. Mass-density transport therefore over-attenuates and
    over-deposits in bone, where Z/A is 5-8% below water's.

    Built from HU, NEVER by transforming density: the challenge density LUT is
    non-monotonic (it steps DOWN at HU -10 -> -9 and 120 -> 121), so density is
    not invertible and a density-keyed correction would alias.

    The returned volume equals ``convert_HU_to_density_lut(hu) *
    relative_zoa_from_hu(hu)`` exactly -- one dimensionless factor and nothing
    else -- so an electron-density run isolates the hypothesis cleanly. Water
    (relative Z/A == 1 by definition) is unchanged, hence a water-phantom
    commissioning is invariant to this switch.
    """
    if isinstance(hu_volume, torch.Tensor):
        np_vol = hu_volume.detach().cpu().numpy()
        out = (np.interp(np_vol, _CHALLENGE_HU, _CHALLENGE_DENSITY)
               * relative_zoa_from_hu(np_vol)).astype(np_vol.dtype, copy=False)
        return torch.from_numpy(out).to(hu_volume.device)
    hu = np.asarray(hu_volume)
    return np.interp(hu, _CHALLENGE_HU, _CHALLENGE_DENSITY) * relative_zoa_from_hu(hu)


def convert_HU_to_transport_density(hu_volume, use_electron_density=False):
    """HU → the density the dose engine should transport with.

    ``use_electron_density=False`` (default) keeps v1 behaviour bit-for-bit so
    the shipped checkpoint and existing training stay reproducible.
    """
    if use_electron_density:
        return convert_HU_to_electron_density_lut(hu_volume)
    return convert_HU_to_density_lut(hu_volume)

def get_patients(data_path, modality, cohort, prefixes):
    modality_path = os.path.join(data_path, modality + "/" + cohort)
    patients = os.listdir(modality_path)
    patients = [patient for patient in patients if any(patient.startswith(p) for p in prefixes)]
    patients = natsorted(patients)

    return patients

# Per-worker single-entry caches for the per-patient artefacts. With
# SameBeamBatchSampler each batch's CPs share a patient (same CT, same
# plan JSON), and consecutive batches of the same group also share a
# patient — so a one-slot LRU keyed on the file path eliminates 8x
# redundant decompression + JSON parsing per batch on the worker side.
# These are module-level dicts; each DataLoader worker is a separate
# process and gets its own copy.
_CT_CACHE: dict = {"path": None, "ct": None, "origin": None, "spacing": None}
_MR_CACHE: dict = {"path": None, "array": None, "origin": None, "spacing": None}
_SCT_CACHE: dict = {"path": None, "array": None, "origin": None, "spacing": None}
_BEAMS_CACHE: dict = {"path": None, "beams": None}


def _read_image_retry(path, attempts=6):
    """sitk.ReadImage with retries + jittered exponential backoff. Concurrent reads of the
    SAME file from multiple cluster jobs (e.g. a hyperparameter sweep sharing the dataset)
    can transiently fail in the ITK/GDCM IO layer; retry rather than crash the run."""
    import time, random
    last = None
    for i in range(attempts):
        try:
            return sitk.ReadImage(path)
        except Exception as e:
            last = e
            time.sleep(0.1 * (2 ** i) + random.uniform(0.0, 0.2))
    raise RuntimeError(f"failed to read {path} after {attempts} attempts: {last!r}")


# The photon cohort is 2 mm isotropic (the challenge's own preprocessing
# resamples every patient onto that grid, and evaluation/doserad2026_evaluator
# scores on it). We nevertheless read the spacing out of each image rather than
# assuming it — see image_spacing_zyx.
EXPECTED_RESOLUTION = (2.0, 2.0, 2.0)
_SPACING_WARNED: set = set()


def image_spacing_zyx(image, path=""):
    """Voxel spacing of a sitk image in ARRAY order (z, y, x), in mm.

    Snapped to EXPECTED_RESOLUTION when it agrees to within 1e-3 mm, so float
    noise in the header can't perturb the engine's grid; anything genuinely
    different is used as-is but warned about once, because the whole pipeline
    (kernel size, crop margins, absolute gain) was commissioned at 2 mm.
    """
    spacing = tuple(float(v) for v in reversed(image.GetSpacing()))
    if all(abs(a - b) < 1e-3 for a, b in zip(spacing, EXPECTED_RESOLUTION)):
        return EXPECTED_RESOLUTION
    if path not in _SPACING_WARNED:
        _SPACING_WARNED.add(path)
        print(f"[loaders] {path or 'image'} has spacing (z,y,x)={spacing}, "
              f"not the expected {EXPECTED_RESOLUTION}; using it as given.", flush=True)
    return spacing


def load_mr(base_path):
    """MR volume, origin (z, y, x) and voxel spacing (z, y, x)."""
    mr_path = f"{base_path}/image/mr.mha"
    if _MR_CACHE["path"] != mr_path:
        image = _read_image_retry(mr_path)
        origin = image.GetOrigin()
        origin = np.array([origin[2], origin[1], origin[0]])
        array = sitk.GetArrayFromImage(image)
        _MR_CACHE["path"] = mr_path
        _MR_CACHE["array"] = array
        _MR_CACHE["origin"] = origin
        _MR_CACHE["spacing"] = image_spacing_zyx(image, mr_path)
    return _MR_CACHE["array"], _MR_CACHE["origin"], _MR_CACHE["spacing"]


def sct_cache_path(base_path):
    """Where the precomputed synthetic CT for this patient lives.

    Next to the MR by default; set ``DOSERAD_SCT_CACHE`` to redirect to a
    writable scratch root when the dataset mount is read-only (the last two
    path components — ``<cohort>/<patient>`` — are preserved there).
    """
    root = os.environ.get("DOSERAD_SCT_CACHE")
    if root:
        parts = os.path.normpath(base_path).split(os.sep)
        return os.path.join(root, *parts[-2:], "sct.mha")
    return f"{base_path}/image/sct.mha"


_SCT_FALLBACK_WARNED = False


def load_sct(base_path):
    """Synthetic CT (HU) for an MR patient, on the MR's own grid.

    The MR arm of the challenge is just "MR -> sCT, then the CT pipeline":
    this is the single place the conversion happens. Reads the precomputed
    volume written by ``precompute_sct.py``; if it is missing, generates it
    inline (slow — the 400 MB bundle gets loaded per DataLoader worker) and
    tries to write it for next time.
    """
    global _SCT_FALLBACK_WARNED
    cache_path = sct_cache_path(base_path)
    if _SCT_CACHE["path"] == cache_path:
        return _SCT_CACHE["array"], _SCT_CACHE["origin"], _SCT_CACHE["spacing"]

    mr_array, origin, spacing = load_mr(base_path)

    if os.path.exists(cache_path):
        array = sitk.GetArrayFromImage(_read_image_retry(cache_path)).astype(np.float32)
        if array.shape != tuple(mr_array.shape):
            raise ValueError(
                f"cached sCT {cache_path} has shape {array.shape}, "
                f"but the MR is {mr_array.shape} — regenerate it")
    else:
        # Inside a DataLoader worker we CANNOT generate the sCT: workers are
        # forked, and torch.jit.load(map_location="cuda") then dies with
        # "Cannot re-initialize CUDA in forked subprocess". Running it on CPU
        # instead would be worse — a 102 M-parameter 3-D U-Net over ~150 patches
        # per patient, every epoch. So fail immediately with the fix, rather
        # than crashing the job 20 minutes in with a CUDA traceback.
        try:
            from torch.utils.data import get_worker_info
            in_worker = get_worker_info() is not None
        except Exception:
            in_worker = False
        if in_worker:
            raise RuntimeError(
                f"No cached synthetic CT at {cache_path}.\n"
                f"It cannot be generated inside a DataLoader worker (forked "
                f"processes cannot initialise CUDA). Precompute it first:\n\n"
                f"    python3 precompute_sct.py --data_path <DATA_PATH> "
                f"--cohort validating\n\n"
                f"Add --cache <writable dir> (or set DOSERAD_SCT_CACHE) if the "
                f"dataset mount is read-only.")
        if not _SCT_FALLBACK_WARNED:
            print(f"[sct] no cached sCT at {cache_path}; generating inline. "
                  f"Run precompute_sct.py once to avoid this.", flush=True)
            _SCT_FALLBACK_WARNED = True
        from sct import mr_to_synthetic_ct
        array = mr_to_synthetic_ct(mr_array, spacing=spacing).astype(np.float32)
        write_sct(cache_path, array, f"{base_path}/image/mr.mha", strict=False)

    _SCT_CACHE["path"] = cache_path
    _SCT_CACHE["array"] = array
    _SCT_CACHE["origin"] = origin
    _SCT_CACHE["spacing"] = spacing
    return array, origin, spacing


def write_sct(cache_path, array, reference_mr_path, strict=True):
    """Write an sCT volume, copying the MR's geometry so it overlays exactly."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        image = sitk.GetImageFromArray(array.astype(np.float32))
        image.CopyInformation(_read_image_retry(reference_mr_path))
        sitk.WriteImage(image, cache_path, useCompression=True)
    except Exception as exc:
        if strict:
            raise
        print(f"[sct] could not cache sCT to {cache_path}: {exc!r}", flush=True)


def load_ct(base_path):
    """CT volume, origin (z, y, x) and voxel spacing (z, y, x)."""
    ct_path = f"{base_path}/image/ct.mha"
    if _CT_CACHE["path"] != ct_path:
        image = _read_image_retry(ct_path)
        origin = image.GetOrigin()
        origin = np.array([origin[2], origin[1], origin[0]])
        array = sitk.GetArrayFromImage(image)
        _CT_CACHE["path"] = ct_path
        _CT_CACHE["ct"] = array
        _CT_CACHE["origin"] = origin
        _CT_CACHE["spacing"] = image_spacing_zyx(image, ct_path)
    return _CT_CACHE["ct"], _CT_CACHE["origin"], _CT_CACHE["spacing"]


def load_dose(base_path, beam_index, cp_index):
    # Per-CP file — no sharing across CPs, no caching.
    dose_path = f"{base_path}/dose/Dose_B{beam_index}_CP{cp_index:03d}.mha"
    image = _read_image_retry(dose_path)
    array = sitk.GetArrayFromImage(image)
    array *= 1e5
    return array


def _load_beams_cached(beam_json_path: str):
    """Worker-local cache of the per-patient plan JSON's ``beams`` list."""
    if _BEAMS_CACHE["path"] != beam_json_path:
        import time, random
        for _i in range(6):
            try:
                with open(beam_json_path) as f:
                    _BEAMS_CACHE["beams"] = json.load(f)["beams"]
                break
            except Exception:
                time.sleep(0.1 * (2 ** _i) + random.uniform(0.0, 0.2))
        else:
            with open(beam_json_path) as f:
                _BEAMS_CACHE["beams"] = json.load(f)["beams"]
        _BEAMS_CACHE["path"] = beam_json_path
    return _BEAMS_CACHE["beams"]


def estimate_h_crop_from_geometry(jaw_lower_mm, jaw_upper_mm, iso_z_mm, H, res_h, margin_mm=30.0):
    """H-axis crop (h_start, h_end) computed from plan geometry only.

    Takes explicit jaw extents (in mm at iso along the leaf-axis
    direction) plus the patient-z iso position; pads by ``margin_mm`` on
    each side. With per-beam (union) jaw extents the result is the
    same for every CP of a beam — required for batched-CP training
    because pydosert.BeamSequence demands a shared iso, which means a
    shared crop.
    """
    h_lower_mm = float(iso_z_mm) + float(jaw_lower_mm) - margin_mm
    h_upper_mm = float(iso_z_mm) + float(jaw_upper_mm) + margin_mm
    h_start = max(0, int(np.floor(h_lower_mm / res_h)))
    h_end = min(H, int(np.ceil(h_upper_mm / res_h)))
    if h_end <= h_start:
        return 0, H
    return h_start, h_end


def body_cylinder_radius_mm(ct_volume, resolution, iso_center, air_hu=-500.0):
    """Largest axial distance from the isocentre to any non-air voxel, in mm.

    The engine rotates each (D, W) slice about the isocentre, so only the
    INSCRIBED CIRCLE of the axial plane survives every gantry angle; anything
    further out leaves the array for some angles and is silently lost.

    Measured on the six validating patients, the default padding clips 0.385%
    of body voxels overall and 1.08% on the worst (1ABB045, whose body reaches
    307 mm against a 275 mm cylinder). Little dose lives there -- 0.41% of plan
    dose and 0.54% of scored voxels on that patient -- but the hidden test set
    is unseen, and a patient whose arms sit further out would be clipped
    harder with no warning.

    Args:
        ct_volume: (H, D, W) HU array.
        resolution: (res_H, res_D, res_W) mm/voxel.
        iso_center: (iso_H, iso_D, iso_W) mm from the grid origin.
    Returns:
        Radius in mm, or 0.0 if the volume is entirely air.
    """
    arr = ct_volume.numpy() if isinstance(ct_volume, torch.Tensor) else np.asarray(ct_volume)
    body = arr > air_hu
    if not body.any():
        return 0.0
    _, D, W = arr.shape
    res_H, res_D, res_W = resolution
    # Collapse H: a voxel's axial radius does not depend on the slice.
    any_dw = body.any(axis=0)                       # (D, W)
    d_idx, w_idx = np.nonzero(any_dw)
    dd = (d_idx + 0.5) * res_D - iso_center[1]
    ww = (w_idx + 0.5) * res_W - iso_center[2]
    return float(np.sqrt(dd * dd + ww * ww).max())


def pad_and_crop_to_iso_center(volumes, resolution, iso_center, fill_value=0, h_bounds=None,
                               min_cylinder_radius_mm=None):
    """
    For the axial (D, W) plane: pad with zeros so iso_center lands at the
    geometric centre of the slice.
    For the H dimension: crop to ``h_bounds = (h_start, h_end)`` — must be
    supplied by the caller. We deliberately do NOT use the GT dose for
    this any more: at submission time the GT dose doesn't exist, so the
    crop must come from plan inputs (MLC/jaws/iso/SAD), computed once in
    ``load_beam_segment`` via ``estimate_h_crop_from_geometry``.

    Args:
        volumes: a single tensor/ndarray of shape (H, D, W), or a sequence.
        resolution: (res_H, res_D, res_W) in mm/voxel.
        iso_center: (iso_H, iso_D, iso_W) in mm, relative to the origin.
        fill_value: scalar applied to all volumes, or a sequence matching
            `volumes` giving the constant-pad value per volume
            (e.g. -1000 for HU-scale CT, 0 for dose).
        h_bounds: (h_start, h_end) inclusive/exclusive voxel indices on
            the H axis. REQUIRED — there is no dose-based fallback.
        min_cylinder_radius_mm: if given, D and W are padded SYMMETRICALLY
            further so the inscribed circle min(D, W)/2 covers this radius.
            Use body_cylinder_radius_mm() so no patient tissue can rotate out
            of the array. Padding stays symmetric about the isocentre, which
            the rotation requires, and is recorded in pad_info so
            crop_and_pad_to_original() still inverts exactly.
    Returns:
        processed, new_iso_center, pad_info.
        pad_info stores H crop as negative padding values so that
        crop_and_pad_to_original() can perfectly invert this operation.
    """
    single = not isinstance(volumes, (list, tuple))
    vol_list = [volumes] if single else list(volumes)
    if isinstance(fill_value, (list, tuple)):
        fills = list(fill_value)
        if len(fills) != len(vol_list):
            raise ValueError(
                f"fill_value length {len(fills)} != volumes length {len(vol_list)}"
            )
    else:
        fills = [fill_value] * len(vol_list)

    H, D, W = vol_list[0].shape[-3:]
    for v in vol_list:
        if v.shape[-3:] != (H, D, W):
            raise ValueError(f"Shape mismatch: {v.shape} vs {(H, D, W)}")

    # ------------------------------------------------------------------ #
    # H dimension: caller-supplied bounds. h_bounds=None skips the crop  #
    # entirely (load_beam_segment defers the H-crop to the collate, so   #
    # every batch can crop to its own union jaw extent).                 #
    # ------------------------------------------------------------------ #
    if h_bounds is None:
        h_start, h_end = 0, H
    else:
        h_start = max(0, int(h_bounds[0]))
        h_end = min(H, int(h_bounds[1]))
        if h_end <= h_start:
            # Degenerate / out-of-range — keep the full axis so we
            # don't crash; downstream metrics will surface the issue.
            h_start, h_end = 0, H

    # Stored as negative padding (crop = negative pad)
    pad_H_before = -(h_start)           # how many rows removed from top
    pad_H_after  = -(H - h_end)         # how many rows removed from bottom

    # ------------------------------------------------------------------ #
    # D / W dimensions: pad so iso_center lands at centre                 #
    # ------------------------------------------------------------------ #
    res_H, res_D, res_W = resolution
    iso_H, iso_D, iso_W = iso_center

    iso_d_vox = iso_D / res_D
    iso_w_vox = iso_W / res_W

    diff_d = 2.0 * iso_d_vox - D
    if diff_d >= 0:
        pad_D_before, pad_D_after = 0, int(math.ceil(diff_d))
    else:
        pad_D_before, pad_D_after = int(math.ceil(-diff_d)), 0

    diff_w = 2.0 * iso_w_vox - W
    if diff_w >= 0:
        pad_W_before, pad_W_after = 0, int(math.ceil(diff_w))
    else:
        pad_W_before, pad_W_after = int(math.ceil(-diff_w)), 0

    # ------------------------------------------------------------------ #
    # Grow the cylinder so no tissue rotates out of the array              #
    # ------------------------------------------------------------------ #
    if min_cylinder_radius_mm:
        # After the centring pad the isocentre is at the middle of D and W, so
        # the inscribed circle is min(D, W)/2. Grow BOTH axes symmetrically --
        # asymmetric padding would move the isocentre off centre and the
        # rotation would no longer be about it.
        D_c = D + pad_D_before + pad_D_after
        W_c = W + pad_W_before + pad_W_after
        # +2 voxels on each axis: the radius is measured to voxel CENTRES, so
        # the outermost voxel's far corner sits up to one voxel beyond it.
        need_D = int(math.ceil(2.0 * min_cylinder_radius_mm / res_D)) + 2 - D_c
        need_W = int(math.ceil(2.0 * min_cylinder_radius_mm / res_W)) + 2 - W_c
        if need_D > 0:
            half = int(math.ceil(need_D / 2.0))
            pad_D_before += half
            pad_D_after += half
        if need_W > 0:
            half = int(math.ceil(need_W / 2.0))
            pad_W_before += half
            pad_W_after += half

    # ------------------------------------------------------------------ #
    # Apply transforms                                                     #
    # ------------------------------------------------------------------ #
    def _process(v, fv):
        # 1. Crop H
        v = v[..., h_start:h_end, :, :]

        # 2. Pad D and W
        if isinstance(v, np.ndarray):
            pad_widths = [(0, 0)] * (v.ndim - 3) + [
                (0, 0),
                (pad_D_before, pad_D_after),
                (pad_W_before, pad_W_after),
            ]
            return np.pad(v, pad_widths, mode="constant", constant_values=fv)
        if isinstance(v, torch.Tensor):
            pad_arg = (pad_W_before, pad_W_after, pad_D_before, pad_D_after, 0, 0)
            return torch.nn.functional.pad(v, pad_arg, mode="constant", value=float(fv))
        raise TypeError(f"Unsupported volume type: {type(v)}")

    processed = [_process(v, fv) for v, fv in zip(vol_list, fills)]

    new_iso_center = (
        iso_H - h_start * res_H,        # shift origin due to H crop
        iso_D + pad_D_before * res_D,
        iso_W + pad_W_before * res_W,
    )

    pad_info = {
        # H: stored as NEGATIVE values (crop = negative pad)
        "pad_H_before": pad_H_before,   # <= 0
        "pad_H_after":  pad_H_after,    # <= 0
        # D / W: stored as POSITIVE values (actual padding)
        "pad_D_before": pad_D_before,
        "pad_D_after":  pad_D_after,
        "pad_W_before": pad_W_before,
        "pad_W_after":  pad_W_after,
        "original_shape": (H, D, W),
        "fill_values": fills,           # needed to invert with correct fill
    }

    if single:
        return processed[0], new_iso_center, pad_info
    return (tuple(processed) if isinstance(volumes, tuple) else processed), new_iso_center, pad_info


def crop_and_pad_to_original(volumes, pad_info, fill_value=None):
    """Invert pad_and_crop_to_iso_center:
      - H axis: was cropped (negative pad) → pad it back with the stored fill value.
      - D / W axes: were padded (positive pad) → crop them back.

    Args:
        volumes: a single volume or list/tuple of volumes to invert.
        pad_info: the pad_info dict returned by pad_and_crop_to_iso_center.
        fill_value: optional override for the H-padding fill. When provided,
            this single value is used for every input volume; when omitted,
            we fall back to ``pad_info["fill_values"]`` positionally. The
            override matters for single-volume calls — without it, the
            function would silently use ``fill_values[0]`` (which is set
            for the CT, HU=-1000) for whatever volume you passed, which
            would, e.g., paint dose with -1000 in the cropped slices and
            make the overlay solid blue instead of transparent.
    Returns:
        restored volume(s) in the same container type as input.
    """
    single = not isinstance(volumes, (list, tuple))
    vol_list = [volumes] if single else list(volumes)

    if fill_value is not None:
        fills = [fill_value] * len(vol_list)
    else:
        fills = pad_info.get("fill_values", [0] * len(vol_list))
        if len(fills) < len(vol_list):              # graceful fallback
            fills = fills + [0] * (len(vol_list) - len(fills))

    # H crop → pad back  (pad_H_before / after are <= 0)
    h_crop_before = -pad_info["pad_H_before"]   # >= 0 rows to re-add at top
    h_crop_after  = -pad_info["pad_H_after"]    # >= 0 rows to re-add at bottom

    # D / W pad → crop back
    D_before = pad_info["pad_D_before"]
    W_before = pad_info["pad_W_before"]
    _, D_orig, W_orig = pad_info["original_shape"]
    # Symmetric lateral crop applied AFTER the centring pad (see
    # crop_lateral_to_body): undo it first by padding those voxels back.
    lat_d = int(pad_info.get("lat_crop_D", 0))
    lat_w = int(pad_info.get("lat_crop_W", 0))

    def _invert(v, fv):
        # 0. Undo the symmetric lateral crop
        if lat_d or lat_w:
            if isinstance(v, np.ndarray):
                pad_widths = [(0, 0)] * (v.ndim - 3) + [(0, 0), (lat_d, lat_d), (lat_w, lat_w)]
                v = np.pad(v, pad_widths, mode="constant", constant_values=fv)
            elif isinstance(v, torch.Tensor):
                v = torch.nn.functional.pad(v, (lat_w, lat_w, lat_d, lat_d, 0, 0),
                                            mode="constant", value=float(fv))
            else:
                raise TypeError(f"Unsupported volume type: {type(v)}")

        # 1. Crop D and W back to original
        v = v[..., :, D_before:D_before + D_orig, W_before:W_before + W_orig]

        # 2. Pad H back
        if h_crop_before == 0 and h_crop_after == 0:
            return v
        if isinstance(v, np.ndarray):
            pad_widths = [(0, 0)] * (v.ndim - 3) + [
                (h_crop_before, h_crop_after),
                (0, 0),
                (0, 0),
            ]
            return np.pad(v, pad_widths, mode="constant", constant_values=fv)
        if isinstance(v, torch.Tensor):
            # torch.nn.functional.pad order: last dim first
            pad_arg = (0, 0, 0, 0, h_crop_before, h_crop_after)
            return torch.nn.functional.pad(v, pad_arg, mode="constant", value=float(fv))
        raise TypeError(f"Unsupported volume type: {type(v)}")

    restored = [_invert(v, fv) for v, fv in zip(vol_list, fills)]

    if single:
        return restored[0]
    return tuple(restored) if isinstance(volumes, tuple) else restored

def crop_lateral_to_body(vol_list, pad_info, body_hu_volume, margin_mm=15.0,
                         resolution=(2.0, 2.0, 2.0), air_hu=-500.0):
    """Crop the D and W axes symmetrically about the centre, down to the body
    extent plus ``margin_mm``.

    NEGATIVE RESULT — off by default, kept only so the reasoning is not lost.
    The idea was that pad_and_crop_to_iso_center PADS D and W so the isocentre
    lands at the geometric centre, which can nearly double a lateral axis, and
    the correction U-Net (84% of the forward pass) pays for every air voxel.
    Cropping to the body's axis-aligned extent is ~1.75x fewer voxels — but it
    is WRONG: it silently destroys the dose at oblique gantry angles (verified,
    max error ~100% of peak) because the BEV rotation needs the box to contain
    the ROTATED body. Making it rotation-safe means cropping to the body's
    radius, which recovers essentially nothing (measured 1.03x, dose-neutral to
    4e-5). Patient-space cropping is a dead end.

    The saving is real but has to be taken in BEV space instead, where the beam
    is axis-aligned and the crop can follow the field: that means cropping
    between the rotation and the corrector inside the engine forward, not here.

    Two constraints make this trickier than a bounding box:

    1. It MUST stay symmetric about the centre, or the isocentre stops being
       centred and the rotation layer misaligns.
    2. It MUST be to the body's RADIUS in the D-W plane, not its per-axis
       extent. The engine rotates the volume into beam's-eye-view, and a
       rotated body only fits inside a box that is at least its diagonal wide.
       Cropping to the axis-aligned extent looks fine at gantry 0/90/180 and
       silently destroys the dose at oblique angles — it clips the corners the
       rotation brings into view.

    Mutates ``pad_info`` with lat_crop_D / lat_crop_W so
    crop_and_pad_to_original inverts it.
    """
    body = body_hu_volume > air_hu
    if not bool(body.any()):
        return vol_list
    body_np = np.asarray(body)
    D, W = body_np.shape[-2], body_np.shape[-1]
    cd, cw = D // 2, W // 2
    # Radial extent of the body about the centre, in mm — the radius the BEV
    # rotation has to be able to sweep without clipping.
    proj = body_np.any(axis=0)                       # [D, W]
    dd = (np.nonzero(proj.any(axis=1))[0] - cd) * resolution[1]
    ww = (np.nonzero(proj.any(axis=0))[0] - cw) * resolution[2]
    if dd.size == 0 or ww.size == 0:
        return vol_list
    radius_mm = float(np.hypot(np.abs(dd).max(), np.abs(ww).max())) + margin_mm
    half_d = min(int(np.ceil(radius_mm / resolution[1])), cd)
    half_w = min(int(np.ceil(radius_mm / resolution[2])), cw)
    lat_d, lat_w = cd - half_d, cw - half_w
    if lat_d <= 0 and lat_w <= 0:
        return vol_list
    lat_d, lat_w = max(lat_d, 0), max(lat_w, 0)
    sl = (Ellipsis, slice(lat_d, D - lat_d) if lat_d else slice(None),
          slice(lat_w, W - lat_w) if lat_w else slice(None))
    pad_info["lat_crop_D"] = lat_d
    pad_info["lat_crop_W"] = lat_w
    return [v[sl] for v in vol_list]


def load_beam_segment(data_path, patient_id, beam_index, cp_index, modality="ct",
                      crop_margin_mm=30.0, lateral_margin_mm=None,
                      use_electron_density=False):
    dose = load_dose(f"{data_path}/{patient_id}", beam_index, cp_index)
    # The MR arm differs from the CT arm in exactly one place: the MR is
    # converted to a synthetic CT first. From here on the two are the same
    # code path — same HU units, same air fill, same density LUT, same
    # corrector.
    if modality == "ct":
        ct, origin, resolution = load_ct(f"{data_path}/{patient_id}")
    elif modality == "mr":
        ct, origin, resolution = load_sct(f"{data_path}/{patient_id}")
    else:
        raise ValueError(f"Unknown modality: {modality}")
    pad_value = -1000

    beam_json = f"{data_path}/{patient_id}/{patient_id}.json"
    beams_raw = _load_beams_cached(beam_json)
    beam_raw = beams_raw[beam_index]
    beam_segment_raw = [bs for bs in beam_raw["control_points"] if bs["cp_idx"] == cp_index][0]

    num_leaf_pairs = beam_raw["num_mlc_leaf_pairs"]
    SAD = beam_raw["SAD"]
    raw_iso_center = np.array(beam_raw["iso_center"])
    raw_iso_center = np.array([raw_iso_center[2], raw_iso_center[1], raw_iso_center[0]])
    original_iso_center = raw_iso_center - origin
    gantry_angle = beam_segment_raw["gantry_angle"]

    # Build the beam segment FIRST so we have jaw_positions before
    # cropping: the H-axis crop is now derived from plan geometry
    # (MLC opening pattern) instead of the GT dose, so the same code
    # path works at submission time when the GT doesn't exist.
    mlc_left = np.array(beam_segment_raw["mlc_left_int_mm"])
    mlc_right = np.array(beam_segment_raw["mlc_right_int_mm"])
    mlc_positions = torch.from_numpy(np.stack([mlc_left, mlc_right], axis=1)).to(torch.float32)

    beam_segment = PDRT.Beam.create(
        gantry_angle_deg=gantry_angle,
        number_of_leaf_pairs=num_leaf_pairs,
        iso_center=tuple(original_iso_center),
        device="cpu"
    )
    beam_segment.leaf_positions = mlc_positions

    jaw_openings_px = (mlc_positions != 0).any(dim=1).nonzero(as_tuple=True)[0]
    if jaw_openings_px.numel() > 0:
        jaw_lower_px = jaw_openings_px[0].item()
        jaw_upper_px = jaw_openings_px[-1].item() + 1
        jaw_lower = (jaw_lower_px - mlc_positions.shape[0] // 2) * 5
        jaw_upper = (jaw_upper_px - mlc_positions.shape[0] // 2) * 5
    else:
        # Fully-closed MLC (no opening): degenerate beam segment; keep
        # a tiny placeholder so downstream code doesn't divide by zero.
        jaw_lower, jaw_upper = -2.5, 2.5
    beam_segment.jaw_positions = torch.tensor([jaw_lower, jaw_upper], dtype=torch.float32)

    # H-crop per CP using THIS CP's own jaw extent. Keeps the volumes
    # going through the DataLoader's worker→main pipe small, which is
    # what gives the GPU something to chew on. The collate aligns the
    # per-CP crops into a shared batch extent by padding (cheap),
    # which is also what keeps pydosert.BeamSequence happy with a
    # single shared iso per batch.
    H_full = ct.shape[0]
    h_bounds = estimate_h_crop_from_geometry(
        float(beam_segment.jaw_positions[0]),
        float(beam_segment.jaw_positions[1]),
        float(original_iso_center[0]),
        H=H_full, res_h=resolution[0], margin_mm=crop_margin_mm,
    )
    ct_t = torch.from_numpy(ct).float()
    dose_t = torch.from_numpy(dose).float()
    # Grow the axial plane so the WHOLE patient stays inside the rotation
    # cylinder. Computed per patient from the CT rather than assumed: the
    # hidden test set is unseen, and a patient whose arms sit further out than
    # any validating patient would be clipped silently.
    cyl_r = body_cylinder_radius_mm(ct_t, resolution, tuple(original_iso_center))
    (ct_pad, dose_pad), iso_center, pad_info = pad_and_crop_to_iso_center(
        [ct_t, dose_t],
        resolution=resolution,
        iso_center=tuple(original_iso_center),
        fill_value=[pad_value, 0],
        h_bounds=h_bounds,
        min_cylinder_radius_mm=cyl_r,
    )
    # Drop the all-air margin the centring pad created. Symmetric about the
    # centre, so the isocentre stays centred and the BEV rotation is unaffected
    # — but its COORDINATE moves, because voxels were removed from the low side
    # of D and W. iso_center is in mm relative to the grid origin, so it has to
    # be shifted by the cropped amount or the engine silently misplaces the beam.
    if lateral_margin_mm is not None:
        ct_pad, dose_pad = crop_lateral_to_body(
            [ct_pad, dose_pad], pad_info, ct_pad,
            margin_mm=lateral_margin_mm, resolution=resolution)
        iso_center = (
            iso_center[0],
            iso_center[1] - pad_info.get("lat_crop_D", 0) * resolution[1],
            iso_center[2] - pad_info.get("lat_crop_W", 0) * resolution[2],
        )

    beam_segment.iso_center = iso_center

    # The body mask is deliberately always derived from MASS density, even when
    # the engine transports with electron density: its 0.1 g/cm3 air threshold
    # is a segmentation heuristic, not physics, and holding it fixed keeps a
    # mass-vs-electron-density A/B comparison scored on identical voxels.
    mass_density = convert_HU_to_density_lut(ct_pad)
    external_mask = body_mask(mass_density)

    density_image = (convert_HU_to_electron_density_lut(ct_pad)
                     if use_electron_density else mass_density)

    return beam_segment, ct_pad, density_image, dose_pad, external_mask, resolution, pad_info, original_iso_center


def body_mask(volume: torch.Tensor, air_threshold: float = 0.1) -> torch.Tensor:
    """Per-slice body mask via external-air flood fill.

    The old "largest connected component" heuristic was wrong: if the
    patient's arms lie at their sides, the cc-labelled body splits into
    three blobs (torso + two arms) and only the torso survives — even
    though dose clearly passes through the arms in the GT.

    New approach, per axial slice:
      1. Threshold for air voxels: ``density < air_threshold``.
      2. Connected-component label the air within that slice.
      3. Mark any component that touches the *slice perimeter* (the 2D
         border of the slice) as "external air".
      4. Body = everything not flagged as external air.

    This keeps:
      * Arms in body (they don't touch the slice perimeter through air).
      * Bowel gas / stomach air / lungs in body (purely internal pockets).
      * Anatomy that connects to the H crop boundaries — because we
        flood-fill *per axial slice*, the H boundary is never seeded as
        air, so lung at the cropped edge doesn't leak.

    Anything that does touch the slice perimeter through a connected air
    corridor is correctly excluded — that's the actual external air.
    """
    if isinstance(volume, torch.Tensor):
        np_vol = volume.detach().cpu().numpy()
        device = volume.device
    else:
        np_vol = np.asarray(volume)
        device = None

    air = np_vol < air_threshold
    out = np.zeros_like(air, dtype=bool)

    for z in range(np_vol.shape[0]):
        sl_air = air[z]
        if not sl_air.any():
            # No air in slice -> everything is body.
            out[z] = True
            continue

        labels, n = cc_label(sl_air)
        if n == 0:
            out[z] = True
            continue

        # Collect labels touching the four edges of the slice.
        perimeter_labels = set()
        for edge in (labels[0, :], labels[-1, :], labels[:, 0], labels[:, -1]):
            perimeter_labels.update(int(v) for v in np.unique(edge) if v != 0)

        if not perimeter_labels:
            # All air is internal (rare — patient fills slice). Body is
            # the inverse of air.
            out[z] = ~sl_air
            continue

        external = np.isin(labels, list(perimeter_labels))
        out[z] = ~external

    result = torch.from_numpy(out)
    if device is not None:
        result = result.to(device)
    return result

def load_baseline(data_path, patient_id, beam_index, cp_index):
    # Imported lazily: baseline.photon pulls pyRadPlan, which the training and
    # submission paths don't need. Only this helper (baseline evaluation) does,
    # so importing loaders.py stays free of the heavy pyRadPlan dependency.
    from baseline.photon import compute_pb_dose
    return 1e5 * compute_pb_dose(
            baseline_pb_dir=data_path,
            output_dir=None,
            patient_id=patient_id,
            modality="photon",
            split="training",
            show_plots=False,
            plot_view_slice="axial",
            plot_shared_max=True,
            beam_idx=beam_index,
            cp_idx=cp_index,
        )

class BeamDataset(Dataset):
    def __init__(
        self,
        data_path,
        cohort,
        modality="ct",
        anatomies=("1THB", "1ABB"),
        patients=None,
        beam_indices=(0,),
        cp_indices=None,
        crop_margin_mm=30.0,
    ):
        self.data_path = data_path
        # modality may be a single string or a sequence — passing both puts CT
        # and MR samples in ONE dataset so a single validation pass covers both
        # arms. Every index entry carries its own modality.
        modalities = [modality] if isinstance(modality, str) else list(modality)
        for m in modalities:
            if m not in ("ct", "mr"):
                raise ValueError(f"Unsupported modality: {m}")
        if cohort not in ["training", "validating"]:
            raise ValueError(f"Unknown cohort: {cohort}")

        self.modality = modalities[0] if len(modalities) == 1 else tuple(modalities)
        self.modalities = modalities
        self.cohort = cohort
        self.crop_margin_mm = crop_margin_mm

        all_patients = get_patients(data_path, modality="photon", cohort=cohort, prefixes=tuple(anatomies))
        if patients is not None:
            patients = set(patients)
            all_patients = [p for p in all_patients if p in patients]

        if cp_indices is None:
            cp_indices = range(0, 180)

        self.index_list = [
            (patient, beam_index, cp_index, mod)
            for mod in modalities
            for patient in all_patients
            for beam_index in beam_indices
            for cp_index in cp_indices
        ]

    def __len__(self):
        return len(self.index_list)

    def __getitem__(self, idx):
        patient, beam_index, cp_index, modality = self.index_list[idx]

        beam, ct, density_image, dose, mask, resolution, pad_info, original_iso_center = load_beam_segment(
            self.data_path + "/photon/" + self.cohort,
            patient,
            beam_index,
            cp_index,
            modality=modality,
            crop_margin_mm=self.crop_margin_mm,
        )

        return {
            "beam": beam,
            "density_image": density_image,
            "image": ct,
            "dose": dose,
            "mask": mask,
            "resolution": resolution,
            "pad_info": pad_info,
            "iso_center": original_iso_center,
            "modality": modality,
            "patient": patient,
            "beam_index": beam_index,
        }

def custom_collate(batch):
    return batch


class SameBeamBatchSampler:
    """Yields batches of dataset indices that all share ``(patient,
    beam_index)``.

    pydosert.BeamSequence requires every CP in a sequence to share
    iso_center / field_size / SID, which only holds within a single
    beam. We bucket the dataset's ``index_list`` by ``(patient, beam)``,
    optionally shuffle bucket order and within-bucket order, and yield
    ``batch_size``-sized chunks from each bucket.
    """

    def __init__(self, index_list, batch_size, shuffle=True, drop_last=False):
        from collections import defaultdict
        groups = defaultdict(list)
        for i, entry in enumerate(index_list):
            p, b, _cp = entry[0], entry[1], entry[2]
            mod = entry[3] if len(entry) > 3 else "ct"
            # Key includes modality: a batch must never mix CT and MR samples,
            # they are different images even though the grid is identical.
            groups[(p, b, mod)].append(i)
        self.groups = list(groups.values())
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last

    def __iter__(self):
        """Yield same-beam chunks, but INTERLEAVED across beams.

        A chunk must stay within one (patient, beam, modality): BeamSequence
        requires every control point in a sequence to share iso_center,
        field_size and SID. Nothing requires CONSECUTIVE chunks to.

        Exhausting one bucket before starting the next meant a bucket of 180
        control points yielded 60 consecutive chunks, so with gradient
        accumulation over 8 steps, 59 of every 60 optimiser updates saw one
        patient, one beam, one anatomy -- 24 control points a couple of degrees
        apart, which is one sample's worth of information wearing a batch's
        clothing. Building every chunk first and shuffling the CHUNK ORDER
        keeps the same-beam constraint and makes each accumulation window span
        different patients, beams and anatomies.
        """
        import random
        chunks = []
        for gi in range(len(self.groups)):
            indices = list(self.groups[gi])
            if self.shuffle:
                random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                chunk = indices[i:i + self.batch_size]
                if self.drop_last and len(chunk) < self.batch_size:
                    continue
                chunks.append(chunk)
        if self.shuffle:
            random.shuffle(chunks)
        for chunk in chunks:
            yield chunk

    def __len__(self):
        total = 0
        for g in self.groups:
            n = len(g)
            if self.drop_last:
                total += n // self.batch_size
            else:
                total += (n + self.batch_size - 1) // self.batch_size
        return total


def _pad_h(volume, pre, post, fill_value):
    """Pad the H axis (axis 0) of a 3-D tensor with ``fill_value``."""
    if pre == 0 and post == 0:
        return volume
    pad_arg = (0, 0, 0, 0, int(pre), int(post))             # F.pad: last dim first
    if volume.dtype == torch.bool:
        out = F.pad(volume.to(torch.uint8), pad_arg, mode="constant", value=int(bool(fill_value)))
        return out.to(torch.bool)
    return F.pad(volume, pad_arg, mode="constant", value=float(fill_value))


def same_beam_collate(samples):
    """Unified collate for both single-CP and G-CP same-beam batches.

    Each sample arrives from ``load_beam_segment`` already H-cropped to
    its OWN per-CP jaw extent (cheap, small payloads through the
    DataLoader pipe — important for GPU utilisation). Here we align
    them to the batch's union H extent by padding each sample with
    a few rows of the right fill value, so all CPs share one grid and
    one in-grid iso_center. pydosert.BeamSequence is happy with that.

    Variation: because the batch composition changes each epoch
    (shuffle=True), the union extent changes too — each CP sees its
    grid extended by slightly different amounts across epochs. Free
    augmentation, no extra IPC cost.
    """
    first = samples[0]

    # SameBeamBatchSampler guarantees same (patient, beam) for G>1;
    # this is a defensive check, not load-bearing.
    if len(samples) > 1:
        first_iso = first["iso_center"]
        first_resolution = first["resolution"]
        for s in samples[1:]:
            assert np.array_equal(s["iso_center"], first_iso), (
                f"Batch mixes isocenters: {s['iso_center']} vs {first_iso}."
            )
            assert tuple(s["resolution"]) == tuple(first_resolution), (
                "Batch mixes voxel resolutions across CPs."
            )

    # Per-CP H-crop bounds are encoded in pad_info as negative paddings.
    h_orig = first["pad_info"]["original_shape"][0]
    h_starts = [-s["pad_info"]["pad_H_before"] for s in samples]
    h_ends   = [h_orig + s["pad_info"]["pad_H_after"] for s in samples]
    h_start_union = min(h_starts)
    h_end_union   = max(h_ends)
    res_h = float(first["resolution"][0])

    # Fill values per volume key. The image is in HU for both arms (real CT
    # or synthetic CT), so -1000 = air; dose and density are 0 outside the
    # beam path, mask is False outside the patient. The image pad value was
    # chosen at load_beam_segment time — read it back from
    # pad_info["fill_values"] (index 0 is the "image" entry from the
    # [image, dose] pair passed to pad_and_crop_to_iso_center).
    for s, h_start_i, h_end_i in zip(samples, h_starts, h_ends):
        pre  = h_start_i - h_start_union     # rows to add on top
        post = h_end_union - h_end_i         # rows to add on bottom
        if pre or post:
            image_fill = float(s["pad_info"]["fill_values"][0])
            fills = {"image": image_fill, "dose": 0.0, "density_image": 0.0, "mask": False}
            for key, fv in fills.items():
                s[key] = _pad_h(s[key], pre, post, fv)
        # Replace pad_info with the union-extent version so
        # crop_and_pad_to_original later inverts back to the original
        # CT shape correctly.
        s["pad_info"] = {
            **s["pad_info"],
            "pad_H_before": -h_start_union,
            "pad_H_after":  -(h_orig - h_end_union),
        }

    # Force every sample's beam.iso_center to the SAME tuple object.
    # pydosert.BeamSequence.from_beams uses exact equality on iso_center
    # tuples; computing each sample's iso_h via a different float path
    # (orig - h_start_i*res + pre_i*res vs orig - h_start_union*res) can
    # leave 1-ULP differences that fail the check, raise in the worker,
    # and hang the main process forever. D/W components are unchanged by
    # the H-pad and were already identical across CPs of the same beam.
    union_iso = (
        float(first["iso_center"][0]) - h_start_union * res_h,
        float(first["beam"].iso_center[1]),
        float(first["beam"].iso_center[2]),
    )
    for s in samples:
        s["beam"].iso_center = union_iso

    if len(samples) > 1:
        # IMPORTANT: do NOT build pydosert.BeamSequence here. Its
        # construction stacks tensors that include machine_config
        # parameters with requires_grad=True; the resulting non-leaf
        # tensors cannot cross a multiprocessing pickle boundary
        # ("Cowardly refusing to serialize non-leaf tensor which
        # requires_grad"), and the failure happens in the worker's
        # background feeder thread — silently — so main hangs forever
        # waiting for a batch that will never arrive.
        #
        # Individual Beam objects pickle fine (that's why G=1 works),
        # so we ship a list of Beams and let the main process build
        # the BeamSequence after IPC.
        beams_list = [s["beam"] for s in samples]
        true_dose = torch.stack([s["dose"] for s in samples], dim=0)   # [G, D, H, W]
        return [{
            "beam": None,                  # populated in main; see below
            "beam_list": beams_list,
            "image": first["image"],
            "density_image": first["density_image"],
            "dose": true_dose,
            "mask": first["mask"],
            "resolution": first["resolution"],
            "pad_info": first["pad_info"],
            "iso_center": first["iso_center"],
            "batched": True,
            "n_cp": len(samples),
        }]
    return [first]


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = default_data_path()
    get_pb_dose = False
    do_plots = True
    plot_errors = False
    patients = get_patients(data_path, modality="photon", cohort="training")
    example_patient = "1ABB020"
    beam_index = 0
    cp_index = 135

    pb_error = []
    pdrt_error = []
    # for example_patient in patients[:3]:
    #     for beam_index in range(3):
    #         for cp_index in range(0, 180, 90):
    beam, ct, density_image, dose, mask, resolution, pad_info, original_iso_center = load_beam_segment(data_path + "/photon/training", example_patient, beam_index, cp_index)

    if get_pb_dose:
        pb_dose = load_baseline(data_path, example_patient, beam_index, cp_index)

    engine = PDRT.DoseEngine(
        machine_config=machine_config,
        kernel_size=55,
        dose_grid_shape=dose.shape,
        dose_grid_spacing=resolution,
    )
    engine = engine.to(device)
    beam = beam.to(device)
    ct = ct.to(device)
    density_image = density_image.to(device)
    dose = dose.to(device)
    pred_dose = engine.compute_dose(beam, density_image).detach().cpu().numpy()[0, ...]
    true_dose = dose.detach().cpu().numpy()
    ct_volume = ct.detach().cpu().numpy()
    pred_dose = crop_and_pad_to_original(pred_dose, pad_info, fill_value=0.0)
    true_dose = crop_and_pad_to_original(true_dose, pad_info, fill_value=0.0)
    image_fill = float(pad_info["fill_values"][0])  # -1000 (HU air) for both arms
    ct_volume = crop_and_pad_to_original(ct_volume, pad_info, fill_value=image_fill)
    
    slice_idx = int(original_iso_center[1] // 2) + 2
    pdrt_error.append(np.mean(np.abs(pred_dose - true_dose)))
    # print(f"Error for beam {beam_index} CP {cp_index}: {np.mean(np.abs(pred_dose - true_dose)):.12f}")
    if get_pb_dose:
        pb_error.append(np.mean(np.abs(pb_dose - true_dose)))
        # print(f"Error for PB dose for beam {beam_index} CP {cp_index}: {np.mean(np.abs(pb_dose - true_dose)):.12f}")
    if do_plots:
        num_images = 3 if get_pb_dose else 2
        plt.subplot(1,num_images,1)
        plt.imshow(ct_volume[:, slice_idx], cmap="gray")
        plt.imshow(true_dose[:, slice_idx], cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.scatter(beam.iso_center[2]//2, beam.iso_center[1]//2, color="red", marker="x")

        plt.subplot(1,num_images,2)
        plt.imshow(ct_volume[:, slice_idx], cmap="gray")
        if plot_errors:
            vmax = np.max(np.abs(pred_dose - true_dose))
            plt.imshow((pred_dose - true_dose)[:, slice_idx], cmap="coolwarm", alpha=0.5, vmax=vmax, vmin=-vmax)
        else:
            plt.imshow(pred_dose[:, slice_idx], cmap="jet", alpha=0.5)
        plt.colorbar()
        plt.scatter(beam.iso_center[2]//2, beam.iso_center[1]//2, color="red", marker="x")

        if get_pb_dose:
            plt.subplot(1,num_images,3)
            plt.imshow(ct_volume[:, slice_idx], cmap="gray")
            if plot_errors:
                vmax = np.max(np.abs(pb_dose - true_dose))
                plt.imshow((pb_dose - true_dose)[:, slice_idx], cmap="coolwarm", alpha=0.5, vmax=vmax, vmin=-vmax)
            else:
                plt.imshow(pb_dose[:, slice_idx], cmap="jet", alpha=0.5)
            plt.colorbar()
            plt.scatter(beam.iso_center[2]//2, beam.iso_center[1]//2, color="red", marker="x")
        plt.show()
                    
    print(f"Average PDRT error across CPs: {np.mean(pdrt_error):.6f}")
    if get_pb_dose:
        print(f"Average PB error across CPs: {np.mean(pb_error):.6f}")