"""Core inference for the DoseRAD2026 Grand Challenge submission (photon).

Reads the GC input mount (10 image slots + one beam-level metadata JSON),
runs the calibrated CorrectedDoseEngine + the trained dose-correction model
per control point, and writes the 10 output dose-map slots in GC's format.

The per-control-point preprocessing (beam construction, HU->density, iso
crop) is kept byte-for-byte identical to loaders.load_beam_segment so the
submission dose equals the validation dose. The only submission-specific
steps are I/O plumbing and undoing the 1e5 training-time dose scale.
"""
import gc
import glob
import json
import os
import time
import re
from typing import NamedTuple

import numpy as np
import torch
import SimpleITK as sitk
import pydosert as PDRT

from configs import (machine_config, machine_config_for, _anatomy_key,
                     DEFAULT_KERNEL_SIZE,
                     MULTISLAB_DEFAULTS)
from engines import CorrectedDoseEngine, DoseCorrectionModel, load_corrector
from loaders import (
    EXPECTED_RESOLUTION,
    convert_HU_to_density_lut,
    image_spacing_zyx,
    estimate_h_crop_from_geometry,
    body_cylinder_radius_mm,
    pad_and_crop_to_iso_center,
    crop_and_pad_to_original,
    body_mask,
)
from sct import mr_to_synthetic_ct

# loaders.load_dose multiplies the GT dose by this before training, so the
# model predicts in 1e5-scaled units. Undo it on output so the challenge sees
# physical dose, which is the scale minimum_cutoff is expressed in.
DOSE_SCALE = 1e5

# H-axis crop: the geometric crop keeps jaw +/- CROP_MARGIN_MM and everything
# outside it is written as a hard ZERO (crop_and_pad_to_original fills 0.0).
#
# 60 mm, NOT the 30 mm the model was trained with, and that mismatch is
# deliberate. The official IDD metric sums the transverse plane along
# beam_axis=0 -- THE SAME AXIS THIS CROP CUTS -- so every discarded slice is a
# slice of the curve being scored. Scoring GT-with-the-crop-zeroed against GT
# gives the floor a PERFECT corrector cannot beat: 0.01010 at 30 mm, 0.00629 at
# 50, 0.00316 at 80. The shipped model achieves 0.01052, i.e. ~96% of that
# leaderboard column was the crop rather than the model.
#
# MEASURED end to end on the e18 checkpoint, 6 patients x 3 beams x 180 CPs,
# changing NOTHING but this number:
#
#            gamma   stratified   beam MAE      IDD
#     30 mm  96.64      0.00292    0.00885   0.01052
#     60 mm  98.66      0.00263    0.00875   0.00553
#    120 mm  98.33      0.00295    0.00878   0.00288
#
# 120 mm is WORSE on gamma and stratified MAE despite the better IDD: the
# corrector was fitted at 30 mm and at 120 it is extrapolating far outside its
# training window. 60 mm is the measured optimum for THIS checkpoint, not a
# compromise. A model retrained at a wider margin may well prefer a different
# one; re-measure with eval_patient_total.py --crop_margin_mm before changing
# it, and override with GC_CROP_MARGIN_MM rather than editing this default.
# Deliberately NOT read from model_config: the checkpoint records the margin it
# was TRAINED at (30 mm), and serving WIDER than trained is the whole point.
#
# It costs runtime: mean H span goes 60.8 -> 80.7 voxels (1.47x) over the
# validation patients, and runtime is 2 of 7 leaderboard ranks. That trade was
# taken on the numbers above.
DEFAULT_CROP_MARGIN_MM = float(os.environ.get("GC_CROP_MARGIN_MM", "60.0"))
N_OUTPUT_SLOTS = 10
# Fallback voxel spacing in ARRAY order (z, y, x). The photon cohort is 2 mm
# isotropic, but the spacing is read off each input image (see run_invoke) and
# threaded through; this is only the default for callers that don't pass one.
RESOLUTION = EXPECTED_RESOLUTION
# Control points of one beam can be run as a batched sequence. There is no
# reason to raise this. Profiled on a 7.05 M-voxel BEV crop (86x278x295),
# chunk 1/2/4/8:
#   throughput  333 / 358 / 348 ms per CP        -- FLAT
#   peak VRAM   2.46 / 4.78 / 9.38 / 18.58 GiB   -- linear, ~2.3 GiB per CP
# The engine already saturates the GPU at G=1 (see
# configs.py), so batching buys no throughput at all while costing linear
# memory. The batching machinery here was built for the pencil-beam era.
#
# The platform T4 has 24 GB, so the old default of 8 would have fit THIS crop
# with ~5 GB to spare -- but only this one. Peak scales linearly with the BEV
# box, thorax beams with wider jaws are larger, and the resident sCT model is
# never evicted. That is a real OOM risk for zero throughput, so 1 it is. The
# env var stays for experiments.
BATCH_CHUNK = int(os.environ.get("GC_BATCH_CHUNK", "1"))

# Autocast dtype for the CORRECTOR forward only (never the physics). Off by
# default: it is worth ~12 ms/CP and -35% peak VRAM, but it perturbs the dose
# by ~4e-4 relative to peak.
#
# THE PLATFORM GPU IS AN A10G (sm_86, Ampere, 24 GB), NOT A T4. An earlier
# version of this comment claimed T4/sm_75 and concluded that bf16 "raises
# there" and that fp16 was the only option. That is wrong on Ampere: bf16 has
# native tensor-core support, and it is also the precision this model was
# TRAINED under (--amp bf16), which makes it the faithful choice rather than
# fp16. Ampere additionally has a TF32 path for fp32 convolutions, which the
# T4 assumption ruled out entirely.
_AMP = os.environ.get("GC_AMP", "").lower()
AMP_DTYPE = {"fp16": torch.float16, "float16": torch.float16,
             "bf16": torch.bfloat16, "": None, "off": None, "fp32": None}[_AMP]

# Build one engine per beam and re-point it at each control point's geometry,
# instead of constructing a fresh CorrectedDoseEngine per chunk (~11 ms a time).
# Verified bit-identical by submission/test/parity_check.py; the switch exists
# so that check can run both ways, and as an escape hatch if a future pydosert
# makes a layer stateful across calls.
REUSE_ENGINE = os.environ.get("GC_REUSE_ENGINE", "1") == "1"

# Every voxel at or below the control point's minimum_cutoff is zeroed before the
# slot is written. The cutoff is per-control-point and comes from the challenge's
# own metadata ("output_info"/"minimum_cutoff"); GC_MIN_CUTOFF overrides it for
# every CP, and is meant for local experiments only -- leave it unset in the
# container so the challenge's value is what gets applied.
_MIN_CUTOFF_OVERRIDE = os.environ.get("GC_MIN_CUTOFF")
MIN_CUTOFF_OVERRIDE = float(_MIN_CUTOFF_OVERRIDE) if _MIN_CUTOFF_OVERRIDE else None

def amp_for(device):
    """AMP_DTYPE, but only where it is SAFE -- which means CUDA only.

    fp16 autocast on CUDA accumulates convolutions in fp32 on the tensor cores.
    On CPU there are no tensor cores and the accumulation happens in fp16,
    which overflows on this pipeline (the model works in 1e5-scaled dose units)
    and yields a silent all-NaN dose -- no exception, no warning, and
    run_local.py's structure and geometry checks still PASS. It was caught only
    because the scale line printed `pred max=nan`.

    The platform always has a GPU, so this costs nothing there; it exists so
    that a CPU fallback degrades to fp32 rather than to NaN, and so the local
    CPU dry run stays a usable end-to-end check.
    """
    if AMP_DTYPE is None:
        return None
    return AMP_DTYPE if torch.device(device).type == "cuda" else None


# tag used in the input directory names -> internal modality string
_TAG_TO_MODALITY = {"ct": "ct", "mri": "mr"}


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
class Corrector(NamedTuple):
    """Everything the engine needs to serve one checkpoint correctly.

    A corrector does not predict dose. It predicts the RESIDUAL of one specific
    physics baseline, so the weights, the feature stack and the engine are a
    single indivisible unit. Passing them around separately is what let the
    container serve a model on a baseline it was not trained on: nothing
    fails, the channel count is identical either way, and the dose is simply
    wrong (per-CP masked MAE 0.0118 -> 0.0228, plan gamma 93.3% -> 63.0%).
    """
    model: object
    feature_cfg: object          # FeatureConfig, or None for a 2-channel model
    engine_kw: dict              # multislab + lateral/TERMA physics
    engine: str                  # "multislab", for logging
    fluence_model: object = None
    lateral_model: object = None


# --------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------
# The leaderboard fits T = t_fix + N_im*t_im + N_dose_map*t_dose_map and scores
# photon at 1 image + 181 dose maps, so what matters is which of those three
# a given cost lands in -- a 2 s saving in per-image work is worth 2 s, the
# same saving per dose map is worth 362 s. These buckets are therefore labelled
# by which coefficient they feed, not by call site.
#
# GC_TIMING=0 disables. Timers around GPU work synchronise, which perturbs the
# very thing being measured, so the totals here run slightly above an untimed
# run -- use them for APPORTIONMENT, not as the runtime number.
# DEFAULT OFF. The sync=True phases below call torch.cuda.synchronize(), which
# is the point when profiling and a pure tax when serving -- and the runtime
# metric is what this submission is scored on. Enable explicitly with
# GC_TIMING=1 when you want the breakdown.
# DEFAULT OFF. The phase timers call torch.cuda.synchronize(), which inserts
# forced serialisation points into the serving path -- small, but non-zero, and
# there is nothing left to diagnose. GC_TIMING=1 brings the breakdown back.
_TIMING = os.environ.get("GC_TIMING", "0") != "0"
_T: dict[str, float] = {}
_TN: dict[str, int] = {}
_T_START = time.time()


class _phase:
    """Accumulate wall time under `name`. `sync` for phases containing GPU work."""

    __slots__ = ("name", "sync", "t0")

    def __init__(self, name, sync=False):
        self.name, self.sync = name, sync

    def __enter__(self):
        if _TIMING and self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        self.t0 = time.time()
        return self

    def __exit__(self, *exc):
        if not _TIMING:
            return False
        if self.sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        _T[self.name] = _T.get(self.name, 0.0) + (time.time() - self.t0)
        _TN[self.name] = _TN.get(self.name, 0) + 1
        return False


# CUDA-event timers. The wall-clock `_phase` above has to synchronise to be
# correct, which serialises the pipeline it is measuring -- fine for coarse
# buckets, useless for a fine breakdown where the syncs would dominate and
# misattribute queued work to whichever call follows. Events are recorded on the
# stream at ~microsecond cost and read ONCE at the end.
_EV: list = []


class _gpu:
    __slots__ = ("name", "a", "b")

    def __init__(self, name):
        self.name = name

    def __enter__(self):
        if _TIMING and torch.cuda.is_available():
            self.a = torch.cuda.Event(enable_timing=True)
            self.b = torch.cuda.Event(enable_timing=True)
            self.a.record()
        else:
            self.a = None
        return self

    def __exit__(self, *exc):
        if self.a is not None:
            self.b.record()
            _EV.append((self.name, self.a, self.b))
        return False


def _drain_events():
    """Fold recorded CUDA events into _T. One synchronise for the whole run."""
    if not _EV:
        return
    torch.cuda.synchronize()
    for name, a, b in _EV:
        _T[name] = _T.get(name, 0.0) + a.elapsed_time(b) / 1000.0
        _TN[name] = _TN.get(name, 0) + 1
    _EV.clear()


def instrument_engine(engine):
    """Wrap the engine's sub-layers so the forward is broken down by operation.

    Only under GC_TIMING. Wrapping is idempotent per engine -- predict_beam_batch
    reuses one engine across chunks, and double-wrapping would double-count.
    """
    if not _TIMING:
        return engine

    # Idempotent PER TARGET, not per engine. _initialize_layers rebuilds the
    # angle-dependent layers but NOT the corrector, so re-instrumenting an
    # engine used to wrap the already-wrapped corrector again; every nesting
    # recorded another full-duration event and the corrector reported 1040% of
    # a forward it is contained in, with unattributed time going negative.
    def wrap_module(obj, attr, label):
        mod = getattr(obj, attr, None)
        if mod is None or not hasattr(mod, "forward"):
            return
        inner = mod.forward
        if getattr(inner, "_timed", False):
            return

        def fwd(*a, **k):
            with _gpu(label):
                return inner(*a, **k)
        fwd._timed = True
        mod.forward = fwd

    def wrap_method(obj, attr, label):
        fn = getattr(obj, attr, None)
        if fn is None or getattr(fn, "_timed", False):
            return

        def call(*a, **k):
            with _gpu(label):
                return fn(*a, **k)
        call._timed = True
        setattr(obj, attr, call)

    wrap_module(engine, "fluence_map_layer", "  fluence_map")
    wrap_module(engine, "fluence_volume_layer", "  fluence_volume")
    wrap_module(engine, "pencil_beam_kernel_layer", "  kernel_build")
    wrap_module(engine, "beam_wise_conv_layer", "  convolution")
    wrap_module(engine, "rotation_layer", "  rotation")
    wrap_module(engine, "inv_rotation_layer", "  inv_rotation")
    wrap_module(engine, "rad_depth_layer", "  rad_depth_layer")
    wrap_module(engine, "terma_layer", "  terma")
    wrap_method(engine, "_lattice_tiles", "  lattice_tiles")
    wrap_method(engine, "_lattice_ray_depths", "  lattice_ray_depths")
    wrap_method(engine, "build_v2_features", "  v2_features")
    # The two FROZEN TRAINED PRIORS. Unlike the physics layers above these are
    # nn.Modules, so they are the only part of the engine an autocast would
    # actually convert -- worth being able to see their share before trying.
    wrap_module(engine, "fluence_correction_model", "  prior_fluence")
    wrap_module(engine, "lateral_scatter_model", "  prior_heterogeneity")
    wrap_method(engine, "patient_to_bev", "  patient_to_bev")
    if getattr(engine, "dose_correction_model", None) is not None:
        wrap_module(engine, "dose_correction_model", "  corrector")
    return engine


# Census of the shapes the corrector actually sees, and of individual forward
# times. A slow outlier against an otherwise flat distribution is the signature
# worth catching -- invisible in a mean, obvious here.
#
# CALIBRATION, measured: eager runs already show 1206 ms calls against a 411 ms
# median, from cuDNN autotuning per shape on first use. So a 3x outlier is
# NORMAL and means "look", not "something is wrong".
_FWD_MS = []
_SHAPES = {}


def note_forward(ms, shape):
    _FWD_MS.append(ms)
    _SHAPES[tuple(shape)] = _SHAPES.get(tuple(shape), 0) + 1


def forward_census():
    if not _FWD_MS:
        return
    v = sorted(_FWD_MS)
    n = len(v)
    med = v[n // 2]
    slow = [x for x in v if x > 3 * med]
    print(f"  engine_fwd calls {n}: min {v[0]:.0f} med {med:.0f} "
          f"p90 {v[int(0.9*(n-1))]:.0f} max {v[-1]:.0f} ms", flush=True)
    if slow:
        print(f"  !! {len(slow)} call(s) over 3x the median "
              f"({', '.join('%.0f' % x for x in slow[:5])} ms) -- expected on the "
              f"first call per box (cuDNN autotune); investigate only if seconds",
              flush=True)
    print(f"  corrector boxes seen: {len(_SHAPES)} distinct", flush=True)
    for shp, cnt in sorted(_SHAPES.items(), key=lambda kv: -kv[1])[:6]:
        print(f"     {str(shp):28s} x{cnt}", flush=True)


def timing_report(n_images, n_dose_maps):
    if not _TIMING:
        return
    _drain_events()
    wall = time.time() - _T_START
    print("\n" + "=" * 78, flush=True)
    print(f"[timing] wall {wall:.1f}s   {n_images} image(s), {n_dose_maps} dose map(s)",
          flush=True)
    print(f"  batch_chunk {BATCH_CHUNK}   amp {os.environ.get('GC_AMP', '-')}"
          f"   sct_amp {os.environ.get('DOSERAD_SCT_AMP', '-')}"
          f"   crop {DEFAULT_CROP_MARGIN_MM:.0f}mm", flush=True)
    forward_census()
    # Sub-operations are nested INSIDE engine_fwd, so they are listed under it
    # and excluded from the bucket sums to avoid double counting.
    subops = [k for k in _T if k.startswith("  ")]
    groups = (("t_fix    (once per run)", ("startup", "model_load")),
              ("t_im     (per image)", ("image_read", "to_hu", "density")),
              ("t_dose_map (per dose map)", ("beam_build", "engine_build",
                                             "engine_fwd", "post", "d2h", "write")))
    for label, keys in groups:
        sub = sum(_T.get(k, 0.0) for k in keys)
        print(f"  {label:30s} {sub:7.2f}s  ({100*sub/max(wall,1e-9):4.1f}%)", flush=True)
        for k in keys:
            if k not in _T:
                continue
            per = _T[k] / max(n_dose_maps, 1)
            print(f"      {k:24s} {_T[k]:7.2f}s  n={_TN[k]:4d}"
                  f"   {1000*per:7.1f} ms/dose_map", flush=True)
            if k == "engine_fwd" and subops:
                tot_sub = sum(_T[x] for x in subops)
                for x in sorted(subops, key=lambda y: -_T[y]):
                    pm = _T[x] / max(n_dose_maps, 1)
                    print(f"        {x.strip():22s} {_T[x]:7.2f}s  n={_TN[x]:5d}"
                          f"   {1000*pm:7.1f} ms/dose_map"
                          f"   {100*_T[x]/max(_T[k],1e-9):5.1f}% of fwd", flush=True)
                print(f"        {'(unattributed in fwd)':22s} "
                      f"{_T[k]-tot_sub:7.2f}s"
                      f"{'':16s}   {100*(_T[k]-tot_sub)/max(_T[k],1e-9):5.1f}% of fwd",
                      flush=True)
    acct = sum(_T.get(k, 0.0) for g in groups for k in g[1])
    print(f"  {'unattributed (outside phases)':30s} {wall-acct:7.2f}s", flush=True)
    if n_dose_maps:
        per_dm = sum(_T.get(k, 0.0) for k in
                     ("beam_build", "engine_build", "engine_fwd", "post", "d2h", "write")) / n_dose_maps
        fix = sum(_T.get(k, 0.0) for k in ("startup", "model_load"))
        # PER IMAGE, not total. The scored metric is fitted at ONE image, so
        # summing across every image in this invoke overstates it by n_images --
        # on a 6-image run that read 101.5 s where the truth was 60.8 s.
        im = (sum(_T.get(k, 0.0) for k in ("image_read", "to_hu", "density"))
              / max(1, n_images))
        print(f"\n  extrapolated leaderboard metric (1 image, 181 dose maps):")
        print(f"    t_fix {fix:.2f}s + t_im {im:.2f}s + 181 x {1000*per_dm:.1f} ms"
              f"  =  {fix + im + 181*per_dm:.1f}s", flush=True)
    if torch.cuda.is_available():
        # Headroom matters as much as the number above it: the platform GPU is
        # a 24 GB A10G, so this says how much room a larger GC_BATCH_CHUNK has.
        print(f"\n  peak VRAM {torch.cuda.max_memory_allocated()/2**30:.2f} GiB"
              f"   (reserved {torch.cuda.max_memory_reserved()/2**30:.2f} GiB)"
              f"   chunk={BATCH_CHUNK}", flush=True)
    print("=" * 78 + "\n", flush=True)


def engine_kwargs(engine):
    """CorrectedDoseEngine kwargs selecting the physics baseline."""
    if engine == "multislab":
        # Pencil beam WITH the heterogeneity model. Both are opt-in; multislab
        # alone drops the lateral scatter the corrector was trained against.
        return dict(multislab=True, lateral_scatter=True, **MULTISLAB_DEFAULTS)
    raise ValueError(f"Unknown engine {engine!r}; expected 'multislab'. "
                     "Collapsed cone has been removed.")


def _engine_kw_from_cfg(cfg):
    """CorrectedDoseEngine kwargs implied by a checkpoint's model_config.

    Shared by every architecture branch in build_model: the physics
    baseline is a property of the CHECKPOINT, not of the corrector class
    that sits on top of it.
    """
    engine = cfg.get("engine", "multislab")
    engine_kw = engine_kwargs(engine)
    if cfg.get("bev_crop"):
        # Same class of hazard as the engine itself: the crop changes what the
        # corrector was trained to SEE, and serving a cropped-trained model on
        # full volumes passes every shape check (the trunk is fully
        # convolutional) while handing it a distribution it never met.
        engine_kw = dict(
            engine_kw,
            bev_crop=True,
            bev_crop_margin_mm=cfg.get("bev_crop_margin_mm", 50.0),
            bev_crop_per_sample=bool(cfg.get("bev_crop_per_sample", False)),
            bev_crop_min=cfg.get("bev_crop_min", 32),
            early_bev_crop=bool(cfg.get("early_bev_crop", False)),
        )
    elif cfg.get("early_bev_crop"):
        # INDEPENDENT of bev_crop, and it has to be: --bev_crop crops what the
        # corrector SEES, --early_bev_crop crops the PHYSICS work volume and
        # scatters back. Reading it only inside the bev_crop branch above meant
        # a model trained with early_bev_crop and no corrector crop -- which is
        # this one -- silently served on the full work volume, doing strictly
        # more work for a bit-identical answer. Measured on the TERMA arm over
        # 6 control points: 3.61x fewer voxels, convolution 3.79 -> 1.21 ms,
        # max error against the uncropped result exactly 0.0
        # (analysis_outputs/early_bev_crop_engine_parity_terma_6cp.json).
        engine_kw = dict(engine_kw, early_bev_crop=True)

    # Lattice and TERMA change the BASELINE the corrector predicts the residual
    # of, which is the indivisibility this whole class documents. A missing key
    # means the checkpoint predates them, so the defaults reproduce the shipped
    # single-ray, no-TERMA physics exactly.
    if int(cfg.get("lattice_size", 1) or 1) > 1 or cfg.get("terma"):
        engine_kw = dict(
            engine_kw,
            lattice_size=int(cfg.get("lattice_size", 1) or 1),
            lattice_depth_mode=cfg.get("lattice_depth_mode", "ray"),
            terma_scaling=bool(cfg.get("terma", False)),
            terma_c1=cfg.get("terma_c1"),
            terma_c2_per_mm=cfg.get("terma_c2_per_mm"),
        )
        # TERMA REPLACES the heuristic lateral-scatter term -- they model the
        # same lateral disequilibrium and the engine RAISES if both are set.
        # engine_kwargs("multislab") turns lateral_scatter on unconditionally,
        # so without this a TERMA checkpoint dies on the first control point
        # inside a scored run. Training measured the pair at 0.0279 against
        # 0.0231 for TERMA alone, so this is also the baseline it was fitted to.
        if cfg.get("terma"):
            engine_kw = dict(engine_kw, lateral_scatter=False)
    return engine_kw


def build_model(weights_path, device):
    """Load a checkpoint into a fully-specified `Corrector`.

    Architecture, feature stack AND physics baseline all come from a
    ``model_config`` dict stored IN the checkpoint rather than being inferred.
    None of the three is recoverable from the weights:

      * ``norm="group"`` changes the state_dict keys;
      * ``bounded_residual`` changes the OUTPUT ALGEBRA and leaves no trace in
        the weights at all, so a shape-matching loader would serve
        ``dose*gain + raw_residual`` instead of the bounded, relu'd form;
      * ``engine`` decides what the residual is a residual OF, and with
        ``v2_features`` on, the corrector takes 13 channels either way -- so a
        mismatch is invisible to every shape check.

    Checkpoints with no ``model_config`` are REJECTED. They used to fall back to
    the inferring loader plus collapsed cone; that engine no longer exists, and
    silently serving such a checkpoint on multislab would be exactly the
    baseline mismatch this docstring warns about.
    """
    ckpt = torch.load(weights_path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if cfg is None:
        raise ValueError(
            f"{os.path.basename(str(weights_path))} has no model_config. Re-bake "
            "it with submission/bake_model.py; the collapsed-cone fallback that "
            "used to handle this has been removed.")

    feature_cfg = None
    in_ch = cfg["in_channels"]
    if cfg.get("v2_features"):
        from features_v2 import FeatureConfig
        # feature_set defaults to "v2" for checkpoints predating it -- which is
        # correct, they were all fitted to the v2 stack. Note v2 and v3 emit the
        # SAME channel count, so the check below cannot catch a swapped set;
        # only the recorded name can.
        #
        # cond_region is the same kind of field and defaults the same way, for
        # the same reason: every checkpoint written before it existed was fitted
        # with the original ("dose") region, so a missing key means "dose". It
        # redefines cond_mean_rho and cond_lung_frac without touching the count
        # OR the names, so neither the check below nor the state_dict can catch
        # a mismatch -- serving a model under the wrong one just returns a
        # plausible wrong dose.
        feature_cfg = FeatureConfig(
            cond_mode=cfg.get("v2_cond_mode", "channels"),
            feature_set=cfg.get("feature_set") or "v2",
            cond_region=cfg.get("cond_region") or "dose",
            crop_relative_position=bool(cfg.get("bev_crop")))
        expect = 1 + feature_cfg.n_scalar_channels()   # raw dose + v2 stack
        if expect != in_ch:
            raise ValueError(
                f"{os.path.basename(weights_path)}: model_config says "
                f"in_channels={in_ch} but FeatureConfig(cond_mode="
                f"{feature_cfg.cond_mode!r}, feature_set="
                f"{feature_cfg.feature_set!r}) emits {expect}. The stem and the "
                f"feature builder must agree or the channels are misaligned.")

    # Hoisted: a local `from fixed_priors import ...` further down made the
    # name local to the whole function, so the architecture branches below hit
    # UnboundLocalError before ever reaching it.
    from fixed_priors import build_embedded_priors

    model = DoseCorrectionModel(
        in_channels=in_ch,
        base_channels=cfg.get("base_channels", 8),
        depth=cfg.get("depth", 4),
        norm=cfg.get("norm", "batch"),
        bounded_residual=cfg.get("bounded_residual", False),
        refine=cfg.get("refine", True),
        # additive_scale_frac changes the OUTPUT ALGEBRA and no tensor at all,
        # so a checkpoint trained with a non-default --v2_alpha loads clean here
        # and is then served under the WRONG bound. Every checkpoint to date used
        # the 0.05 default, which is why this never bit; the first arm that moves
        # it would have been silently mis-served. use_gain does change the
        # state_dict (the gain head appears or does not), so a mismatch there is
        # caught by the load below -- passed anyway rather than relied on.
        additive_scale_frac=cfg.get("additive_scale_frac", 0.05),
        use_gain=bool(cfg.get("use_gain", True)),
        material_embedding_dim=cfg.get("material_embedding_dim", 0),
        # Separable/anisotropic convolutions. sep_levels, lateral_kernel,
        # depth_kernel, separable_refine and refine_hidden all change the
        # state_dict, so the strict-ish load below catches a mismatch. The
        # depth dilations do NOT: they change the receptive field and no
        # tensor, so a wrong value here serves a plausible wrong dose with a
        # clean load. Missing keys default to the dense model, which is what
        # every checkpoint written before this existed actually is.
        pool_kernels=cfg.get("pool_kernels"),
        width_growth=cfg.get("width_growth", 2.0),
        downsample_mode=cfg.get("downsample_mode", "maxpool"),
        activation_checkpointing=False,
        sep_levels=cfg.get("sep_levels", 0),
        lateral_kernel=cfg.get("lateral_kernel", 3),
        depth_kernel=cfg.get("depth_kernel", 5),
        separable_refine=bool(cfg.get("separable_refine", False)),
        refine_scale_to_peak=bool(cfg.get("refine_scale_to_peak", False)),
        refine_hidden=cfg.get("refine_hidden", 16),
        stem_stride=cfg.get("stem_stride", 1),
        refine_mode=cfg.get("refine_mode", "parallel"),
        refine_depth_dilations=cfg.get("refine_depth_dilations"),
        # The aux heads are training-only at inference time, but they are
        # PARAMETERS: omitting them here makes the strict-ish load below report
        # them as unexpected and refuse the checkpoint.
        deep_supervision=cfg.get("deep_supervision", False),
    )
    sd = ckpt["dose_model_state_dict"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        # Loud: a partial load here serves a part-random network.
        raise RuntimeError(
            f"{os.path.basename(weights_path)}: state_dict mismatch, "
            f"{len(missing)} missing / {len(unexpected)} unexpected. "
            f"First missing {missing[:3]}, first unexpected {unexpected[:3]}")
    engine_kw = _engine_kw_from_cfg(cfg)
    engine = cfg.get("engine", "multislab")
    print(f"[model] {os.path.basename(str(weights_path))}: {len(sd)} tensors, "
          f"in_channels={in_ch}, base_channels={cfg.get('base_channels', 8)}, "
          f"norm={cfg.get('norm', 'batch')}, "
          f"bounded_residual={cfg.get('bounded_residual', False)}, "
          f"v2_features={bool(cfg.get('v2_features'))}, "
          f"feature_set={cfg.get('feature_set') or 'v2'}, "
          f"cond_region={cfg.get('cond_region') or 'dose'}, "
          f"lattice={cfg.get('lattice_size', 1) or 1}, "
          f"terma={bool(cfg.get('terma', False))}, "
          f"material_embed={cfg.get('material_embedding_dim', 0)} "
          f"({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
    print(f"[model]   engine: {engine}", flush=True)
    for k in ("experiment_name", "epoch", "best_val_lv1_beam_mae"):
        if k in ckpt:
            print(f"[model]   {k}: {ckpt[k]}", flush=True)
    fluence_model, lateral_model = build_embedded_priors(ckpt, device)
    if fluence_model is not None or lateral_model is not None:
        print(f"[model]   embedded priors: fluence="
              f"{sum(p.numel() for p in fluence_model.parameters()) if fluence_model else 0:,}, "
              f"heterogeneity="
              f"{sum(p.numel() for p in lateral_model.parameters()) if lateral_model else 0:,}",
              flush=True)
    return Corrector(model.to(device).to(torch.float32).eval(), feature_cfg,
                     engine_kw, engine, fluence_model, lateral_model)


def open_area_mm2(beams):
    """Open MLC area per control point in mm^2 -- the v2 conditioning scalar.

    Must match train_correction._open_area_mm2 exactly: the model was
    conditioned on this and a different definition silently shifts the
    conditioning channels. Leaf pitch is 5 mm (beam_parameters.json).
    """
    widths = torch.stack([
        (b.leaf_positions[..., 1] - b.leaf_positions[..., 0]).clamp_min(0.0).sum()
        for b in beams])
    return (widths * 5.0).detach()                      # [G]


# --------------------------------------------------------------------------
# Modality handling: the MR arm is the CT arm with one extra step in front
# --------------------------------------------------------------------------
_HU_CACHE: dict = {"key": None, "src": None, "array": None}
# Depth-1 cache of the decoded source image, keyed on image_file_idx. See
# run_invoke for why it is depth 1 and why the MR arm depends on it.
_IMG_CACHE: dict = {}


def to_hu(image_arr, modality, resolution=RESOLUTION):
    """Return the source image in HU: identity for CT, MR -> synthetic CT.

    This is the ONLY difference between the CT and MR algorithms. Everything
    downstream (density LUT, body mask, engine, correction model) is shared.

    A slot's image is passed through here once per beam group, so the result is
    memoised — the sCT network is ~15 s a volume and must not run per beam. The
    key is the array's IDENTITY, not a content hash: run_invoke holds one
    decoded array alive for the whole slot, and blake2b over a 60 M-voxel volume
    cost more per beam group than it saved. Keeping `src` in the cache pins that
    identity, so a recycled id() cannot alias a different volume.
    """
    if modality == "ct":
        return image_arr
    if modality != "mr":
        raise ValueError(f"Unknown modality: {modality}")

    key = (id(image_arr), image_arr.shape, image_arr.dtype.str, tuple(resolution))
    if _HU_CACHE.get("key") != key:
        print("[sct] converting MR -> synthetic CT ...", flush=True)
        _HU_CACHE["array"] = mr_to_synthetic_ct(image_arr, spacing=resolution)
        _HU_CACHE["src"] = image_arr
        _HU_CACHE["key"] = key
    return _HU_CACHE["array"]


# --------------------------------------------------------------------------
# Per-control-point preprocessing (mirrors loaders.load_beam_segment, minus GT)
# --------------------------------------------------------------------------
def build_cp_sample(image_arr, origin_zyx, beam_meta, cp_meta, modality,
                    crop_margin_mm=None, resolution=RESOLUTION):
    """Build the beam + density + body-mask for one control point.

    image_arr : np.ndarray from sitk.GetArrayFromImage (z, y, x).
    origin_zyx: image origin already reordered to (z, y, x).
    Returns (beam, image_pad, density, mask, pad_info).
    """
    image_arr = to_hu(image_arr, modality, resolution)
    num_leaf_pairs = beam_meta["num_mlc_leaf_pairs"]
    raw_iso = np.array(beam_meta["iso_center"])
    raw_iso = np.array([raw_iso[2], raw_iso[1], raw_iso[0]])
    original_iso_center = raw_iso - origin_zyx
    gantry_angle = cp_meta["gantry_angle"]

    mlc_left = np.array(cp_meta["mlc_left_int_mm"])
    mlc_right = np.array(cp_meta["mlc_right_int_mm"])
    mlc_positions = torch.from_numpy(
        np.stack([mlc_left, mlc_right], axis=1)
    ).to(torch.float32)

    beam = PDRT.Beam.create(
        gantry_angle_deg=gantry_angle,
        number_of_leaf_pairs=num_leaf_pairs,
        iso_center=tuple(original_iso_center),
        device="cpu",
    )
    beam.leaf_positions = mlc_positions

    jaw_openings_px = (mlc_positions != 0).any(dim=1).nonzero(as_tuple=True)[0]
    if jaw_openings_px.numel() > 0:
        jaw_lower_px = jaw_openings_px[0].item()
        jaw_upper_px = jaw_openings_px[-1].item() + 1
        jaw_lower = (jaw_lower_px - mlc_positions.shape[0] // 2) * 5
        jaw_upper = (jaw_upper_px - mlc_positions.shape[0] // 2) * 5
    else:
        jaw_lower, jaw_upper = -2.5, 2.5
    beam.jaw_positions = torch.tensor([jaw_lower, jaw_upper], dtype=torch.float32)

    H_full = image_arr.shape[0]
    h_bounds = estimate_h_crop_from_geometry(
        float(beam.jaw_positions[0]), float(beam.jaw_positions[1]),
        float(original_iso_center[0]), H=H_full, res_h=resolution[0],
        margin_mm=(DEFAULT_CROP_MARGIN_MM if crop_margin_mm is None
                   else crop_margin_mm),
    )

    pad_value = -1000        # HU air, both arms (the MR is already an sCT here)
    img_t = torch.from_numpy(image_arr).float()
    # SAME cylinder padding as training. The engine rotates each (D, W) slice
    # about the isocentre, so only the inscribed circle survives every gantry
    # angle; without this the arms of a wide patient rotate out of the array.
    # Training pads per patient (loaders.load_beam_segment), so the container
    # must too or it serves a geometry the model never saw.
    _cyl_r = body_cylinder_radius_mm(img_t, resolution, tuple(original_iso_center))
    processed, iso_center, pad_info = pad_and_crop_to_iso_center(
        [img_t], resolution=resolution,
        iso_center=tuple(original_iso_center),
        fill_value=[pad_value], h_bounds=h_bounds,
        min_cylinder_radius_mm=_cyl_r,
    )
    image_pad = processed[0] if isinstance(processed, (list, tuple)) else processed
    beam.iso_center = iso_center

    density = convert_HU_to_density_lut(image_pad)
    mask = body_mask(density)

    return beam, image_pad, density, mask, pad_info


@torch.inference_mode()
def predict_cp_dose(corr, beam, image_pad, density, mask, modality, pad_info,
                    device, resolution=RESOLUTION):
    """Run the engine for one CP; return physical dose on the ORIGINAL grid.

    Not used by run_invoke (which goes through predict_beam_batch); kept for
    submission/test/plot_validation.py and eval_official.py.
    """
    beam = beam.to(device)
    density = density.to(device)
    mask = mask.to(device)

    engine = CorrectedDoseEngine(
        machine_config=machine_config,
        kernel_size=DEFAULT_KERNEL_SIZE,
        dose_grid_shape=density.shape[-3:],
        dose_grid_spacing=resolution,
        beam_template=beam,
        fluence_correction_model=corr.fluence_model,
        lateral_scatter_model=corr.lateral_model,
        dose_correction_model=corr.model,
        v2_features=corr.feature_cfg is not None,
        v2_feature_cfg=corr.feature_cfg,
        amp_dtype=amp_for(device),
        **corr.engine_kw,
        device=device,
    ).to(device)

    leaf_pos_in = beam.leaf_positions.unsqueeze(0).unsqueeze(0)
    mus_in = beam.mu.unsqueeze(0).unsqueeze(0)
    jaws_in = beam.jaw_positions.unsqueeze(0).unsqueeze(0)

    fwd = {}
    if corr.feature_cfg is not None:
        fwd["hu_image"] = image_pad.to(device).unsqueeze(0)
        fwd["open_area_mm2"] = open_area_mm2([beam]).to(device)

    pred = engine.forward(
        leaf_positions=leaf_pos_in, mus=mus_in, jaw_positions=jaws_in,
        density_image=density.unsqueeze(0),
        return_per_beam=False, **fwd,
    ) * mask
    pred = pred.squeeze(0)  # [D, H, W] on the cropped grid

    full = crop_and_pad_to_original(pred, pad_info, fill_value=0.0)
    arr = full.detach().cpu().numpy().astype(np.float32) / DOSE_SCALE

    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return arr


# --------------------------------------------------------------------------
# Batched (per-beam) preprocessing + inference
# --------------------------------------------------------------------------
def build_beam_batch(image_arr, origin_zyx, beam_meta, cp_list, modality,
                     crop_margin_mm=None, resolution=RESOLUTION):
    """Like build_cp_sample but for a whole beam at once: builds one Beam per
    control point sharing a SINGLE crop (union of the CPs' jaw extents), so the
    density/mask/engine are computed once for the beam and the control points
    can be run as a batched sequence. Returns (beams, image_pad, density, mask,
    pad_info)."""
    image_arr = to_hu(image_arr, modality, resolution)
    num_leaf_pairs = beam_meta["num_mlc_leaf_pairs"]
    raw_iso = np.array(beam_meta["iso_center"])
    raw_iso = np.array([raw_iso[2], raw_iso[1], raw_iso[0]])
    original_iso_center = raw_iso - origin_zyx

    beams, jaw_lo, jaw_hi = [], [], []
    for cp_meta in cp_list:
        mlc_left = np.array(cp_meta["mlc_left_int_mm"])
        mlc_right = np.array(cp_meta["mlc_right_int_mm"])
        mlc_positions = torch.from_numpy(
            np.stack([mlc_left, mlc_right], axis=1)).to(torch.float32)
        beam = PDRT.Beam.create(
            gantry_angle_deg=cp_meta["gantry_angle"],
            number_of_leaf_pairs=num_leaf_pairs,
            iso_center=tuple(original_iso_center), device="cpu")
        beam.leaf_positions = mlc_positions
        op = (mlc_positions != 0).any(dim=1).nonzero(as_tuple=True)[0]
        if op.numel() > 0:
            lo = (op[0].item() - mlc_positions.shape[0] // 2) * 5
            hi = (op[-1].item() + 1 - mlc_positions.shape[0] // 2) * 5
        else:
            lo, hi = -2.5, 2.5
        beam.jaw_positions = torch.tensor([lo, hi], dtype=torch.float32)
        beams.append(beam); jaw_lo.append(lo); jaw_hi.append(hi)

    h_bounds = estimate_h_crop_from_geometry(
        min(jaw_lo), max(jaw_hi), float(original_iso_center[0]),
        H=image_arr.shape[0], res_h=resolution[0],
        margin_mm=(DEFAULT_CROP_MARGIN_MM if crop_margin_mm is None
                   else crop_margin_mm))
    pad_value = -1000        # HU air, both arms (the MR is already an sCT here)
    _img_t = torch.from_numpy(image_arr).float()
    _cyl_r = body_cylinder_radius_mm(_img_t, resolution, tuple(original_iso_center))
    processed, iso_center, pad_info = pad_and_crop_to_iso_center(
        [_img_t], resolution=resolution,
        iso_center=tuple(original_iso_center), fill_value=[pad_value],
        h_bounds=h_bounds, min_cylinder_radius_mm=_cyl_r)
    image_pad = processed[0] if isinstance(processed, (list, tuple)) else processed
    for beam in beams:
        beam.iso_center = iso_center

    density = convert_HU_to_density_lut(image_pad)
    mask = body_mask(density)
    return beams, image_pad, density, mask, pad_info


@torch.inference_mode()
def predict_beam_batch(corr, beams, image_pad, density, mask, pad_info, device,
                       cutoffs=None, chunk=None,
                       resolution=RESOLUTION, reuse_engine=None, region=None):
    """Run a beam's control points and return per-CP physical dose arrays on the
    ORIGINAL grid, in the same order as `beams`.

    `cutoffs` is the per-CP minimum_cutoff (physical units), applied here so the
    threshold runs on the GPU over the CROPPED grid rather than on the CPU over
    the full one. That is exact, not an approximation: crop_and_pad_to_original
    fills the re-added slices with 0.0, and thresholding a zero at a
    non-negative cutoff leaves it zero either way.

    ONE engine is built per beam and re-pointed at each control point's geometry
    with `_initialize_layers`. Rebuilding it per chunk cost ~11 ms a time (3-4 ms
    of construction plus a 7.5 ms cone setup thrown away with the `_dir_cache`);
    reuse is bit-identical because every angle-dependent layer -- rotation,
    inverse rotation, radiological depth, fluence volume -- is rebuilt on a
    gantry/iso/field-size change, and `bev_dose` takes only
    BEV tensors and carries no beam geometry at all.
    """
    density = density.to(device)
    mask = mask.to(device)
    hu = image_pad.to(device).unsqueeze(0) if corr.feature_cfg is not None else None
    dens_in = density.unsqueeze(0)
    if chunk is None:
        chunk = BATCH_CHUNK
    if reuse_engine is None:
        reuse_engine = REUSE_ENGINE

    engine = None
    out = []
    i = 0
    while i < len(beams):
        n = min(chunk, len(beams) - i)
        with _phase("beam_build"):
            grp = [b.to(device) for b in beams[i:i + n]]
            seq = PDRT.BeamSequence.from_beams(grp)
        if engine is not None and not reuse_engine:
            del engine
            engine = None
        if engine is None:
          with _phase("engine_build", sync=True):
            engine = CorrectedDoseEngine(
                machine_config=machine_config_for(region),
                kernel_size=DEFAULT_KERNEL_SIZE,
                dose_grid_shape=density.shape[-3:], dose_grid_spacing=resolution,
                beam_template=seq, fluence_correction_model=corr.fluence_model,
                lateral_scatter_model=corr.lateral_model,
                dose_correction_model=corr.model,
                v2_features=corr.feature_cfg is not None,
                v2_feature_cfg=corr.feature_cfg,
                amp_dtype=amp_for(device), **corr.engine_kw,
                device=device).to(device)
            instrument_engine(engine)
        else:
            with _phase("engine_build", sync=True):
                engine._initialize_layers(seq)
                # Rebuilt layers arrive unwrapped; already-wrapped ones are
                # skipped by the _timed guard, so this cannot nest.
                instrument_engine(engine)

        fwd = {}
        if corr.feature_cfg is not None:
            # The same inputs training used. Withholding them does not fail --
            # it silently degrades the material embedding to a constant and
            # zeroes the aperture conditioning, for a ~28% worse dose.
            fwd["hu_image"] = hu
            fwd["open_area_mm2"] = open_area_mm2(grp).to(device)
        # OOM SAFETY NET, restored 2026-08-31. It was removed earlier so that an
        # over-large chunk would fail loudly and make the A10G's ceiling
        # discoverable -- which it did: chunk 16 crashed, then chunk 8 crashed,
        # so the ceiling is below 8 and 4 is the shipped value.
        #
        # With that answered and no submissions left to spend, the trade
        # reverses. A crash scores ZERO; a halved chunk finishes and scores. So
        # this catches OutOfMemoryError and retries at half, logging each step,
        # and `chunk` is reassigned so the reduction PERSISTS for the remaining
        # beam groups instead of being rediscovered on every one.
        #
        # It raises only at n == 1, where there is nothing left to halve.
        try:
            _t_fwd = time.time()
            with _phase("engine_fwd", sync=True):
                pred = engine.forward(
                    leaf_positions=seq.leaf_positions.unsqueeze(0),
                    mus=seq.mus.unsqueeze(0),
                    jaw_positions=seq.jaw_positions.unsqueeze(0),
                    density_image=dens_in,
                    return_per_beam=True, **fwd,
                )[0] * mask                                  # [G, D, H, W]
            note_forward(1000 * (time.time() - _t_fwd), (n,) + tuple(dens_in.shape[-3:]))
        except torch.cuda.OutOfMemoryError:
            if n == 1:
                raise
            print(f"[oom] chunk {n} did not fit; retrying at {n // 2}", flush=True)
            del seq
            engine = None
            torch.cuda.empty_cache()
            chunk = n // 2
            continue

        pred = pred / DOSE_SCALE                             # -> physical units
        for g in range(pred.shape[0]):
            frame = pred[g]
            if cutoffs is not None:
                cut = (MIN_CUTOFF_OVERRIDE if MIN_CUTOFF_OVERRIDE is not None
                       else float(cutoffs[i + g]))
                frame = torch.where(frame <= cut,
                                    frame.new_zeros(()), frame)
            # D2H the CROPPED volume and pad on the host: the padded grid is
            # ~2.7x the transfer (27.4 ms vs 10.0 ms measured) for slices that
            # are all zeros. `copy=False` because the tensor is already fp32 and
            # numpy's astype copies by default.
            with _phase("d2h", sync=True):
                arr = frame.cpu().numpy().astype(np.float32, copy=False)
            with _phase("post"):
                out.append(crop_and_pad_to_original(arr, pad_info, fill_value=0.0))
        del seq, pred
        i += n
    # empty_cache() per chunk cost ~14 ms (2.5 s over a beam) and bought
    # nothing: the allocator reuses these blocks immediately, since every
    # control point of a beam has the same shape. Release at the beam boundary
    # instead, where the next beam may want a different crop.
    del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# GC input discovery
# --------------------------------------------------------------------------
def find_metadata(input_dir):
    hits = glob.glob(os.path.join(input_dir, "stacked-*-beam-level-metadata.json"))
    if not hits:
        raise FileNotFoundError(
            f"No stacked-*-beam-level-metadata.json under {input_dir}")
    path = hits[0]
    treatment = "photon" if "photon" in os.path.basename(path) else "proton"
    return path, treatment


def detect_modality_tag(input_dir):
    dirs = glob.glob(os.path.join(
        input_dir, "images", "radiation-dose-calculation-source-*-image-*"))
    for d in dirs:
        m = re.search(r"source-(ct|mri)-image-", os.path.basename(d))
        if m:
            return m.group(1)
    raise FileNotFoundError(
        f"Could not detect ct/mri from image dirs under {input_dir}/images")


def read_image_slot(input_dir, modality_tag, image_file_idx):
    slot = image_file_idx + 1
    d = os.path.join(
        input_dir, "images",
        f"radiation-dose-calculation-source-{modality_tag}-image-{slot}")
    mhas = sorted(glob.glob(os.path.join(d, "*.mha")))
    if not mhas:
        raise FileNotFoundError(f"No .mha in image slot {slot}: {d}")
    return sitk.ReadImage(mhas[0])


# --------------------------------------------------------------------------
# GC output writing
# --------------------------------------------------------------------------
def _slot_output_path(output_dir, slot):
    """<output>/images/stacked-radiation-dose-map-<slot>/output.mha (matches example)."""
    outdir = os.path.join(output_dir, "images", f"stacked-radiation-dose-map-{slot}")
    os.makedirs(outdir, exist_ok=True)
    return os.path.join(outdir, "output.mha")


def _write_placeholder(output_dir, slot):
    """Trivial placeholder for an unused slot (same as the example submission)."""
    sitk.WriteImage(sitk.Image(1, 1, sitk.sitkFloat32),
                    _slot_output_path(output_dir, slot))


# --------------------------------------------------------------------------
# Top-level invoke
# --------------------------------------------------------------------------
def run_invoke(input_dir, output_dir, models, device):
    """Process the currently-mounted input and write the full output.

    Memory: work is grouped by output slot from the metadata alone, then each
    slot is computed, stacked and written before the next one starts — so peak
    memory is a SINGLE slot's frames, never all ten at once. (Accumulating
    every slot's full-volume dose in RAM was OOM-ing large runs.)
    """
    _invoke_t0 = time.time()
    meta_path, treatment = find_metadata(input_dir)
    if treatment != "photon":
        raise ValueError(f"This container only handles photon; got {treatment}")
    modality_tag = detect_modality_tag(input_dir)          # 'ct' / 'mri'
    modality = _TAG_TO_MODALITY[modality_tag]              # 'ct' / 'mr'
    corr = models[modality]

    with open(meta_path) as f:
        metadata = json.load(f)

    # Group per-CP work by output slot (metadata only, no dose yet). Each entry:
    # (idx_in_output, image_file_idx, beam_meta, cp_meta, minimum_cutoff).
    slot_work = {i: [] for i in range(N_OUTPUT_SLOTS)}
    _regions = {}
    for entry in metadata:
        ifi = entry["image_file_idx"]
        # The challenge supplies the site per entry. It selects the engine
        # calibration, so an unrecognised value is worth a warning rather than
        # a silent fallback -- see machine_config_for.
        region = entry.get("anatomical_region")
        _regions[str(region)] = _regions.get(str(region), 0) + 1
        for beam_meta in entry["beams"]:
            for cp_meta in beam_meta["control_points"]:
                oi = cp_meta["output_info"]
                slot_work[oi["output_file_idx"]].append(
                    (oi["idx_in_output"], ifi, beam_meta, cp_meta,
                     oi["minimum_cutoff"], region))
    print(f"[anatomy] regions in this invoke: "
          f"{', '.join(f'{k}:{v}' for k, v in sorted(_regions.items()))}", flush=True)
    for r in _regions:
        if _anatomy_key(r) is None:
            print(f"[anatomy] WARN unrecognised anatomical_region {r!r}; "
                  f"using the shipped calibration", flush=True)

    for slot in range(1, N_OUTPUT_SLOTS + 1):
        work = slot_work[slot - 1]
        if not work:
            _write_placeholder(output_dir, slot)
            continue

        work.sort(key=lambda w: w[0])
        idxs = [w[0] for w in work]
        if idxs != list(range(len(idxs))):
            raise ValueError(
                f"slot {slot}: idx_in_output not contiguous 0..n-1: {idxs}")

        # Every CP in one slot shares the same source image (as in the example).
        # Consecutive slots often share one too — a patient's three beams write
        # three slots off the same volume — so hold the last decode. That saves
        # a re-read and re-decode, and on the MR arm it is what keeps the sCT
        # cache warm: to_hu keys on array IDENTITY, so a fresh decode of the
        # same file would re-run a ~15 s network. Depth 1 on purpose: each entry
        # is ~57 MiB and holding all ten slots' volumes is what used to OOM.
        ifi = work[0][1]
        if _IMG_CACHE.get("key") != ifi:
          with _phase("image_read"):
            ref_img = read_image_slot(input_dir, modality_tag, ifi)
            _IMG_CACHE.clear()
            _IMG_CACHE.update(
                key=ifi, img=ref_img,
                arr=sitk.GetArrayFromImage(ref_img),
                origin=np.array(ref_img.GetOrigin())[::-1],
                # The engine grid IS the input image grid — read it, don't assume it.
                res=image_spacing_zyx(ref_img, f"slot {slot}"))
        ref_img = _IMG_CACHE["img"]
        image_arr = _IMG_CACHE["arr"]
        origin_zyx = _IMG_CACHE["origin"]
        resolution = _IMG_CACHE["res"]

        # Group this slot's CPs by their owning beam (shared geometry), so each
        # beam is preprocessed once and its control points run as one batched
        # sequence. Preserve idx_in_output so frames land in the right order.
        beam_groups = {}
        for idx, _ifi, beam_meta, cp_meta, cutoff, region in work:
            _, _r, entries = beam_groups.setdefault(
                id(beam_meta), (beam_meta, region, []))
            entries.append((idx, cp_meta, cutoff))

        frames = [None] * len(work)
        for beam_meta, region, entries in beam_groups.values():
            cp_list = [cp for (_i, cp, _c) in entries]
            cutoffs = [c for (_i, _cp, c) in entries]
            with _phase("to_hu"):
                beams, image_pad, density, mask, pad_info = build_beam_batch(
                    image_arr, origin_zyx, beam_meta, cp_list, modality,
                    resolution=resolution)
            doses = predict_beam_batch(
                corr, beams, image_pad, density, mask, pad_info, device,
                cutoffs=cutoffs, resolution=resolution, region=region)
            for (idx, _cp, _cutoff), dose in zip(entries, doses):
                frame = sitk.GetImageFromArray(dose)
                frame.CopyInformation(ref_img)   # match input grid exactly
                frames[idx] = frame
            del beams, image_pad, density, mask, doses

        # Compressed. MetaImage compression is zlib -- lossless, so it cannot
        # change a dose value -- and it is FASTER, not slower: post-cutoff
        # frames are ~99.8% zeros, so the write shrinks far more than the
        # compressor costs. Measured over 20 frames of a 246x246x249 grid,
        # uncompressed 527 ms / 1149.6 MiB against compressed 205 ms / 3.6 MiB,
        # bit-identical on read-back. (This file used to carry a comment
        # asserting the opposite; it was never measured.)
        with _phase("write"):
            sitk.WriteImage(sitk.JoinSeries(frames),
                            _slot_output_path(output_dir, slot), useCompression=True)
        # image_arr / ref_img stay alive in _IMG_CACHE for the next slot; the
        # cache itself is released after the loop.
        del frames
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Release the two per-invoke caches: together they pin up to two ~57 MiB
    # volumes plus a synthetic CT, and the next invoke has different patients.
    _IMG_CACHE.clear()
    _HU_CACHE.clear()
    gc.collect()
    _n_im = len({e["image_file_idx"] for e in metadata})
    _n_dm = sum(len(b["control_points"]) for e in metadata for b in e["beams"])
    # ALWAYS ON, and free: one wall-clock read around work that has already been
    # forced to completion by writing the outputs. No torch.cuda.synchronize()
    # of its own, so unlike the GC_TIMING breakdown below it cannot inflate the
    # number it reports. This is the quantity the leaderboard bills
    # (T = t_fix + N_im*t_im + N_dose_map*t_dose_map), so it belongs in every
    # run's log, not behind a flag that ships off.
    _wall = time.time() - _invoke_t0
    _per = 1000.0 * _wall / max(_n_dm, 1)
    _msg = (f"[timing] invoke: {_wall:.1f}s total | {_n_im} image(s), "
            f"{_n_dm} dose map(s) | {_per:.0f} ms per dose map")
    # Only extrapolate when there are enough dose maps for the per-beam engine
    # build to have amortised. At small N the fixed cost dominates -- a 1-dose-
    # map invoke reported 1567 ms against a ~577 ms steady state, and scaling
    # that to 181 gave a scary, meaningless "284s (hard limit 181s)".
    if _n_dm >= 20:
        _msg += (f" | 181-dose-map projection {_wall * 181.0 / _n_dm:.0f}s "
                 f"(hard limit 181s)")
    else:
        _msg += (f" | no projection: {_n_dm} dose map(s) is too few, the "
                 f"per-beam engine build has not amortised")
    print(_msg, flush=True)
    timing_report(n_images=_n_im, n_dose_maps=_n_dm)
