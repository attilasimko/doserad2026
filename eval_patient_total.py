#!/usr/bin/env python3
"""All-beam (patient-total) Level-2 evaluation.

This is the quantity the challenge scores at Level 2: every control point of
EVERY beam summed into one dose volume per patient, then compared with the
reference. It is strictly harder than the per-beam number -- beams have
different isocentres, so their errors do not cancel. Measured historically, the
gap is about 2 points (89.51 per-beam vs 87.51 patient-total) and on the worst
patient it is 7 (83.64 vs 76.62).

Nothing else in the repo computes it. `eval_plans.py` and the validation pass
in `train_correction.py` both treat ONE BEAM summed over its control points as
a "plan". A script that produced `analysis_outputs/*/PATIENT_TOTAL/` existed in
August 2026 but was never committed and is gone, which is why this one is.

Unlike `eval_plans.py` this takes the model configuration explicitly rather
than inferring it from tensor shapes. Inference cannot work for the current
models: `--v1_norm group` changes the state_dict keys, and
`--v1_bounded_residual` changes the OUTPUT ALGEBRA while leaving no trace in
the weights at all, so a shape-matching loader would silently evaluate
`dose*gain + raw_residual` instead of the bounded, relu'd form and report a
wrong dose with no warning.

    python3 eval_patient_total.py \
        --data_path ~/doserad_f \
        --ckpt ~/doserad_local/ckpt/cmp_v1v2fix_556407_best.pt \
        --v2_features --v1_bounded_residual --v1_norm group

Gamma is ~350 s per patient on an 11 M-voxel volume; --no_gamma gives a fast
MAE-only pass.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from configs import machine_config, DEFAULT_KERNEL_SIZE, MULTISLAB_DEFAULTS  # noqa: E402
from engines import CorrectedDoseEngine, DoseCorrectionModel  # noqa: E402
from loaders import crop_and_pad_to_original, get_patients, load_beam_segment  # noqa: E402
from train_correction import _gamma_pass_rate_3d, _stratified_plan_mae  # noqa: E402
# The OFFICIAL Level-1 implementations, not a reimplementation. These are the
# two metrics the leaderboard scores per control point, so scoring against a
# local copy of the formula would defeat the point of the exercise.
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "evaluation"))
from doserad2026_evaluator.metrics_beam import (  # noqa: E402
    masked_beam_mae, idd_curve_distance)


def _official_gt(cp_path, patient, beam_index, cp):
    """The GT exactly as the evaluator sees it: full grid, no crop.

    `load_beam_segment` crops to the beam's work region, which discards ~3% of
    the dose in the low-level tail. Both pred and GT lose it symmetrically, so
    masked beam MAE is unaffected (it only looks at >=10% of max) but the IDD
    curve integrates the WHOLE transverse plane and comes out ~2x optimistic:
    measured 0.0030 against the official harness's 0.0059 on 1THB043 B0.
    Level 1 is therefore scored against this, not against the cropped array.

    The 1e5 matches `loaders.load_dose`, so pred and GT share units.
    """
    import SimpleITK as sitk
    f = os.path.join(cp_path, patient, "dose",
                     f"Dose_B{beam_index}_CP{cp:03d}.mha")
    return sitk.GetArrayFromImage(sitk.ReadImage(f)).astype(np.float64) * 1e5

N_CPS = 180


class _ZeroHead(torch.nn.Module):
    """Keep trunk execution intact but remove one trained output head's term."""

    def forward(self, features):
        return features[:, :1] * 0.0


def build_model(args, device):
    if not args.ckpt:
        return None, None
    feature_cfg = None
    in_ch = 3                                       # dose, density, rad_depth
    if args.v2_features:
        from features_v2 import FeatureConfig
        feature_cfg = FeatureConfig(cond_mode=args.v2_cond_mode,
                                    feature_set=args.feature_set,
                                    cond_region=args.cond_region,
                                    crop_relative_position=args.bev_crop)
        in_ch = 1 + feature_cfg.n_scalar_channels()  # raw dose + v2 stack
    # Read the checkpoint FIRST: the separable-convolution geometry is
    # architecture, not a knob, and there is no CLI flag for it here on
    # purpose -- taking it from anywhere but the checkpoint would let this
    # script evaluate a different network than the one that was trained.
    # Absent keys mean the dense 3x3x3 model, which is what every checkpoint
    # written before SeparableConv3d existed actually is.
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    _arch = ckpt.get("model_config") or {}
    if _arch.get("in_channels") is not None and int(_arch["in_channels"]) != in_ch:
        raise SystemExit(
            f"checkpoint expects {_arch['in_channels']} corrector channels but "
            f"the recorded feature configuration builds {in_ch}")
    model = DoseCorrectionModel(
        in_channels=in_ch,
        base_channels=args.base_channels,
        depth=args.unet_depth,
        norm=args.v1_norm,
        bounded_residual=args.v1_bounded_residual,
        refine=bool(_arch.get("refine", True)),
        # additive_scale_frac changes the OUTPUT ALGEBRA and no tensor at all,
        # so a checkpoint trained with a non-default --v2_alpha loads clean here
        # and is then served under the WRONG bound. Every checkpoint to date used
        # the 0.05 default, which is why this never bit; the first arm that moves
        # it would have been silently mis-served. use_gain does change the
        # state_dict (the gain head appears or does not), so a mismatch there is
        # caught by the load below -- passed anyway rather than relied on.
        additive_scale_frac=_arch.get("additive_scale_frac", 0.05),
        refine_mode=_arch.get("refine_mode", "parallel"),
        stem_stride=_arch.get("stem_stride", 1),
        use_gain=bool(_arch.get("use_gain", True)),
        material_embedding_dim=args.v1_material_embed_dim,
        # Aux heads are training-only but they are parameters, so the model has
        # to be built with them or the load reports them as unexpected.
        deep_supervision=args.v1_deep_supervision,
        pool_kernels=_arch.get("pool_kernels"),
        width_growth=_arch.get("width_growth", 2.0),
        downsample_mode=_arch.get("downsample_mode", "maxpool"),
        activation_checkpointing=False,
        sep_levels=_arch.get("sep_levels", 0),
        lateral_kernel=_arch.get("lateral_kernel", 3),
        depth_kernel=_arch.get("depth_kernel", 5),
        separable_refine=bool(_arch.get("separable_refine", False)),
        refine_scale_to_peak=bool(_arch.get("refine_scale_to_peak", False)),
        refine_hidden=_arch.get("refine_hidden", 16),
        refine_depth_dilations=_arch.get("refine_depth_dilations"),
    )
    sd = ckpt.get("dose_model_state_dict", ckpt)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    inherited = len(sd) - len(unexpected)
    print(f"[model] {os.path.basename(args.ckpt)}: {inherited}/{len(sd)} tensors, "
          f"in_channels={in_ch}, norm={args.v1_norm}, "
          f"bounded_residual={args.v1_bounded_residual}, "
          f"cond_region={args.cond_region}", flush=True)
    # cond_region redefines two input channels without changing the count or
    # the names, so a mismatch produces a plausible wrong dose and no error.
    # The checkpoint records what it was trained with; honour it or say why.
    _mc = ckpt.get("model_config") or {}
    _want = _mc.get("cond_region")
    if args.v2_features and _want and _want != args.cond_region:
        raise SystemExit(
            f"checkpoint was trained with cond_region={_want!r} but "
            f"--cond_region {args.cond_region!r} was passed. Two of the "
            f"thirteen input channels would differ by more than their whole "
            f"between-cohort range and nothing downstream would notice. Pass "
            f"--cond_region {_want} to evaluate this model honestly.")
    # The BASELINE the residual is predicted against. A mismatch here is the
    # failure that measured plan gamma 93.3 -> 63.0, and it leaves no trace in
    # the weights: the channel count and every shape are identical.
    for _key, _got, _flag in (
            ("lattice_size", args.lattice_size, "--lattice_size"),
            ("lattice_depth_mode", args.lattice_depth_mode,
             "--lattice_depth_mode"),
            ("terma", bool(args.terma), "--terma"),
            ("feature_set", args.feature_set, "--feature_set")):
        _exp = _mc.get(_key)
        if _key == "lattice_size":
            _exp = int(_exp or 1)
        elif _key == "lattice_depth_mode":
            _exp = _exp or "ray"
        elif _key == "terma":
            _exp = bool(_exp)
        elif _exp is None or not args.v2_features:
            continue
        if _exp != _got:
            raise SystemExit(
                f"checkpoint was trained with {_key}={_exp!r} but "
                f"{_flag} says {_got!r}. The corrector predicts the residual "
                f"of ONE baseline; evaluating it against another returns a "
                f"plausible wrong dose with no error. Pass {_flag} {_exp}.")
    for k in ("experiment_name", "epoch", "best_val_lv1_beam_mae"):
        if k in ckpt:
            print(f"[model]   {k}: {ckpt[k]}", flush=True)
    if missing or unexpected:
        # Loud, because a partial load here silently evaluates a part-random net.
        raise SystemExit(f"state_dict mismatch: {len(missing)} missing, "
                         f"{len(unexpected)} unexpected. First missing: "
                         f"{missing[:3]}, first unexpected: {unexpected[:3]}")
    return model.to(device).eval(), feature_cfg


def build_embedded_engine_priors(checkpoint_path, device):
    if not checkpoint_path:
        return None, None
    from fixed_priors import build_embedded_priors

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    fluence, lateral = build_embedded_priors(checkpoint, device)
    if fluence is not None:
        print(f"[prior] embedded fluence: "
              f"{sum(p.numel() for p in fluence.parameters()):,} params", flush=True)
    if lateral is not None:
        print(f"[prior] embedded heterogeneity: "
              f"{sum(p.numel() for p in lateral.parameters()):,} params", flush=True)
    return fluence, lateral


def apply_checkpoint_runtime_config(args):
    """Make an embedded model_config authoritative for evaluation."""
    if not args.ckpt:
        return args
    checkpoint = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = checkpoint.get("model_config") or {}
    if not cfg:
        return args
    for key, attr in (
            ("base_channels", "base_channels"), ("depth", "unet_depth"),
            ("norm", "v1_norm"), ("bounded_residual", "v1_bounded_residual"),
            ("material_embedding_dim", "v1_material_embed_dim"),
            ("deep_supervision", "v1_deep_supervision"),
            ("v2_features", "v2_features"), ("v2_cond_mode", "v2_cond_mode"),
            ("feature_set", "feature_set"), ("cond_region", "cond_region"),
            ("lattice_size", "lattice_size"),
            ("lattice_depth_mode", "lattice_depth_mode"),
            ("terma", "terma"), ("terma_c1", "terma_c1"),
            ("terma_c2_per_mm", "terma_c2"), ("bev_crop", "bev_crop"),
            ("bev_crop_margin_mm", "bev_crop_margin_mm"),
            ("bev_crop_min", "bev_crop_min"),
            ("early_bev_crop", "early_bev_crop")):
        if cfg.get(key) is not None:
            setattr(args, attr, cfg[key])
    return args


def build_fluence_model(args, device):
    """Load the focused tiny-fluence experiment without a dose model."""
    if not args.fluence_ckpt:
        return None
    from fixed_priors import build_tiny_fluence

    ckpt = torch.load(args.fluence_ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt.get("config") or {}
    sd = ckpt.get("model_state_dict", ckpt)
    model = build_tiny_fluence(cfg, sd, device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[fluence] {os.path.basename(args.fluence_ckpt)}: "
          f"epoch={ckpt.get('epoch', '?')} params={n_params:,} "
          f"base={cfg.get('base_channels', 8)} "
          f"gain_bound=[{1.0/float(cfg.get('max_gain', 1.5)):.3f},"
          f"{float(cfg.get('max_gain', 1.5)):.3f}]",
          flush=True)
    return model


def build_learned_scatter_model(args, device):
    """Load the focused conservative lateral-scatter experiment."""
    if not args.lateral_scatter_ckpt:
        return None
    from lateral_scatter import (
        TinyLateralHeterogeneityCorrection,
        TinyLateralScatterMixture,
    )

    ckpt = torch.load(args.lateral_scatter_ckpt, map_location="cpu",
                      weights_only=False)
    cfg = ckpt.get("config") or {}
    model_cls = (TinyLateralHeterogeneityCorrection
                 if cfg.get("architecture") == "heterogeneity"
                 else TinyLateralScatterMixture)
    model = model_cls(
        hidden_channels=int(cfg.get("hidden_channels", 4)),
        sigmas_mm=tuple(float(x) for x in cfg.get("sigmas_mm", (2, 5, 10))),
        max_fraction=float(cfg.get("max_fraction", 0.4)),
    )
    model.load_state_dict(ckpt.get("model_state_dict", ckpt), strict=True)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[scatter] {os.path.basename(args.lateral_scatter_ckpt)}: "
          f"epoch={ckpt.get('epoch', '?')} params={n_params:,} "
          f"sigmas_mm={model.sigmas_mm}", flush=True)
    return model.to(device).eval()


@torch.no_grad()
def patient_total(cp_path, patient, modality, model, feature_cfg, device,
                  engine_kw, beams, cps, beam_axis=0, crop_margin_mm=60.0,
                  fluence_model=None, lateral_scatter_model=None):
    """Sum EVERY control point of EVERY beam into one patient-total volume.

    Also returns the OFFICIAL Level-1 metrics per control point. Level 1 is
    scored per CP, not per summed beam, so these are collected inside the loop
    on the same arrays that feed the plan sum -- one pass, no second forward.
    """
    pred = true = None
    n = 0
    lv1_mae, lv1_idd = [], []
    seen = 0
    for beam_index in beams:
        for cp in cps:
            try:
                beam, img, dens, dose, mask, res, pad, iso = load_beam_segment(
                    cp_path, patient, beam_index, cp, modality,
                    crop_margin_mm=crop_margin_mm)
            except Exception:
                continue
            b = beam.to(device)
            engine = CorrectedDoseEngine(
                machine_config=machine_config, kernel_size=DEFAULT_KERNEL_SIZE,
                dose_grid_shape=dens.shape[-3:], dose_grid_spacing=res,
                beam_template=b, fluence_correction_model=fluence_model,
                lateral_scatter_model=lateral_scatter_model,
                dose_correction_model=model, device=device,
                v2_features=feature_cfg is not None,
                v2_feature_cfg=feature_cfg, **engine_kw).to(device)
            fwd = {}
            if feature_cfg is not None:
                # Same inputs training used. Withholding them silently degrades
                # the material embedding and zeroes the aperture conditioning.
                fwd["hu_image"] = img.to(device)[None]
                fwd["open_area_mm2"] = float(
                    (b.leaf_positions[..., 1] - b.leaf_positions[..., 0])
                    .clamp_min(0.0).sum().item() * 5.0)
            P = (engine.forward(
                    leaf_positions=b.leaf_positions[None, None],
                    mus=b.mu[None, None],
                    jaw_positions=b.jaw_positions[None, None],
                    density_image=dens.to(device)[None], **fwd
                 ) * mask.to(device)).squeeze(0).cpu().numpy()
            del engine
            pf = crop_and_pad_to_original(P, pad, 0.0)
            tf = crop_and_pad_to_original(dose.numpy(), pad, 0.0)
            try:
                gt_full = _official_gt(cp_path, patient, beam_index, cp)
            except Exception:
                gt_full = tf          # fall back rather than lose the CP
            lv1_mae.append(masked_beam_mae(pf, gt_full))
            lv1_idd.append(idd_curve_distance(pf, gt_full, beam_axis))
            pred = pf if pred is None else pred + pf
            # The plan sum takes the UNCROPPED gt, for the same reason the
            # Level-1 metrics do. `tf` is load_beam_segment's dose, which has
            # already been through the H-crop -- summing it compares a cropped
            # prediction against a cropped GT, so the dose the crop throws away
            # cancels on both sides and the crop becomes free. It is not free:
            # scored against the real GT the 30 mm crop alone costs 3.4 gamma
            # points and 0.00067 mae_low on 1ABB045, with a PERFECT corrector.
            true = gt_full if true is None else true + gt_full
            n += 1
        # Per-beam progress. Level 1 is scored PER CONTROL POINT, so the mean
        # over this beam's control points is the scored quantity restricted to
        # one beam, not an approximation of it. Gamma and the stratified plan
        # MAE are plan-level and stay on the patient line.
        k = len(lv1_mae) - seen
        if k:
            print(f"[beam] {patient} b{beam_index} n={k:3d} "
                  f"lv1_mae={float(np.mean(lv1_mae[-k:])):.5f} "
                  f"lv1_idd={float(np.mean(lv1_idd[-k:])):.5f}", flush=True)
            seen = len(lv1_mae)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return pred, true, n, lv1_mae, lv1_idd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--cohort", default="validating")
    ap.add_argument("--modality", default="ct")
    ap.add_argument("--anatomies", default="1THB,1ABB")
    ap.add_argument("--patients", default=None)
    ap.add_argument("--beams", default="0,1,2")
    ap.add_argument("--beam_axis", type=int, default=0,
                    help="Axis the beam travels along, for the official "
                         "IDD curve. 0 matches the evaluator default.")
    ap.add_argument("--cp_stride", type=int, default=1,
                    help="Debug only. Gamma on a strided sum is meaningless -- "
                         "the plan is then a fraction of the real dose.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--fluence_ckpt", default=None,
                    help="TinyFluenceCorrector checkpoint. May be evaluated "
                         "alone (--ckpt omitted) or before a dose corrector.")
    ap.add_argument("--lateral_scatter_ckpt", default=None,
                    help="TinyLateralScatterMixture checkpoint. Applied after "
                         "multislab/lattice/TERMA and before rotation back.")
    ap.add_argument("--base_channels", type=int, default=8)
    ap.add_argument("--unet_depth", type=int, default=4)
    ap.add_argument("--v2_features", action="store_true")
    ap.add_argument("--v2_cond_mode", choices=("film", "channels", "off"),
                    default="channels")
    ap.add_argument("--feature_set", choices=("v2", "v3", "v4"), default="v2")
    # Defaults to the ORIGINAL region: this script evaluates existing
    # checkpoints, and every one of them predates the correction. Pass
    # --cond_region aperture only for a model trained with it.
    ap.add_argument("--cond_region", choices=("dose", "aperture"),
                    default="dose")
    ap.add_argument("--lattice_size", type=int, default=1)
    ap.add_argument("--lattice_depth_mode", choices=("ray", "tile_mean"),
                    default="ray",
                    help="Radiological-depth representative used by each "
                         "lattice tile. Use tile_mean for the cheap surface "
                         "correction experiment.")
    ap.add_argument("--terma", action="store_true")
    ap.add_argument("--terma_c1", type=float, default=0.9)
    ap.add_argument("--terma_c2", type=float, default=0.28)
    ap.add_argument("--v1_material_embed_dim", type=int, default=0)
    ap.add_argument("--v1_deep_supervision", action="store_true")
    ap.add_argument("--bev_crop", action="store_true")
    ap.add_argument("--bev_crop_margin_mm", type=float, default=50.0)
    ap.add_argument("--bev_crop_min", type=int, default=32)
    ap.add_argument("--early_bev_crop", action="store_true")
    ap.add_argument("--bev_crop_per_sample", action="store_true")
    ap.add_argument("--v1_norm", choices=("batch", "group", "group_masked"),
                    default="batch")
    ap.add_argument("--v1_bounded_residual", action="store_true")
    ap.add_argument("--amp", choices=("off", "fp16", "bf16"), default="off",
                    help="Autocast dtype for the CORRECTOR forward, matching "
                         "gc_inference's GC_AMP. The physics stays fp32. Use "
                         "this to score what a serving-side speedup costs.")
    ap.add_argument("--crop_margin_mm", type=float, default=60.0,   # match the container, not training
                    help="Safety margin on the geometric H-axis crop, matching "
                         "loaders.load_beam_segment and gc_inference. Dose "
                         "outside it is a hard zero in the prediction, and the "
                         "official IDD curve is summed along THAT SAME axis, "
                         "so this is the dominant term in the IDD column: the "
                         "floor a perfect corrector cannot beat is 0.01010 at "
                         "30 mm, 0.00629 at 50, 0.00316 at 80, 0.00108 at 120.")
    ap.add_argument("--ablate_refine", action="store_true",
                    help="Drop the full-resolution dilated refine branch at "
                         "eval. It is 60%% of the corrector's MACs for 1.8%% of "
                         "its parameters, and refine=True in every checkpoint "
                         "this repo has ever produced -- it has never once been "
                         "ablated. This measures how much the TRAINED branch "
                         "actually contributes; it does not measure what a "
                         "model trained without it would do.")
    ap.add_argument("--ablate_trunk", action="store_true",
                    help="Zero the trained U-Net gain and residual heads while "
                         "leaving the refine branch untouched. This isolates "
                         "the refiner's accuracy contribution, but deliberately "
                         "does not skip trunk computation and therefore is not "
                         "a runtime benchmark. As with --ablate_refine, the "
                         "checkpoint was trained with both branches present.")
    ap.add_argument("--no_gamma", action="store_true")
    ap.add_argument("--out", default=None, help="Write a JSON summary here.")
    args = apply_checkpoint_runtime_config(ap.parse_args())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp_path = f"{args.data_path}/photon/{args.cohort}"
    prefixes = tuple(p.strip() for p in args.anatomies.split(",") if p.strip())
    patients = get_patients(args.data_path, "photon", args.cohort, prefixes)
    if args.patients:
        want = {p.strip() for p in args.patients.split(",")}
        patients = [p for p in patients if p in want]
    beams = [int(b) for b in args.beams.split(",")]
    cps = list(range(0, N_CPS, args.cp_stride))

    engine_kw = dict(multislab=True, lateral_scatter=True, **MULTISLAB_DEFAULTS)
    if args.bev_crop:
        engine_kw.update(bev_crop=True,
                         bev_crop_margin_mm=args.bev_crop_margin_mm,
                         bev_crop_per_sample=args.bev_crop_per_sample,
                         bev_crop_min=args.bev_crop_min,
                         early_bev_crop=args.early_bev_crop)
    elif args.early_bev_crop:
        # Independent of --bev_crop, exactly as in train_correction.py: this one
        # crops the physics work volume and scatters back bit-identically, so it
        # buys speed without changing what the corrector sees.
        engine_kw.update(early_bev_crop=True)
    if args.lattice_size > 1 or args.terma:
        engine_kw.update(
            lattice_size=args.lattice_size,
            lattice_depth_mode=args.lattice_depth_mode,
            terma_scaling=bool(args.terma),
            terma_c1=(args.terma_c1 if args.terma else None),
            terma_c2_per_mm=(args.terma_c2 if args.terma else None))
        # TERMA REPLACES the heuristic lateral-scatter term and the engine
        # raises if both are set. Training ran --terma without
        # --lateral_scatter; evaluating with both would not just crash, it would
        # be the wrong baseline for this corrector's residual.
        if args.terma:
            engine_kw["lateral_scatter"] = False
    if args.amp != "off":
        # Same knob gc_inference exposes as GC_AMP, applied the same way: the
        # corrector forward autocasts, the physics does not. Scoring it here is
        # the only way to find out whether the raw dose deviation it introduces
        # is coherent (which 1%/1mm gamma punishes) or scattered (which it
        # tolerates) -- the deviation magnitude alone cannot tell them apart.
        engine_kw["amp_dtype"] = (torch.float16 if args.amp == "fp16"
                                  else torch.bfloat16)
    model, feature_cfg = build_model(args, device)
    fluence_model = build_fluence_model(args, device)
    lateral_scatter_model = build_learned_scatter_model(args, device)
    embedded_fluence, embedded_lateral = build_embedded_engine_priors(args.ckpt, device)
    if fluence_model is not None and embedded_fluence is not None:
        raise SystemExit("checkpoint embeds a fluence prior; do not also pass --fluence_ckpt")
    if lateral_scatter_model is not None and embedded_lateral is not None:
        raise SystemExit("checkpoint embeds a heterogeneity prior; do not also pass "
                         "--lateral_scatter_ckpt")
    fluence_model = fluence_model or embedded_fluence
    lateral_scatter_model = lateral_scatter_model or embedded_lateral
    if args.ablate_refine and args.ablate_trunk:
        raise SystemExit("choose only one of --ablate_refine or --ablate_trunk")
    if args.ablate_refine:
        if getattr(model, "refine", None) is None:
            raise SystemExit("--ablate_refine but this model has no refine branch")
        # `correction = correction + self.refine(x)` (engines.py) is the branch's
        # ONLY contribution, so dropping the module removes exactly that term and
        # nothing else. The trunk is untouched and its weights were fitted WITH
        # the branch present, so this is an upper bound on the damage removing it
        # does -- a model retrained without it gets to compensate.
        n_ref = sum(q.numel() for q in model.refine.parameters())
        n_all = sum(q.numel() for q in model.parameters())
        model.refine = None
        print(f"[ablate] refine branch dropped: {n_ref} of {n_all} params "
              f"({100.0 * n_ref / n_all:.1f}%), ~60% of the corrector's MACs",
              flush=True)
    if args.ablate_trunk:
        if getattr(model, "refine", None) is None:
            raise SystemExit("--ablate_trunk needs a model with a refine branch")
        # The trunk reaches the final correction only through these two heads.
        # Zeroing them leaves exactly the trained refine contribution. Keeping
        # the trunk execution makes this an output/accuracy ablation rather
        # than a misleading runtime measurement.
        n_trunk = sum(q.numel() for q in model.trunk.parameters())
        n_heads = sum(q.numel() for q in model.residual_head.parameters())
        model.residual_head = _ZeroHead()
        if model.gain_head is not None:
            n_heads += sum(q.numel() for q in model.gain_head.parameters())
            model.gain_head = _ZeroHead()
        print(f"[ablate] U-Net contribution zeroed: trunk + heads contain "
              f"{n_trunk + n_heads} parameters; refine branch retained "
              f"(trunk still executes, so timing is not meaningful)", flush=True)

    print(f"{len(patients)} patients x {len(beams)} beams x {len(cps)} CPs "
          f"= {len(patients)*len(beams)*len(cps)} forwards", flush=True)

    rows, results = [], {}
    for patient in patients:
        pred, true, n, l1_mae, l1_idd = patient_total(
            cp_path, patient, args.modality, model, feature_cfg, device,
            engine_kw, beams, cps, args.beam_axis, args.crop_margin_mm,
            fluence_model=fluence_model,
            lateral_scatter_model=lateral_scatter_model)
        if pred is None:
            print(f"{patient}: no control points loaded", flush=True)
            continue
        rx = float(true.max())
        strat = _stratified_plan_mae(pred, true, rx)
        g = float("nan")
        if not args.no_gamma:
            # 1.1 preserves the exact pass/fail result while stopping the
            # search earlier for voxels that are already known to fail.
            g = _gamma_pass_rate_3d(pred, true, voxel_mm=2.0, prescription=rx,
                                    max_gamma=1.1)
        # nanmean: masked_beam_mae/idd_curve_distance return nan on an empty
        # beam, which is a real thing at the extreme CPs of a small aperture.
        m_mae = float(np.nanmean(l1_mae)) if l1_mae else float("nan")
        m_idd = float(np.nanmean(l1_idd)) if l1_idd else float("nan")
        rows.append((patient, g, strat["combined"], n, m_mae, m_idd))
        results[patient] = {"gamma_1_1": g, "stratified": strat, "n_cp": n,
                            "lv1_beam_mae": m_mae, "lv1_idd": m_idd}
        print(f"  {patient}: n={n:4d}  gamma={g:6.2f}%  "
              f"stratified_combined={strat['combined']:.5f}  "
              f"lv1_mae={m_mae:.5f}  lv1_idd={m_idd:.5f}", flush=True)

    if rows:
        gs = [r[1] for r in rows if r[1] == r[1]]
        cs = [r[2] for r in rows]
        ms = [r[4] for r in rows if r[4] == r[4]]
        ds = [r[5] for r in rows if r[5] == r[5]]
        print(f"\nPATIENT-TOTAL over {len(rows)} patients "
              f"({len(beams)} beams x {len(cps)} CPs each)")
        if gs:
            print(f"  gamma 1%/1mm   {np.mean(gs):.2f} +/- {np.std(gs):.2f}")
        print(f"  stratified MAE {np.mean(cs):.5f} +/- {np.std(cs):.5f}")
        # Level 1 is per-CP and anatomy-imbalanced cohorts skew a flat mean, so
        # the per-anatomy split is printed too -- it is what the leaderboard
        # spread actually reflects.
        if ms:
            print(f"  LEVEL-1 masked beam MAE {np.mean(ms):.5f} "
                  f"+/- {np.std(ms):.5f}")
        if ds:
            print(f"  LEVEL-1 IDD distance    {np.mean(ds):.5f} "
                  f"+/- {np.std(ds):.5f}")
        for pref in sorted({r[0][:4] for r in rows}):
            sub = [r for r in rows if r[0].startswith(pref)]
            sg = [r[1] for r in sub if r[1] == r[1]]
            gtxt = f"{np.mean(sg):6.2f}" if sg else "   nan"
            print(f"    [{pref}] n={len(sub):3d}  gamma={gtxt}  "
                  f"lv1_mae={np.nanmean([r[4] for r in sub]):.5f}  "
                  f"lv1_idd={np.nanmean([r[5] for r in sub]):.5f}")
        results["_mean"] = {"gamma_1_1": float(np.mean(gs)) if gs else None,
                            "stratified_combined": float(np.mean(cs)),
                            "lv1_beam_mae": float(np.mean(ms)) if ms else None,
                            "lv1_idd": float(np.mean(ds)) if ds else None,
                            "n_patients": len(rows)}
    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=1)
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
