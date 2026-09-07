#!/usr/bin/env python3
"""Stamp a training checkpoint into the servable weights the container ships.

A checkpoint out of `train_correction.py` records the weights but not the
architecture, and for the current models the architecture is NOT recoverable
from the weights:

  * `--v1_norm group` changes the state_dict KEYS, so a shape-matching loader
    that guesses `batch` fails outright;
  * `--v1_bounded_residual` changes the OUTPUT ALGEBRA and leaves no trace in
    the weights at all, so a loader that guesses wrong silently serves
    `dose*gain + raw_residual` instead of the bounded, relu'd form;
  * `--v2_features` decides whether the engine builds the 12-channel BEV stack
    and passes `hu_image`/`open_area_mm2`. Get it wrong and the stem sees 2
    channels where it wants 13.
  * `--engine` decides WHICH PHYSICS BASELINE the residual is a residual of.
    With `--v2_features` the corrector takes 13 channels under either engine,
    so a mismatch passes every shape check and simply produces a wrong dose:
    serving this model on a different physics baseline than it was trained on
    it was trained on takes per-CP masked MAE from 0.0118 to 0.0228 and plan
    gamma from 93.3% to 63.0%.

So the config is written INTO the checkpoint here, once, by the person who
knows which flags produced it. `gc_inference.build_model` then reads it back
and asserts the channel count against the FeatureConfig.

The shipped model is the 10-epoch `cmp_v1v2fix_556407` run: v1 trunk, v2
feature stack, v2 composite loss, multislab pencil beam with lateral scatter.

    python3 submission/bake_model.py \
        --ckpt ~/doserad_local/ckpt/final_v1v2fix.pt \
        --out models/photon_corrector.pt \
        --v2_features --v1_norm group --v1_bounded_residual \
        --engine multislab

Verify what a baked file claims without loading torch weights twice:

    python3 submission/bake_model.py --inspect models/photon_corrector.pt
"""
import argparse
import os
import sys

import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)


def inspect(path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt.get("model_config")
    sd = ckpt["dose_model_state_dict"]
    print(f"{os.path.basename(path)}")
    print(f"  tensors     : {len(sd)}  "
          f"({sum(v.numel() for v in sd.values()):,} parameters)")
    for k in ("experiment_name", "epoch", "best_val_lv1_beam_mae"):
        if k in ckpt:
            print(f"  {k:12s}: {ckpt[k]}")
    if cfg is None:
        print("  model_config: MISSING — gc_inference will fall back to the "
              "shape-inferring loader, which cannot represent group norm or "
              "bounded residual. Re-bake this file.")
        return 1
    print("  model_config:")
    for k in sorted(cfg):
        print(f"    {k:18s}= {cfg[k]!r}")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inspect", default=None,
                    help="Print a baked checkpoint's config and exit.")
    ap.add_argument("--ckpt", default=None, help="Training checkpoint to bake.")
    ap.add_argument("--out", default=None, help="Where to write the baked file.")
    ap.add_argument("--base_channels", type=int, default=8)
    ap.add_argument("--unet_depth", type=int, default=4)
    ap.add_argument("--v2_features", action="store_true")
    ap.add_argument("--v2_cond_mode", choices=("film", "channels", "off"),
                    default="channels")
    ap.add_argument("--feature_set", choices=("v2", "v3", "v4"), default="v2")
    ap.add_argument("--lattice_size", type=int, default=1)
    ap.add_argument("--terma", action="store_true")
    ap.add_argument("--terma_c1", type=float, default=0.9)
    ap.add_argument("--terma_c2", type=float, default=0.28)
    ap.add_argument("--cond_region", choices=("dose", "aperture"),
                    default="dose")
    ap.add_argument("--v1_material_embed_dim", type=int, default=0)
    ap.add_argument("--v1_deep_supervision", action="store_true")
    ap.add_argument("--v1_norm", choices=("batch", "group", "group_masked"),
                    default="batch")
    ap.add_argument("--v1_bounded_residual", action="store_true")
    ap.add_argument("--no_refine", action="store_true")
    ap.add_argument("--engine", choices=("multislab",), default="multislab",
                    help="The physics baseline the run was TRAINED on: "
                         "--multislab --lateral_scatter -> multislab, "
                         "Collapsed cone has been removed. "
                         "that is safe to guess.")
    args = ap.parse_args()

    if args.inspect:
        raise SystemExit(inspect(args.inspect))
    if not (args.ckpt and args.out):
        raise SystemExit("need --ckpt and --out (or --inspect)")

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)

    # Prefer the config the TRAINING RUN recorded. Re-deriving it from CLI flags
    # means retyping seven booleans that leave no trace in the weights, and
    # getting one wrong produces a servable model that returns a wrong dose. The
    # flags remain for checkpoints written before train_correction stored this.
    trained_cfg = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    if trained_cfg:
        cfg = {k: v for k, v in trained_cfg.items() if k != "arch"}
        if cfg.get("engine") and cfg["engine"] != args.engine:
            raise SystemExit(
                f"--engine {args.engine} contradicts the checkpoint, which was "
                f"trained on {cfg['engine']}. The corrector predicts the residual "
                f"of one specific baseline; serving it on the other is not a "
                f"small perturbation. Fix the flag, or retrain.")
        cfg["engine"] = args.engine
        print(f"[bake] using model_config recorded by "
              f"{ckpt.get('experiment_name', 'the training run')}")
    else:
        in_ch = 2                                     # base dose, density
        if args.v2_features:
            from features_v2 import FeatureConfig
            in_ch = 1 + FeatureConfig(
                cond_mode=args.v2_cond_mode,
                feature_set=args.feature_set,
                cond_region=args.cond_region).n_scalar_channels()
        cfg = {
            "in_channels": in_ch,
            "base_channels": args.base_channels,
            "depth": args.unet_depth,
            "norm": args.v1_norm,
            "bounded_residual": bool(args.v1_bounded_residual),
            "refine": not args.no_refine,
            "v2_features": bool(args.v2_features),
            "v2_cond_mode": args.v2_cond_mode,
            "feature_set": args.feature_set if args.v2_features else None,
            "cond_region": args.cond_region if args.v2_features else None,
            "lattice_size": int(args.lattice_size),
            "terma": bool(args.terma),
            "terma_c1": (float(args.terma_c1) if args.terma else None),
            "terma_c2_per_mm": (float(args.terma_c2) if args.terma else None),
            "material_embedding_dim": args.v1_material_embed_dim,
            "deep_supervision": args.v1_deep_supervision,
            "engine": args.engine,
        }
        print("[bake] checkpoint carries no model_config; using CLI flags")
    # Drop everything the server does not need. Optimiser state and the unused
    # fluence head are dead weight in a container image.
    baked = {"dose_model_state_dict": ckpt["dose_model_state_dict"],
             "model_config": cfg,
             "source_checkpoint": os.path.basename(args.ckpt)}
    for key in ("fluence_model_state_dict",
                "lateral_heterogeneity_model_state_dict"):
        if ckpt.get(key) is not None:
            baked[key] = ckpt[key]
    for k in ("experiment_name", "epoch", "best_val_lv1_beam_mae"):
        if k in ckpt:
            baked[k] = ckpt[k]

    # Build it once here so a mismatch is caught at bake time rather than at the
    # first control point of a scored submission.
    from engines import DoseCorrectionModel
    model = DoseCorrectionModel(
        in_channels=cfg["in_channels"], base_channels=cfg["base_channels"],
        depth=cfg["depth"], norm=cfg["norm"],
        bounded_residual=cfg["bounded_residual"], refine=cfg["refine"],
        additive_scale_frac=cfg.get("additive_scale_frac", 0.05),
        refine_mode=cfg.get("refine_mode", "parallel"),
        stem_stride=cfg.get("stem_stride", 1),
        use_gain=bool(cfg.get("use_gain", True)),
        material_embedding_dim=cfg.get("material_embedding_dim", 0),
        deep_supervision=cfg.get("deep_supervision", False),
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
        refine_depth_dilations=cfg.get("refine_depth_dilations"))
    model.load_state_dict(baked["dose_model_state_dict"], strict=True)
    # Rebuild embedded fixed priors at bake time as well, so a malformed or
    # incomplete training checkpoint cannot become a seemingly valid artifact.
    from fixed_priors import build_embedded_priors
    build_embedded_priors(baked, torch.device("cpu"))

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(baked, args.out)
    print(f"wrote {args.out}  ({os.path.getsize(args.out) / 1e6:.2f} MB)")
    inspect(args.out)


if __name__ == "__main__":
    main()
