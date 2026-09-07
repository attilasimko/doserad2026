#!/usr/bin/env python3
"""Plan-level (Level-2) evaluation on the validating cohort, CT and MR.

For each patient and each modality it sums ALL control points of one beam
(default beam 0) into a full 3-D plan dose, then scores that against the
reference with the challenge's Level-2 metrics:

  * stratified plan MAE   (high >=80% Rx, mid 30-80%, low 10-30%, equal weight)
  * 3D local gamma 1%/1mm, full volumes, settings identical to
    evaluation/doserad2026_evaluator/metrics_plan.py

Six patients x two arms = 12 plans, 12 gamma calls. The MR arm feeds the
synthetic CT from sct.py — nothing else differs between the arms, so the
CT->MR gap is exactly what the sCT costs.

    # engine only, both arms
    python3 eval_plans.py

    # with a trained corrector
    python3 eval_plans.py --ckpt out/pb_ct_best.pt --refine --residual --attention se

    # quick look: CT only, one patient
    python3 eval_plans.py --modalities ct --patients 1ABB045

Gamma is genuinely slow (~350 s per plan on an 11 M-voxel volume; that is what
1%/1mm with interp_fraction=10 on a 2 mm grid costs, and it is the same cost
the official evaluator pays). Use --no_gamma for a fast MAE-only pass.
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
from configs import (machine_config, DEFAULT_KERNEL_SIZE,  # noqa: E402
                     MULTISLAB_DEFAULTS)
from engines import CorrectedDoseEngine, load_corrector                       # noqa: E402
from loaders import (                                                        # noqa: E402
    crop_and_pad_to_original, get_patients, load_beam_segment,
)
from train_correction import (                                               # noqa: E402
    _gamma_pass_rate_3d, _ls_scale, _stratified_plan_mae,
)

N_CPS = 180


def build_corrector(args, device):
    """Load a corrector checkpoint; its architecture is inferred from the weights,
    so --refine/--attention/... need not be passed and cannot be got wrong."""
    if not args.ckpt:
        return None
    return load_corrector(args.ckpt, device)


@torch.no_grad()
def plan_dose(cp_path, patient, beam_index, modality, corrector, device, engine_kw):
    """Sum every control point of one beam into a full plan dose on the original grid."""
    pred = true = None
    n = 0
    for cp in range(N_CPS):
        try:
            beam, img, dens, dose, mask, res, pad, iso = load_beam_segment(
                cp_path, patient, beam_index, cp, modality)
        except Exception:
            continue
        b = beam.to(device)
        engine = CorrectedDoseEngine(
            machine_config=machine_config, kernel_size=DEFAULT_KERNEL_SIZE,
            dose_grid_shape=dens.shape[-3:], dose_grid_spacing=res, beam_template=b,
            fluence_correction_model=None, dose_correction_model=corrector,
            device=device, **engine_kw).to(device)
        P = (engine.forward(
                leaf_positions=b.leaf_positions[None, None], mus=b.mu[None, None],
                jaw_positions=b.jaw_positions[None, None],
                density_image=dens.to(device)[None]) * mask.to(device)
             ).squeeze(0).cpu().numpy()
        del engine
        pf = crop_and_pad_to_original(P, pad, 0.0)
        tf = crop_and_pad_to_original(dose.numpy(), pad, 0.0)
        pred = pf if pred is None else pred + pf
        true = tf if true is None else true + tf
        n += 1
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, true, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path",
                    default=None)
    ap.add_argument("--cohort", default="validating")
    ap.add_argument("--modalities", default="ct,mr")
    ap.add_argument("--anatomies", default="1THB,1ABB")
    ap.add_argument("--patients", default=None)
    ap.add_argument("--beam", type=int, default=0)
    ap.add_argument("--no_gamma", action="store_true",
                    help="Skip gamma (~350 s/plan) and report stratified MAE only.")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--in_channels", type=int, default=2,
                    help="3 for multislab (dose, density, radiological depth).")
    ap.add_argument("--base_channels", type=int, default=8)
    ap.add_argument("--unet_depth", type=int, default=4)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--no_gain_head", action="store_true")
    ap.add_argument("--attention", default="none")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cp_path = f"{args.data_path}/photon/{args.cohort}"
    mods = [m.strip() for m in args.modalities.split(",") if m.strip()]
    prefixes = tuple(p.strip() for p in args.anatomies.split(",") if p.strip())
    patients = get_patients(args.data_path, "photon", args.cohort, prefixes)
    if args.patients:
        want = set(p.strip() for p in args.patients.split(","))
        patients = [p for p in patients if p in want]

    engine_kw = dict(multislab=True, lateral_scatter=True, **MULTISLAB_DEFAULTS)
    corrector = build_corrector(args, device)

    print(f"engine: multislab pencil beam | "
          f"tpr={machine_config.tpr_20_10} gain={machine_config.mean_photon_energy_MeV} "
          f"penumbra={machine_config.penumbra_fwhm} multislab+lateral_scatter")
    print(f"corrector: {args.ckpt or 'none (engine only)'}")
    print(f"{len(patients)} patients x {mods}, beam {args.beam}, all {N_CPS} control points "
          f"summed -> {len(patients)*len(mods)} plans\n")
    print(f"{'patient':>10} {'arm':>4} {'CPs':>4} {'scale':>7} {'planMAE':>8} "
          f"{'high':>7} {'mid':>7} {'low':>7} {'gamma':>8} {'t':>7}")

    rows, scales, t_all = [], [], time.time()
    for patient in patients:
        for mod in mods:
            t0 = time.time()
            pred, true, n = plan_dose(cp_path, patient, args.beam, mod,
                                      corrector, device, engine_kw)
            if pred is None or true.max() <= 0:
                print(f"{patient:>10} {mod:>4}   -- no dose, skipped")
                continue
            # One global scale per plan, as the training loop does. The corrector
            # predicts absolute dose, so this is ~1 for a trained model and shows
            # up as a residual gain error when it is not.
            s = 1.0 if corrector is not None else _ls_scale(pred, true)
            rx = float(true.max())
            strat = _stratified_plan_mae(pred * s, true, rx)
            g = float("nan")
            if not args.no_gamma:
                g = _gamma_pass_rate_3d(pred * s, true, voxel_mm=2.0, prescription=rx,
                                        random_subset=None, return_map=False)
            scales.append(s)
            anat = "thorax" if patient.startswith("1THB") else "abdomen"
            rows.append((patient, anat, mod, strat, g))
            print(f"{patient:>10} {mod:>4} {n:>4} {s:7.3f} {strat['combined']:8.4f} "
                  f"{strat['high']:7.4f} {strat['mid']:7.4f} {strat['low']:7.4f} "
                  f"{g:7.2f}% {time.time()-t0:6.0f}s", flush=True)

    if not rows:
        print("nothing scored")
        return
    if corrector is None and scales:
        med = float(np.median(scales))
        if abs(med - 1.0) > 0.10:
            print(f"\n  !! median plan scale {med:.3f}, not ~1.0: the engine's ABSOLUTE "
                  f"gain is off by {med:.2f}x -- check mean_photon_energy_MeV "
                  f"in configs.py.\n"
                  f"     Every number above is scale-corrected, so this is invisible "
                  f"in the MAE and gamma columns.")
    print(f"\n===== summary ({time.time()-t_all:.0f}s total) =====")
    print(f"{'group':>16} {'n':>3} {'planMAE':>9} {'high':>8} {'mid':>8} {'low':>8} {'gamma':>9}")

    def summarise(label, sel):
        r = [x for x in rows if sel(x)]
        if not r:
            return
        g = [x[4] for x in r if x[4] == x[4]]
        print(f"{label:>16} {len(r):>3} {np.mean([x[3]['combined'] for x in r]):9.4f} "
              f"{np.mean([x[3]['high'] for x in r]):8.4f} "
              f"{np.mean([x[3]['mid'] for x in r]):8.4f} "
              f"{np.mean([x[3]['low'] for x in r]):8.4f} "
              f"{(np.mean(g) if g else float('nan')):8.2f}%")

    for mod in mods:
        summarise(mod.upper(), lambda x, m=mod: x[2] == m)
        for anat in ("thorax", "abdomen"):
            summarise(f"  {mod}/{anat}", lambda x, m=mod, a=anat: x[2] == m and x[1] == a)
    summarise("ALL", lambda x: True)

    if len(mods) > 1 and "ct" in mods and "mr" in mods:
        print("\nsCT cost (MR minus CT, per patient):")
        by = {}
        for patient, anat, mod, strat, g in rows:
            by.setdefault(patient, {})[mod] = (strat["combined"], g)
        dm, dg = [], []
        for patient, d in sorted(by.items()):
            if "ct" in d and "mr" in d:
                dm.append(d["mr"][0] - d["ct"][0])
                if d["ct"][1] == d["ct"][1] and d["mr"][1] == d["mr"][1]:
                    dg.append(d["mr"][1] - d["ct"][1])
                print(f"  {patient:>10}  MAE {d['ct'][0]:.4f} -> {d['mr'][0]:.4f} "
                      f"({dm[-1]:+.4f})   gamma {d['ct'][1]:.2f}% -> {d['mr'][1]:.2f}%")
        if dm:
            print(f"  mean dMAE {np.mean(dm):+.4f}"
                  + (f"   mean dgamma {np.mean(dg):+.2f}pp" if dg else ""))


if __name__ == "__main__":
    main()
