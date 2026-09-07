#!/usr/bin/env python3
"""Validation-cohort evaluation of the bare pencil-beam engine, CT and MR.

Runs the same CorrectedDoseEngine the training and submission paths use — with
the stripped machine config from configs.py (calibrated tpr, no penumbra, no
head scatter, no profile corrections, no MLC transmission) — over every
validating patient, once per modality, and scores the challenge's level-1
metric (masked_beam_mae: MAE inside GT >= 10% of the beam max, normalised by
that max).

The CT and MR arms run the identical pipeline; the MR arm just feeds the
synthetic CT produced by sct.py. So the CT->MR delta printed at the end is a
direct measurement of what the sCT costs in dose accuracy, with nothing else
varying.

Voxel spacing is whatever each patient's own image header says (loaders reads
it per patient now); the spacing actually used is printed per patient so a
stray non-2 mm case is visible rather than silent.

Absolute scale: two numbers are reported —

  fixed   the MAE as-is, i.e. including whatever absolute-dose error the
          machine config's current gain leaves;
  LS      the MAE after ONE global scale, applied to every patient and both
          arms (the deployed reality: one gain for everyone).

That global scale is fitted to the AVERAGE PATIENT, not to the pooled voxel
population: an L1-optimal scale is computed per patient and the median across
patients is taken. Pooling raw voxels instead would let the biggest, hottest
plans set the gain for everyone. The per-patient spread is printed so you can
see how much of the residual is gain error that no single number can fix.

    # engine only, all three beams, every 4th control point
    python3 eval_validation.py --data_path $DATA_PATH

    # with a trained corrector, all control points, CT only
    python3 eval_validation.py --data_path $DATA_PATH --ckpt out/pb_ct_best.pt \\
        --cp_stride 1 --modalities ct
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "evaluation"))
from doserad2026_evaluator.metrics_beam import masked_beam_mae          # noqa: E402
from configs import (machine_config, DEFAULT_KERNEL_SIZE, MULTISLAB_DEFAULTS,
                     default_data_path)  # noqa: E402
from engines import CorrectedDoseEngine, DoseCorrectionModel            # noqa: E402
from loaders import (                                                   # noqa: E402
    crop_and_pad_to_original, get_patients, load_beam_segment,
)

N_CPS = 180


def l1_optimal_scale(pred, true):
    """Scale s minimising sum|true - s*pred| — the pred-weighted median of true/pred."""
    p = np.asarray(pred, np.float64).ravel()
    t = np.asarray(true, np.float64).ravel()
    keep = p > 0
    p, t = p[keep], t[keep]
    if p.size == 0:
        return 1.0
    order = np.argsort(t / p)
    r, w = (t / p)[order], p[order]
    cw = np.cumsum(w)
    idx = int(np.searchsorted(cw, cw[-1] * 0.5))
    return float(r[min(idx, r.size - 1)])


def build_corrector(args, device):
    if not args.ckpt:
        return None
    model = DoseCorrectionModel(
        in_channels=args.in_channels, base_channels=args.base_channels,
        depth=args.unet_depth, use_gain=not args.no_gain_head,
        attention=args.attention, refine=args.refine, residual=args.residual,
    )
    ckpt = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(ckpt["dose_model_state_dict"], strict=True)
    return model.to(device).to(torch.float32).eval()


ENGINE_KW = {}


@torch.no_grad()
def beam_dose(segs, corrector, device):
    """Accumulate one beam's predicted and ground-truth dose on the original grid."""
    orig = segs[0][6]["original_shape"]
    pred = np.zeros(orig, np.float32)
    true = np.zeros(orig, np.float32)
    for beam, _image, density, dose, mask, resolution, pad_info, _iso in segs:
        true += crop_and_pad_to_original(dose.numpy(), pad_info, 0.0)
        beam = beam.to(device)
        engine = CorrectedDoseEngine(
            machine_config=machine_config, kernel_size=DEFAULT_KERNEL_SIZE,
            dose_grid_shape=density.shape[-3:], dose_grid_spacing=resolution,
            beam_template=beam, fluence_correction_model=None,
            dose_correction_model=corrector, device=device, **ENGINE_KW,
        ).to(device)
        d = engine.forward(
            leaf_positions=beam.leaf_positions.unsqueeze(0).unsqueeze(0),
            mus=beam.mu.unsqueeze(0).unsqueeze(0),
            jaw_positions=beam.jaw_positions.unsqueeze(0).unsqueeze(0),
            density_image=density.to(device).unsqueeze(0),
            return_per_beam=False,
        ) * mask.to(device)
        pred += crop_and_pad_to_original(d.squeeze(0).cpu().numpy(), pad_info, 0.0)
        del engine
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pred, true


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_path", default=None)
    ap.add_argument("--cohort", default="validating")
    ap.add_argument("--modalities", default="ct,mr")
    ap.add_argument("--anatomies", default="1THB,1ABB")
    ap.add_argument("--patients", default=None, help="Comma-separated ids; overrides --anatomies.")
    ap.add_argument("--beams", default="0,1,2")
    ap.add_argument("--cp_stride", type=int, default=4,
                    help="Score every Nth control point. Pred and GT are strided "
                         "identically, so the metric is consistent but the beam is a "
                         "subsample of the clinical one. Use 1 for the real thing.")
    ap.add_argument("--max_patients", type=int, default=None)
    ap.add_argument("--sub_n", type=int, default=20000,
                    help="Voxels sampled per beam for the pooled gain fit.")
    # corrector (optional)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--in_channels", type=int, default=2)
    ap.add_argument("--base_channels", type=int, default=8)
    ap.add_argument("--unet_depth", type=int, default=4)
    ap.add_argument("--refine", action="store_true")
    ap.add_argument("--residual", action="store_true")
    ap.add_argument("--no_gain_head", action="store_true")
    ap.add_argument("--attention", default="none")
    # engine overrides (the point of an engine evaluation is to sweep these)
    ap.add_argument("--tpr", type=float, default=None)
    ap.add_argument("--gain", type=float, default=None,
                    help="Override mean_photon_energy_MeV (the absolute-dose gain).")
    ap.add_argument("--multislab", action="store_true",
                    help="Radiological-depth heterogeneity correction on the PB baseline.")
    ap.add_argument("--mu_eff", type=float, default=MULTISLAB_DEFAULTS["mu_eff"])
    ap.add_argument("--lateral_scatter", action="store_true",
                    help="Density-scaled lateral spreading (the lung fix).")
    ap.add_argument("--lat_sigma_mm", default=None,
                    help="Scalar, or comma-separated per-density-bin sigmas in mm. "
                         "Default: the fitted vector in configs.MULTISLAB_DEFAULTS.")
    ap.add_argument("--lat_cap_mm", type=float, default=MULTISLAB_DEFAULTS["lat_cap_mm"])
    ap.add_argument("--lat_coarse", action="store_true",
                    help="Use pydosert's 3-bin lateral scatter instead of the fine bins.")
    args = ap.parse_args()

    global machine_config
    upd = {}
    if args.tpr is not None:
        upd["tpr_20_10"] = args.tpr
    if args.gain is not None:
        upd["mean_photon_energy_MeV"] = args.gain
    if upd:
        machine_config = machine_config.model_copy(update=upd)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data_path = args.data_path or default_data_path()

    modalities = [m.strip() for m in args.modalities.split(",") if m.strip()]
    beams = [int(b) for b in args.beams.split(",") if b.strip()]
    prefixes = tuple(p.strip() for p in args.anatomies.split(",") if p.strip())
    cp_path = f"{data_path}/photon/{args.cohort}"

    patients = get_patients(data_path, "photon", args.cohort, prefixes)
    if args.patients:
        wanted = set(p.strip() for p in args.patients.split(","))
        patients = [p for p in patients if p in wanted]
    if args.max_patients:
        patients = patients[: args.max_patients]

    print(f"config: tpr={machine_config.tpr_20_10} "
          f"E_MeV={machine_config.mean_photon_energy_MeV} "
          f"penumbra={machine_config.penumbra_fwhm} "
          f"head_scatter={machine_config.head_scatter_amplitude} "
          f"profile={machine_config.profile_corrections is not None} "
          f"mlc_t={machine_config.mlc_transmission} kernel={DEFAULT_KERNEL_SIZE}")
    print(f"corrector: {args.ckpt or 'none (engine only)'}")
    print(f"{len(patients)} {args.cohort} patients x {modalities} x beams {beams} "
          f"@ cp_stride {args.cp_stride}\n", flush=True)

    global ENGINE_KW
    if args.lat_sigma_mm is None:
        lat_sigma = MULTISLAB_DEFAULTS["lat_sigma_mm"]
    elif "," in str(args.lat_sigma_mm):
        lat_sigma = [float(v) for v in str(args.lat_sigma_mm).split(",")]
    else:
        lat_sigma = float(args.lat_sigma_mm)
    ENGINE_KW = dict(multislab=args.multislab, mu_eff=args.mu_eff,
                     lateral_scatter=args.lateral_scatter,
                     lat_sigma_mm=lat_sigma, lat_cap_mm=args.lat_cap_mm,
                     lat_fine=not args.lat_coarse)
    print(f"engine: multislab={args.multislab} mu_eff={args.mu_eff} "
          f"lateral_scatter={args.lateral_scatter} "
          f"sigma={lat_sigma} cap={args.lat_cap_mm} "
          f"bins={'coarse(pydosert)' if args.lat_coarse else 'fine'}")

    corrector = build_corrector(args, device)
    rng = np.random.default_rng(0)
    rows = []          # (patient, anat, modality, beam, mae_fixed, pred_s, true_s, beam_max)
    spacings = {}
    t0 = time.time()

    for pi, patient in enumerate(patients, 1):
        anat = "thorax" if patient.startswith("1THB") else "abdomen"
        for modality in modalities:
            for b in beams:
                segs = []
                for cp in range(0, N_CPS, args.cp_stride):
                    try:
                        segs.append(load_beam_segment(cp_path, patient, b, cp, modality))
                    except Exception:
                        continue
                if not segs:
                    continue
                spacings[(patient, modality)] = tuple(segs[0][5])
                pred, true = beam_dose(segs, corrector, device)
                if true.max() <= 0:
                    continue
                bmax = float(true.max())
                m = true >= 0.1 * bmax
                pm, tm = pred[m], true[m]
                idx = rng.choice(pm.size, min(args.sub_n, pm.size), replace=False)
                rows.append((patient, anat, modality, b,
                             masked_beam_mae(pred, true),
                             pm[idx].astype(np.float32), tm[idx].astype(np.float32), bmax))
                del segs, pred, true
        sp = spacings.get((patient, modalities[0]))
        print(f"[{pi:3d}/{len(patients)}] {patient} ({anat}) spacing(z,y,x)={sp} "
              f"({time.time() - t0:.0f}s)", flush=True)

    if not rows:
        print("nothing scored")
        return

    # ONE global gain for everyone, fitted to the average patient: per-patient
    # L1-optimal scale first, then the median across patients. (Concatenating
    # every patient's voxels and fitting once would weight patients by how many
    # hot voxels they have, i.e. let the biggest plans set the gain.)
    fit_mod = "ct" if any(r[2] == "ct" for r in rows) else modalities[0]
    per_patient_scale = {}
    for patient, _anat, modality, _b, _f, pm, tm, _bmax in rows:
        if modality == fit_mod:
            per_patient_scale.setdefault(patient, [[], []])
            per_patient_scale[patient][0].append(pm)
            per_patient_scale[patient][1].append(tm)
    scales = {p: l1_optimal_scale(np.concatenate(v[0]), np.concatenate(v[1]))
              for p, v in per_patient_scale.items()}
    S = float(np.median(list(scales.values())))

    def agg(sel):
        fixed = [r[4] for r in rows if sel(r)]
        ls = [float(np.mean(np.abs(S * r[5] - r[6])) / r[7]) for r in rows if sel(r)]
        return (np.mean(fixed) if fixed else float("nan"),
                np.mean(ls) if ls else float("nan"), len(fixed))

    print(f"\n===== masked_beam_mae, {args.cohort} cohort "
          f"({'engine only' if corrector is None else os.path.basename(args.ckpt)}) =====")
    sv = np.array(list(scales.values()))
    print(f"global gain (median of per-patient L1 scales on '{fit_mod}', "
          f"n={sv.size}): scale={S:.4f}")
    print(f"  -> mean_photon_energy_MeV = "
          f"{S * float(machine_config.mean_photon_energy_MeV):.4f}  "
          f"(current {float(machine_config.mean_photon_energy_MeV):.4f})")
    print(f"  per-patient scale spread: min {sv.min():.4f}  p25 {np.percentile(sv, 25):.4f}  "
          f"p75 {np.percentile(sv, 75):.4f}  max {sv.max():.4f}  "
          f"(+-{100 * 0.5 * (sv.max() - sv.min()) / S:.1f}% about the median)")
    print(f"\n{'modality':>9} {'anatomy':>9} {'beams':>6} {'MAE fixed':>10} {'MAE LS':>9}")
    for modality in modalities:
        for anat in ("thorax", "abdomen", "all"):
            sel = (lambda r, mo=modality, a=anat:
                   r[2] == mo and (a == "all" or r[1] == a))
            f_, l_, n = agg(sel)
            if n:
                print(f"{modality:>9} {anat:>9} {n:>6} {f_:>10.4f} {l_:>9.4f}")

    if len(modalities) > 1 and "ct" in modalities and "mr" in modalities:
        print("\n===== sCT cost: per-patient MR minus CT (LS-scaled MAE) =====")
        per = {}
        for patient, anat, modality, b, _f, pm, tm, bmax in rows:
            per.setdefault((patient, anat), {}).setdefault(modality, []).append(
                float(np.mean(np.abs(S * pm - tm)) / bmax))
        deltas = []
        for (patient, anat), d in sorted(per.items()):
            if "ct" in d and "mr" in d:
                c, m_ = np.mean(d["ct"]), np.mean(d["mr"])
                deltas.append(m_ - c)
                print(f"  {patient:>10} ({anat:>7})  CT {c:.4f}  MR {m_:.4f}  "
                      f"delta {m_ - c:+.4f}")
        if deltas:
            print(f"  mean delta {np.mean(deltas):+.4f}   "
                  f"median {np.median(deltas):+.4f}   worst {np.max(deltas):+.4f}")

    odd = {k: v for k, v in spacings.items() if v != (2.0, 2.0, 2.0)}
    print(f"\nspacings used: {sorted(set(spacings.values()))}"
          + (f"  NON-2mm: {odd}" if odd else ""))


if __name__ == "__main__":
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    main()
