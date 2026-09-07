"""
doserad.evaluate
----------------
Per-patient and full-cohort evaluation pipeline, plus the CLI entry point.

Usage (CLI)
-----------
    python -m doserad.evaluate \
        --pred_dir   /data/predictions \
        --gt_dir     /data/ground_truth \
        --struct_dir /data/structures \
        --task       photon_ct \
        --output_dir ./results
"""

import os
import json
import argparse
import numpy as np
import SimpleITK as sitk
import pandas as pd
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from .io import load_dose_array, load_dose_sitk, load_structures, find_dose_file
from .metrics_beam import evaluate_beam_level
from .metrics_plan import (
    stratified_plan_mae,
    gamma_pass_rate_3d,
    dvh_clinical_score,
    PRESCRIPTION_DOSE_GY,
)

# ---------------------------------------------------------------------------
# Voxel spacings per task  (SimpleITK order: x, y, z)
# ---------------------------------------------------------------------------
SPACING: Dict[str, Tuple[float, ...]] = {
    'photon_ct':  (2.0, 2.0, 2.0),
    'photon_mri': (2.0, 2.0, 2.0),
    'proton_ct':  (1.0, 1.0, 3.0),
    'proton_mri': (1.0, 1.0, 3.0),
}


# ---------------------------------------------------------------------------
# Single patient
# ---------------------------------------------------------------------------

def evaluate_patient(pred_plan_path: str,
                     gt_plan_path: str,
                     struct_path: str,
                     structure_names: List[str],
                     ptv_index: int,
                     oar_indices: List[int],
                     spacing: Tuple[float, ...] = (2.0, 2.0, 2.0),
                     prescription_dose: float = PRESCRIPTION_DOSE_GY,
                     pred_beams: Optional[List[np.ndarray]] = None,
                     gt_beams: Optional[List[np.ndarray]] = None,
                     beam_axis: int = 0) -> Dict:
    """Run all Level 1 and Level 2 metrics for a single patient.

    Parameters
    ----------
    pred_plan_path:
        Path to the predicted full-plan dose file.
    gt_plan_path:
        Path to the ground-truth full-plan dose file.
    struct_path:
        Path to the structure mask array (``*.npy``).
    structure_names:
        Ordered list of structure names matching the last axis of the mask array.
    ptv_index:
        Index of the PTV in the structure array.
    oar_indices:
        Indices of OAR structures to consider (3 closest to PTV are used).
    spacing:
        Voxel spacing (x, y, z) in mm.
    prescription_dose:
        Prescription dose in Gy.
    pred_beams / gt_beams:
        Optional per-beam dose arrays for Level 1 metrics.
    beam_axis:
        Array axis along which each beam propagates.

    Returns
    -------
    dict
        All metric results for this patient.
    """
    results: Dict = {}

    # Load arrays
    gt_arr   = load_dose_array(gt_plan_path)
    pred_arr = load_dose_array(pred_plan_path)
    pred_arr = np.clip(pred_arr, 0.0, None)

    # Level 1 — beam metrics (optional)
    if pred_beams is not None and gt_beams is not None:
        results.update(evaluate_beam_level(pred_beams, gt_beams, beam_axis))

    # Level 2 — stratified MAE
    results.update(stratified_plan_mae(pred_arr, gt_arr, prescription_dose))

    # Level 2 — gamma pass rate (uses SimpleITK images)
    gt_sitk   = load_dose_sitk(gt_plan_path, spacing)
    pred_sitk = load_dose_sitk(pred_plan_path, spacing)

    # Propagate clamp into the sitk image without re-loading from disk
    pred_arr_clean = sitk.GetImageFromArray(np.clip(sitk.GetArrayFromImage(pred_sitk), 0.0, None))
    pred_arr_clean.CopyInformation(pred_sitk)

    results['gamma_pass_rate'] = gamma_pass_rate_3d(
        pred_arr_clean, gt_sitk
    )

    # Level 2 — DVH clinical score
    structures = load_structures(struct_path)
    results.update(dvh_clinical_score(
        pred_arr, gt_arr, structures, structure_names,
        ptv_index, oar_indices, prescription_dose,
    ))

    return results


# ---------------------------------------------------------------------------
# Full cohort
# ---------------------------------------------------------------------------

def evaluate_all_patients(pred_dir: str,
                          gt_dir: str,
                          struct_dir: str,
                          patient_ids: List[str],
                          structure_names: List[str],
                          ptv_index: int,
                          oar_indices: List[int],
                          task: str = 'photon_ct',
                          output_dir: str = './doserad_results') -> pd.DataFrame:
    """Evaluate an entire cohort and write per-patient and summary CSVs.

    Expected directory layout::

        <pred_dir>/<patient_id>/plan_dose.npy   (or .nii.gz / .mha)
        <gt_dir>/<patient_id>/plan_dose.npy
        <struct_dir>/<patient_id>/structures.npy

    Returns
    -------
    DataFrame
        One row per patient with all metric values.
    """
    os.makedirs(output_dir, exist_ok=True)
    spacing = SPACING.get(task, (2.0, 2.0, 2.0))

    all_results: List[Dict] = []

    for pid in tqdm(patient_ids, desc="Evaluating patients"):
        pred_path   = find_dose_file(pred_dir,   pid)
        gt_path     = find_dose_file(gt_dir,     pid)
        struct_path = os.path.join(struct_dir, pid, 'structures.npy')

        if pred_path is None or gt_path is None:
            print(f"[skip] {pid}: missing dose file")
            continue
        if not os.path.exists(struct_path):
            print(f"[skip] {pid}: missing structure file")
            continue

        try:
            row = evaluate_patient(
                pred_path, gt_path, struct_path,
                structure_names, ptv_index, oar_indices,
                spacing=spacing,
            )
            row['patient_id'] = pid
            all_results.append(row)
        except Exception as exc:
            print(f"[error] {pid}: {exc}")

    df = pd.DataFrame(all_results)

    per_patient_path = os.path.join(output_dir, f'per_patient_{task}.csv')
    df.to_csv(per_patient_path, index=False)

    # Summary statistics
    metric_cols = [
        'mae_high', 'mae_mid', 'mae_low', 'mae_stratified',
        'gamma_pass_rate', 'dvh_score', 'ptv_score', 'oar_group_score',
    ]
    if 'beam_mae_mean' in df.columns:
        metric_cols += ['beam_mae_mean', 'idd_distance_mean']

    summary = {
        col: {
            'mean':   df[col].mean(),
            'std':    df[col].std(),
            'median': df[col].median(),
            'q1':     df[col].quantile(0.25),
            'q3':     df[col].quantile(0.75),
        }
        for col in metric_cols if col in df.columns
    }

    summary_path = os.path.join(output_dir, f'summary_{task}.csv')
    pd.DataFrame(summary).T.to_csv(summary_path)

    print(f"\nResults written to {output_dir}/")
    print(pd.DataFrame(summary).T.to_string())

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='python -m doserad.evaluate',
        description='DoseRAD2026 Challenge — evaluation pipeline',
    )
    p.add_argument('--pred_dir',   required=True,  help='Predicted dose directory')
    p.add_argument('--gt_dir',     required=True,  help='Ground-truth dose directory')
    p.add_argument('--struct_dir', required=True,  help='Structure mask directory')
    p.add_argument('--output_dir', default='./doserad_results')
    p.add_argument('--task', default='photon_ct',
                   choices=list(SPACING.keys()))
    p.add_argument('--patient_list', default=None,
                   help='Text file with one patient ID per line')
    p.add_argument('--ptv_index', type=int, default=0)
    p.add_argument('--structure_names', default=None,
                   help='JSON file with ordered list of structure names')
    return p


def main() -> None:
    args = _build_parser().parse_args()

    # Patient IDs
    if args.patient_list and os.path.exists(args.patient_list):
        with open(args.patient_list) as f:
            patient_ids = [l.strip() for l in f if l.strip()]
    else:
        patient_ids = sorted(
            d for d in os.listdir(args.gt_dir)
            if os.path.isdir(os.path.join(args.gt_dir, d))
        )

    # Structure names
    if args.structure_names and os.path.exists(args.structure_names):
        with open(args.structure_names) as f:
            structure_names = json.load(f)
    else:
        structure_names = ['PTV'] + [f'OAR_{i}' for i in range(1, 20)]

    oar_indices = list(range(1, len(structure_names)))

    print(f"Task:     {args.task}")
    print(f"Patients: {len(patient_ids)}")
    print(f"Spacing:  {SPACING[args.task]} mm (x, y, z)")

    evaluate_all_patients(
        args.pred_dir, args.gt_dir, args.struct_dir,
        patient_ids, structure_names, args.ptv_index, oar_indices,
        task=args.task, output_dir=args.output_dir,
    )


if __name__ == '__main__':
    main()
