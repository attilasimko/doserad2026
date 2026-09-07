"""
doserad.metrics_plan
--------------------
Level 2 full-plan evaluation metrics.

Metrics
-------
stratified_plan_mae   MAE split into high / mid / low dose strata.
gamma_pass_rate_3d    3D local gamma index pass rate (1 % / 1 mm).
dvh_clinical_score    DVH-based clinical score (PTV + 3 closest OARs).
"""

import numpy as np
import SimpleITK as sitk
import pymedphys
from typing import Dict, List

from .io import DVH

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PRESCRIPTION_DOSE_GY = 60.0
GAMMA_DOSE_DIFF = 1.0    # 1 % local
GAMMA_DTA       = 1.0    # 1 mm
GAMMA_THRESHOLD = 0.1    # 10 % of Dmax


# ---------------------------------------------------------------------------
# Stratified MAE
# ---------------------------------------------------------------------------

def stratified_plan_mae(pred_plan: np.ndarray,
                        gt_plan: np.ndarray,
                        prescription_dose: float = PRESCRIPTION_DOSE_GY) -> Dict:
    """MAE split into three dose strata, each normalised by *prescription_dose*.

    Strata boundaries (fraction of prescription dose):
        high  >= 80 %
        mid    30 % – 80 %
        low    10 % – 30 %

    Returns
    -------
    dict with keys: mae_high, mae_mid, mae_low, mae_stratified
        All values are fractions (multiply by 100 for percentages).
        ``mae_stratified`` is the mean of the three strata MAEs.
    """
    boundaries = {
        'high': gt_plan >= 0.80 * prescription_dose,
        'mid':  (gt_plan >= 0.30 * prescription_dose) & (gt_plan < 0.80 * prescription_dose),
        'low':  (gt_plan >= 0.10 * prescription_dose) & (gt_plan < 0.30 * prescription_dose),
    }

    results: Dict = {}
    for name, mask in boundaries.items():
        if mask.any():
            mae = float(np.mean(np.abs(pred_plan[mask] - gt_plan[mask])) / prescription_dose)
        else:
            mae = float('nan')
        results[f'mae_{name}'] = mae

    valid = [results[k] for k in ('mae_high', 'mae_mid', 'mae_low')
             if not np.isnan(results[k])]
    results['mae_stratified'] = float(np.mean(valid)) if valid else float('nan')
    return results


# ---------------------------------------------------------------------------
# Gamma pass rate
# ---------------------------------------------------------------------------

def gamma_pass_rate_3d(pred_plan: sitk.Image,
                       gt_plan: sitk.Image,
                       dose_diff: float = GAMMA_DOSE_DIFF,
                       dta: float = GAMMA_DTA,
                       threshold: float = GAMMA_THRESHOLD
                       ) -> float:
    """3D local gamma index pass rate.

    Criteria: *dose_diff* % local / *dta* mm.
    Only voxels with GT dose >= *threshold* x Dmax are evaluated.

    Returns
    -------
    float
        Pass rate as a percentage (0-100), or ``nan`` if no voxels are evaluated.
    """
    gt_arr   = sitk.GetArrayFromImage(gt_plan).astype(np.float64)
    pred_arr = sitk.GetArrayFromImage(pred_plan).astype(np.float64)
    if float(gt_arr.max()) <= 0:
        return float('nan')

    # Real mm axes, not voxel indices: sitk spacing is (x, y, z), numpy is
    # (z, y, x). With bare indices, dta silently scales by the voxel size.
    spacing = tuple(reversed(gt_plan.GetSpacing()))
    coords  = tuple(np.arange(n, dtype=float) * s
                    for n, s in zip(gt_arr.shape, spacing))

    gamma = pymedphys.gamma(
        coords, gt_arr,             # reference
        coords, pred_arr,           # evaluation
        dose_percent_threshold=dose_diff,
        distance_mm_threshold=dta,
        lower_percent_dose_cutoff=threshold * 100.0,
        interp_fraction=10,
        max_gamma=2,
        local_gamma=True,
        ram_available=5 * 10**9, # x * gigabytes
        quiet=True,
    )

    valid = gamma[~np.isnan(gamma)]     # below-cutoff voxels are nan
    return float((valid <= 1.0).mean() * 100.0) if valid.size else float('nan')


# ---------------------------------------------------------------------------
# DVH clinical score
# ---------------------------------------------------------------------------

def dvh_clinical_score(pred_plan: np.ndarray,
                       gt_plan: np.ndarray,
                       structures: np.ndarray,
                       structure_names: List[str],
                       ptv_index: int,
                       oar_indices: List[int],
                       prescription_dose: float = PRESCRIPTION_DOSE_GY) -> Dict:
    """DVH-based clinical metric.

    PTV targets
    -----------
    D98  — dose to the coldest 2 % of PTV (coverage metric).
    V95  — volume fraction receiving >= 95 % of prescription dose.

    OAR constraints (3 OARs closest to PTV centroid)
    -------------------------------------------------
    D2   — dose to the hottest 2 % of OAR volume (near-max).
    Dmean — mean OAR dose.

    Scoring
    -------
    Each metric error is normalised by *prescription_dose* and expressed as a
    percentage.  PTV score and OAR group score are averaged with equal weight.

    Returns
    -------
    dict
        Per-structure errors and the combined ``dvh_score``.
    """
    results: Dict = {}

    # --- PTV ------------------------------------------------------------------
    ptv_mask = structures[..., ptv_index]
    ptv_centroid = None

    if ptv_mask.any():
        gt_dvh_ptv   = DVH(gt_plan,   ptv_mask)
        pred_dvh_ptv = DVH(pred_plan, ptv_mask)

        d98_error = abs(pred_dvh_ptv.D(0.98) - gt_dvh_ptv.D(0.98)) / prescription_dose * 100
        v95_error = abs(
            pred_dvh_ptv.V(0.95 * prescription_dose) -
            gt_dvh_ptv.V(0.95 * prescription_dose)
        ) * 100

        results['ptv_d98_error'] = d98_error
        results['ptv_v95_error'] = v95_error
        results['ptv_score']     = (d98_error + v95_error) / 2.0

        ptv_coords   = np.argwhere(ptv_mask > 0)
        ptv_centroid = ptv_coords.mean(axis=0)
    else:
        results['ptv_score'] = float('nan')

    # --- Select 3 closest OARs -----------------------------------------------
    valid_oars      = []
    oar_centroids   = []

    for oi in oar_indices:
        oar_mask = structures[..., oi]
        if oar_mask.any():
            coords = np.argwhere(oar_mask > 0)
            valid_oars.append(oi)
            oar_centroids.append(coords.mean(axis=0))

    if ptv_centroid is not None and valid_oars:
        distances = [np.linalg.norm(c - ptv_centroid) for c in oar_centroids]
        selected  = [valid_oars[i] for i in np.argsort(distances)[:3]]
    else:
        selected = valid_oars[:3]

    # --- OAR metrics ----------------------------------------------------------
    oar_scores: List[float] = []

    for oi in selected:
        oar_mask     = structures[..., oi]
        gt_dvh_oar   = DVH(gt_plan,   oar_mask)
        pred_dvh_oar = DVH(pred_plan, oar_mask)

        d2_error    = abs(pred_dvh_oar.D(0.02) - gt_dvh_oar.D(0.02))    / prescription_dose * 100
        dmean_error = abs(pred_plan[oar_mask > 0].mean() -
                          gt_plan[oar_mask > 0].mean())                   / prescription_dose * 100

        oar_score = (d2_error + dmean_error) / 2.0
        oar_scores.append(oar_score)

        name = structure_names[oi] if oi < len(structure_names) else f"OAR_{oi}"
        results[f'{name}_d2_error']    = d2_error
        results[f'{name}_dmean_error'] = dmean_error
        results[f'{name}_score']       = oar_score

    results['oar_group_score'] = float(np.mean(oar_scores)) if oar_scores else float('nan')

    # --- Combined score -------------------------------------------------------
    ptv_s = results.get('ptv_score', float('nan'))
    oar_s = results['oar_group_score']

    if not np.isnan(ptv_s) and not np.isnan(oar_s):
        results['dvh_score'] = (ptv_s + oar_s) / 2.0
    else:
        results['dvh_score'] = ptv_s if not np.isnan(ptv_s) else oar_s

    return results