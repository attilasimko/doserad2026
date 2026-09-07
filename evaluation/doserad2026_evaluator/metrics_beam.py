"""
doserad.metrics_beam
--------------------
Level 1 single-beam evaluation metrics.

Metrics
-------
masked_beam_mae     MAE restricted to the high-dose region, normalised by beam max.
idd_curve_distance  Normalised RMS difference of integrated depth-dose curves.
evaluate_beam_level Aggregate both metrics over a list of beams.
"""

import numpy as np
from typing import Dict, List, Tuple


def masked_beam_mae(pred_beam: np.ndarray, gt_beam: np.ndarray) -> float:
    """Masked MAE for a single beam.

    Computed only in voxels where the ground-truth dose is >= 10 % of the
    beam's max dose, then normalised by that max dose.

    Returns
    -------
    float
        MAE as a fraction (multiply by 100 for a percentage).
        ``nan`` if the beam is empty or the mask contains no voxels.
    """
    beam_max = float(np.max(gt_beam))
    if beam_max <= 0:
        return float('nan')

    mask = gt_beam >= 0.1 * beam_max
    if not mask.any():
        return float('nan')

    return float(np.mean(np.abs(pred_beam[mask] - gt_beam[mask])) / beam_max)


def compute_idd_curve(dose_3d: np.ndarray, beam_axis: int = 0) -> np.ndarray:
    """Integrated Depth-Dose (IDD) curve along *beam_axis*.

    Each value is the sum of dose across the transverse plane at that depth.
    """
    axes_to_sum = tuple(i for i in range(3) if i != beam_axis)
    return np.sum(dose_3d, axis=axes_to_sum)


def idd_curve_distance(pred_beam: np.ndarray, gt_beam: np.ndarray,
                       beam_axis: int = 0) -> float:
    """Normalised RMS difference between predicted and ground-truth IDD curves.

    Both curves are normalised by the GT IDD maximum before computing the RMS,
    so the result is dimensionless.

    Returns
    -------
    float
        Normalised RMS IDD distance, or ``nan`` if the GT curve is flat.
    """
    idd_pred = compute_idd_curve(pred_beam, beam_axis)
    idd_gt   = compute_idd_curve(gt_beam,   beam_axis)

    idd_max = float(np.max(idd_gt))
    if idd_max <= 0:
        return float('nan')

    return float(np.sqrt(np.mean((idd_pred / idd_max - idd_gt / idd_max) ** 2)))


def evaluate_beam_level(pred_beams: List[np.ndarray],
                        gt_beams: List[np.ndarray],
                        beam_axis: int = 0) -> Dict:
    """Evaluate Level 1 metrics over all beams in a plan.

    Parameters
    ----------
    pred_beams:
        Predicted dose array for each beam.
    gt_beams:
        Ground-truth dose array for each beam (same ordering).
    beam_axis:
        Array axis along which the beam propagates (0 = z, 1 = y, 2 = x).

    Returns
    -------
    dict with keys:
        beam_mae_per_beam, beam_mae_mean, beam_mae_std,
        idd_distance_per_beam, idd_distance_mean, idd_distance_std
    """
    maes:      List[float] = []
    idd_dists: List[float] = []

    for pred_b, gt_b in zip(pred_beams, gt_beams):
        maes.append(masked_beam_mae(pred_b, gt_b))
        idd_dists.append(idd_curve_distance(pred_b, gt_b, beam_axis))

    return {
        'beam_mae_per_beam':      maes,
        'beam_mae_mean':          float(np.nanmean(maes)),
        'beam_mae_std':           float(np.nanstd(maes)),
        'idd_distance_per_beam':  idd_dists,
        'idd_distance_mean':      float(np.nanmean(idd_dists)),
        'idd_distance_std':       float(np.nanstd(idd_dists)),
    }
