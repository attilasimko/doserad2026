"""Composite loss for the v2 photon corrector.

Deliberately the SAME shape as the proton ``_loss`` with ``loss_mask="additive"``
-- a full-support MAE plus a high-dose MAE added on top rather than traded
against it, so raising the high-dose weight cannot silently abandon the
periphery:

    L = support_mae + w_high10 * high10_mae + w_deep * deep_supervision

Weight provenance, so these are not mystery numbers:

  * ``w_high10 = 0.15`` -- the proton ``_loss`` default for the ``additive``
    branch (``loss_high10_weight``).
  * ``w_deep = 0.1`` -- the proton ablation found 0.2 best; deliberately backed
    off to 0.1 here since the photon decoder differs. The shipped proton
    checkpoint used 0.05.

NO IDD TERM BY DEFAULT. Level 1.2 is a scored metric and is always REPORTED
(see ``idd_distance_per_sample`` below). Optimising it was measured on the
proton side to degrade MAE, so ``w_idd`` defaults to 0 and the term is opt-in.
It exists because the photon evidence differs from the proton evidence: at
matched epochs v2 trails v1 by 49% on pointwise MAE but 377% on IDD, which
means its residual is spatially COHERENT, and no pointwise term can see that.

Why not plain L1, which is what v1 shipped with: absolute whole-volume L1 is
median-seeking and shaves peaks. The v1 model under-predicts the per-CP maximum
on 85% of control points despite already having a gain head -- the loss, not the
head, was the binding constraint.

Everything is normalised by the PER-SAMPLE reference peak and reduced as a mean
over samples, matching how the challenge scores each control point separately
and then takes a nanmean. A batch mixing a large-aperture and a small-aperture
CP must not let the larger one dominate.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

EPS = 1e-12


@dataclass
class LossConfig:
    w_high10: float = 0.15      # proton `additive` default
    w_deep: float = 0.1         # proton ablation optimum was 0.2; backed off
    w_grad: float = 0.0         # off; ablation knob only
    high_dose_frac: float = 0.1
    min_peak: float = 1e-6      # below this a CP is degenerate (closed MLC)
    # Which voxels the support term covers when no explicit support_mask is
    # given. "all" = every voxel; "positive" = only t > 0.
    #
    # A photon target volume is 77-82% EXACT zero (measured on 1ABB006 /
    # 1THB063 / 1THB122, B0 CP090), and under "positive" none of those voxels
    # gets any gradient -- verified: the gradient there is exactly 0.0.
    # That sounds alarming and MEASURES AS HARMLESS. Trained checkpoints put
    # essentially nothing in that region either way:
    #
    #            leak_mass   leak_idd     (fraction of true dose / IDD peak)
    #   v1        4.07e-06   4.76e-06     unmasked L1, IS trained there
    #   v2nopool  1.58e-06   3.40e-06     masked, is NOT trained there
    #
    # v2 leaks LESS while training on 20% of the volume. The reason is the
    # relu: `relu(D_base*(1+a*tanh g) + alpha*s*tanh r)` returns exactly 0 for
    # any negative argument, so zero is a wide flat attractor and the
    # unsupervised region solves itself. Covering it with "all" only dilutes
    # the support mean over ~5x more voxels that are already right, which
    # down-weights the in-field region against high10. Hence the default
    # stays "positive"; "all" is kept as the measured control.
    #
    # Where the IDD gap actually lives, therefore, is INSIDE t > 0. See w_idd.
    support_mode: str = "positive"

    # --- LOW-DOSE EMPHASIS -------------------------------------------------
    # All default to 0.0, i.e. OFF, so the shipped loss is unchanged.
    #
    # Why any of this: the scored Level-2 metrics live where the current loss
    # does not look. Stratified plan MAE averages three dose bands EQUALLY, and
    # local gamma 1%/1mm sets a tolerance of 1% of the LOCAL dose -- so a voxel
    # at 15% of prescription gets a ~7x tighter absolute tolerance than one at
    # peak. Measured on 1THB218: the 10-30% band holds 264,389 scored voxels
    # failing at 18.36%, against 4,908 failures in 30-80% and 381 above 80% --
    # about 90% of all gamma failures. The existing loss adds w_high10 ON TOP
    # of the support term, so it leans the other way.
    w_strata: float = 0.0       # equal-weight band MAE, mirrors stratified plan MAE
    w_rel: float = 0.0          # relative error, mirrors LOCAL gamma
    w_logw: float = 0.0         # log-weighted MAE, smooth low-dose emphasis
    # Squared error. BOTH published MR-linac dose networks optimise a squared
    # error -- Tsekas et al 2022 (PMB 67 225020) use RMSE, Tseng et al 2023
    # (PMB 68 175004) use MSE -- while this loss is MAE-based throughout. MSE
    # weights a 4x error 16x rather than 4x, and gamma failures ARE the large
    # errors, so it targets the metric we rank worst on. Peak-normalised so it
    # stays commensurate with the MAE terms.
    w_mse: float = 0.0
    strata_edges: tuple = (0.1, 0.3, 0.8)   # fractions of the sample peak
    rel_floor_frac: float = 0.05            # caps relative error at 1/0.05 = 20x
    logw_cap: float = 6.0                   # caps the log weight
    # Weight on the integrated-profile term. OFF by default -- the proton side
    # measured that optimising IDD degrades MAE (see the module docstring), so
    # this is opt-in and has to earn its place per modality.
    #
    # Why it is here at all. At matched epochs v2nopool is 49% worse than v1 on
    # pointwise masked MAE (0.0200 vs 0.0134) but 377% worse on IDD (0.0444 vs
    # 0.0093). Integration cancels zero-mean error at ~1/sqrt(N) and preserves
    # COHERENT error, so an 8x amplification from MAE to IDD means v2's residual
    # is spatially correlated where v1's is not. Nothing in the pointwise
    # objective distinguishes the two: a mean over voxels treats every error as
    # independent. This term is the only one that sees the difference.
    w_idd: float = 0.0


def _flatten_samples(x: torch.Tensor) -> torch.Tensor:
    """Collapse every leading dim into one sample axis: [..., D,H,W] -> [N,D,H,W]."""
    return x.reshape(-1, *x.shape[-3:])


def _masked_mae_per_sample(diff: torch.Tensor, mask: torch.Tensor,
                           peak: torch.Tensor) -> torch.Tensor:
    """Peak-normalised MAE inside `mask`, one value per sample."""
    m = mask.to(diff.dtype)
    n = m.flatten(1).sum(dim=1).clamp_min(1.0)
    return (diff * m).flatten(1).sum(dim=1) / n / peak.flatten()


def _gradient_l1(pred: torch.Tensor, target: torch.Tensor,
                 peak: torch.Tensor) -> torch.Tensor:
    """Peak-normalised L1 on spatial gradients, per sample. Ablation knob."""
    total = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    for ax in (-3, -2, -1):
        d = (pred.diff(dim=ax) - target.diff(dim=ax)).abs()
        total = total + d.flatten(1).mean(dim=1)
    return total / 3.0 / peak.flatten()


def _profile_l1_per_sample(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Differentiable penalty on COHERENT error, per sample.

    For each of the three axes, integrate out the other two and compare the
    resulting 1-D profile, normalised by that profile's own peak. Averaged over
    the three axes.

    Why all three and not just the scored one: Level 1.2 profiles along numpy
    axis 0 of the PATIENT volume, while ``train_correction._idd_distance``
    profiles BEV depth -- two different quantities, as the axis warning on
    ``idd_distance_per_sample`` records. Penalising every axis is agnostic to
    that unresolved question and costs three cheap reductions.

    L1 rather than the scored RMS: RMS lets one bad depth slice dominate the
    gradient, and the surrounding terms are all L1-scaled, so a squared term
    would need its own weight calibration to sit alongside them.
    """
    total = torch.zeros(pred.shape[0], device=pred.device, dtype=pred.dtype)
    for axis in (-3, -2, -1):
        others = tuple(a for a in (-3, -2, -1) if a != axis)
        p = pred.sum(dim=others)
        t = target.sum(dim=others)
        peak = t.detach().abs().amax(dim=-1, keepdim=True).clamp_min(EPS)
        total = total + ((p - t).abs() / peak).mean(dim=-1)
    return total / 3.0


@torch.no_grad()
def idd_distance_per_sample(pred: torch.Tensor, target: torch.Tensor,
                            idd_axis: int = -3) -> torch.Tensor:
    """Challenge Level 1.2, as a REPORTING metric only -- never in the loss.

    AXIS WARNING. Level 1.2 profiles along numpy axis 0 of the PATIENT volume
    (z, y, x). In beam's-eye view that axis is H, not depth. The default -3 is
    correct for patient-space tensors. ``train_correction._idd_distance`` feeds
    the same reduction with BEV tensors and so profiles along beam depth, which
    is a different quantity from the scored one.
    """
    axes = tuple(a for a in (-3, -2, -1) if a % pred.dim() != idd_axis % pred.dim())
    p = pred.sum(dim=axes)
    t = target.sum(dim=axes)
    peak = t.abs().amax(dim=-1, keepdim=True).clamp_min(EPS)
    return torch.sqrt((((p - t) / peak) ** 2).mean(dim=-1))


def photon_corrector_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    support_mask: torch.Tensor | None = None,
    deep_predictions: tuple[torch.Tensor, ...] = (),
    deep_target: torch.Tensor | None = None,
    cfg: LossConfig | None = None,
    report_idd: bool = True,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return (scalar loss, diagnostics).

    ``pred``/``target`` are patient-space ``[..., D, H, W]``. Deep-supervision
    predictions live in BEV and need ``deep_target`` in BEV; they are scored
    with the same support + high10 combination.
    """
    cfg = cfg or LossConfig()
    p = _flatten_samples(pred)
    t = _flatten_samples(target)
    peak = t.detach().amax(dim=(-3, -2, -1)).clamp_min(EPS)          # [N]

    # A closed-MLC control point has an all-zero target. Normalising by its ~0
    # peak produces a huge finite loss that overflows fp16 under AMP; such
    # samples carry no information, so they are dropped from every mean.
    valid = peak > cfg.min_peak
    if not bool(valid.any()):
        return pred.sum() * 0.0, {"n_valid": 0.0}

    peak4 = peak.view(-1, 1, 1, 1)
    diff = (p - t).abs()
    if support_mask is not None:
        supp = support_mask.reshape(p.shape)
    elif cfg.support_mode == "all":
        supp = torch.ones_like(t, dtype=torch.bool)
    else:
        supp = t > 0
    high = t >= cfg.high_dose_frac * peak4

    support_mae = _masked_mae_per_sample(diff, supp, peak)
    high10_mae = _masked_mae_per_sample(diff, high, peak)

    def mean_valid(v: torch.Tensor) -> torch.Tensor:
        return v[valid].mean()

    total = mean_valid(support_mae) + cfg.w_high10 * mean_valid(high10_mae)
    terms = {
        "support_mae": float(mean_valid(support_mae).detach()),
        "high10_mae": float(mean_valid(high10_mae).detach()),
        "n_valid": float(int(valid.sum())),
    }

    def _weighted_mae(mask_or_w: torch.Tensor) -> torch.Tensor:
        """Peak-normalised MAE with an arbitrary non-negative weight map."""
        w = mask_or_w.to(diff.dtype)
        n = w.flatten(1).sum(dim=1).clamp_min(1.0)
        return (diff * w).flatten(1).sum(dim=1) / n / peak.flatten()

    if cfg.w_mse > 0.0:
        sq = ((p - t) / peak4) ** 2
        m = supp.to(sq.dtype)
        n = m.flatten(1).sum(dim=1).clamp_min(1.0)
        mse = (sq * m).flatten(1).sum(dim=1) / n
        total = total + cfg.w_mse * mean_valid(mse)
        terms["mse"] = float(mean_valid(mse).detach())

    if cfg.w_strata > 0.0:
        lo, mid, hi = cfg.strata_edges
        bands = ((t >= lo * peak4) & (t < mid * peak4),
                 (t >= mid * peak4) & (t < hi * peak4),
                 t >= hi * peak4)
        # Equal weight per band, exactly as the metric averages its three
        # stratified MAEs -- the low band is a third of the score however few
        # or many voxels it contains.
        strat = torch.stack([_masked_mae_per_sample(diff, b, peak)
                             for b in bands]).mean(0)
        total = total + cfg.w_strata * mean_valid(strat)
        terms["strata_mae"] = float(mean_valid(strat).detach())

    if cfg.w_rel > 0.0:
        # |p - t| / t on scored voxels: dimensionless, and the quantity local
        # gamma actually thresholds. Denominator floored at a fraction of peak
        # so a near-zero target cannot produce an unbounded gradient.
        scored = t >= cfg.high_dose_frac * peak4
        den = t.clamp_min(cfg.rel_floor_frac * peak4)
        r = diff / den
        m = scored.to(r.dtype)
        n = m.flatten(1).sum(dim=1).clamp_min(1.0)
        rel = (r * m).flatten(1).sum(dim=1) / n
        total = total + cfg.w_rel * mean_valid(rel)
        terms["rel_mae"] = float(mean_valid(rel).detach())

    if cfg.w_logw > 0.0:
        # Smooth version of the same idea: weight 1 at the peak, growing as
        # log(peak/t) toward low dose, capped so the tail cannot dominate.
        ratio = (peak4 / t.clamp_min(EPS)).clamp_min(1.0)
        w = torch.log(ratio).clamp(max=cfg.logw_cap) + 1.0
        w = w * (t >= cfg.high_dose_frac * peak4).to(w.dtype)
        lw = _weighted_mae(w)
        total = total + cfg.w_logw * mean_valid(lw)
        terms["logw_mae"] = float(mean_valid(lw).detach())

    if cfg.w_grad > 0.0:
        g = _gradient_l1(p, t, peak)
        total = total + cfg.w_grad * mean_valid(g)
        terms["grad"] = float(mean_valid(g).detach())

    if cfg.w_idd > 0.0:
        prof = _profile_l1_per_sample(p, t)
        total = total + cfg.w_idd * mean_valid(prof)
        terms["profile"] = float(mean_valid(prof).detach())

    if deep_predictions and deep_target is not None and cfg.w_deep > 0.0:
        # Aux heads live at REDUCED depth, so the target comes down to meet them
        # -- never the prediction up. Upsampling cost a full-resolution volume
        # per head and OOM'd an 80 GB card. Mirrors the proton
        # `_bev_deep_supervision_loss`.
        dt = _flatten_samples(deep_target)
        # Peak is the FULL-resolution one for every scale (proton does the same),
        # so coarse levels are normalised on the same footing as the main head
        # rather than against their own smoothed peak.
        dpeak = dt.detach().amax(dim=(-3, -2, -1)).clamp_min(EPS)
        dvalid = dpeak > cfg.min_peak
        dsupp_full = (torch.ones_like(dt, dtype=torch.bool)
                      if cfg.support_mode == "all" else dt > 0)
        acc = torch.zeros((), device=pred.device, dtype=pred.dtype)
        wsum = 0.0
        for i, dp in enumerate(deep_predictions):
            p_i = _flatten_samples(dp)
            if p_i.shape[-3:] != dt.shape[-3:]:
                tgt = F.adaptive_avg_pool3d(dt.unsqueeze(1), p_i.shape[-3:]).squeeze(1)
                # MAX-pool the mask: a coarse voxel is valid if ANY fine voxel in
                # it was. Average-pooling a boolean would invent fractional
                # validity and then threshold it arbitrarily.
                supp = F.adaptive_max_pool3d(
                    dsupp_full.unsqueeze(1).to(dt.dtype), p_i.shape[-3:]).squeeze(1) > 0
            else:
                tgt, supp = dt, dsupp_full
            high = tgt >= cfg.high_dose_frac * dpeak.view(-1, 1, 1, 1)
            ddiff = (p_i - tgt).abs()
            aux = (_masked_mae_per_sample(ddiff, supp, dpeak)
                   + cfg.w_high10 * _masked_mae_per_sample(ddiff, high, dpeak))
            # deep_predictions arrives FINEST-first (the model reverses it), so
            # 0.5**i discounts coarser scales.
            w = 0.5 ** i
            acc = acc + w * aux[dvalid].mean()
            wsum += w
        acc = acc / wsum
        total = total + cfg.w_deep * acc
        terms["deep"] = float(acc.detach())

    # Diagnostics, not optimised. peak_ratio is the bias tracker: v1 sits at
    # 0.979 with 85% of CPs under-predicting, and that coherent error is what
    # survives summation into the Level-2 plan metrics.
    with torch.no_grad():
        ppeak = p.amax(dim=(-3, -2, -1))
        terms["peak_ratio"] = float((ppeak[valid] / peak[valid]).mean())
        if report_idd:
            terms["idd"] = float(idd_distance_per_sample(p, t)[valid].mean())
        terms["total"] = float(total.detach())

    return total, terms
