import matplotlib.pyplot as plt
import numpy as np


def _safe_idx(value, axis_size):
    return int(max(0, min(axis_size - 1, value)))


def _alpha_mask(arr: np.ndarray, base_alpha: float, vmax: float, eps_frac: float = 0.10) -> np.ndarray:
    """Per-pixel alpha array: ``base_alpha`` where |arr| is above
    ``eps_frac * vmax``, zero everywhere else.

    This is what lets the CT underneath show through cleanly in
    near-zero regions — without it, each cmap paints its "zero colour"
    (jet=blue, magma=black, coolwarm/seismic=white) over the CT and the
    background brightness flickers between panels. The 10% cutoff
    matches the clinical "scored region" convention: voxels below 10%
    of the panel's peak are treated as background and hidden.
    """
    eps = eps_frac * max(vmax, 1e-9)
    out = np.where(np.abs(arr) > eps, base_alpha, 0.0)
    return out.astype(np.float32)


#: Display window (HU) for the greyscale backdrop. Fixed rather than
#: autoscaled so a CT plot and an sCT plot of the same patient look the same.
HU_WINDOW = (-1000.0, 1000.0)


def plot_dose_comparison(
    ct_volume,
    true_dose,
    pb_dose,
    beam,
    original_iso_center,
    save_path,
    title=None,
    correction=None,
    body_mask=None,
    gamma_map=None,
):
    """Render true / predicted / |error| in three planes plus a 1-D profile.

    The greyscale backdrop is always the volume the dose was actually computed
    on, in HU: the real CT for the CT arm, the SYNTHETIC CT for the MR arm (the
    raw MR is never plotted — it is not what the engine saw). Windowed
    explicitly so both arms render identically.

    Args:
        ct_volume, true_dose, pb_dose: matched 3-D arrays (H, D, W).
        beam: beam segment (unused; kept for backward compatibility).
        original_iso_center: (H, D, W) physical coords (mm).
        save_path: png path.
        title: optional suptitle text.
        correction: optional total model correction (same shape) shown in
            an extra column.
        body_mask: optional boolean mask (same shape) drawn as a thin red
            contour on every panel.
        gamma_map: optional dense 3-D gamma array from pymedphys.gamma
            (same shape, NaN where the test wasn't evaluated). Rendered
            as an extra column with a fixed [0, 2] scale so γ ≤ 1
            (passing) sits in the green half and γ > 1 (failing) in red.
    """
    H, D, W = true_dose.shape
    iso_h = _safe_idx(original_iso_center[0] // 2, H)
    iso_d = _safe_idx(original_iso_center[1] // 2, D)
    iso_w = _safe_idx(original_iso_center[2] // 2, W)

    rows_config = [
        ("Axial",    0, iso_h, iso_w, iso_d),
        ("Coronal",  1, iso_d, iso_w, iso_h),
        ("Sagittal", 2, iso_w, iso_d, iso_h),
    ]

    diff_volume = pb_dose - true_dose
    err_vmax = float(np.max(np.abs(diff_volume))) or 1e-9
    dose_vmax = max(float(true_dose.max()), float(pb_dose.max()), 1e-9)
    corr_vmax = float(np.max(np.abs(correction))) if correction is not None else 0.0
    if correction is not None and corr_vmax == 0.0:
        corr_vmax = 1e-9

    # Base columns: True, Pred, signed diff, |diff|. Then optional
    # Model Δ (correction) and optional Gamma map, then always the
    # 1-D profile column.
    n_cols = 5 + (1 if correction is not None else 0) + (1 if gamma_map is not None else 0)
    fig, axes = plt.subplots(
        3, n_cols,
        figsize=(3.4 * n_cols, 9.6),
        gridspec_kw={"hspace": 0.15, "wspace": 0.28},
    )

    col_titles = ["True", "Pred", "Pred − True", "|Pred − True|"]
    if correction is not None:
        col_titles.append("Model Δ (gain·dose + residual)")
    if gamma_map is not None:
        col_titles.append("γ (1%/1mm)  pass≤1 · fail>1")
    col_titles.append("Profile @ iso")
    for ax, t in zip(axes[0], col_titles):
        ax.set_title(t, fontsize=10, fontweight="bold", pad=3)

    mae = float(np.mean(np.abs(diff_volume)))
    suptitle = f"MAE={mae:.4f}  vmax(err)={err_vmax:.3f}"
    if title:
        suptitle = f"{title}   |   {suptitle}"
    fig.suptitle(suptitle, fontsize=11, fontweight="bold", y=0.995)

    base_alpha = 0.65

    for row_idx, (label, axis, slice_idx, sx, sy) in enumerate(rows_config):
        slicer = [slice(None)] * 3
        slicer[axis] = slice_idx
        slicer = tuple(slicer)

        ct_slice   = ct_volume[slicer]
        true_slice = true_dose[slicer]
        pred_slice = pb_dose[slicer]
        diff_slice = diff_volume[slicer]
        abs_slice  = np.abs(diff_slice)
        mask_slice = body_mask[slicer] if body_mask is not None else None

        # (data, cmap, vmin, vmax, scale-used-for-alpha-threshold).
        # scale == "nan_mask" is a sentinel meaning "alpha = base_alpha
        # where the value is finite, 0 where it's NaN"; used for the
        # gamma map so passing voxels (γ near 0) aren't hidden by the
        # standard 10%-of-vmax threshold.
        panels = [
            (true_slice, "jet",      0,          dose_vmax, dose_vmax),
            (pred_slice, "jet",      0,          dose_vmax, dose_vmax),
            (diff_slice, "coolwarm", -err_vmax,  err_vmax,  err_vmax),
            (abs_slice,  "magma",    0,          err_vmax,  err_vmax),
        ]
        if correction is not None:
            panels.append(
                (correction[slicer], "seismic", -corr_vmax, corr_vmax, corr_vmax)
            )
        if gamma_map is not None:
            panels.append(
                (gamma_map[slicer], "RdYlGn_r", 0.0, 2.0, "nan_mask")
            )

        for col, (arr, cmap, vmin, vmax, scale) in enumerate(panels):
            ax = axes[row_idx, col]
            ax.imshow(ct_slice, cmap="gray", aspect="auto",
                      vmin=HU_WINDOW[0], vmax=HU_WINDOW[1])
            if scale == "nan_mask":
                # NaN -> transparent, finite -> visible.
                finite = np.isfinite(arr)
                alpha_arr = np.where(finite, base_alpha, 0.0).astype(np.float32)
                # imshow doesn't render NaN under any cmap; replace with 0
                # which is already alpha-masked out.
                arr_display = np.where(finite, arr, 0.0)
                im = ax.imshow(arr_display, cmap=cmap, alpha=alpha_arr, vmin=vmin, vmax=vmax)
            else:
                alpha_arr = _alpha_mask(arr, base_alpha=base_alpha, vmax=scale)
                im = ax.imshow(arr, cmap=cmap, alpha=alpha_arr, vmin=vmin, vmax=vmax)
            if mask_slice is not None and mask_slice.any() and not mask_slice.all():
                # Thin red contour at the mask boundary, drawn last so it
                # sits on top of the dose overlay.
                ax.contour(mask_slice.astype(float), levels=[0.5],
                           colors="red", linewidths=0.5, alpha=0.9)
            ax.scatter(sx, sy, color="lime", marker="+", s=60, linewidths=1.0, zorder=5)
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(label, fontsize=10, fontweight="bold")
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
            cbar.ax.tick_params(labelsize=6)

        ax = axes[row_idx, len(panels)]
        line_axes = [a for a in (0, 1, 2) if a != axis]
        slicer_line = [slice(None)] * 3
        slicer_line[axis] = slice_idx
        slicer_line[line_axes[1]] = (iso_h, iso_d, iso_w)[line_axes[1]]
        slicer_line = tuple(slicer_line)
        true_line = true_dose[slicer_line]
        pred_line = pb_dose[slicer_line]
        x = np.arange(true_line.shape[0])
        ax.plot(x, true_line, label="true", lw=1.1, color="black")
        ax.plot(x, pred_line, label="pred", lw=1.1, color="tab:red", alpha=0.85)
        ax.fill_between(x, true_line, pred_line, color="tab:red", alpha=0.15)
        ax.tick_params(labelsize=6)
        if row_idx == 0:
            ax.legend(fontsize=6, loc="upper right")

    plt.savefig(save_path, bbox_inches="tight", dpi=120)
    plt.close()
