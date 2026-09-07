from typing import Literal

import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

from pyRadPlan import plot_slice

# ---------------------------------------------------------------------------
# Dataset directory layout
# ---------------------------------------------------------------------------
DIR_IMAGE            = "image"
DIR_DOSE             = "dose"
CT_NAME              = "ct.mha"
BEAM_PARAMS_FILENAME = "beam_parameters.json"


def plot_dose_comparison(
    ct,
    cst,
    pb_dose: sitk.Image,
    ref_dose: sitk.Image,
    pb_title: str,
    ref_title: str,
    suptitle: str,
    plane: Literal["axial", "sagittal", "coronal"] = "axial",
    use_shared_max: bool = True,
) -> None:
    """Three-panel PB dose | MC reference | difference figure.

    The slice with the highest dose in the reference volume is selected
    automatically.  ``slice=N`` is appended to both panel titles so the
    displayed index is always visible.

    Parameters
    ----------
    ct, cst:
        CT and structure-set objects passed through to ``plot_slice``.
    pb_dose, ref_dose:
        Pencil-beam and Monte-Carlo reference dose images.
    pb_title, ref_title:
        Panel titles for the PB and reference columns (without slice info).
    suptitle:
        Figure super-title.
    plane:
        Viewing plane ("axial", "sagittal", or "coronal").
    use_shared_max:
        When True both dose panels share the same colour-scale maximum.
    """
    pb_arr  = sitk.GetArrayFromImage(pb_dose)
    ref_arr = sitk.GetArrayFromImage(ref_dose)

    # SimpleITK arrays are ordered (z, y, x).
    # Pick the slice index with the highest dose in the reference volume and
    # build a correct numpy index expression for the diff panel.
    if plane == "axial":
        view_slice = int(np.argmax(ref_arr.max(axis=(1, 2))))
        sl = np.s_[view_slice]
    elif plane == "sagittal":
        view_slice = int(np.argmax(ref_arr.max(axis=(0, 1))))
        sl = np.s_[:, :, view_slice]
    else:  # coronal
        view_slice = int(np.argmax(ref_arr.max(axis=(0, 2))))
        sl = np.s_[:, view_slice, :]

    shared_max = max(float(pb_arr.max()), float(ref_arr.max()))

    _, axes = plt.subplots(1, 3, figsize=(18, 5))

    plot_slice(
        ct=ct, cst=cst, overlay=pb_dose,
        view_slice=view_slice, plane=plane,
        overlay_unit="Gy", show_plot=False, ax=axes[0],
    )
    axes[0].set_title(f"{pb_title}  slice={view_slice}")

    plot_slice(
        ct=ct, cst=cst, overlay=ref_dose,
        view_slice=view_slice, plane=plane,
        overlay_unit="Gy", show_plot=False, ax=axes[1],
    )
    axes[1].set_title(f"{ref_title}  slice={view_slice}")

    if use_shared_max:
        for ax in axes[:2]:
            imgs = ax.get_images()
            if len(imgs) > 1:  # CT background + dose overlay
                imgs[-1].set_clim(0, shared_max)

    # Difference panel: PB minus MC reference, masked to the dose region.
    diff_slice = pb_arr[sl] - ref_arr[sl]
    dose_mask  = ref_arr[sl] > 0.01 * shared_max  # ignore near-zero voxels
    abs_max    = float(np.abs(diff_slice[dose_mask]).max()) if dose_mask.any() else 1.0

    plot_slice(ct=ct, cst=cst, view_slice=view_slice, plane=plane,
               show_plot=False, ax=axes[2])
    im_diff = axes[2].imshow(
        np.where(dose_mask, diff_slice, np.nan),
        cmap="RdBu_r", interpolation="nearest",
        alpha=0.7, vmin=-abs_max, vmax=abs_max,
    )
    plt.colorbar(im_diff, ax=axes[2], label="Gy")
    axes[2].set_title(f"Difference (PB − MC) slice={view_slice}")

    plt.suptitle(suptitle, fontsize=12)
    plt.tight_layout()
    plt.show()
