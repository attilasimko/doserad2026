"""
doserad.io
----------
File loading utilities and the DVH helper class.
"""

import os
import numpy as np
import SimpleITK as sitk
from typing import Tuple, Optional


# ---------------------------------------------------------------------------
# Dose file loading
# ---------------------------------------------------------------------------

def load_dose_sitk(file_path: str,
                   spacing: Optional[Tuple[float, ...]] = None) -> sitk.Image:
    """Load a dose file and return it as a SimpleITK image.

    Parameters
    ----------
    file_path:
        Path to a .nii.gz, .npy, or .mha file.
    spacing:
        Voxel spacing (x, y, z) in mm.  Applied only when loading .npy files,
        which carry no spatial metadata.  SimpleITK expects (x, y, z) order.
    """
    ext = _extension(file_path)

    if ext in ('.nii.gz', '.mha'):
        return sitk.ReadImage(file_path)

    if ext == '.npy':
        arr = np.load(file_path).astype(np.float64)
        img = sitk.GetImageFromArray(arr)   # arr is (z, y, x); sitk handles the flip
        if spacing is not None:
            img.SetSpacing(spacing)         # expects (x, y, z)
        return img

    raise ValueError(f"Unsupported file format: {ext}")


def load_dose_array(file_path: str) -> np.ndarray:
    """Load a dose file and return it as a float64 numpy array (z, y, x)."""
    ext = _extension(file_path)

    if ext in ('.nii.gz', '.mha'):
        return sitk.GetArrayFromImage(sitk.ReadImage(file_path)).astype(np.float64)

    if ext == '.npy':
        return np.load(file_path).astype(np.float64)

    raise ValueError(f"Unsupported file format: {ext}")


def load_structures(struct_path: str) -> np.ndarray:
    """Load structure masks.

    Returns an array whose last axis indexes structures, i.e. shape
    ``(*volume_shape, N_structures)``.
    """
    return np.load(struct_path, allow_pickle=True)


def find_dose_file(base_dir: str, patient_id: str) -> Optional[str]:
    """Locate a dose file for *patient_id* under *base_dir*.

    Checks ``<base_dir>/<patient_id>/<stem><ext>`` for a set of common stems
    and extensions, then falls back to ``<base_dir>/<patient_id><ext>``.
    """
    patient_dir = os.path.join(base_dir, patient_id)
    stems = ['plan_dose', 'dose', 'final_dose', patient_id]
    extensions = ['.npy', '.nii.gz', '.mha']

    if os.path.isdir(patient_dir):
        for stem in stems:
            for ext in extensions:
                candidate = os.path.join(patient_dir, f"{stem}{ext}")
                if os.path.exists(candidate):
                    return candidate
    else:
        for ext in extensions:
            candidate = os.path.join(base_dir, f"{patient_id}{ext}")
            if os.path.exists(candidate):
                return candidate

    return None


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _extension(path: str) -> str:
    """Return the file extension, handling .nii.gz as a unit."""
    if path.endswith('.nii.gz'):
        return '.nii.gz'
    return os.path.splitext(path)[1].lower()


# ---------------------------------------------------------------------------
# DVH
# ---------------------------------------------------------------------------

class DVH:
    """Cumulative Dose-Volume Histogram.

    Oncology convention
    -------------------
    ``D(v)`` — dose [Gy] received by *at least* fraction *v* of the masked
    volume.

    * ``D(0.98)`` → dose exceeded by 98 % of voxels → **D98** (near-minimum)
    * ``D(0.02)`` → dose exceeded by only 2 % of voxels → **D2** (near-maximum)

    ``V(d)`` — fraction of masked volume receiving >= dose *d*.
    """

    def __init__(self, dose_array: np.ndarray, mask: np.ndarray) -> None:
        self._doses = dose_array[mask > 0].astype(np.float64)

    def D(self, volume_fraction: float) -> float:
        """Dose received by at least *volume_fraction* of the volume."""
        if self._doses.size == 0:
            return float('nan')
        percentile = (1.0 - volume_fraction) * 100.0
        return float(np.percentile(self._doses, percentile))

    def V(self, dose_threshold: float) -> float:
        """Fraction of volume receiving >= *dose_threshold*."""
        if self._doses.size == 0:
            return float('nan')
        return float(np.mean(self._doses >= dose_threshold))
