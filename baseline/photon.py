import json
import logging
from pathlib import Path

from typing import Literal, Optional
from numpy.typing import NDArray

import numpy as np
import SimpleITK as sitk

from pyRadPlan import PhotonPlan, generate_stf, calc_dose_forward
from pyRadPlan.ct import ct_from_file
from pyRadPlan.cst import StructureSet
from pyRadPlan.machines import Jaw, MLC

from .utils import DIR_IMAGE, DIR_DOSE, BEAM_PARAMS_FILENAME, CT_NAME, plot_dose_comparison
from configs import default_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Fixed machine / geometry constants for the photon baseline.
_MC_SCALE       = 19755.0  # weight normalisation matched on central-axis depth-dose curve
_MLC_LEAF_WIDTH = 5.0     # mm
_JAW_HALF       = 200.0   # mm
_ENERGY         = 6.0     # MeV (fixed for the generic photon machine)"


def _compute_cp_dose(
    ct,
    cst,
    beam_json: dict,
    cp_json: dict,
    hlut: Optional[NDArray] = None,
) -> sitk.Image:
    """Compute pencil-beam dose for a single control point.

    Reconstructs MLC and jaw geometry from the plan JSON, builds a
    field-based STF via pyRadPlan's photon BLD generator, normalises
    beamlet weights to the MC scale, and runs the AAA forward dose engine.

    Returns the dose image (physical_dose).
    """
    iso_center   = np.array(beam_json["iso_center"], dtype=float)
    gantry_angle = float(cp_json["gantry_angle"])
    n_leaf_pairs = int(beam_json["num_mlc_leaf_pairs"])

    # Compared to pyRadPlan's convention, MLC positions seem to be rotated by 180°
    mlc_left  = -np.flip(np.array(cp_json["mlc_right_int_mm"], dtype=float))
    mlc_right = -np.flip(np.array(cp_json["mlc_left_int_mm"], dtype=float))

    leaf_position_boundaries = np.arange(
        -n_leaf_pairs / 2 * _MLC_LEAF_WIDTH,
         n_leaf_pairs / 2 * _MLC_LEAF_WIDTH,
        _MLC_LEAF_WIDTH,
    )  # shape (n_leaf_pairs,)

    mlc = MLC(
        device_orientation="X",
        leaf_position_boundaries=leaf_position_boundaries,
        leaf_positions=np.column_stack([mlc_left, mlc_right]),
        leaf_width=_MLC_LEAF_WIDTH,
        leaf_leakage=0.015,
    )

    jaw_x = Jaw(
        device_orientation="X",
        positions=[-_JAW_HALF, _JAW_HALF],
        field_width=2 * _JAW_HALF,
        leakage=0.015,
    )
    jaw_y = Jaw(
        device_orientation="Y",
        positions=[-_JAW_HALF, _JAW_HALF],
        field_width=2 * _JAW_HALF,
        leakage=0.015,
    )

    pln = PhotonPlan(machine="Generic")
    pln.prop_stf = {
        "gantry_angles": [gantry_angle],
        "couch_angles":  [0.0],
        "iso_center":    iso_center,
        "generator":     "photonSingleBixel",
        "field_based":   True,
        "blds":          [mlc, jaw_x, jaw_y],
        "resolution":    1.0,   # 1 mm mask sampling for accurate leaf-edge rendering
        "energy":        _ENERGY,
    }
    pln.prop_dose_calc = {
        "engine":                "SVDPB",
        "dose_grid":             ct.grid,
        "force_penumbra":        1.8,   # approximated to match MC well
        "force_uniform_fluence": True,  # match uniform rejection sampling
        "geometric_lateral_cutoff": 0.0, # field based dose calculation automatically determine larger calculation cones, so this can be zero here
        "hlut": hlut,
    }

    stf = generate_stf(ct, cst, pln)
    logger.info(
        "  STF created: %d beam, %d ray",
        len(stf.beams), len(stf.beams[0].rays),
    )

    # Normalise beamlet weights to the MC fluence scale.
    for beam in stf.beams:
        for ray in beam.rays:
            for beamlet in ray.beamlets:
                beamlet.weight /= _MC_SCALE

    result   = calc_dose_forward(ct, cst, stf, pln)
    pb_dose: sitk.Image = result["physical_dose"]

    return pb_dose


def compute_pb_dose(
    baseline_pb_dir: str | Path,
    patient_id: str,
    modality: str,
    split: str,
    output_dir: Optional[str | Path] = None,
    beam_idx: int = 0,
    cp_idx: int = 0,
    show_plots: bool = False,
    plot_view_slice: Literal["axial", "sagittal", "coronal"] = "axial",
    plot_shared_max: bool = True,
) -> None:
    """
    Run the photon pencil-beam baseline for a single control point.

    Parameters
    ----------
    baseline_pb_dir:
        Root directory of the dataset (e.g. ``/home/user/doserad2026/baseline_PB``).
    patient_id:
        Patient identifier string (e.g. ``"1THB002"``).
    modality:
        Radiation modality folder name (e.g. ``"photon"``).
    split:
        Dataset split folder name (e.g. ``"train"`` or ``"test"``).
    beam_idx:
        List index of the beam in the plan JSON (0-based).
    cp_idx:
        List index of the control point within the selected beam (0-based).
    show_plots:
        Whether to display a comparison figure for the control point.
    plot_view_slice:
        The plane in which to display the slice ("axial", "sagittal", "coronal").
    plot_shared_max:
        Whether to use a shared maximum for the colour scale across all plots.
    """
    baseline_pb_dir = Path(baseline_pb_dir)

    # Fixed path layout expected by the challenge dataset.
    ct_path        = baseline_pb_dir / modality / split / patient_id / DIR_IMAGE / CT_NAME
    plan_json_path = baseline_pb_dir / modality / split / patient_id / f"{patient_id}.json"
    dose_dir       = baseline_pb_dir / modality / split / patient_id / DIR_DOSE

    # General beam parameters
    beam_params_json_path = baseline_pb_dir / modality / split  / BEAM_PARAMS_FILENAME
    with open(beam_params_json_path) as f:
        beam_params = json.load(f)

    hlut_entries = beam_params["hu_to_density"]["entries"]
    hlut_array = np.array([tuple(each.values()) for each in hlut_entries], dtype=float)

    ct = ct_from_file(ct_path)

    with open(plan_json_path) as f:
        pln_json = json.load(f)

    # An empty structure set is sufficient for forward dose calculation;
    # body segmentation is created automatically from the CT HU values.
    cst = StructureSet(vois=[], ct_image=ct)
    cst.create_body_seg(voi_type="TARGET")

    # Select the requested beam and control point by their list positions.
    beam_json    = pln_json["beams"][beam_idx]
    cp_json      = beam_json["control_points"][cp_idx]

    # Read back the plan-internal IDs used in the reference dose filenames.
    b_idx        = beam_json["beam_idx"]
    c_idx        = cp_json["cp_idx"]
    gantry_angle = float(cp_json["gantry_angle"])

    pb_dose = _compute_cp_dose(ct, cst, beam_json, cp_json, hlut=hlut_array)

    # Load the Monte-Carlo ground-truth dose for this control point.
    ref_path = dose_dir / f"Dose_B{b_idx}_CP{c_idx:03d}.mha"
    ref_dose: sitk.Image = sitk.ReadImage(str(ref_path))

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # construct filename
        filename = f"{patient_id}_{modality}_{split}_PB_Dose_B{b_idx}_CP{c_idx:03d}.mha"
        write_path = output_dir / filename

        sitk.WriteImage(pb_dose, str(write_path), useCompression=True)

    if show_plots:
        plot_dose_comparison(
            ct=ct, cst=cst,
            pb_dose=pb_dose, ref_dose=ref_dose,
            pb_title=f"pyRadPlan SVDPB (BLD)  |  {_ENERGY:.0f} MeV  |  MLC + Jaws",
            ref_title=f"MC Reference  |  B{b_idx} CP{c_idx:03d}",
            suptitle=(
                f"Patient: {patient_id}  |  gantry={gantry_angle:.0f}°  |  "
                f"B{b_idx} CP{c_idx:03d}"
            ),
            plane=plot_view_slice,
            use_shared_max=plot_shared_max,
        )

    return sitk.GetArrayFromImage(pb_dose)


if __name__ == "__main__":
    baseline_dir = default_data_path()
    write_dir = "out/"    
    compute_pb_dose(
        baseline_pb_dir=baseline_dir,
        output_dir=write_dir,
        patient_id="1ABB169",
        modality="photon",
        split="training",
        show_plots=True,
        plot_view_slice="axial",
        plot_shared_max=True,
        beam_idx=0,
        cp_idx=1,
    )