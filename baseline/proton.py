import json
import logging
from pathlib import Path

from typing import Literal, Optional
from numpy.typing import NDArray

import numpy as np
import SimpleITK as sitk

from pyRadPlan import IonPlan, calc_dose_forward
from pyRadPlan.ct import ct_from_file
from pyRadPlan.cst import StructureSet
from pyRadPlan.stf import SteeringInformation
from pyRadPlan.stf._beam import Beam
from pyRadPlan.geometry import get_beam_rotation_matrix
from pyRadPlan.machines import load_from_name, IonAccelerator

from utils import DIR_IMAGE, DIR_DOSE, BEAM_PARAMS_FILENAME, CT_NAME, plot_dose_comparison
from configs import default_data_path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



def _compute_beamlet_dose(
    ct,
    cst,
    pln,
    beam_json: dict,
    ray_json: dict,
    bl_json: dict,
    sad: float,
    machine_energies: NDArray,
) -> tuple[sitk.Image, float]:
    """Compute pencil-beam dose for a single beamlet.

    Reconstructs the beam geometry from the plan JSON and runs a forward
    dose calculation via pyRadPlan's Hong pencil-beam algorithm.

    Returns the dose image and the matched machine energy (MeV).
    """
    gantry_angle = float(beam_json["gantry_angle"])

    # Rotation matrix that maps beam-eye-view (BEV) coordinates to world
    # coordinates for this gantry angle (couch angle fixed to 0°).
    R = get_beam_rotation_matrix(gantry_angle, 0.0)

    # The ray target in world coordinates doubles as the iso-center for this
    # individual beamlet setup, matching how the reference dose was generated.
    ray_target_world = np.array(ray_json["ray_target"], dtype=float)
    iso_center_beam = ray_target_world

    # In the single-beamlet STF the ray sits exactly on the beam axis (origin
    # in BEV space), so both world and BEV ray positions are zero vectors.
    ray_pos = np.zeros(3)
    ray_pos_bev = np.zeros(3)

    # Source lies SAD mm upstream along the BEV y-axis; rotate to world frame.
    source_point_bev = np.array([0.0, -sad, 0.0])
    source_point = R @ source_point_bev

    # Snap the requested energy to the nearest value available in the machine
    # model, since the machine LUT may not contain every arbitrary energy.
    energy = float(
        machine_energies[np.argmin(np.abs(machine_energies - bl_json["energy"]))]
    )

    # Build a minimal single-beam steering information object for this beamlet.
    beam = Beam.model_validate({
        "gantry_angle":     gantry_angle,
        "couch_angle":      0.0,
        "radiation_mode":   "protons",
        "machine":          "Generic",
        "SAD":              sad,
        "iso_center":       iso_center_beam,
        "source_point_bev": source_point_bev,
        "source_point":     source_point,
        "rays": [{
            "ray_pos_bev": ray_pos_bev,
            "ray_pos":     ray_pos,
            "beamlets":    [{"energy": energy}],
        }],
    })
    stf = SteeringInformation(beams=[beam])

    result = calc_dose_forward(ct, cst, stf, pln)
    pb_dose: sitk.Image = result["physical_dose"]

    return pb_dose, energy


def main(
    baseline_pb_dir: str | Path,
    patient_id: str,
    modality: str,
    split: str,
    output_dir: Optional[str | Path] = None,
    beam_idx: int = 0,
    ray_idx: int = 0,
    beamlet_idx: int = 0,
    show_plots: bool = False,
    plot_view_slice: Literal["axial", "sagittal", "coronal"] = "axial",
    plot_shared_max: bool = True,
) -> None:
    """Run the proton pencil-beam baseline for a single beamlet and compare
    against the Monte-Carlo ground truth.

    Parameters
    ----------
    baseline_pb_dir:
        Root directory of the dataset (e.g. ``/home/user/doserad2026``).
    patient_id:
        Patient identifier string (e.g. ``"1THB008"``).
    modality:
        Radiation modality folder name (e.g. ``"proton"``).
    split:
        Dataset split folder name (e.g. ``"train"`` or ``"test"``).
    beam_idx:
        List index of the beam in the plan JSON (0-based).
    output_dir:
        Optional directory to write the computed PB dose image
    ray_idx:
        List index of the ray within the selected beam (0-based).
    beamlet_idx:
        List index of the beamlet within the selected ray (0-based).
    show_plots:
        Whether to display a comparison figure for the beamlet.
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

    ct = ct_from_file(ct_path)
    logger.info("CT loaded: size=%s  spacing=%s mm", ct.size, ct.resolution)

    with open(plan_json_path) as f:
        pln_json = json.load(f)

    # General beam parameters
    beam_params_json_path = baseline_pb_dir / modality / split / BEAM_PARAMS_FILENAME
    with open(beam_params_json_path) as f:
        beam_params = json.load(f)

    hlut_entries = beam_params["hu_to_density"]["entries"]
    hlut_array = np.array([tuple(each.values()) for each in hlut_entries], dtype=float)

    # SAD is stored per plan and may differ between patients/institutions.
    sad = float(1000.0)#pln_json["SAD"])

    # Load the generic proton machine model once and sort its energy levels so
    # we can do nearest-neighbour energy matching for each beamlet.
    machine: IonAccelerator = load_from_name("protons", "Generic")
    machine_energies = np.array(sorted(machine.energies), dtype=np.float64)

    if machine.sad != sad:
        logger.warning(
            "Machine SAD (%.1f mm) does not match plan JSON SAD (%.1f mm)! "
            "This comes from different interpretations of the SAD parameter "
            "between the DoseRad dataset (nozzle) and pyRadPlan (virtual scan" \
            " source). We override with the pyRadPlan machine SAD.",
            machine.sad, sad,
        )
        sad = machine.sad

    # An empty structure set is sufficient for forward dose calculation;
    # body segmentation is created automatically from the CT HU values.
    cst = StructureSet(vois=[], ct_image=ct)
    cst.create_body_seg()

    # Use the CT's native resolution for the dose grid to avoid resampling
    # artefacts when comparing against the per-voxel MC reference doses.
    pln = IonPlan(radiation_mode="protons", machine="Generic")
    pln.prop_dose_calc = {
        "dose_grid": {"resolution": ct.resolution},
        "air_offset_correction": True,
        "geometric_lateral_cutoff": 25.0,
        "trace_on_dose_grid": True,
        "hlut": hlut_array,
    }

    # Select the requested beam, ray, and beamlet by their list positions.
    beam_json    = pln_json["beams"][beam_idx]
    ray_json     = beam_json["rays"][ray_idx]
    bl_json      = ray_json["beamlets"][beamlet_idx]

    # Read back the plan-internal IDs used in the reference dose filenames.
    b_idx        = beam_json["beam_idx"]
    r_idx        = ray_json["ray_idx"]
    l_idx        = bl_json["beamlet_idx"]
    gantry_angle = float(beam_json["gantry_angle"])

    logger.info(
        "Processing B%d R%d L%d  gantry=%.1f°",
        b_idx, r_idx, l_idx, gantry_angle,
    )

    pb_dose, energy = _compute_beamlet_dose(
        ct, cst, pln,
        beam_json, ray_json, bl_json,
        sad, machine_energies,
    )
    logger.info(
        "  PB dose computed  |  matched energy=%.4f MeV  (JSON: %.4f MeV)",
        energy, bl_json["energy"],
    )

    # Load the Monte-Carlo ground-truth dose for this beamlet.
    ref_path = dose_dir / f"Dose_B{b_idx}_R{r_idx}_L{l_idx}.mha"
    ref_dose: sitk.Image = sitk.ReadImage(str(ref_path))
    logger.info("  Reference dose loaded: %s", ref_path)

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # construct filename
        filename = f"{patient_id}_{modality}_{split}_PB_Dose_B{b_idx}_R{r_idx}_L{l_idx}.mha"
        write_path = output_dir / filename

        sitk.WriteImage(pb_dose, str(write_path), useCompression=True)
        logger.info("  PB dose written: %s", write_path)

    if show_plots:
        plot_dose_comparison(
            ct=ct, cst=cst,
            pb_dose=pb_dose, ref_dose=ref_dose,
            pb_title=f"pyRadPlan HongPB  |  {energy:.1f} MeV",
            ref_title=f"MC Reference  |  B{b_idx} R{r_idx} L{l_idx}",
            suptitle=(
                f"Patient: {patient_id}  |  gantry={gantry_angle:.0f}°  |  {energy:.1f} MeV"
            ),
            plane=plot_view_slice,
            use_shared_max=plot_shared_max,
        )


if __name__ == "__main__":
    baseline_dir = default_data_path()
    write_dir = "out/"
    main(
        baseline_pb_dir=baseline_dir,
        output_dir=write_dir,
        patient_id="1THB008",
        modality="proton",
        split="training",
        show_plots=True,
        plot_view_slice="axial",
        plot_shared_max=True,
    )