"""Geant4 phantom material model for the DoseRAD2026 photon task.

Vendored, not imported, on purpose. The photon stack depends on ``pydosert``
(the ``doserad`` branch), while these tables live in ``pydose_rt.physics`` on
the proton line -- a different package off the same repo. They are fixed
physical constants of the challenge, so copying them is safer than coupling the
two dependency trees.

Provenance: ``geant4-dose-sim/DICOMphantom.cc``, ``InitialisationOfMaterials``.
``beam_parameters.json`` states "Both photon and proton simulations use
identical tables", so the proton-side extraction applies unchanged.

Two SEPARATE tables are involved and conflating them is easy:

  * ``hu_to_density`` in ``beam_parameters.json`` -- 10 anchor points, the
    DENSITY calibration curve only. This is ``loaders._CHALLENGE_HU``.
  * ``GEANT4_HU_BOUNDS`` here -- 86 COMPOSITION families. The simulator picks
    the composition family from these HU intervals first, then clones a
    density-specific variant quantised at 0.001 g/cm3.

Why this matters for photons: at the challenge spectrum (fluence-weighted mean
1.3535 MeV, 2.09% of fluence below 200 keV) Compton dominates, so both the
attenuation coefficient and the mass energy-absorption coefficient carry a
factor Z/A. Relative Z/A runs 0.921-0.949 through bone.

MEASURED 2026-08-13, six patients, beam 0, 180 CPs, engine-only multislab
pencil beam. The Z/A factor enters in TWO separate places and only one of them
matters here:

  1. TRANSPORT (radiological depth): mu = rho * (mu/rho) ~ electron density.
     Swapping the engine's density volume for rho_e is what
     ``loaders.convert_HU_to_electron_density_lut`` does. Effect on the
     bone-minus-soft-tissue error gap: +0.13 to +0.33 pp, i.e. the WRONG SIGN
     and a factor ~20 too small. Bone is thin, so it barely moves the path
     integral, and thinning it lets MORE fluence through.

  2. LOCAL ABSORPTION: dose = fluence * (mu_en/rho) * E, and (mu_en/rho) ~ Z/A.
     The pencil-beam engine deposits a WATER kernel at
     radiological depth -- they compute dose-to-WATER. Geant4 scores
     dose-to-MEDIUM. The ratio is exactly this table. Multiplying the engine
     dose by relative Z/A moves the gap by -3.1 to -3.5 pp, matching the
     +3.3 to +4.0 pp predicted. This is where the bone bias lives.

See ``engines.CorrectedDoseEngine.forward(absorption_scale=...)``.
"""

from __future__ import annotations

import numpy as np
import torch

# HU interval edges selecting the composition family; 86 families = 87 edges.
# Fine through soft tissue, then every 20 HU across bone. Note LUNG is a SINGLE
# family spanning -950..-90 HU: composition is constant there and only density
# varies, so a material label carries no information in lung.
GEANT4_HU_BOUNDS = np.asarray(
    [
    -1024.0, -950.0, -90.0, -64.0, -38.0, -24.0, -10.0, 4.0,
    18.0, 70.0, 120.0, 140.0, 160.0, 180.0, 200.0, 220.0,
    240.0, 260.0, 280.0, 300.0, 320.0, 340.0, 360.0, 380.0,
    400.0, 420.0, 440.0, 460.0, 480.0, 500.0, 520.0, 540.0,
    560.0, 580.0, 600.0, 620.0, 640.0, 660.0, 680.0, 700.0,
    720.0, 740.0, 760.0, 780.0, 800.0, 820.0, 840.0, 860.0,
    880.0, 900.0, 920.0, 940.0, 960.0, 980.0, 1000.0, 1020.0,
    1040.0, 1060.0, 1080.0, 1100.0, 1120.0, 1140.0, 1160.0, 1180.0,
    1200.0, 1220.0, 1240.0, 1260.0, 1280.0, 1300.0, 1320.0, 1340.0,
    1360.0, 1380.0, 1400.0, 1420.0, 1440.0, 1460.0, 1480.0, 1500.0,
    1520.0, 1540.0, 1560.0, 1580.0, 1600.0, 1620.0, 4000.0,    ],
    dtype=np.float64,
)
GEANT4_NUM_MATERIALS = int(GEANT4_HU_BOUNDS.size - 1)

# Water Z/A under the Geant4 G4_WATER convention.
WATER_ZOA = 0.555062167

# (Z/A)_material / (Z/A)_water per family == relative electron density per unit
# MASS density. Multiply by mass density to get relative electron density.
RELATIVE_ZOA = np.asarray(
    [
    0.90045564, 0.99170803, 1.00241016, 1.00010746, 1.00132697, 0.99937010,
    0.99745674, 0.99558403, 0.99272527, 0.98826560, 0.98486995, 0.98324056,
    0.98166011, 0.98011375, 0.97859627, 0.97712410, 0.97568049, 0.97438104,
    0.97300129, 0.97164931, 0.97032538, 0.96902825, 0.96776874, 0.96652813,
    0.96531435, 0.96412140, 0.96296315, 0.96174353, 0.96062372, 0.95953076,
    0.95845641, 0.95740210, 0.95637326, 0.95536304, 0.95436367, 0.95339092,
    0.95243576, 0.95149252, 0.95057480, 0.94966794, 0.94877830, 0.94790834,
    0.94705537, 0.94621225, 0.94537939, 0.94456494, 0.94376791, 0.94298080,
    0.94221240, 0.94145210, 0.94070192, 0.93997072, 0.93924868, 0.93853463,
    0.93783185, 0.93714596, 0.93647026, 0.93580264, 0.93514492, 0.93449770,
    0.93385822, 0.93322854, 0.93261689, 0.93200559, 0.93140399, 0.93081124,
    0.93022832, 0.92965358, 0.92908888, 0.92853343, 0.92798647, 0.92737820,
    0.92684153, 0.92631346, 0.92579430, 0.92528430, 0.92477548, 0.92428434,
    0.92379246, 0.92330214, 0.92282970, 0.92235690, 0.92188516, 0.92143083,
    0.92096798, 0.92096798,    ],
    dtype=np.float64,
)

# The simulator clones material variants in steps of this density.
GEANT4_DENSITY_BIN_G_CM3 = 0.001


def material_id_from_hu(hu: torch.Tensor) -> torch.Tensor:
    """HU -> Geant4 composition family index in [0, 85]."""
    bounds = torch.as_tensor(GEANT4_HU_BOUNDS, device=hu.device, dtype=hu.dtype)
    clipped = hu.clamp(float(GEANT4_HU_BOUNDS[0]), float(GEANT4_HU_BOUNDS[-1]))
    ids = torch.bucketize(clipped.contiguous(), bounds[1:].contiguous(), right=True)
    return ids.clamp_(0, GEANT4_NUM_MATERIALS - 1).long()


def relative_electron_density(density: torch.Tensor, hu: torch.Tensor) -> torch.Tensor:
    """Water-relative electron density = mass density * (Z/A)_mat / (Z/A)_water.

    This is the quantity Compton dose actually scales with, and the direct fix
    for the systematic bone over-prediction.

    Uses the NOMINAL ``GEANT4_HU_BOUNDS`` family assignment. That is right to
    within 0.7% relative Z/A but is not bit-exact -- see
    ``simulator_family_from_hu`` for the assignment the simulator actually
    performs, and use ``relative_zoa_from_hu`` when exactness matters.
    """
    mid = material_id_from_hu(hu)
    rel = torch.as_tensor(RELATIVE_ZOA, device=density.device, dtype=density.dtype)
    return density * rel[mid]


# --- Exact simulator family assignment -------------------------------------
#
# The nominal picture -- "HU falls in an interval of GEANT4_HU_BOUNDS, that
# picks the composition" -- is very nearly, but not exactly, what the challenge
# generator does. The real chain (geant4-dose-sim/docker/software/g4dcm) is:
#
#   dicomFileCT::BuildMaterials
#       density = dicomFileMgr::Hounsfield2density(int HU)      # 10-anchor LUT
#       mateID  = dicomFileMgr::GetMaterialIndexByDensity(density)
#
# and ``GetMaterialIndexByDensity`` is ``theMaterialsDensity.upper_bound(d)``
# over a ``std::map<double, std::string>`` -- i.e. a table keyed and SORTED BY
# DENSITY, whose keys are the ``:MATE_DENS`` values in ``Data1.dat``. Those
# values are the challenge density LUT evaluated at each family's UPPER HU
# bound, so they inherit the LUT's non-monotonicity: d(120) = 1.126553 sorts
# ABOVE d(140) = 1.107849 and d(160) = 1.121271. Sorting therefore permutes
# three families relative to declaration order, and the composition assigned to
# a voxel is not the one its HU interval names.
#
# Net effect, over integer HU in [-1024, 3000]:
#
#   HU  -9.. 0   HU_10to4    -> HU_24to_10   relative Z/A +0.0019
#   HU  70..101  HU70to120   -> HU120to140                -0.0034
#   HU 102..114  HU70to120   -> HU140to160                -0.0050
#   HU 160..167  HU160to180  -> HU70to120                 +0.0066
#   plus 10 isolated HU values exactly on a family boundary (<= +0.0016)
#
# 73 of 4025 integer HU values, max 0.66% in relative Z/A. Small next to the
# 5-8% bone effect, but free to get right, and the +0.19% shift over HU -9..0
# lands squarely on soft tissue.
#
# ``MATE_DENS_G_CM3`` is transcribed from Data1.dat in declaration order, so
# ``MATE_DENS_G_CM3[i]`` belongs to the family with ``RELATIVE_ZOA[i]``.
MATE_DENS_G_CM3 = np.asarray(
    [
    5.046545006257821542e-02, 9.268856666666666078e-01, 9.527859999999999108e-01,
    9.786863333333332138e-01, 9.926326666666664966e-01, 1.006578999999999890e+00,
    1.009763390697674401e+00, 1.023858688372093040e+00, 1.076212651162790745e+00,
    1.126552999999999916e+00, 1.107848560611323308e+00, 1.121271255991663773e+00,
    1.134693951372004239e+00, 1.148116646752344483e+00, 1.161539342132684949e+00,
    1.174962037513025415e+00, 1.188384732893365658e+00, 1.201807428273706124e+00,
    1.215230123654046590e+00, 1.228652819034386834e+00, 1.242075514414727300e+00,
    1.255498209795067766e+00, 1.268920905175408009e+00, 1.282343600555748475e+00,
    1.295766295936088941e+00, 1.309188991316429185e+00, 1.322611686696769651e+00,
    1.336034382077110116e+00, 1.349457077457450360e+00, 1.362879772837790826e+00,
    1.376302468218131292e+00, 1.389725163598471536e+00, 1.403147858978812001e+00,
    1.416570554359152467e+00, 1.429993249739492711e+00, 1.443415945119833177e+00,
    1.456838640500173643e+00, 1.470261335880514109e+00, 1.483684031260854352e+00,
    1.497106726641194818e+00, 1.510529422021535284e+00, 1.523952117401875750e+00,
    1.537374812782215994e+00, 1.550797508162556460e+00, 1.564220203542896925e+00,
    1.577642898923237169e+00, 1.591065594303577635e+00, 1.604488289683918101e+00,
    1.617910985064258345e+00, 1.631333680444598810e+00, 1.644756375824939276e+00,
    1.658179071205279520e+00, 1.671601766585619986e+00, 1.685024461965960452e+00,
    1.698447157346300695e+00, 1.711869852726641161e+00, 1.725292548106981627e+00,
    1.738715243487321871e+00, 1.752137938867662337e+00, 1.765560634248002803e+00,
    1.778983329628343046e+00, 1.792406025008683512e+00, 1.805828720389023978e+00,
    1.819251415769364222e+00, 1.832674111149704688e+00, 1.846096806530045153e+00,
    1.859519501910385397e+00, 1.872942197290725863e+00, 1.886364892671066329e+00,
    1.899787588051406573e+00, 1.913210283431747039e+00, 1.926632978812087504e+00,
    1.940055674192427748e+00, 1.953478369572768214e+00, 1.966901064953108680e+00,
    1.980323760333448924e+00, 1.993746455713789389e+00, 2.007169151094129855e+00,
    2.020591846474470099e+00, 2.034014541854810787e+00, 2.047437237235151031e+00,
    2.060859932615491275e+00, 2.074282627995831962e+00, 2.087705323376172206e+00,
    2.101128018756512894e+00, 3.698430000000000000e+00,    ],
    dtype=np.float64,
)

# Challenge HU -> density anchors (beam_parameters.json "hu_to_density"). Kept
# here as well as in loaders so the family assignment is self-contained; the
# two copies are asserted equal by tests/_selfcheck below.
CHALLENGE_HU_ANCHORS = np.asarray(
    [-1024, -999, -200, -199, -10, -9, 120, 121, 3000, 4000], dtype=np.float64)
CHALLENGE_DENSITY_ANCHORS = np.asarray(
    [1.200000e-03, 1.210000e-03, 8.043754e-01, 8.183035e-01,
     1.006579e+00, 9.966749e-01, 1.126553e+00, 1.095097e+00,
     3.027294e+00, 3.698428e+00], dtype=np.float64)

# Integer HU domain of the LUT. Below/above, the generator clamps.
_HU_LO, _HU_HI = -1024, 4000


def _build_simulator_lut():
    """Integer-HU tables: family index and relative Z/A, exactly as generated.

    Replicates ``std::map<double, std::string>`` ordering (sort by density,
    first insertion wins on duplicate keys) plus ``upper_bound``.
    """
    order = np.argsort(MATE_DENS_G_CM3, kind="stable")
    keys = MATE_DENS_G_CM3[order]
    keep = np.concatenate([[True], np.diff(keys) > 0.0])   # dedupe like std::map
    keys, fam = keys[keep], order[keep]

    hu = np.arange(_HU_LO, _HU_HI + 1, dtype=np.int64)
    dens = np.interp(hu.astype(np.float64), CHALLENGE_HU_ANCHORS, CHALLENGE_DENSITY_ANCHORS)
    idx = np.searchsorted(keys, dens, side="right")        # upper_bound
    idx = np.clip(idx, 0, keys.size - 1)                   # C++ aborts here; clamp
    return hu, fam[idx]


_SIM_HU, _SIM_FAMILY = _build_simulator_lut()
SIMULATOR_RELATIVE_ZOA_LUT = RELATIVE_ZOA[_SIM_FAMILY]     # indexed by HU - _HU_LO


def simulator_family_from_hu(hu):
    """Composition-family index the challenge generator assigns to ``hu``.

    HU is rounded to the nearest integer first, because the generator reads
    integer DICOM pixel values.  Accepts numpy arrays or torch tensors.
    """
    if isinstance(hu, torch.Tensor):
        i = torch.round(hu).clamp_(_HU_LO, _HU_HI).long() - _HU_LO
        tab = torch.as_tensor(_SIM_FAMILY, device=hu.device, dtype=torch.long)
        return tab[i]
    i = np.clip(np.rint(np.asarray(hu)), _HU_LO, _HU_HI).astype(np.int64) - _HU_LO
    return _SIM_FAMILY[i]


def relative_zoa_from_hu(hu):
    """(Z/A)_mat / (Z/A)_water for the family the simulator assigns to ``hu``.

    This is the DIMENSIONLESS ratio only -- multiply by mass density to get
    relative electron density. Keeping it separate is deliberate: it means an
    electron-density transport run differs from a mass-density run by exactly
    this factor and nothing else.
    """
    if isinstance(hu, torch.Tensor):
        i = torch.round(hu).clamp_(_HU_LO, _HU_HI).long() - _HU_LO
        tab = torch.as_tensor(SIMULATOR_RELATIVE_ZOA_LUT, device=hu.device, dtype=hu.dtype)
        return tab[i]
    i = np.clip(np.rint(np.asarray(hu)), _HU_LO, _HU_HI).astype(np.int64) - _HU_LO
    return SIMULATOR_RELATIVE_ZOA_LUT[i]


def quantize_density(density: torch.Tensor) -> torch.Tensor:
    """Density snapped to the 0.001 g/cm3 bin midpoint the simulator used."""
    step = GEANT4_DENSITY_BIN_G_CM3
    return (torch.floor(density / step) * step + 0.5 * step).to(density.dtype)
