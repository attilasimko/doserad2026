# DoseRAD2026 — Ground truth tarball format

The evaluation container reads the ground truth from
`/opt/ml/input/data/ground_truth` (Grand Challenge extracts the tarball you
upload under *Admin → Phase Settings → Ground Truths* to this location).

Pack it with, from inside the directory that contains the case folders:

```bash
tar -czf ground_truth.tar.gz -C <ground_truth_dir> .
```

## Layout

One directory per case. The directory name must equal the case id that the
evaluator resolves for that case — the `case_id` field you put in each entry
of the stacked beam-level metadata JSON (fallback: the original upload
filename of the input image, without extension).

```
ground_truth/
└── <case_id>/
    ├── case.json            # required — see below
    ├── gt_beam_doses.mha    # required*  4D stack of per-beam GT doses
    │                        #   numpy order (beam, z, y, x); the beam order
    │                        #   must match the "beams" list in case.json
    ├── beams/<uuid>.mha     # *alternative to gt_beam_doses.mha:
    │                        #   one 3D file per beam, named by beam uuid
    ├── plan_dose.mha        # optional   3D GT plan dose on the reference
    │                        #   grid; if absent it is composed as
    │                        #   sum(weight_i * beam_i) from the GT beams
    └── structures.npy       # required   bool/uint8 array, shape
                             #   (z, y, x, N_structures)
```

## case.json

```jsonc
{
  "case_id": "case0",
  "prescription_dose_gy": 60.0,        // optional, default from task_config
  "ptv_index": 0,                      // index into structures.npy last axis
  "structure_names": ["PTV", "Liver", "Stomach", "Duodenum", "Kidney_L"],
  "oar_indices": [1, 2, 3, 4],         // optional, default: all non-PTV
  "beam_axis": 0,                      // optional, IDD axis in numpy (z,y,x)
                                       //   order; default from task_config
  "beams": [                           // defines GT beam order and weights
    {"uuid": "9e627303-4af7-48be-9921-7e163ca9400c", "weight": 1.85},
    {"uuid": "2b97a613-21e0-4cad-bccc-8a6c793bcff7", "weight": 2.10}
  ]
}
```

Notes:

- `uuid` is the photon `cp_uuid` or proton `beamlet_uuid` from the archive
  metadata. The evaluator matches predictions to GT exclusively through
  these uuids, so prediction stacking order does not matter.
- `weight` is the clinical weight (VMAT MU weight / pencil-beam spot weight)
  used to compose the full plan for Level 2 metrics. Scale the weights so
  the composed GT plan is in Gy relative to `prescription_dose_gy`.
- All volumes (GT beams, plan dose, structures, and participant outputs)
  must be on the task's reference dose grid: 2×2×2 mm (photon) or
  1×1×3 mm (proton). The gamma computation takes spacing from
  `plan_dose.mha` when present, otherwise from `task_config.json`.
- Missing predicted beams, shape mismatches, or a missing case directory
  fail the evaluation with an explicit error message (visible in the
  evaluation logs), rather than silently producing NaN scores.
