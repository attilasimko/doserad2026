# DoseRAD2026 — photon dose prediction

Photon dose prediction for the
[DoseRAD2026 Grand Challenge](https://doserad2026.grand-challenge.org/),
on both the CT and MR arms.

The method is a **calibrated analytical dose engine with a small learned
corrector on top**, rather than an end-to-end network. A 1×1 divergent
ray-lattice pencil beam with TERMA scaling produces a physics baseline; two
frozen priors and a 1.36 M-parameter 3D U-Net correct it. The MR arm converts
MR → synthetic CT first and then runs the identical CT pipeline.

The challenge ranking weights runtime at 29% (it is counted twice of seven
slots), so the design trades accuracy against inference cost explicitly.

---

## What the components contribute

Cumulative ablation, Level-1 masked beam MAE, over the full validation protocol
— 6 patients × 3 beams × 180 control points = 3240 forwards per stage:

| stage | params added | CT | MR |
|---|---|---|---|
| bare dose engine | 0 | 0.04896 | 0.04891 |
| + fluence prior | 39,081 | 0.03745 | 0.03746 |
| + heterogeneity prior | **48** | 0.03506 | 0.03521 |
| + U-Net corrector | 1,361,187 | **0.00874** | **0.01023** |
| | | **−82.1%** | **−79.1%** |

Two things that table shows. The 48-parameter heterogeneity prior is
anatomically targeted — it does almost nothing on abdomen (−1.1%) and −8.4% on
thorax, which is where lateral electronic disequilibrium lives. And the physics
stages are identical across modalities to within 0.1%, so the synthetic CT is
dosimetrically equivalent to real CT for the engine; the entire MR/CT gap
appears only at the corrector, which never saw a synthetic CT during training.

Final validation numbers for the submitted model:

| | CT | MR |
|---|---|---|
| local gamma 1%/1mm | 99.12 ± 0.42 | 96.84 ± 2.59 |
| stratified plan MAE | 0.00237 | 0.00474 |
| Level-1 beam MAE | 0.00855 | 0.01010 |
| Level-1 IDD | 0.00532 | 0.00539 |

The DVH clinical score is one of the six ranked metrics and is **not** measured
here: it needs PTV/OAR contours, and none ship with the released data.

---

## Layout

| path | what |
|---|---|
| `engines.py` | `CorrectedDoseEngine` — the lattice/TERMA engine plus corrector orchestration |
| `features_v2.py` | the 7-channel beam's-eye-view feature stack the corrector consumes |
| `sct.py` | MR → synthetic CT (TorchScript bundle, fetched from Hugging Face) |
| `train_correction.py` | corrector training |
| `eval_patient_total.py`, `eval_plans.py`, `eval_validation.py` | scoring |
| `models/` | the shipped corrector and the two frozen priors — see `models/README.md` |
| `pydosert/` | vendored physics engine (MIT, upstream UMU-DDI/PyDoseRT) |
| `submission/` | the Grand Challenge container — see `submission/README.md` |
| `Dockerfile` | the invoke-API image (at the root so GC's GitHub build finds it) |

The physics comes from [PyDoseRT](https://github.com/UMU-DDI/PyDoseRT) and is
**vendored** at `pydosert/` (MIT, commit `d74b4038`) rather than installed from
GitHub. The corrector predicts the residual of ONE baseline dose, so a rebuild
that picked up a moved branch would change the physics underneath a corrector
trained on the old one — with no error anywhere, since the shapes all match.
See `pydosert/VENDORED.md`. Anything with a number fitted to this dataset —
the TERMA constants, the multislab calibration, both priors — lives here.

---

## Setup

```bash
pip install -r requirements.txt                  # pydosert is vendored, not installed

export DOSERAD_DATA=/path/to/DoseRAD2026        # contains photon/ and proton/
python -c "import sct; sct.fetch_bundle()"      # 390 MB sCT weights
```

The sCT TorchScript weights are too large for git and live at
**[zimmeryWo/MRI-sCT_converter-DoseRAD2026](https://huggingface.co/zimmeryWo/MRI-sCT_converter-DoseRAD2026)**.
`fetch_bundle()` downloads them into `mrtoct_bundle/` and verifies the size; the
Dockerfile fetches them at build time so the image stays self-contained (the
scored container has no network).

## Training the corrector

The dose engine and both priors are fixed; only the corrector is trained.

```bash
python train_correction.py --data_path "$DOSERAD_DATA" \
    --modality ct --anatomies both --num_epochs 50 \
    --num_beams 3 --val_num_beams 3 --val_cp_stride 1 \
    --batched_cp_size 3 --batch_size 8 \
    --optimizer adamw --lrate 5e-4 --lr_schedule constant --grad_clip 1.0 \
    --amp --v2_features --v2_loss --feature_set v4 --v1_norm none \
    --refine_mode sequential --v2_alpha 0.20 \
    --run_tag my_run
```

`--warm_start models/photon_corrector.pt` continues from the released model
instead of starting cold. Training logs to Comet if `COMET_API` is set and
falls back to an offline experiment otherwise.

Two flags that are easy to misread: `--batched_cp_size` is the real batch
(control points stacked into one forward, so it drives memory), while
`--batch_size` is gradient accumulation and costs none.

## Evaluating

```bash
python eval_patient_total.py --data_path "$DOSERAD_DATA" --cohort validating \
    --modality ct --beams 0,1,2 --cp_stride 1 --ckpt <checkpoint>.pt \
    --crop_margin_mm 60 --amp fp16
```

`--cp_stride 1` and all three beams are not optional. Subsampling control points
at a fixed stride samples the same gantry angles every time, so whatever those
angles do wrong stays invisible — that produced a local beam MAE of 0.0111
against 0.0196 on the leaderboard, and two design decisions went the wrong way
on the strength of it.

## Building the container

```bash
./submission/do_save.sh ct     # -> doserad-photon_ct.tar.gz
```

See `submission/README.md` for the model-baking and verification steps.

---

## Licence

Apache-2.0 (`LICENSE`), except `pydosert/`, which is vendored from
[UMU-DDI/PyDoseRT](https://github.com/UMU-DDI/PyDoseRT) and remains under its
own MIT licence (`pydosert/LICENSE`).
