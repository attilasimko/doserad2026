# The inference container (photon, CT + MRI)

A Grand Challenge **invoke-API** container: it reads the mounted `/input`,
computes per-control-point dose with the calibrated engine plus the trained
corrector, and writes dose maps in the challenge's format.

**One image serves both algorithms.** Nothing is baked in per modality — the
container inspects `/input` at invoke time and picks the arm from the image
directory name (`…source-ct-image-*` vs `…source-mri-image-*`). The MR arm
converts MR → synthetic CT (`sct.py`); everything downstream is identical,
including the corrector.

---

## What ships

| file | role |
|---|---|
| `../Dockerfile` | the image, at the repo root so a GitHub-linked build finds it |
| `serve.py` | FastAPI app: `GET /health`, `POST /invoke`. Loads the corrector and the sCT bundle at startup, so `/invoke` never pays for them |
| `gc_inference.py` | read GC inputs → per-CP dose via `CorrectedDoseEngine` → write GC outputs |
| `bake_model.py` | turn a training checkpoint into servable weights |
| `../models/` | the corrector and the two frozen priors |

The image also carries the repo modules the inference imports, the vendored
`pydosert/`, and the sCT TorchScript bundle downloaded at build time.

## The engine is part of the model

The corrector predicts the residual of **one** baseline dose. Serving it on a
different engine is not a configuration change — it is a different model, and
nothing detects it: the shapes all match and the dose is simply wrong.

So the engine settings travel *inside* the checkpoint's `model_config`
(`lattice_size`, `terma`, `engine`, the feature set) rather than on the command
line, and `bake_model.py` writes them there. `build_model()` reconstructs the
engine from the checkpoint and refuses a mismatch.

## Baking a checkpoint

```bash
python3 submission/bake_model.py --ckpt out/<run>_best.pt \
    --out models/photon_corrector.pt

python3 submission/bake_model.py --inspect models/photon_corrector.pt
```

`bake_model.py` writes a `model_config` dict into the checkpoint, rebuilds the
network from it with a strict load, and drops the optimiser state. The frozen
priors are embedded too, so serving never reads them from disk.

## Building the image

```bash
./submission/do_save.sh ct     # -> doserad-photon_ct.tar.gz
./submission/do_save.sh mri    # -> doserad-photon_mri.tar.gz
```

`do_save.sh` builds, runs an in-image sanity check, and saves one `.tar.gz`.
The sanity check imports the lazily-imported feature modules explicitly, loads
the baked checkpoint and prints its identity — failures it covers are otherwise
invisible until the first control point of a scored run.

The `BUILD_ID` argument (`ct` / `mri`) only changes a label so the two images
hash differently: Grand Challenge refuses a second upload of the same image
SHA, and the two algorithms need distinct images even though the code is
identical.

### The manifest format matters

The build passes `--provenance=false --sbom=false --platform linux/amd64`.
Without them, buildx emits an OCI image *index* with an attestation manifest
attached, `docker save` writes an index whose one entry is itself an index, and
the platform importer expects a single plain manifest for a single platform:

```bash
tar -xzOf doserad-photon_ct.tar.gz index.json | python3 -m json.tool
# the one entry's mediaType must be
#   application/vnd.docker.distribution.manifest.v2+json
```

### Docker Desktop / WSL

`docker build` may fail with `error getting credentials … A specified logon
session does not exist` — that is `credsStore: desktop.exe` failing to reach the
Windows credential store. The base image is public, so build with an isolated
config rather than editing the global one:

```bash
export DOCKER_CONFIG=$(mktemp -d) && echo '{}' > $DOCKER_CONFIG/config.json
```

Note also that `torch 2.6 (cu124)` supports up to `sm_90`; on a newer GPU the
container cannot use the local device and `--gpus all` fails at startup with
`no kernel image is available`. Verify on CPU in that case.

## Runtime

Ranking weights runtime at 29% of the final position (counted twice of seven
slots), fitted as

```
T = t_fix + N_images * t_im + N_dose_maps * t_dose_map
```

and scored at one image and 181 dose maps, with a hard limit of 181 s.

The container is shaped around that per-dose-map term: one engine per beam
re-pointed per control point rather than rebuilt, the dose cutoff and the `/1e5`
rescale applied on the GPU over the cropped grid, compressed slot writes, and
half precision for the corrector and the sCT network.

### Environment variables

| variable | default | meaning |
|---|---|---|
| `MODEL_NAME`, `MODEL_NAME_CT`, `MODEL_NAME_MR` | `photon_corrector.pt` | which checkpoint in `MODEL_DIR` to serve; the per-arm names allow a dedicated MR model |
| `MODEL_DIR` | `models/` | where to find them |
| `GC_CROP_MARGIN_MM` | `60` | safety margin on the geometric H-axis crop |
| `GC_AMP` | `fp16` | autocast dtype for the corrector forward (the physics stays fp32) |
| `DOSERAD_SCT_AMP` | `fp16` | autocast dtype for the sCT network (MR arm only) |
| `GC_BATCH_CHUNK` | `4` | control points stacked into one corrector forward |
| `GC_REUSE_ENGINE` | `1` | re-point one engine per beam instead of rebuilding per chunk |
| `GC_TIMING` | `0` | print a per-phase breakdown of each invoke |
| `INPUT_PATH`, `OUTPUT_PATH`, `PORT` | GC defaults | |

`GC_MIN_CUTOFF` exists for experiments and should be left unset, so the
challenge's own per-control-point `minimum_cutoff` applies.

`GC_BATCH_CHUNK` trades memory for throughput and is bounded by the GPU: peak
VRAM is roughly linear in the chunk and in the BEV box, and the box is set by
patient anatomy. An over-large value raises `OutOfMemoryError`, which
`predict_beam_batch` catches, halves the chunk, and retries.
