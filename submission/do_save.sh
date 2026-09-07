#!/usr/bin/env bash
# Build the submission container and save it as ONE file for Grand Challenge.
#
#   ./submission/do_save.sh
#
# Produces  doserad-photon_<build-timestamp>.tar.gz  in the repo root — upload
# that single file to Grand Challenge ("upload a container image" rather than
# linking a GitHub repo).
#
# ONE image serves BOTH interfaces. Nothing is baked in per modality: the
# container inspects its own /input at invoke time and picks the arm.
#   * photon-ct   -> images/radiation-dose-calculation-source-ct-image-N
#   * photon-mri  -> images/radiation-dose-calculation-source-mri-image-N
# gc_inference.detect_modality_tag() reads that directory name; the MR arm then
# converts MR -> synthetic CT (sct.py) and everything downstream is identical.
# NOTE the beam-level metadata JSON distinguishes photon from proton, NOT ct
# from mri, so the image directory is the thing to look at.
#
# TASK is deliberately NOT set in the Dockerfile — setting it would pin the
# image to one interface, which is exactly what we do not want.
set -euo pipefail
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
REPO_DIR=$( cd -- "${SCRIPT_DIR}/.." &> /dev/null && pwd )
IMAGE_TAG="${IMAGE_TAG:-doserad-photon}"
# Grand Challenge rejects a second upload of the same image SHA. The CT and MR
# Algorithms therefore need distinct images even though the code is identical.
# Pass a name to stamp one:  ./submission/do_save.sh ct   /   ... mri
BUILD_ID="${1:-${BUILD_ID:-$(date +%Y%m%d%H%M%S)}}"

command -v docker >/dev/null || { echo "docker not found"; exit 1; }

echo "= STEP 1 = Build ${IMAGE_TAG}  (BUILD_ID=${BUILD_ID})"
# --provenance=false --sbom=false: modern buildx defaults to emitting an OCI
# image INDEX with an extra attestation manifest attached. `docker save` then
# writes an index whose single entry is itself an index, and Grand Challenge's
# importer wants one plain manifest for one platform. Turning both off gives a
# single-manifest linux/amd64 image, which is what GC ingests.
# --platform is explicit for the same reason: never hand it a multi-arch list.
docker build --provenance=false --sbom=false --platform linux/amd64 \
    --build-arg "BUILD_ID=${BUILD_ID}" -t "${IMAGE_TAG}:${BUILD_ID}" "${REPO_DIR}"
docker tag "${IMAGE_TAG}:${BUILD_ID}" "${IMAGE_TAG}"

echo
echo "= STEP 2 = Sanity-check what is inside the image"
# This runs on CPU, so it cannot catch a CUDA problem — but it DOES catch the
# two failure modes that have actually bitten us: a repo module the Dockerfile
# forgot to COPY (which only surfaces on the first control point, i.e. inside a
# scored run), and a checkpoint whose architecture the server cannot rebuild.
docker run --rm --entrypoint /bin/sh "${IMAGE_TAG}:${BUILD_ID}" -c '
  set -e
  test -f /opt/app/mrtoct_bundle/model.pt     || { echo "MISSING sCT bundle"; exit 1; }
  test -f /opt/app/mrtoct_bundle/metadata.json|| { echo "MISSING sCT metadata"; exit 1; }
  ls /opt/app/models/*.pt >/dev/null|| { echo "MISSING corrector weights"; exit 1; }
  python -c "
import os, sys, torch
sys.path.insert(0,\"/opt/app\"); sys.path.insert(0,\"/opt/app/submission\")
import gc_inference, sct, configs, pydosert
# Import the v2 feature modules explicitly. engines only imports them lazily,
# inside the corrector forward, so a missing COPY is invisible until inference.
import features_v2, materials_v2
name = os.environ.get(\"MODEL_NAME\", \"photon_corrector.pt\")
corr = gc_inference.build_model(\"/opt/app/models/\"+name, torch.device(\"cpu\"))
n_in = 1 + corr.feature_cfg.n_scalar_channels() if corr.feature_cfg else 2
print(\"  machine  : tpr\", configs.machine_config.tpr_20_10, \"| MeV\", configs.machine_config.mean_photon_energy_MeV)
print(\"  pydosert :\", pydosert.__version__)
print(\"  corrector:\", sum(p.numel() for p in corr.model.parameters()), \"params,\", n_in, \"input channels\")
print(\"  features : v2 stack\" if corr.feature_cfg else \"  features : v1 (dose, density)\")
print(\"  engine   :\", corr.engine, \"(the baseline the corrector was TRAINED on)\")
print(\"  chunk    : GC_BATCH_CHUNK =\", gc_inference.BATCH_CHUNK, \"(halves on OOM and keeps going; 16 and 8 both crashed the A10G)\")
print(\"  H crop   :\", gc_inference.DEFAULT_CROP_MARGIN_MM, \"mm margin (trained at 30; 60 measured +2.02 gamma, -47% IDD)\")
assert gc_inference.DEFAULT_CROP_MARGIN_MM >= 60.0, \"crop margin regressed below the measured optimum\"
print(\"  modality : auto-detected per invoke, both arms in one image\")
print(\"  sCT      : bundle loads ->\", sct.DEFAULT_BUNDLE_DIR)
assert gc_inference.BATCH_CHUNK in (1, 2, 4, 8, 16), \"GC_BATCH_CHUNK must be 1, 2, 4, 8 or 16\"
assert os.environ.get(\"GC_BATCH_CHUNK\"), \"GC_BATCH_CHUNK must be DECLARED in the Dockerfile ENV, not left to the code default -- the image has to carry its own settings\"
"' || { echo "image sanity check FAILED"; exit 1; }

echo
echo "= STEP 3 = Save the image as a single file"
out="${REPO_DIR}/${IMAGE_TAG}_${BUILD_ID}.tar.gz"
docker save "${IMAGE_TAG}:${BUILD_ID}" | gzip -c > "${out}"
echo "  image SHA: $(docker inspect --format='{{.Id}}' "${IMAGE_TAG}:${BUILD_ID}")"

echo
printf 'Upload this single file to Grand Challenge:\n  \033[32m%s\033[0m\n' "${out}"
du -h "${out}" | awk '{print "  size: " $1}'
