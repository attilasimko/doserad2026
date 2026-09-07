# DoseRAD2026 Grand Challenge invoke container (photon; CT + MRI).
# Lives at the repo ROOT so Grand Challenge's GitHub build finds it.
# Local build:  docker build -t doserad-photon .
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime

WORKDIR /opt/app

# Python deps first (better layer caching). torch/CUDA come from the base image.
#
# pydosert is VENDORED in this repo (pydosert/, see pydosert/VENDORED.md), not
# installed from GitHub. The corrector predicts the residual of ONE baseline
# dose, and the engine calibration it ships against was fitted to that exact
# PyDoseRT -- a rebuild that picked up a moved branch would change the physics
# under a corrector trained on the old one, with no error anywhere. Pinning a
# commit fixed that for the image but still depended on the upstream repo
# staying reachable; vendoring makes it part of this repo's history.
COPY submission/requirements.txt /opt/app/submission/requirements.txt
RUN pip install --no-cache-dir -r /opt/app/submission/requirements.txt

# Only the repo modules the inference actually imports, plus the submission
# package (server, inference, baked-in weights).
#
# features_v2/materials_v2 are NOT optional: the shipped corrector takes the
# 13-channel v2 BEV feature stack, and engines.build_v2_features imports them
# on the corrector forward. Leaving them out builds an image that passes every
# sanity check and then dies with ModuleNotFoundError on the first control
# point of a scored run.
COPY configs.py engines.py loaders.py sct.py features_v2.py materials_v2.py terma_scaling.py fixed_priors.py lateral_scatter.py /opt/app/
COPY pydosert /opt/app/pydosert
# Corrector weights and the two frozen priors it was built on.
COPY models /opt/app/models
COPY submission /opt/app/submission
# MR -> synthetic CT bundle (TorchScript, 390 MB). Only the MRI algorithm uses
# it, but one image serves both arms.
#
# FETCHED AT BUILD TIME, not COPYd: the weights are too large for git and live
# on Hugging Face, so a fresh clone does not have them. Downloading here rather
# than at inference is deliberate -- the scored container has no network, and a
# 390 MB download inside /invoke would blow the 181 s runtime limit even if it
# did. The build fails loudly if the file is short, so a truncated layer cannot
# reach a submission.
ARG SCT_BUNDLE_URL=https://huggingface.co/zimmeryWo/MRI-sCT_converter-DoseRAD2026/resolve/main/mrtoct_bundle
RUN mkdir -p /opt/app/mrtoct_bundle \
 && python -c "import urllib.request as u, os, sys; \
base='${SCT_BUNDLE_URL}'; \
[u.urlretrieve(base+'/'+n, '/opt/app/mrtoct_bundle/'+n) for n in ('metadata.json','model.pt')]; \
sz=os.path.getsize('/opt/app/mrtoct_bundle/model.pt'); \
sys.exit(0) if sz==408644663 else sys.exit('sCT model.pt is %d bytes, expected 408644663' % sz)"

# Serving configuration. Every value here is measured over the full validation
# protocol (6 patients x 3 beams x 180 control points, both modalities), not
# assumed.
#
# GC_MIN_CUTOFF is intentionally NOT set: the dose cutoff is read per control
# point from the challenge's own metadata ("output_info"/"minimum_cutoff"), and
# setting it here would override that for every control point.
#
# MODEL_NAME is the canonical baked filename -- what bake_model.py writes and
# what do_save.sh's in-image check loads. Re-baking to it is the whole update
# procedure; no Dockerfile change is needed to swap model or epoch. The
# in-image check prints experiment_name and epoch, so which weights are inside
# is visible at build time rather than inferred from a filename.
#
# GC_AMP=fp16 autocasts the CORRECTOR forward; the physics stays fp32. Worth
# 1.50x on the serving path (181.6 -> 120.7 ms per dose map) for 0.01 gamma,
# with beam MAE, IDD and stratified plan MAE unchanged to five decimals. fp16
# rather than bf16: measured both faster and more accurate here (10 mantissa
# bits against 7), and inference does not need bf16's exponent range.
#
# DOSERAD_SCT_AMP=fp16 autocasts the sCT network (MR arm only; the CT arm never
# builds one). The sliding-window accumulator and Gaussian blend stay float32 --
# ~150 patches summed in half precision is where an sCT would actually lose HU,
# not in any single forward. 22.0 -> 8.1 s per image for 0.17 HU of body
# deviation, against a model whose own error against real CT is 27.6-40.7 HU.
#
# The ENGINE is not half precision. Its cost is grid_sample, cumsum and
# elementwise work, none of which autocast converts, so peak memory does not
# move and the only effect is cast overhead on the few ops that do convert.
#
# GC_BATCH_CHUNK stacks control points into one corrector forward. Safe because
# the corrector uses no normalisation, so a control point carries no
# cross-sample statistics and stacking cannot leak between them -- this would
# NOT hold under group or instance norm. Peak VRAM is roughly linear in the
# chunk and in the BEV box, and the box is set by patient anatomy: the largest
# of the six validation patients is 9.9M voxels, 1.30x the median, and it is an
# abdomen case rather than a thorax one. 4 fits a 24 GB card with margin; 8 and
# above do not. An over-large value raises OutOfMemoryError, which
# predict_beam_batch catches, halves, and keeps reduced for the remaining beam
# groups -- so a bad value costs time rather than the run.
#
# GC_CROP_MARGIN_MM sets the safety margin on the geometric H-axis crop. The
# official IDD metric sums the transverse plane along the same axis this crop
# cuts, so every discarded slice is a slice of the scored curve. Measured on
# the shipped checkpoint over the full protocol, changing nothing else:
#
#            gamma   stratified   beam MAE      IDD
#     30 mm  96.64      0.00292    0.00885   0.01052
#     60 mm  98.66      0.00263    0.00875   0.00553
#    120 mm  98.33      0.00295    0.00878   0.00288
#
# 120 mm is worse on gamma and stratified MAE -- the corrector extrapolates far
# outside its training window there -- so 60 is the operating point, at ~1.47x
# the H span of 30.
ENV GC_CROP_MARGIN_MM=60 \
    GC_AMP=fp16 \
    DOSERAD_SCT_AMP=fp16 \
    GC_BATCH_CHUNK=4 \
    GC_TIMING=0 \
    MODEL_NAME=photon_corrector.pt \
    MODEL_NAME_CT=photon_corrector.pt \
    MODEL_NAME_MR=photon_corrector.pt \
    INPUT_PATH=/input \
    OUTPUT_PATH=/output \
    MODEL_DIR=/opt/app/models \
    DOSERAD_SCT_BUNDLE=/opt/app/mrtoct_bundle \
    PORT=4743 \
    PYTHONUNBUFFERED=1
EXPOSE 4743

# Grand Challenge rejects containers that run as root. Create an unprivileged
# user and switch to it (all root-only steps above are already done).
RUN groupadd -r user && useradd -m --no-log-init -r -u 1000 -g user user
USER user

# Required for Grand Challenge to recognise the HTTP invoke interface.
LABEL org.grand-challenge.api-method="invoke"

# Grand Challenge deduplicates uploads by image SHA256, so the SAME tarball
# cannot be uploaded to two Algorithms ("This container image has already been
# uploaded"). The photon-ct and photon-mri Algorithms need distinct images even
# though the code is identical and auto-detects the modality at invoke time.
# BUILD_ID makes the final layer differ so the two uploads hash differently.
# It changes nothing at runtime.
ARG BUILD_ID=dev
LABEL org.doserad.build-id="${BUILD_ID}"

CMD ["python", "-u", "/opt/app/submission/serve.py"]
