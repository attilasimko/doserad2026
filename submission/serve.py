"""Grand Challenge invoke-API server for the DoseRAD2026 photon submission.

Matches the official DoseRAD2026 example-submission contract: a FastAPI app
served by uvicorn on host 0.0.0.0 port 4743, with

    GET  /health  -> 200 once the models are loaded (503 while loading),
    POST /invoke  -> 201 after the output is written.

Models load in the lifespan startup so they're ready before the first invoke.
One image serves both CT and MRI algorithms — modality is auto-detected per
invoke from the input image directory names — and both share ONE correction
model: the MR arm converts the MR to a synthetic CT first (sct.py), after
which the two arms are the same pipeline. The sCT network is loaded at startup
too, so /invoke never pays for it.
"""
import os
import sys
from contextlib import asynccontextmanager

import torch
import uvicorn
from fastapi import FastAPI, Response, status

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gc_inference as gci  # noqa: E402
import sct  # noqa: E402

INPUT_PATH = os.environ.get("INPUT_PATH", "/input")
OUTPUT_PATH = os.environ.get("OUTPUT_PATH", "/output")
MODEL_DIR = os.environ.get(
    "MODEL_DIR",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models"))
# ONE corrector for both arms (see module docstring). Its architecture comes
# from the `model_config` dict inside the checkpoint, so a new sweep winner is
# dropped in with no code change — but it must be stamped by
# submission/bake_model.py, not copied straight out of out/.
MODEL_NAME = os.environ.get("MODEL_NAME", "photon_corrector.pt")
# PER-ARM MODELS. CT and MR are SEPARATELY RANKED tracks, so there is no reason
# to serve one corrector for both: any MR-specific adaptation otherwise has to
# be paid for out of CT accuracy. Both default to MODEL_NAME, so a single-model
# image behaves exactly as before; set MODEL_NAME_MR to a second baked
# checkpoint (e.g. one fine-tuned on synthetic CT) to split them.
MODEL_NAME_CT = os.environ.get("MODEL_NAME_CT", MODEL_NAME)
MODEL_NAME_MR = os.environ.get("MODEL_NAME_MR", MODEL_NAME)
PORT = int(os.environ.get("PORT", "4743"))

_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODELS = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models before serving so /invoke never waits on model init.
    # A gc_inference.Corrector: weights + feature config + physics baseline.
    # All three travel together because none of them is recoverable from the
    # others, and a mismatch produces a wrong dose rather than an error.
    #
    # This runs in lifespan, BEFORE /health goes green, so Grand Challenge has
    # already waited for it and it is outside the /invoke the runtime model
    # fits. The same is true of the sCT bundle warm-up below.
    _cache = {}
    for arm, name in (("ct", MODEL_NAME_CT), ("mr", MODEL_NAME_MR)):
        if name not in _cache:
            _cache[name] = gci.build_model(os.path.join(MODEL_DIR, name), _device)
        MODELS[arm] = _cache[name]
    # Warm the MR->sCT bundle now rather than inside the first MRI invoke.
    sct.get_predictor(device=_device)
    yield
    MODELS.clear()


app = FastAPI(lifespan=lifespan)


@app.get("/health")
async def health():
    if "ct" in MODELS and "mr" in MODELS:
        return Response(status_code=status.HTTP_200_OK)
    return Response(status_code=status.HTTP_503_SERVICE_UNAVAILABLE)


@app.post("/invoke")
def invoke():
    # Deliberately sync (`def`, not `async def`). run_invoke is minutes of
    # blocking CPU+CUDA work; on an async handler it would occupy the uvicorn
    # event loop for the whole invoke and /health could not answer while we
    # compute. A sync handler is dispatched to the threadpool instead.
    gci.run_invoke(INPUT_PATH, OUTPUT_PATH, MODELS, _device)
    return Response(status_code=status.HTTP_201_CREATED)


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=PORT)
