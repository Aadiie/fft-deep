"""FastAPI service: image -> CNN score (main answer) + FFT-only score (second opinion)."""
from __future__ import annotations

import io
from contextlib import asynccontextmanager

import keras
import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image
from pydantic import BaseModel

from fftdeep.frequency.spectrum import log_spectrum, standardize
from fftdeep.paths import PROJECT_ROOT

MODELS_DIR = PROJECT_ROOT / "models"
CNN_PATH = MODELS_DIR / "cnn_deeper_lrdecay.keras"
FREQ_PATH = MODELS_DIR / "cnn_freq.keras"
STATS_PATH = MODELS_DIR / "spec_stats.npz"
THRESHOLD = 0.5
MAX_BYTES = 5 * 1024 * 1024
NOTE = (
    "Trained only on 32x32 CIFAKE images (CIFAR-10 vs Stable Diffusion). Your image was "
    "resized to 32x32. Scores are not calibrated probabilities and are unreliable for "
    "other kinds of images. The frequency model is weaker (about 84% accurate on test)."
)


class ModelScore(BaseModel):
    label: str
    score_fake: float


class Prediction(BaseModel):
    label: str
    score_fake: float
    threshold: float
    frequency_model: ModelScore
    original_size: list[int]
    note: str


def label_from_score(score: float) -> str:
    return "FAKE" if score >= THRESHOLD else "REAL"


def preprocess_image(data: bytes):
    """bytes -> ((1, 32, 32, 3) uint8 array, original (width, height))."""
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except Exception as exc:
        raise ValueError("could not read the file as an image") from exc
    original_size = img.size
    img = img.convert("RGB").resize((32, 32), Image.Resampling.BICUBIC)
    return np.asarray(img, dtype="uint8")[None], original_size


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.cnn = keras.models.load_model(CNN_PATH, compile=False)
    app.state.freq = keras.models.load_model(FREQ_PATH, compile=False)
    stats = np.load(STATS_PATH)
    app.state.mu, app.state.sd = stats["mu"], stats["sd"]
    yield


app = FastAPI(title="FFT-Deep: real vs AI-generated image detector", lifespan=lifespan)


@app.get("/health")
def health(request: Request):
    return {"status": "ok", "models_loaded": hasattr(request.app.state, "cnn")}

from fastapi.responses import RedirectResponse


@app.get("/", include_in_schema=False)
def root():
    return RedirectResponse(url="/docs")

@app.post("/predict", response_model=Prediction)
def predict(request: Request, file: UploadFile = File(...)):
    data = file.file.read()
    if len(data) > MAX_BYTES:
        raise HTTPException(status_code=413, detail="file too large (max 5 MB)")
    try:
        arr, original_size = preprocess_image(data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    state = request.app.state
    cnn_score = float(state.cnn.predict(arr.astype("float32"), verbose=0)[0, 0])
    spec = standardize(log_spectrum(arr), state.mu, state.sd)
    freq_score = float(state.freq.predict({"spectrum": spec}, verbose=0)[0, 0])

    return Prediction(
        label=label_from_score(cnn_score),
        score_fake=cnn_score,
        threshold=THRESHOLD,
        frequency_model=ModelScore(label=label_from_score(freq_score), score_fake=freq_score),
        original_size=[original_size[0], original_size[1]],
        note=NOTE,
    )