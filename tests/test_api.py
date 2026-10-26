import io

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from fftdeep.api.main import CNN_PATH, FREQ_PATH, STATS_PATH, app, preprocess_image


def _png_bytes(size=(64, 48)):
    arr = np.random.default_rng(0).integers(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_preprocess_resizes_to_32x32():
    arr, size = preprocess_image(_png_bytes((64, 48)))
    assert arr.shape == (1, 32, 32, 3) and arr.dtype == np.uint8
    assert size == (64, 48)


def test_preprocess_rejects_garbage():
    with pytest.raises(ValueError):
        preprocess_image(b"not an image")


@pytest.mark.skipif(
    not (CNN_PATH.exists() and FREQ_PATH.exists() and STATS_PATH.exists()),
    reason="model files not present",
)
def test_predict_endpoint():
    with TestClient(app) as client:
        assert client.get("/health").json()["models_loaded"] is True
        r = client.post("/predict", files={"file": ("x.png", _png_bytes(), "image/png")})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in {"REAL", "FAKE"}
        assert 0.0 <= body["score_fake"] <= 1.0
        bad = client.post("/predict", files={"file": ("x.txt", b"hello", "text/plain")})
        assert bad.status_code == 400