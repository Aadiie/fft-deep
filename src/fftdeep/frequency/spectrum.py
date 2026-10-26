#converting original images into FFT frequency spectra, standardizes those spectra
from __future__ import annotations

import numpy as np

from fftdeep.data.arrays import get_arrays
from fftdeep.paths import PROCESSED_DIR

GRAY = np.array([0.299, 0.587, 0.114], dtype="float64")


def log_spectrum(images: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """(N, 32, 32, 3) uint8 -> (N, 32, 32) float32 log-magnitude, DC removed, centred."""
    out = np.empty(images.shape[:3], dtype="float32")
    for i in range(0, len(images), chunk):
        gray = images[i:i + chunk].astype("float64") @ GRAY          # (n, 32, 32) size
        gray -= gray.mean(axis=(1, 2), keepdims=True)               # dropping mean brightness
        F = np.fft.fftshift(np.fft.fft2(gray, axes=(1, 2)), axes=(1, 2))
        out[i:i + chunk] = np.log1p(np.abs(F))
    return out


def fit_stats(spec_train: np.ndarray):
    #mean and sd for train data 
    return spec_train.mean(axis=0), np.maximum(spec_train.std(axis=0), 1e-3)


def standardize(spec: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> np.ndarray:
    return ((spec - mu) / sd)[..., None].astype("float32")          # (N, 32, 32, 1)


def _stats():
    path = PROCESSED_DIR / "spec_stats.npz"
    if not path.exists():
        Xtr, _ = get_arrays("train")
        mu, sd = fit_stats(log_spectrum(Xtr))
        np.savez(path, mu=mu, sd=sd)
    d = np.load(path)
    return d["mu"], d["sd"]


def get_spectra(subset: str) -> np.ndarray:
    
    path = PROCESSED_DIR / f"spec_{subset}.npy"
    if path.exists():
        return np.load(path)
    mu, sd = _stats()
    X, _ = get_arrays(subset)
    spec = standardize(log_spectrum(X), mu, sd)
    np.save(path, spec)
    return spec