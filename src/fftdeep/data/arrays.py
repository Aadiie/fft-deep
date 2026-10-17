#Loading manifest images into NumPy arrays and cache them as .npy files.
from __future__ import annotations

from pathlib import Path
from fftdeep.paths import PROCESSED_DIR, RAW_DIR
import numpy as np
import pandas as pd
from PIL import Image


def load_subset(manifest: pd.DataFrame, subset: str, root: Path):
    #Return (X, y): X is (N, 32, 32, 3) uint8, y is (N,) with 0=REAL, 1=FAKE.
    rows = manifest[manifest.subset == subset]
    X = np.empty((len(rows), 32, 32, 3), dtype=np.uint8)
    for i, rel in enumerate(rows.path):
        with Image.open(root / rel) as im:
            X[i] = np.asarray(im.convert("RGB"))   
    return X, rows.label.to_numpy()


def get_arrays(subset: str, root: Path = RAW_DIR, processed: Path = PROCESSED_DIR):
    
    x_path, y_path = processed / f"{subset}_x.npy", processed / f"{subset}_y.npy"
    if x_path.exists() and y_path.exists():
        return np.load(x_path), np.load(y_path)
    manifest = pd.read_csv(processed / "manifest.csv")
    X, y = load_subset(manifest, subset, root)
    np.save(x_path, X)
    np.save(y_path, y)
    return X, y