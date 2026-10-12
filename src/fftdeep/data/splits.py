"""Building a deduplicated train/val/test manifest from the CIFAKE folder """
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

LABELS = {"REAL": 0, "FAKE": 1}  # sigmoid output = P(FAKE)


def pixel_hash(path: Path) -> str:
    """creating md5 hash"""
    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    return hashlib.md5(arr.tobytes()).hexdigest()


def build_inventory(root: Path) -> pd.DataFrame:
    """One row per image: path (relative to root), split, label, md5."""
    rows = []
    for split in ["train", "test"]:
        for cls, label in LABELS.items():
            for p in sorted((root / split / cls).glob("*.jpg")):
                rows.append({
                    "path": str(p.relative_to(root)),
                    "split": split,
                    "label": label,
                    "md5": pixel_hash(p),
                })
    return pd.DataFrame(rows)


def clean_and_split(inventory: pd.DataFrame, val_fraction: float = 0.20,
                    seed: int = 42) -> pd.DataFrame:
    train = inventory[inventory.split == "train"]
    test = inventory[inventory.split == "test"]

    # 1) dropping train images whose pixels also appear in the official test set
    train = train[~train.md5.isin(set(test.md5))]
    # 2) keeping one image per fingerprint inside train
    train = train.drop_duplicates(subset="md5", keep="first")
    # 3) taking out validation out of the cleaned train set, preserving the label ratio
    train, val = train_test_split(
        train, test_size=val_fraction, stratify=train.label, random_state=seed
    )
    return pd.concat([
        train.assign(subset="train"),
        val.assign(subset="val"),
        test.assign(subset="test"),
    ], ignore_index=True)


def check_no_leakage(manifest: pd.DataFrame) -> None:
    """Raise if any pixel fingerprint appears in two different subsets."""
    hashes = {s: set(g.md5) for s, g in manifest.groupby("subset")}
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        overlap = hashes[a] & hashes[b]
        if overlap:
            raise ValueError(f"{len(overlap)} images appear in both {a} and {b}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("data/processed/manifest.csv"))
    args = ap.parse_args()

    manifest = clean_and_split(build_inventory(args.root))
    check_no_leakage(manifest)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)
    print(manifest.groupby(["subset", "label"]).size())