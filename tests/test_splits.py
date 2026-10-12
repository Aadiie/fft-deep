import numpy as np
import pandas as pd
import pytest
from PIL import Image

from fftdeep.data.splits import build_inventory, check_no_leakage, clean_and_split


def _save(path, seed):
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(arr).save(path)


def make_dataset(root):
    for i in range(20):
        _save(root / "train" / "REAL" / f"r{i}.jpg", i)
        _save(root / "train" / "FAKE" / f"f{i}.jpg", 100 + i)
    _save(root / "train" / "FAKE" / "f_dup.jpg", 100)    # duplicate inside train
    for i in range(5):
        _save(root / "test" / "REAL" / f"r{i}.jpg", 300 + i)
        _save(root / "test" / "FAKE" / f"f{i}.jpg", 200 + i)
    _save(root / "test" / "FAKE" / "f_leak.jpg", 105)    # also exists in train


def test_no_overlap_between_subsets(tmp_path):
    make_dataset(tmp_path)
    manifest = clean_and_split(build_inventory(tmp_path))
    check_no_leakage(manifest)  # raises if there is any leakage


def test_leak_and_duplicate_removed_and_test_untouched(tmp_path):
    make_dataset(tmp_path)
    manifest = clean_and_split(build_inventory(tmp_path))
    non_test = manifest[manifest.subset != "test"]
    assert non_test.md5.is_unique
    assert len(non_test) == 39                          # 20 REAL + 20 FAKE - leaked - duplicate
    assert (manifest.subset == "test").sum() == 11      # official test kept as is


def test_check_detects_leakage():
    bad = pd.DataFrame({"md5": ["a", "a"], "subset": ["train", "val"]})
    with pytest.raises(ValueError):
        check_no_leakage(bad)