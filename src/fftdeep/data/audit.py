"""Auditig  image dataset laid out as <root>/<split>/<class>/<image>."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def audit(root: Path) -> dict:
    stats = defaultdict(lambda: {
        "count": 0, "sizes": Counter(), "modes": Counter(),
        "formats": Counter(), "extensions": Counter(),
        "file_bytes": [], "channel_sum": np.zeros(3), "pixel_count": 0,
    })
    corrupted, non_images, unexpected = [], [], []
    hashes = defaultdict(list)  # pixel-hash -> [(split, class, relpath)]

    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(root)
        if p.suffix.lower() not in IMG_EXT:
            non_images.append(str(rel))
            continue
        if len(rel.parts) < 3:
            unexpected.append(str(rel))
            continue
        split, cls = rel.parts[0], rel.parts[1]
        key = f"{split}/{cls}"
        try:
            with Image.open(p) as im:
                im.load()  # force full decode so truncated files fail here
                fmt, mode, size = im.format, im.mode, im.size
                arr = np.asarray(im.convert("RGB"))
        except Exception as e:  
            corrupted.append((str(rel), repr(e)))
            continue
        s = stats[key]
        s["count"] += 1
        s["sizes"][f"{size[0]}x{size[1]}"] += 1
        s["modes"][mode] += 1
        s["formats"][fmt] += 1
        s["extensions"][p.suffix.lower()] += 1
        s["file_bytes"].append(p.stat().st_size)
        s["channel_sum"] += arr.reshape(-1, 3).sum(axis=0) #
        s["pixel_count"] += arr.shape[0] * arr.shape[1]
        hashes[hashlib.md5(arr.tobytes()).hexdigest()].append((split, cls, str(rel)))

    dup_groups = [g for g in hashes.values() if len(g) > 1]
    cross_split = [g for g in dup_groups if len({x[0] for x in g}) > 1]
    cross_class = [g for g in dup_groups if len({x[1] for x in g}) > 1]

    report = {"groups": {}, "n_corrupted": len(corrupted), "corrupted": corrupted[:20],
              "n_non_images": len(non_images), "non_image_examples": non_images[:10],
              "n_unexpected_layout": len(unexpected), "unexpected_examples": unexpected[:10],
              "n_duplicate_groups": len(dup_groups),
              "n_dup_groups_across_splits": len(cross_split),
              "n_dup_groups_across_classes": len(cross_class),
              "dup_examples": dup_groups[:5]}
    for key, s in sorted(stats.items()):
        fb = np.array(s["file_bytes"])
        report["groups"][key] = {
            "count": s["count"], "sizes": dict(s["sizes"]), "modes": dict(s["modes"]),
            "formats": dict(s["formats"]), "extensions": dict(s["extensions"]),
            "file_bytes": {"mean": float(fb.mean()), "std": float(fb.std()),
                           "min": int(fb.min()), "max": int(fb.max())},
            "mean_rgb": (s["channel_sum"] / s["pixel_count"]).round(2).tolist(),
        }
    return report


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root", type=Path)
    ap.add_argument("--out", type=Path, default=Path("reports/data_audit.json"))
    args = ap.parse_args()
    rep = audit(args.root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))