"""Train image-only / frequency-only / fused models with an identical recipe."""
from __future__ import annotations

import argparse
import json

import keras

from fftdeep.data.arrays import get_arrays
from fftdeep.frequency.spectrum import get_spectra
from fftdeep.models.cnn import compile_model
from fftdeep.models.fusion import build_model
from fftdeep.paths import PROJECT_ROOT


def make_inputs(mode: str, subset: str):
    X, y = get_arrays(subset)
    inputs = {}
    if mode in ("image", "both"):
        inputs["image"] = X.astype("float32")
    if mode in ("freq", "both"):
        inputs["spectrum"] = get_spectra(subset)
    return inputs, y.astype("float32")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["image", "freq", "both"], required=True)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    xtr, ytr = make_inputs(args.mode, "train")
    xva, yva = make_inputs(args.mode, "val")

    model = compile_model(build_model(args.mode))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-5),
    ]
    history = model.fit(
        xtr, ytr,
        validation_data=(xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    (PROJECT_ROOT / "models").mkdir(exist_ok=True)
    (PROJECT_ROOT / "reports").mkdir(exist_ok=True)
    model.save(PROJECT_ROOT / "models" / f"cnn_{args.mode}.keras")
    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (PROJECT_ROOT / "reports" / f"history_{args.mode}.json").write_text(json.dumps(hist, indent=2))

    best = min(range(len(hist["val_loss"])), key=hist["val_loss"].__getitem__)
    print(f"best epoch: {best + 1}  val_loss={hist['val_loss'][best]:.4f}  "
          f"val_acc={hist['val_accuracy'][best]:.4f}  val_auc={hist['val_auc'][best]:.4f}")


if __name__ == "__main__":
    main()
