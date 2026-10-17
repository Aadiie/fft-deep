#Train a CNN on the cleaned train split, validate on val, save model + history
from __future__ import annotations

import argparse
import json

import keras

from fftdeep.data.arrays import get_arrays
from fftdeep.models.cnn import build_cnn, compile_model
from fftdeep.paths import PROJECT_ROOT

VARIANTS = {
    "baseline": dict(convs_per_block=1, batch_norm=False),  
    "deeper": dict(convs_per_block=2, batch_norm=True),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, default="baseline")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    keras.utils.set_random_seed(args.seed)
    Xtr, ytr = get_arrays("train")
    Xva, yva = get_arrays("val")
    Xtr, Xva = Xtr.astype("float32"), Xva.astype("float32")
    ytr, yva = ytr.astype("float32"), yva.astype("float32")

    model = compile_model(build_cnn(name=f"cnn_{args.variant}", **VARIANTS[args.variant]))
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)
    ]
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xva, yva),
        epochs=args.epochs,
        batch_size=args.batch_size,
        callbacks=callbacks,
        verbose=2,
    )

    models_dir = PROJECT_ROOT / "models"
    reports_dir = PROJECT_ROOT / "reports"
    models_dir.mkdir(exist_ok=True)
    reports_dir.mkdir(exist_ok=True)
    model.save(models_dir / f"cnn_{args.variant}.keras")

    hist = {k: [float(v) for v in vals] for k, vals in history.history.items()}
    (reports_dir / f"history_{args.variant}.json").write_text(json.dumps(hist, indent=2))

    best = min(range(len(hist["val_loss"])), key=hist["val_loss"].__getitem__)
    print(f"best epoch: {best + 1}  val_loss={hist['val_loss'][best]:.4f}  "
          f"val_acc={hist['val_accuracy'][best]:.4f}  val_auc={hist['val_auc'][best]:.4f}")


if __name__ == "__main__":
    main()