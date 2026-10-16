#Binary-classification metrics and plots. Positive class = Fake label =1 
# Negative Class = Real, label = 0 
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)


def binary_metrics(y_true, probs, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true).astype(int)
    probs = np.asarray(probs)
    pred = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "specificity": float(tn / (tn + fp)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probs)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def youden_threshold(y_true, probs) -> float:
    #Threshold maximising TPR - FPR on the data given (use validation data only)
    fpr, tpr, thresholds = roc_curve(y_true, probs)
    return float(thresholds[np.argmax(tpr - fpr)])


def plot_confusion(m: dict, ax, title: str = "") -> None:
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    ax.imshow(cm, cmap="Blues")
    for (i, j), v in np.ndenumerate(cm):
        ax.text(j, i, f"{v:,}", ha="center", va="center",
                color="white" if v > cm.max() / 2 else "black")
    ax.set_xticks([0, 1], ["REAL", "FAKE"])
    ax.set_yticks([0, 1], ["REAL", "FAKE"])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)


def plot_roc(y_true, probs, label: str, ax) -> None:
    fpr, tpr, _ = roc_curve(y_true, probs)
    ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc_score(y_true, probs):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("false positive rate")
    ax.set_ylabel("true positive rate")
    ax.legend()


def plot_history(history: dict, title: str = ""):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    for ax, key in zip(axes, ["loss", "accuracy"]):
        ax.plot(history[key], label="train")
        ax.plot(history[f"val_{key}"], label="val")
        ax.set_xlabel("epoch")
        ax.set_title(f"{title} {key}")
        ax.legend()
    fig.tight_layout()
    return fig