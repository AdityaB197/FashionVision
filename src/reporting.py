"""Reporting helpers: metrics, plots, and JSON saving."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # non-interactive backend, works without a display
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def compute_metrics(y_true, y_pred, class_names):
    acc = accuracy_score(y_true, y_pred)

    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted", zero_division=0
    )

    per_class_p, per_class_r, per_class_f, per_class_s = precision_recall_fscore_support(
        y_true, y_pred, average=None, zero_division=0
    )

    per_class = {}
    for i, name in enumerate(class_names):
        per_class[name] = {
            "precision": float(per_class_p[i]),
            "recall": float(per_class_r[i]),
            "f1": float(per_class_f[i]),
            "support": int(per_class_s[i]),
        }

    return {
        "accuracy": float(acc),
        "macro_precision": float(precision_macro),
        "macro_recall": float(recall_macro),
        "macro_f1": float(f1_macro),
        "weighted_precision": float(precision_weighted),
        "weighted_recall": float(recall_weighted),
        "weighted_f1": float(f1_weighted),
        "per_class": per_class,
    }


def plot_training_curves(history, save_path):
    """Two side-by-side subplots: accuracy and loss, train vs val."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(history["accuracy"], label="train")
    ax1.plot(history["val_accuracy"], label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(history["loss"], label="train")
    ax2.plot(history["val_loss"], label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # write counts in each cell
    thresh = cm.max() / 2
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_misclassified(x_raw, y_true, y_pred, class_names, save_path, n=16):
    """Grid of up to n wrongly predicted test images."""
    wrong = np.where(y_true != y_pred)[0]
    wrong = wrong[:n]

    cols = 4
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, wrong):
        img = x_raw[idx].squeeze()
        ax.imshow(img, cmap="gray")
        ax.set_title(
            f"true: {class_names[y_true[idx]]}\npred: {class_names[y_pred[idx]]}",
            fontsize=8,
        )
        ax.axis("off")

    # hide unused subplots
    for ax in axes[len(wrong):]:
        ax.axis("off")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)
