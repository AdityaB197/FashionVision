"""Evaluation script for FashionVision v2.

Evaluates the trained MobileNetV2 model exclusively on the test split,
computes metrics, and saves evaluation reports and visualization plots.
"""

import argparse
import json
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib
matplotlib.use("Agg")  # Non-interactive plotting backend
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report
import tensorflow as tf

from src.reporting import compute_metrics, plot_confusion_matrix, save_json


def plot_v2_training_curves(history_data, save_path):
    """Plot accuracy and loss with a vertical line marking the start of fine-tuning."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    total_epochs = len(history_data.get("accuracy", []))
    epochs = range(1, total_epochs + 1)
    finetune_epoch = history_data.get("finetune_start_epoch", 0)

    # 1. Accuracy
    ax1.plot(epochs, history_data.get("accuracy", []), label="train")
    ax1.plot(epochs, history_data.get("val_accuracy", []), label="val")
    ax1.set_title("Accuracy")
    ax1.set_xlabel("Epoch")
    if 0 < finetune_epoch < total_epochs:
        ax1.axvline(x=finetune_epoch + 0.5, color="red", linestyle="--", label="Fine-tuning starts")
    ax1.legend()

    # 2. Loss
    ax2.plot(epochs, history_data.get("loss", []), label="train")
    ax2.plot(epochs, history_data.get("val_loss", []), label="val")
    ax2.set_title("Loss")
    ax2.set_xlabel("Epoch")
    if 0 < finetune_epoch < total_epochs:
        ax2.axvline(x=finetune_epoch + 0.5, color="red", linestyle="--", label="Fine-tuning starts")
    ax2.legend()

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def plot_misclassified_real_photos(images, y_true, y_pred, class_names, save_path, n=16):
    """Plot a grid of misclassified real clothing photos with true and predicted labels."""
    wrong = np.where(y_true != y_pred)[0]
    wrong = wrong[:n]

    if len(wrong) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "100% Test Accuracy - No Misclassified Samples", ha="center", va="center")
        ax.axis("off")
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path)
        plt.close(fig)
        return

    cols = min(4, len(wrong))
    rows = (len(wrong) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = np.array(axes).reshape(-1)

    for ax, idx in zip(axes, wrong):
        img = images[idx]
        if img.max() <= 1.0:
            ax.imshow(img)
        else:
            ax.imshow(img.astype(np.uint8))
        ax.set_title(
            f"true: {class_names[y_true[idx]]}\npred: {class_names[y_pred[idx]]}",
            fontsize=8,
        )
        ax.axis("off")

    for ax in axes[len(wrong):]:
        ax.axis("off")

    fig.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(save_path)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Evaluate FashionVision v2 model on test split.")
    parser.add_argument("--data-dir", default="data/clothing_prepared", help="Prepared dataset directory.")
    parser.add_argument("--output-dir", default=".", help="Root output directory.")
    args = parser.parse_args()

    out_root = Path(args.output_dir)
    models_v2_dir = out_root / "models_v2"
    results_v2_dir = out_root / "results_v2"
    results_v2_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_v2_dir / "real_model.keras"
    class_names_path = models_v2_dir / "class_names.json"

    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found at '{model_path.resolve()}'. Train the model first.")
    if not class_names_path.is_file():
        raise FileNotFoundError(f"Class names file not found at '{class_names_path.resolve()}'.")

    # Load artifacts
    print(f"Loading model from '{model_path.resolve()}'...")
    model = tf.keras.models.load_model(str(model_path))

    with open(class_names_path, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    # Load test split (strictly shuffle=False)
    test_dir = Path(args.data_dir) / "test"
    print(f"Loading TEST set from '{test_dir.resolve()}'...")
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=32,
        image_size=(224, 224),
        shuffle=False,
    )

    # Gather images and ground truth labels
    test_images_list = []
    test_labels_list = []
    for imgs, lbls in test_ds:
        test_images_list.append(imgs.numpy())
        test_labels_list.append(lbls.numpy())

    x_test = np.concatenate(test_images_list, axis=0)
    y_true = np.concatenate(test_labels_list, axis=0)

    # Predict
    print("\nRunning model predictions on test split...")
    probs = model.predict(test_ds, verbose=1)
    y_pred = np.argmax(probs, axis=1)

    # 1. Compute metrics
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics_path = results_v2_dir / "metrics.json"
    save_json(metrics, metrics_path)

    # 2. Classification report
    clf_report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )
    clf_report_path = results_v2_dir / "classification_report.json"
    save_json(clf_report, clf_report_path)

    # 3. Confusion matrix plot
    cm_path = results_v2_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_true, y_pred, class_names, str(cm_path))

    # 4. Misclassified examples plot
    misclassified_path = results_v2_dir / "misclassified_examples.png"
    plot_misclassified_real_photos(x_test, y_true, y_pred, class_names, str(misclassified_path))

    # 5. Training curves plot (if history.json exists)
    history_path = results_v2_dir / "history.json"
    curves_path = results_v2_dir / "training_curves.png"
    if history_path.is_file():
        with open(history_path, "r", encoding="utf-8") as f:
            history_data = json.load(f)
        plot_v2_training_curves(history_data, str(curves_path))

    # Print summary
    print("\n" + "=" * 50)
    print(f"Test Accuracy: {metrics['accuracy']:.4f}")
    print(f"Macro F1:      {metrics['macro_f1']:.4f}")
    print(f"Weighted F1:   {metrics['weighted_f1']:.4f}")
    print("=" * 50)
    print("\nFiles written to results_v2/:")
    print(f"  - {metrics_path.name}")
    print(f"  - {clf_report_path.name}")
    print(f"  - {cm_path.name}")
    print(f"  - {misclassified_path.name}")
    if history_path.is_file():
        print(f"  - {curves_path.name}")

    print("\nIMPORTANT NOTE:")
    print("These evaluation numbers reflect only this dataset's test split, not all real-world clothing photos.")


if __name__ == "__main__":
    main()
