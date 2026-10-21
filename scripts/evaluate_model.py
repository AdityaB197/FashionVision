"""Evaluation script for FashionVision CNN."""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from src.data import load_fashion_mnist, preprocess, CLASS_NAMES
from src.reporting import (
    compute_metrics,
    plot_confusion_matrix,
    plot_misclassified,
    plot_training_curves,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate FashionVision CNN on test set")
    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--model-name", type=str, default="fashion_cnn.keras")
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / args.model_name
    print(f"Loading model from {model_path} ...")
    model = tf.keras.models.load_model(str(model_path))

    print("Loading test set ...")
    (_, _), (_, _), (x_test, y_test) = load_fashion_mnist()
    x_test_p = preprocess(x_test)

    print(f"Predicting on {len(x_test)} test images ...")
    y_prob = model.predict(x_test_p, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)

    # Metrics
    metrics = compute_metrics(y_test, y_pred, CLASS_NAMES)
    metrics_path = results_dir / "metrics.json"
    save_json(metrics, metrics_path)

    # sklearn classification report as a dict
    from sklearn.metrics import classification_report
    report = classification_report(y_test, y_pred,
                                   target_names=CLASS_NAMES,
                                   output_dict=True,
                                   zero_division=0)
    report_path = results_dir / "classification_report.json"
    save_json(report, report_path)

    # Confusion matrix
    cm_path = results_dir / "confusion_matrix.png"
    plot_confusion_matrix(y_test, y_pred, CLASS_NAMES, cm_path)

    # Misclassified examples
    mis_path = results_dir / "misclassified_examples.png"
    plot_misclassified(x_test, y_test, y_pred, CLASS_NAMES, mis_path)

    # Training curves (loaded from history.json written during training)
    history_path = results_dir / "history.json"
    curves_path = results_dir / "training_curves.png"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)
        plot_training_curves(history, curves_path)
    else:
        print(f"WARNING: {history_path} not found — skipping training curves plot.")

    print(f"\n=== Evaluation complete ===")
    print(f"  Test accuracy: {metrics['accuracy']:.4f}")
    print(f"\nFiles written:")
    for p in [metrics_path, report_path, cm_path, mis_path, curves_path]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
