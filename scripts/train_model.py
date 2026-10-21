"""Training script for FashionVision CNN."""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path so imports work when run as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import tensorflow as tf

from src.data import load_fashion_mnist, preprocess, CLASS_NAMES
from src.model import build_cnn
from src.reporting import save_json


def parse_args():
    p = argparse.ArgumentParser(description="Train FashionVision CNN")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=str, default=".")
    p.add_argument("--patience", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()

    # Reproducibility: fix all random sources
    tf.keras.utils.set_random_seed(args.seed)

    output_dir = Path(args.output_dir)
    models_dir = output_dir / "models"
    results_dir = output_dir / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # GPU check — training still works on CPU but will be slow
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"GPU(s) detected: {[g.name for g in gpus]}")
    else:
        print("WARNING: No GPU found. Training on CPU — this will be slow.")

    print("\nLoading data ...")
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_fashion_mnist(seed=args.seed)

    # preprocess converts uint8 -> float32 in [0,1] with shape (N,28,28,1)
    # test set is NOT used here — it is reserved for evaluate_model.py
    x_train_p = preprocess(x_train)
    x_val_p = preprocess(x_val)

    print(f"  train: {x_train_p.shape}  val: {x_val_p.shape}")

    model = build_cnn()
    model.summary()

    best_model_path = str(models_dir / "best_model.keras")
    callbacks = [
        # Stop early if val_loss stops improving; restore the best weights afterward
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=args.patience,
            restore_best_weights=True,
        ),
        # Save the checkpoint with the lowest val_loss seen so far
        tf.keras.callbacks.ModelCheckpoint(
            filepath=best_model_path,
            monitor="val_loss",
            save_best_only=True,
        ),
    ]

    start_time = time.time()
    history = model.fit(
        x_train_p, y_train,
        epochs=args.epochs,
        batch_size=args.batch_size,
        validation_data=(x_val_p, y_val),
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = time.time() - start_time

    epochs_run = len(history.history["val_loss"])
    best_epoch = int(np.argmin(history.history["val_loss"])) + 1  # 1-indexed

    # Save final model (best weights restored by EarlyStopping)
    final_model_path = models_dir / "fashion_cnn.keras"
    model.save(str(final_model_path))

    # Save class names
    save_json(CLASS_NAMES, models_dir / "class_names.json")

    # Save training history as plain Python lists
    history_data = {k: [float(v) for v in vals]
                    for k, vals in history.history.items()}
    save_json(history_data, results_dir / "history.json")

    # Build metadata from real runtime values only
    metadata = {
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs_requested": args.epochs,
        "epochs_run": epochs_run,
        "best_epoch": best_epoch,
        "input_shape": [28, 28, 1],
        "num_classes": 10,
        "train_size": int(x_train_p.shape[0]),
        "val_size": int(x_val_p.shape[0]),
        "preprocessing": "float32, divided by 255.0, shape (N,28,28,1)",
        "tensorflow_version": tf.__version__,
        "python_version": sys.version,
        "gpu_devices": [g.name for g in gpus],
        "training_seconds": round(elapsed, 1),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }
    save_json(metadata, models_dir / "metadata.json")

    print("\n=== Training complete ===")
    print(f"  Epochs run      : {epochs_run} / {args.epochs}")
    print(f"  Best epoch      : {best_epoch}")
    print(f"  Training time   : {elapsed:.1f}s")
    print(f"\nArtifacts saved:")
    for p in [best_model_path, final_model_path,
               models_dir / "class_names.json",
               models_dir / "metadata.json",
               results_dir / "history.json"]:
        print(f"  {p}")


if __name__ == "__main__":
    main()
