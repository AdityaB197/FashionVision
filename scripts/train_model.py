"""Training script for FashionVision v2 (MobileNetV2 transfer learning).

Executes two-stage transfer learning:
  Stage 1: Frozen base model feature extractor, training classifier head.
  Stage 2: Fine-tuning top 30 layers of MobileNetV2 with low learning rate (1e-5).

The test split is strictly forbidden from being loaded or referenced in this script.
"""

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import tensorflow as tf

from src.data import compute_class_weights, load_datasets
from src.model import build_real_model, enable_fine_tuning


def main():
    parser = argparse.ArgumentParser(description="Train FashionVision v2 MobileNetV2 model.")
    parser.add_argument("--data-dir", default="data/clothing_prepared", help="Prepared dataset directory.")
    parser.add_argument("--output-dir", default=".", help="Root output directory for models and results.")
    parser.add_argument("--epochs-head", type=int, default=10, help="Epochs for Stage 1 (head only).")
    parser.add_argument("--epochs-finetune", type=int, default=10, help="Epochs for Stage 2 (fine-tuning).")
    parser.add_argument("--batch-size", type=int, default=32, help="Mini-batch size.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--patience", type=int, default=4, help="Early stopping patience.")
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test mode (~2 batches, 1 epoch per stage).")
    args = parser.parse_args()

    # 1. Deterministic seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)

    # 2. Hardware inspection and GPU optimization
    gpus = tf.config.list_physical_devices("GPU")
    gpu_names = [gpu.name for gpu in gpus]
    print(f"=== Hardware Environment ===")
    if gpus:
        print(f"GPUs detected: {gpu_names}")
        # Allow GPU memory to grow on demand instead of grabbing all VRAM upfront
        for gpu in gpus:
            try:
                tf.config.experimental.set_memory_growth(gpu, True)
            except RuntimeError:
                pass  # Memory growth must be set before GPUs are initialized
        # Mixed precision: use float16 on GPU for ~2x throughput on T4/A100
        tf.keras.mixed_precision.set_global_policy("mixed_float16")
        print("Mixed precision enabled (float16 compute, float32 storage)")
    else:
        print("WARNING: No GPU detected. Training will run on CPU, which will be significantly slower.")

    # 3. Load datasets — skip the test split during training (saves time, prevents leakage)
    data_path = Path(args.data_dir)
    print(f"\nLoading training and validation sets from '{data_path.resolve()}'...")
    train_ds, val_ds, _, class_names = load_datasets(
        data_dir=data_path,
        img_size=(224, 224),
        batch_size=args.batch_size,
        seed=args.seed,
        load_splits=("train", "val"),
    )
    print(f"Loaded {len(class_names)} classes: {class_names}")

    # Compute class weights strictly from training split
    class_weights = compute_class_weights(data_path / "train")
    print(f"Computed class weights: {class_weights}")

    # Configure smoke test mode if requested
    epochs_head = 1 if args.smoke else args.epochs_head
    epochs_finetune = 1 if args.smoke else args.epochs_finetune

    if args.smoke:
        print("\n[SMOKE MODE] Limiting dataset to 2 batches per epoch and 1 epoch per stage.")
        train_ds = train_ds.take(2)
        val_ds = val_ds.take(2)

    # Prepare output directories
    out_root = Path(args.output_dir)
    models_dir = out_root / "models"
    results_dir = out_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_file = models_dir / "real_model.keras"

    # 4. Build Model
    print("\nBuilding MobileNetV2 transfer learning model...")
    model = build_real_model(num_classes=len(class_names), img_size=(224, 224))

    # Shared callbacks
    early_stop_head = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )
    checkpoint_head = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_file),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    # 5. Stage 1: Train Head Only
    print(f"\n=== Stage 1: Training Classification Head ({epochs_head} epochs max) ===")
    history_head = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs_head,
        class_weight=class_weights,
        callbacks=[early_stop_head, checkpoint_head],
    )
    actual_epochs_head = len(history_head.history["loss"])
    print(f"Stage 1 finished in {actual_epochs_head} epochs.")

    # 6. Stage 2: Fine-Tuning
    print("\n=== Stage 2: Fine-Tuning Top 30 Layers (Learning Rate: 1e-5) ===")
    model = enable_fine_tuning(model, num_layers=30)

    early_stop_finetune = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=args.patience,
        restore_best_weights=True,
    )
    checkpoint_finetune = tf.keras.callbacks.ModelCheckpoint(
        filepath=str(checkpoint_file),
        monitor="val_loss",
        save_best_only=True,
        verbose=1,
    )

    history_finetune = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=actual_epochs_head + epochs_finetune,
        initial_epoch=actual_epochs_head,
        class_weight=class_weights,
        callbacks=[early_stop_finetune, checkpoint_finetune],
    )
    actual_epochs_finetune = len(history_finetune.history["loss"])
    print(f"Stage 2 finished in {actual_epochs_finetune} epochs.")

    # Save final best weights model
    print(f"\nSaving final model artifact to '{checkpoint_file.resolve()}'...")
    model.save(str(checkpoint_file))

    # 7. Save class_names.json
    class_names_file = models_dir / "class_names.json"
    with open(class_names_file, "w", encoding="utf-8") as f:
        json.dump(class_names, f, indent=2)
    print(f"Saved class names to '{class_names_file.resolve()}'.")

    # 8. Save combined history.json
    combined_history = {
        "finetune_start_epoch": actual_epochs_head,
    }
    all_metric_keys = set(history_head.history.keys()).union(history_finetune.history.keys())
    for key in all_metric_keys:
        head_vals = [float(v) for v in history_head.history.get(key, [])]
        finetune_vals = [float(v) for v in history_finetune.history.get(key, [])]
        combined_history[key] = head_vals + finetune_vals

    history_file = results_dir / "history.json"
    with open(history_file, "w", encoding="utf-8") as f:
        json.dump(combined_history, f, indent=2)
    print(f"Saved combined training history to '{history_file.resolve()}'.")

    # 9. Load split summary details if available for metadata
    split_summary_path = data_path / "split_summary.json"
    image_counts = {}
    split_method_name = "unknown"
    dataset_source_str = "kaggle: agrigorev/clothing-dataset-full"
    if split_summary_path.is_file():
        try:
            with open(split_summary_path, "r", encoding="utf-8") as f:
                s_data = json.load(f)
                image_counts = s_data.get("counts_per_split", {})
                split_method_name = s_data.get("split_method", "unknown")
                dataset_source_str = s_data.get("dataset_source", dataset_source_str)
        except Exception:
            pass

    # 10. Save metadata.json
    metadata = {
        "model_name": "FashionVision v2 (Real Photos MobileNetV2)",
        "base_model_name": "MobileNetV2",
        "dataset_source": dataset_source_str,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "epochs_requested": {
            "head": epochs_head,
            "finetune": epochs_finetune,
        },
        "epochs_actually_run": {
            "head": actual_epochs_head,
            "finetune": actual_epochs_finetune,
        },
        "img_size": [224, 224],
        "class_names": class_names,
        "image_counts_per_class_per_split": image_counts,
        "split_method": split_method_name,
        "tensorflow_version": tf.__version__,
        "python_version": sys.version,
        "gpus": gpu_names,
        "unfrozen_layers": 30,
        "preprocessing_description": (
            "PIL LANCZOS resize to 224x224 RGB; internal Keras Rescaling(scale=1/127.5, offset=-1)"
        ),
        "utc_timestamp": datetime.now(timezone.utc).isoformat(),
    }

    metadata_file = models_dir / "metadata.json"
    with open(metadata_file, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved runtime metadata to '{metadata_file.resolve()}'.")

    print("\n[SUCCESS] Training pipeline completed successfully.")


if __name__ == "__main__":
    main()
