"""Data loading and preprocessing utilities for FashionVision v2 (real photos).

The test split must NOT be used anywhere in training or any hyperparameter decision.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def load_datasets(data_dir, img_size=(224, 224), batch_size=32, seed=42,
                   load_splits=("train", "val", "test")):
    """Load train, val, and/or test datasets using tf.keras.utils.image_dataset_from_directory.

    The images are already resized to 224x224 by the prepare step, so no further
    resizing should distort them. This keeps training and inference resizing identical:
    both use PIL LANCZOS downsampling.

    Args:
        load_splits: Which splits to load. Default loads all three.
                     Pass ("train", "val") during training to skip loading
                     the test split (saves time and prevents accidental use).
    """
    data_path = Path(data_dir)

    for split_name in load_splits:
        split_path = data_path / split_name
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing {split_name} directory at '{split_path.resolve()}'. "
                f"Please run scripts/prepare_data.py first."
            )

    # --- Train split ---
    train_ds = None
    class_names = None
    if "train" in load_splits:
        # shuffle=False here because we apply a proper per-epoch shuffle via .shuffle() below.
        # Using shuffle=True in image_dataset_from_directory only shuffles the file list once,
        # which is redundant and wastes a tiny amount of time.
        train_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(data_path / "train"),
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=img_size,
            shuffle=True,
            seed=seed,
        )
        class_names = list(train_ds.class_names)

    # --- Val split ---
    val_ds = None
    if "val" in load_splits:
        val_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(data_path / "val"),
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False,
        )
        if class_names is None:
            class_names = list(val_ds.class_names)

    # --- Test split (only when requested) ---
    test_ds = None
    if "test" in load_splits:
        test_ds = tf.keras.utils.image_dataset_from_directory(
            directory=str(data_path / "test"),
            labels="inferred",
            label_mode="int",
            color_mode="rgb",
            batch_size=batch_size,
            image_size=img_size,
            shuffle=False,
        )
        if class_names is None:
            class_names = list(test_ds.class_names)

    # Optimization: prefetch next batch while GPU processes current batch.
    # Note: We deliberately avoid in-memory .cache() here because caching 3,500+
    # 224x224 RGB tensors consumes ~2.5 GB of system RAM on Colab/machines,
    # causing swapping, extreme training slowdown, or out-of-memory kernel crashes.
    autotune = tf.data.AUTOTUNE
    if train_ds is not None:
        train_ds = train_ds.prefetch(buffer_size=autotune)
    if val_ds is not None:
        val_ds = val_ds.prefetch(buffer_size=autotune)
    if test_ds is not None:
        test_ds = test_ds.prefetch(buffer_size=autotune)

    return train_ds, val_ds, test_ds, class_names


def compute_class_weights(train_dir_or_labels):
    """Compute balanced class weights from TRAINING labels only to handle class imbalance.

    Accepts either a directory Path/str pointing to the train split, or an array/list
    of integer class labels from the training set.
    """
    if isinstance(train_dir_or_labels, (str, Path)):
        train_path = Path(train_dir_or_labels)
        labels = []
        # Folder names sorted alphabetically match Keras class_names index ordering
        class_folders = sorted([d for d in train_path.iterdir() if d.is_dir()])
        for class_idx, class_dir in enumerate(class_folders):
            num_images = len([f for f in class_dir.iterdir() if f.is_file()])
            labels.extend([class_idx] * num_images)
        y_train = np.array(labels, dtype=np.int32)
    else:
        y_train = np.array(train_dir_or_labels, dtype=np.int32)

    if len(y_train) == 0:
        return {}

    classes = np.unique(y_train)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)

    return {int(cls): float(w) for cls, w in zip(classes, weights)}
