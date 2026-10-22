"""Data loading and preprocessing utilities for FashionVision v2 (real photos).

The test split must NOT be used anywhere in training or any hyperparameter decision.
"""

from pathlib import Path
import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight


def load_datasets(data_dir, img_size=(224, 224), batch_size=32, seed=42):
    """Load train, val, and test datasets using tf.keras.utils.image_dataset_from_directory.

    The images are already resized to 224x224 by the prepare step, so no further
    resizing should distort them. This keeps training and inference resizing identical:
    both use PIL LANCZOS downsampling.
    """
    data_path = Path(data_dir)
    train_dir = data_path / "train"
    val_dir = data_path / "val"
    test_dir = data_path / "test"

    for split_name, split_path in [("train", train_dir), ("val", val_dir), ("test", test_dir)]:
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing {split_name} directory at '{split_path.resolve()}'. "
                f"Please run scripts/prepare_real_data.py first."
            )

    # Load datasets from subfolder directory structure
    train_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(train_dir),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=True,
        seed=seed,
    )

    class_names = list(train_ds.class_names)

    val_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(val_dir),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    # The test set must NOT be shuffled to preserve deterministic evaluation alignment
    test_ds = tf.keras.utils.image_dataset_from_directory(
        directory=str(test_dir),
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=img_size,
        shuffle=False,
    )

    # Optimization: cache in memory and prefetch next batch while GPU processes current batch
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.cache().shuffle(buffer_size=1000, seed=seed).prefetch(buffer_size=autotune)
    val_ds = val_ds.cache().prefetch(buffer_size=autotune)
    test_ds = test_ds.cache().prefetch(buffer_size=autotune)

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
