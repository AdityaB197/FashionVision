"""Shared inference module for FashionVision.

This model was trained on standardized Fashion-MNIST images and is not reliable
on arbitrary real-world clothing photographs.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf

from src.data import preprocess


def load_artifacts(model_dir="models"):
    model_path_dir = Path(model_dir)

    model_file = model_path_dir / "fashion_cnn.keras"
    class_names_file = model_path_dir / "class_names.json"
    metadata_file = model_path_dir / "metadata.json"

    # Check for missing artifact files
    required_files = [
        ("fashion_cnn.keras", model_file),
        ("class_names.json", class_names_file),
        ("metadata.json", metadata_file),
    ]
    missing = [name for name, path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required artifact(s): {', '.join(missing)} in folder '{model_path_dir.resolve()}'"
        )

    # Load class names and metadata JSON files
    with open(class_names_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    # Compare major.minor TensorFlow versions
    saved_tf_ver = metadata.get("tensorflow_version", "")
    current_tf_ver = tf.__version__
    saved_major_minor = ".".join(saved_tf_ver.split(".")[:2]) if saved_tf_ver else ""
    current_major_minor = ".".join(current_tf_ver.split(".")[:2])

    if saved_major_minor and saved_major_minor != current_major_minor:
        print(
            f"WARNING: TensorFlow version mismatch. Model was trained on TF {saved_tf_ver}, "
            f"but current environment is running TF {current_tf_ver}."
        )

    model = tf.keras.models.load_model(str(model_file))

    return model, class_names, metadata


def prepare_image(pil_image, invert=False):
    # Convert image to grayscale (single channel)
    img = pil_image.convert("L")

    # Fashion-MNIST items are light objects on a black background, while ordinary
    # user photos and sketches usually have dark objects on a white background.
    # The invert option flips the pixel values (255 - pixel) to match Fashion-MNIST.
    if invert:
        img = ImageOps.invert(img)

    # Resize to 28x28 using LANCZOS interpolation for sharp downsampling
    img = img.resize((28, 28), Image.Resampling.LANCZOS)

    # Convert to uint8 numpy array of shape (28, 28)
    arr = np.array(img, dtype=np.uint8)

    # Reusing preprocess() from src.data ensures consistent scaling (0-255 -> 0.0-1.0)
    # and tensor reshaping (1, 28, 28, 1) identical to how the model was trained.
    return preprocess(arr)


def predict_top3(model, x, class_names, k=3):
    preds = model.predict(x, verbose=0)
    scores = preds[0]

    # Find indices of highest probabilities in descending order
    top_indices = np.argsort(scores)[::-1][:k]

    predicted_index = int(top_indices[0])
    predicted_class = str(class_names[predicted_index])
    confidence = float(scores[predicted_index])

    top_k = [
        {
            "class_index": int(idx),
            "class_name": str(class_names[idx]),
            "score": float(scores[idx]),
        }
        for idx in top_indices
    ]

    all_scores = [float(s) for s in scores]

    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "confidence": confidence,
        "top_k": top_k,
        "all_scores": all_scores,
    }
