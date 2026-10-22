"""Real clothing photo inference module for FashionVision v2.

This model was trained on a small public dataset of real clothing photos;
it works best on a single garment, centered, on a plain background;
high confidence is not a guarantee of correctness.
"""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import tensorflow as tf


def load_real_artifacts(model_dir="models_v2"):
    """Load v2 model artifacts: real_model.keras, class_names.json, and metadata.json.

    Raises FileNotFoundError with a clear message if any artifact is missing.
    Warns if there is a major.minor mismatch between current TensorFlow and training environment.
    """
    model_path_dir = Path(model_dir)

    model_file = model_path_dir / "real_model.keras"
    class_names_file = model_path_dir / "class_names.json"
    metadata_file = model_path_dir / "metadata.json"

    required_files = [
        ("real_model.keras", model_file),
        ("class_names.json", class_names_file),
        ("metadata.json", metadata_file),
    ]
    missing = [name for name, path in required_files if not path.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing required v2 artifact(s): {', '.join(missing)} in folder '{model_path_dir.resolve()}'"
        )

    with open(class_names_file, "r", encoding="utf-8") as f:
        class_names = json.load(f)

    with open(metadata_file, "r", encoding="utf-8") as f:
        metadata = json.load(f)

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


def prepare_real_image(pil_image):
    """Prepare a user-provided PIL image for v2 MobileNetV2 inference.

    Applies EXIF orientation normalization, converts to RGB (handling RGBA, grayscale,
    and palette modes), and downsamples to (224, 224) using high-quality LANCZOS filtering.
    Returns a float32 NumPy array with shape (1, 224, 224, 3) and pixel values in [0, 255].
    Pixel normalization [-1.0, 1.0] is handled internally by the model's Rescaling layer.
    """
    # Fix orientation based on camera EXIF metadata (essential for smartphone photos)
    img = ImageOps.exif_transpose(pil_image)

    # Standardize to 3-channel RGB (flattens RGBA alpha, converts grayscale/palette)
    img = img.convert("RGB")

    # Resize to 224x224 using LANCZOS interpolation
    img = img.resize((224, 224), Image.Resampling.LANCZOS)

    # Convert to float32 array in [0, 255] and add batch dimension
    arr = np.array(img, dtype=np.float32)
    return np.expand_dims(arr, axis=0)


def predict_top_k_real(model, x, class_names, k=3):
    """Predict class probabilities and return top-k predictions.

    Returns the same JSON-serializable dictionary format as v1:
    - predicted_index (int)
    - predicted_class (str)
    - confidence (float)
    - top_k (list of dicts with class_index, class_name, score)
    - all_scores (list of 10 floats)
    """
    preds = model.predict(x, verbose=0)
    scores = preds[0]

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
