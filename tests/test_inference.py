"""Tests for inference functions in src/inference.py."""

import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from src.inference import load_real_artifacts, prepare_real_image, predict_top_k_real


@pytest.mark.parametrize(
    "pil_image",
    [
        Image.new("L", (300, 200), color=120),
        Image.new("RGB", (100, 100), color=(120, 150, 180)),
        Image.new("RGBA", (64, 48), color=(100, 150, 200, 255)),
    ],
    ids=["grayscale_300x200", "rgb_100x100", "rgba_64x48"],
)
def test_prepare_real_image_output_properties(pil_image):
    # Protects against dimension and mode mismatches across various PIL image formats and sizes
    processed = prepare_real_image(pil_image)

    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32
    assert np.all(processed >= 0.0) and np.all(processed <= 255.0)


class FakeModel:
    """Minimal fake model providing predict() with fixed probabilities."""

    def __init__(self, probabilities):
        self._probabilities = np.array(probabilities, dtype=np.float32)

    def predict(self, x, verbose=0):
        return np.array([self._probabilities], dtype=np.float32)


def test_predict_top_k_real_with_fake_model():
    # Protects against ranking errors, incorrect top-k slicing, or non-serializable outputs
    dummy_probs = [0.10, 0.60, 0.30]  # 3 classes: dress, shirt, t-shirt
    fake_model = FakeModel(dummy_probs)
    dummy_input = np.zeros((1, 224, 224, 3), dtype=np.float32)
    class_names = ["dress", "shirt", "t-shirt"]

    result = predict_top_k_real(fake_model, dummy_input, class_names, k=3)

    # Check predicted index and class
    assert result["predicted_index"] == 1
    assert result["predicted_class"] == "shirt"
    assert result["confidence"] == pytest.approx(0.60)

    # Check top_k length and ordering (descending)
    assert len(result["top_k"]) == 3
    scores = [item["score"] for item in result["top_k"]]
    assert scores == sorted(scores, reverse=True)
    assert result["top_k"][0]["class_index"] == 1  # shirt (0.60)
    assert result["top_k"][1]["class_index"] == 2  # t-shirt (0.30)
    assert result["top_k"][2]["class_index"] == 0  # dress (0.10)

    # Check all_scores contains exactly 3 values
    assert len(result["all_scores"]) == 3

    # Check result is JSON-serializable without errors
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


REQUIRED_MODEL_FILES = [
    Path("models/real_model.keras"),
    Path("models/class_names.json"),
    Path("models/metadata.json"),
]
missing_artifacts = [str(p) for p in REQUIRED_MODEL_FILES if not p.exists()]


@pytest.mark.skipif(
    bool(missing_artifacts),
    reason=f"Missing model artifacts in models/: {missing_artifacts}",
)
def test_real_artifacts_inference_integration():
    # Protects against runtime regressions when loading and executing real trained artifacts
    model, class_names, metadata = load_real_artifacts(model_dir="models")

    # Create test RGB images
    test_samples = [
        Image.new("RGB", (300, 250), color=(100, 120, 140)),
        Image.new("RGB", (200, 400), color=(50, 80, 100)),
        Image.new("RGBA", (150, 150), color=(200, 100, 50, 255)),
    ]

    expected_keys = {"predicted_index", "predicted_class", "confidence", "top_k", "all_scores"}

    for pil_img in test_samples:
        x_prep = prepare_real_image(pil_img)
        res = predict_top_k_real(model, x_prep, class_names, k=min(3, len(class_names)))

        assert expected_keys.issubset(res.keys())
        assert res["predicted_class"] in class_names
        assert 0 <= res["predicted_index"] < len(class_names)
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["top_k"]) <= len(class_names)
        for item in res["top_k"]:
            assert item["class_name"] in class_names
            assert 0 <= item["class_index"] < len(class_names)
            assert 0.0 <= item["score"] <= 1.0
