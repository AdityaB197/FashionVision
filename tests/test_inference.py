"""Tests for inference functions in src/inference.py."""

import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import pytest

from src.data import load_fashion_mnist
from src.inference import load_artifacts, prepare_image, predict_top3


@pytest.mark.parametrize(
    "pil_image",
    [
        Image.new("L", (28, 28), color=120),
        Image.new("RGB", (100, 100), color=(120, 150, 180)),
        Image.new("RGBA", (64, 48), color=(100, 150, 200, 255)),
    ],
    ids=["grayscale_28x28", "rgb_100x100", "rgba_64x48"],
)
def test_prepare_image_output_properties(pil_image):
    # Protects against dimension and mode mismatches across various PIL image formats and sizes
    processed = prepare_image(pil_image)

    assert processed.shape == (1, 28, 28, 1)
    assert processed.dtype == np.float32
    assert np.all(processed >= 0.0) and np.all(processed <= 1.0)


def test_prepare_image_invert():
    # Protects against incorrect color-inversion transformations
    # Generate non-uniform 28x28 grayscale pattern
    pattern = (np.arange(28 * 28) % 256).astype(np.uint8).reshape((28, 28))
    img = Image.fromarray(pattern, mode="L")

    normal = prepare_image(img, invert=False)
    inverted = prepare_image(img, invert=True)

    # Inverting non-uniform image must differ from uninverted image
    assert not np.allclose(normal, inverted)

    # Invert input image beforehand, then call prepare_image with invert=True
    pre_inverted = ImageOps.invert(img)
    restored = prepare_image(pre_inverted, invert=True)

    # Applying invert twice must restore the original result
    assert np.allclose(normal, restored)


class FakeModel:
    """Minimal fake model providing predict() with fixed probabilities."""

    def __init__(self, probabilities):
        self._probabilities = np.array(probabilities, dtype=np.float32)

    def predict(self, x, verbose=0):
        return np.array([self._probabilities], dtype=np.float32)


def test_predict_top3_with_fake_model():
    # Protects against ranking errors, incorrect top-k slicing, or non-serializable outputs
    dummy_probs = [0.05, 0.10, 0.02, 0.60, 0.01, 0.03, 0.07, 0.04, 0.02, 0.06]
    fake_model = FakeModel(dummy_probs)
    dummy_input = np.zeros((1, 28, 28, 1), dtype=np.float32)
    class_names = [f"class_{i}" for i in range(10)]

    result = predict_top3(fake_model, dummy_input, class_names, k=3)

    # Check predicted index and class
    assert result["predicted_index"] == 3
    assert result["predicted_class"] == "class_3"
    assert result["confidence"] == pytest.approx(0.60)

    # Check top_k length and ordering (descending)
    assert len(result["top_k"]) == 3
    scores = [item["score"] for item in result["top_k"]]
    assert scores == sorted(scores, reverse=True)
    assert result["top_k"][0]["class_index"] == 3
    assert result["top_k"][1]["class_index"] == 1
    assert result["top_k"][2]["class_index"] == 6

    # Check all_scores contains exactly 10 values
    assert len(result["all_scores"]) == 10

    # Check result is JSON-serializable without errors
    serialized = json.dumps(result)
    assert isinstance(serialized, str)


REQUIRED_MODEL_FILES = [
    Path("models/fashion_cnn.keras"),
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
    model, class_names, metadata = load_artifacts(model_dir="models")

    (_, _), (_, _), (x_test, _) = load_fashion_mnist()
    test_samples = x_test[:3]

    expected_keys = {"predicted_index", "predicted_class", "confidence", "top_k", "all_scores"}

    for img_arr in test_samples:
        pil_img = Image.fromarray(img_arr)
        x_prep = prepare_image(pil_img, invert=False)
        res = predict_top3(model, x_prep, class_names, k=3)

        assert expected_keys.issubset(res.keys())
        assert res["predicted_class"] in class_names
        assert 0 <= res["predicted_index"] < len(class_names)
        assert 0.0 <= res["confidence"] <= 1.0
        assert len(res["top_k"]) == 3
        for item in res["top_k"]:
            assert item["class_name"] in class_names
            assert 0 <= item["class_index"] < len(class_names)
            assert 0.0 <= item["score"] <= 1.0
