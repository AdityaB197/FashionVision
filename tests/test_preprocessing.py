"""Tests for preprocessing functions and dataset constants in src/data.py."""

from pathlib import Path
import numpy as np

from src.data import CLASS_NAMES, preprocess


def test_preprocess_batch_output():
    # Protects against incorrect tensor shape, non-float32 dtype, or pixel values outside [0, 1]
    batch_size = 4
    synthetic_batch = np.random.randint(0, 256, size=(batch_size, 28, 28), dtype=np.uint8)

    processed = preprocess(synthetic_batch)

    assert processed.dtype == np.float32
    assert processed.shape == (batch_size, 28, 28, 1)
    assert np.all(processed >= 0.0) and np.all(processed <= 1.0)


def test_preprocess_single_image():
    # Protects against single 28x28 images failing to receive batch and channel dimensions
    single_image = np.zeros((28, 28), dtype=np.uint8)

    processed = preprocess(single_image)

    assert processed.shape == (1, 28, 28, 1)
    assert processed.dtype == np.float32
    assert np.all(processed >= 0.0) and np.all(processed <= 1.0)


def test_preprocess_boundary_mapping():
    # Protects against scaling errors by ensuring 0 maps to 0.0 and 255 maps to 1.0
    extremes = np.array([[0, 255], [255, 0]], dtype=np.uint8)

    processed = preprocess(extremes)

    assert processed[0, 0, 0, 0] == 0.0
    assert processed[0, 0, 1, 0] == 1.0


def test_preprocess_does_not_modify_input_in_place():
    # Protects against mutating input arrays, preserving caller's raw data
    original = np.array([[0, 128], [200, 255]], dtype=np.uint8)
    original_copy = original.copy()

    _ = preprocess(original)

    assert np.array_equal(original, original_copy)
    assert original.dtype == np.uint8


def test_class_names_ten_unique_entries():
    # Protects against duplicate, missing, or mismatched class definitions
    assert len(CLASS_NAMES) == 10
    assert len(set(CLASS_NAMES)) == 10
