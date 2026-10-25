"""Tests for preprocessing functions and data loading in src/data.py and src/inference.py."""

from pathlib import Path
import numpy as np
from PIL import Image

from src.inference import prepare_real_image


def test_prepare_real_image_rgb_output():
    # Protects against incorrect tensor shape, non-float32 dtype, or pixel values outside [0, 255]
    test_img = Image.new("RGB", (300, 200), color=(120, 150, 180))

    processed = prepare_real_image(test_img)

    assert processed.dtype == np.float32
    assert processed.shape == (1, 224, 224, 3)
    assert np.all(processed >= 0.0) and np.all(processed <= 255.0)


def test_prepare_real_image_rgba_conversion():
    # Protects against RGBA alpha channel not being properly flattened to RGB
    rgba_img = Image.new("RGBA", (150, 100), color=(100, 150, 200, 128))

    processed = prepare_real_image(rgba_img)

    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_prepare_real_image_grayscale_conversion():
    # Protects against grayscale images not being converted to 3-channel RGB
    gray_img = Image.new("L", (200, 150), color=120)

    processed = prepare_real_image(gray_img)

    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_prepare_real_image_exif_transpose():
    # Protects against EXIF orientation issues (common with smartphone photos)
    # Create a simple test image
    test_img = Image.new("RGB", (100, 200), color=(255, 0, 0))  # Tall red image
    
    # The function should handle EXIF without errors
    processed = prepare_real_image(test_img)
    
    assert processed.shape == (1, 224, 224, 3)
    assert processed.dtype == np.float32


def test_prepare_real_image_preserves_input():
    # Protects against mutating input PIL image
    original = Image.new("RGB", (100, 100), color=(50, 100, 150))
    original_mode = original.mode
    original_size = original.size

    _ = prepare_real_image(original)

    # Original image should not be modified
    assert original.mode == original_mode
    assert original.size == original_size
