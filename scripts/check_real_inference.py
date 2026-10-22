"""Manual sanity check script for FashionVision v2 inference pipeline."""

import argparse
from pathlib import Path
import sys

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from PIL import Image

from src.real_inference import load_real_artifacts, prepare_real_image, predict_top_k_real


def main():
    parser = argparse.ArgumentParser(description="Sanity check FashionVision v2 inference.")
    parser.add_argument("--model-dir", default="models_v2", help="Folder containing real_model.keras, class_names.json, metadata.json.")
    parser.add_argument("--data-dir", default="data/clothing_prepared", help="Prepared dataset directory.")
    args = parser.parse_args()

    print("=== Loading FashionVision v2 Artifacts ===")
    model, class_names, metadata = load_real_artifacts(model_dir=args.model_dir)
    print(f"Loaded model successfully with {len(class_names)} classes.")
    print(f"Base model: {metadata.get('base_model_name', 'MobileNetV2')}\n")

    # 1. Sanity check on sample images from the test folder
    test_dir = Path(args.data_dir) / "test"
    test_image_files = []
    if test_dir.is_dir():
        for class_dir in sorted([d for d in test_dir.iterdir() if d.is_dir()]):
            images = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
            if images:
                # Pick up to 2 images per class
                test_image_files.extend([(img_file, class_dir.name) for img_file in images[:2]])

    if not test_image_files:
        print(f"Warning: No test images found in '{test_dir.resolve()}'. Skipping sample test inference.")
    else:
        # Evaluate up to 10 sample test images
        sample_subset = test_image_files[:10]
        print(f"=== Testing Sample Images from Test Split ({len(sample_subset)} images) ===")
        header = f"{'Index':<6} {'True Class':<18} {'Predicted Class':<18} {'Confidence':>10}  {'Result':<6}"
        print(header)
        print("-" * len(header))

        correct_count = 0
        for idx, (img_path, true_label) in enumerate(sample_subset):
            with Image.open(img_path) as pil_img:
                x_prep = prepare_real_image(pil_img)
                res = predict_top_k_real(model, x_prep, class_names, k=3)

            pred_class = res["predicted_class"]
            conf = res["confidence"]
            is_ok = (pred_class == true_label)
            if is_ok:
                correct_count += 1
            status = "OK" if is_ok else "WRONG"

            print(f"{idx:<6} {true_label:<18} {pred_class:<18} {conf:>10.4f}  {status:<6}")

        print("-" * len(header))
        print(f"Correct: {correct_count} / {len(sample_subset)}")
        print("NOTE: This is a tiny sanity check on a few samples, not an overall accuracy measurement.\n")

    # 2. Verify prepare_real_image on arbitrary RGBA image
    print("=== Testing prepare_real_image with Arbitrary RGBA Input ===")
    rand_rgba = np.random.randint(0, 256, (140, 95, 4), dtype=np.uint8)
    rgba_img = Image.fromarray(rand_rgba, mode="RGBA")

    prepared = prepare_real_image(rgba_img)
    print(f"Output shape: {prepared.shape} (expected: (1, 224, 224, 3))")
    print(f"Output dtype: {prepared.dtype} (expected: float32)")
    print(f"Value range:  min={prepared.min():.1f}, max={prepared.max():.1f} (expected: within [0.0, 255.0])")

    assert prepared.shape == (1, 224, 224, 3), f"Unexpected shape: {prepared.shape}"
    assert prepared.dtype == np.float32, f"Unexpected dtype: {prepared.dtype}"
    assert np.all(prepared >= 0.0) and np.all(prepared <= 255.0), "Values out of bounds [0, 255]"
    print("[PASS] Arbitrary RGBA image preprocessing verified successfully.")

    print("\n[SUCCESS] All v2 inference sanity checks passed.")


if __name__ == "__main__":
    main()
