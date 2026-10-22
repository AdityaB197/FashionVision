"""Manual sanity check script for FashionVision inference pipeline."""

import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import load_fashion_mnist, preprocess
from src.inference import load_artifacts, prepare_image, predict_top3


def main():
    print("=== Loading Artifacts ===")
    model, class_names, metadata = load_artifacts()
    print("Model, class names, and metadata loaded successfully.\n")

    print("=== Loading Test Set (First 20 Images) ===")
    (_, _), (_, _), (x_test, y_test) = load_fashion_mnist()
    test_subset = x_test[:20]
    true_labels = y_test[:20]

    pil_preds = []
    correct_count = 0

    header = f"{'Index':<6} {'True Class':<15} {'Predicted Class':<15} {'Confidence':>10}  {'Result':<6}"
    print(header)
    print("-" * len(header))

    sample_prepared = None
    for idx, (img_arr, true_label) in enumerate(zip(test_subset, true_labels)):
        pil_img = Image.fromarray(img_arr)
        x_prep = prepare_image(pil_img, invert=False)
        if sample_prepared is None:
            sample_prepared = x_prep

        res = predict_top3(model, x_prep, class_names)
        pred_idx = res["predicted_index"]
        pred_class = res["predicted_class"]
        conf = res["confidence"]
        pil_preds.append(pred_idx)

        is_ok = (pred_idx == true_label)
        if is_ok:
            correct_count += 1
        status = "OK" if is_ok else "WRONG"
        true_name = class_names[true_label]

        print(f"{idx:<6} {true_name:<15} {pred_class:<15} {conf:>10.4f}  {status:<6}")

    print("-" * len(header))
    print(f"Correct: {correct_count} / 20")
    print("NOTE: This is a tiny sanity check on 20 samples, not an overall accuracy measurement.\n")

    print("=== Verifying Preprocessing Consistency ===")
    batch_prep = preprocess(test_subset)
    for i in range(20):
        single_x = batch_prep[i:i + 1]
        direct_res = predict_top3(model, single_x, class_names)
        assert direct_res["predicted_index"] == pil_preds[i], (
            f"Mismatch at index {i}: PIL prediction ({pil_preds[i]}) != direct prediction ({direct_res['predicted_index']})"
        )
    print("Assertion passed: prepare_image and src.data.preprocess yield identical predictions.\n")

    print("=== Checking Array Properties ===")
    print(f"Shape : {sample_prepared.shape} (expected: (1, 28, 28, 1))")
    print(f"Dtype : {sample_prepared.dtype} (expected: float32)")
    print(f"Min   : {sample_prepared.min():.4f} (expected: >= 0.0)")
    print(f"Max   : {sample_prepared.max():.4f} (expected: <= 1.0)\n")

    print("=== Testing Arbitrary RGB Image Input ===")
    random_rgb_arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    rgb_img = Image.fromarray(random_rgb_arr, mode="RGB")
    rgb_prep = prepare_image(rgb_img, invert=False)
    print(f"RGB image input shape after prepare_image: {rgb_prep.shape}, dtype: {rgb_prep.dtype}")
    assert rgb_prep.shape == (1, 28, 28, 1), f"Expected (1, 28, 28, 1), got {rgb_prep.shape}"
    print("RGB image test passed.\n")

    print("[SUCCESS] All inference checks passed.")


if __name__ == "__main__":
    main()
