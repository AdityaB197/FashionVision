"""Project verification script for FashionVision v2.

Validates required repository structure, artifacts, configuration consistency,
and executes an end-to-end inference smoke test.
"""

import json
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main():
    passed_count = 0
    warn_count = 0
    fail_count = 0

    print("=== FashionVision v2 Project Verification ===\n")

    # 1. Required files and directories (FAIL if missing)
    required_paths = [
        ("app.py", PROJECT_ROOT / "app.py", False),
        ("requirements.txt", PROJECT_ROOT / "requirements.txt", False),
        (".gitignore", PROJECT_ROOT / ".gitignore", False),
        ("src/data.py", PROJECT_ROOT / "src" / "data.py", False),
        ("src/model.py", PROJECT_ROOT / "src" / "model.py", False),
        ("src/inference.py", PROJECT_ROOT / "src" / "inference.py", False),
        ("src/reporting.py", PROJECT_ROOT / "src" / "reporting.py", False),
        ("scripts/train_model.py", PROJECT_ROOT / "scripts" / "train_model.py", False),
        ("scripts/evaluate_model.py", PROJECT_ROOT / "scripts" / "evaluate_model.py", False),
        ("scripts/verify_project.py", PROJECT_ROOT / "scripts" / "verify_project.py", False),
        ("models/real_model.keras", PROJECT_ROOT / "models" / "real_model.keras", False),
        ("models/class_names.json", PROJECT_ROOT / "models" / "class_names.json", False),
        ("models/metadata.json", PROJECT_ROOT / "models" / "metadata.json", False),
        ("results/metrics.json", PROJECT_ROOT / "results" / "metrics.json", False),
        ("results/history.json", PROJECT_ROOT / "results" / "history.json", False),
        ("results/confusion_matrix.png", PROJECT_ROOT / "results" / "confusion_matrix.png", False),
        ("results/misclassified_examples.png", PROJECT_ROOT / "results" / "misclassified_examples.png", False),
        ("results/training_curves.png", PROJECT_ROOT / "results" / "training_curves.png", False),
        ("tests/ folder", PROJECT_ROOT / "tests", True),
    ]

    print("-- Checking Required Files & Folders --")
    for name, path, is_dir in required_paths:
        exists = path.is_dir() if is_dir else path.is_file()
        if exists:
            print(f"[PASS] Found required item: {name}")
            passed_count += 1
        else:
            print(f"[FAIL] Missing required item: {name}")
            fail_count += 1

    # 2. Files that only produce a WARN if missing
    warn_paths = [
        ("README.md", PROJECT_ROOT / "README.md"),
        ("docs/ARCHITECTURE.md", PROJECT_ROOT / "docs" / "ARCHITECTURE.md"),
        ("docs/REPORT_OUTLINE.md", PROJECT_ROOT / "docs" / "REPORT_OUTLINE.md"),
        ("notebooks/colab_train.ipynb", PROJECT_ROOT / "notebooks" / "colab_train.ipynb"),
    ]

    print("\n-- Checking Optional / Pending Documentation & Notebooks --")
    for name, path in warn_paths:
        if path.is_file():
            print(f"[PASS] Optional file present: {name}")
            passed_count += 1
        else:
            print(f"[WARN] Optional file not found: {name}")
            warn_count += 1

    # 3. Check inference with 224x224 RGB test tensor
    print("\n-- Running End-to-End Sample Prediction Check (224x224 RGB) --")
    try:
        import numpy as np
        from PIL import Image
        from src.inference import load_real_artifacts, prepare_real_image, predict_top_k_real

        model, class_names, metadata = load_real_artifacts(str(PROJECT_ROOT / "models"))
        
        # Create a test RGB image (random)
        test_array = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
        test_img = Image.fromarray(test_array, mode="RGB")
        
        prep_img = prepare_real_image(test_img)
        pred_res = predict_top_k_real(model, prep_img, class_names, k=min(3, len(class_names)))

        expected_keys = {"predicted_index", "predicted_class", "confidence", "top_k", "all_scores"}
        is_valid = (
            isinstance(pred_res, dict)
            and expected_keys.issubset(pred_res.keys())
            and pred_res["predicted_class"] == class_names[pred_res["predicted_index"]]
            and 0 <= pred_res["predicted_index"] < len(class_names)
            and 0.0 <= pred_res["confidence"] <= 1.0
            and len(pred_res["top_k"]) <= len(class_names)
            and len(pred_res["all_scores"]) == len(class_names)
            and prep_img.shape == (1, 224, 224, 3)
            and prep_img.dtype == np.float32
        )

        if is_valid:
            print(
                f"[PASS] Sample prediction succeeded: predicted '{pred_res['predicted_class']}' "
                f"(index {pred_res['predicted_index']}) with confidence {pred_res['confidence']:.4f}."
            )
            print(f"       Preprocessed tensor shape: {prep_img.shape}, dtype: {prep_img.dtype}")
            passed_count += 1
        else:
            print(f"[FAIL] Sample prediction output did not meet schema expectations: {pred_res}")
            fail_count += 1

    except Exception as e:
        print(f"[FAIL] Sample prediction failed with exception: {e}")
        fail_count += 1

    # 5. Summary and exit
    total_checks = passed_count + warn_count + fail_count
    print("\n" + "=" * 50)
    print("Verification Summary:")
    print(f"  Total Checks: {total_checks}")
    print(f"  Passed:       {passed_count}")
    print(f"  Warnings:     {warn_count}")
    print(f"  Failed:       {fail_count}")
    print("=" * 50)

    if fail_count > 0:
        print("\n[RESULT] Verification FAILED.")
        sys.exit(1)
    else:
        print("\n[RESULT] Verification PASSED (warnings do not fail build).")
        sys.exit(0)


if __name__ == "__main__":
    main()
