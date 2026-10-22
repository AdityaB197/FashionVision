"""Project verification script for FashionVision.

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

    print("=== FashionVision Project Verification ===\n")

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
        ("models/fashion_cnn.keras", PROJECT_ROOT / "models" / "fashion_cnn.keras", False),
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

    # 3. Check class_names.json equals CLASS_NAMES from src.data
    print("\n-- Checking Class Names Consistency --")
    class_names_file = PROJECT_ROOT / "models" / "class_names.json"
    if class_names_file.is_file():
        try:
            from src.data import CLASS_NAMES

            with open(class_names_file, "r", encoding="utf-8") as f:
                saved_class_names = json.load(f)

            if saved_class_names == CLASS_NAMES:
                print("[PASS] models/class_names.json matches src.data.CLASS_NAMES exactly.")
                passed_count += 1
            else:
                print(
                    f"[FAIL] Class names mismatch between models/class_names.json and src.data.CLASS_NAMES.\n"
                    f"       Saved: {saved_class_names}\n"
                    f"       Code:  {CLASS_NAMES}"
                )
                fail_count += 1
        except Exception as e:
            print(f"[FAIL] Error comparing class names: {e}")
            fail_count += 1
    else:
        print("[FAIL] Cannot check class names consistency: models/class_names.json is missing.")
        fail_count += 1

    # 4. End-to-end sample prediction check
    print("\n-- Running End-to-End Sample Prediction Check --")
    try:
        from PIL import Image
        from src.data import load_fashion_mnist
        from src.inference import load_artifacts, prepare_image, predict_top3

        model, class_names, metadata = load_artifacts(str(PROJECT_ROOT / "models"))
        (_, _), (_, _), (x_test, y_test) = load_fashion_mnist()

        test_img = Image.fromarray(x_test[0])
        prep_img = prepare_image(test_img, invert=False)
        pred_res = predict_top3(model, prep_img, class_names, k=3)

        expected_keys = {"predicted_index", "predicted_class", "confidence", "top_k", "all_scores"}
        is_valid = (
            isinstance(pred_res, dict)
            and expected_keys.issubset(pred_res.keys())
            and pred_res["predicted_class"] == class_names[pred_res["predicted_index"]]
            and 0 <= pred_res["predicted_index"] < len(class_names)
            and 0.0 <= pred_res["confidence"] <= 1.0
            and len(pred_res["top_k"]) == 3
            and len(pred_res["all_scores"]) == 10
        )

        if is_valid:
            print(
                f"[PASS] Sample prediction succeeded: predicted '{pred_res['predicted_class']}' "
                f"(index {pred_res['predicted_index']}) with confidence {pred_res['confidence']:.4f}."
            )
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
