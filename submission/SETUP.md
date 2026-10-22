# Setup and Reproduction Guide — FashionVision

This document provides step-by-step instructions to set up, run, and verify the **FashionVision** project on Windows, macOS, and Linux.

---

## 1. Quick Start (Running the Streamlit Web Application)

### 1.1 Prerequisites
- **Python**: Version 3.10, 3.11, 3.12, or 3.13.
- **Git** installed on your system.

### 1.2 Windows (PowerShell) Setup

1. Open PowerShell and navigate to the project directory:
   ```powershell
   cd "C:\path\to\ML mini Project"
   ```

2. Create a virtual environment:
   ```powershell
   python -m venv .venv
   ```

3. Activate the virtual environment:
   ```powershell
   .venv\Scripts\Activate.ps1
   ```
   *(If script execution is disabled on your system, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` first).*

4. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

5. Launch the Streamlit dashboard:
   ```powershell
   python -m streamlit run app.py
   ```
   The application will automatically open in your default browser at **http://localhost:8501**.

---

### 1.3 macOS / Linux (Bash / Zsh) Setup

1. Open Terminal and clone/navigate to the project:
   ```bash
   cd /path/to/ML\ mini\ Project
   ```

2. Create and activate a virtual environment:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install -r requirements.txt
   ```

4. Launch the application:
   ```bash
   python -m streamlit run app.py
   ```

---

## 2. Running Automated Verifications and Test Suites

To verify that the complete machine learning pipeline, dataset ingestion, tensor preprocessing, and model inference are operating correctly:

### 2.1 Run the Full Pytest Suite (11 Tests)
```bash
python -m pytest -v
```
- **Tests Covered**:
  - `tests/test_preprocessing.py`: Batch tensor shapes, float32 precision, [0.0, 1.0] scaling, single-image dimension expansion, boundary mapping (0 -> 0.0, 255 -> 1.0), input immutability, and class name consistency (10 unique entries).
  - `tests/test_inference.py`: PIL image format handling (Grayscale, RGB, RGBA), invert logic idempotency, top-k ranking and score ordering with mock model, JSON serializability, and real artifact integration tests.

### 2.2 Run the Comprehensive System Verifier (25 Checks)
```bash
python scripts/verify_project.py
```
- **Checks Performed**:
  - Validates presence of all 19 required repository files and folders.
  - Verifies presence of documentation files (`README.md`, `docs/ARCHITECTURE.md`, `docs/REPORT_OUTLINE.md`, `notebooks/colab_train.ipynb`).
  - Verifies that `models/class_names.json` matches `src.data.CLASS_NAMES` exactly.
  - Executes an end-to-end sample prediction through the real trained model, validating the output schema and confidence bounds.

---

## 3. Command-Line Sanity Checks

### 3.1 FashionVision v1 (Fashion-MNIST CNN)
```bash
python scripts/check_inference.py
```
- Loads `models/fashion_cnn.keras`, runs predictions on the first 20 images of the test set, compares against true labels, and verifies preprocessing consistency.

### 3.2 FashionVision v2 (MobileNetV2 Real Clothing Classifier)
```bash
python scripts/check_real_inference.py
```
- Loads `models_v2/real_model.keras` (when present), tests real photographic downsampling, and verifies that arbitrary RGBA inputs are converted to shape `(1, 224, 224, 3)`.

---

## 4. Reproducing Model Training

### 4.1 FashionVision v1 Training
The v1 model was trained on Google Colab using a GPU in under 60 seconds:
- **Notebook**: Open and execute `notebooks/colab_training.ipynb` (or run `python scripts/train_model.py` locally).
- **Evaluation**: `python scripts/evaluate_model.py` recomputes all metrics and regenerates `results/` plots.

### 4.2 FashionVision v2 Training (Colab GPU Recommended)
1. Prepare data with:
   ```bash
   python scripts/prepare_real_data.py --output-dir data/clothing_prepared
   ```
2. Train head and fine-tune:
   ```bash
   python scripts/train_real.py --data-dir data/clothing_prepared --epochs-head 10 --epochs-finetune 10
   ```
3. Evaluate test split:
   ```bash
   python scripts/evaluate_real.py --data-dir data/clothing_prepared
   ```
