# Setup and Reproduction Guide — FashionVision v2

This document provides step-by-step instructions to set up, run, and verify the **FashionVision v2** project on Windows, macOS, and Linux.

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

### 2.1 Run the Full Pytest Suite (10 Tests)
```bash
python -m pytest -v
```
- **Tests Covered**:
  - `tests/test_preprocessing.py`: RGB output shapes `(1, 224, 224, 3)`, float32 precision, `[0.0, 255.0]` tensor range, RGBA alpha flattening to RGB, grayscale-to-RGB expansion, and EXIF transposition safety.
  - `tests/test_inference.py`: Output properties across arbitrary PIL formats (Grayscale, RGB, RGBA), top-k score ordering and ranking with mock model, and real artifact integration tests against `models/real_model.keras`.

### 2.2 Run the Comprehensive System Verifier (24 Checks)
```bash
python scripts/verify_project.py
```
- **Checks Performed**:
  - Validates presence of all mandatory repository files, models, and results.
  - Verifies presence of documentation files (`README.md`, `docs/ARCHITECTURE.md`, `docs/REPORT_OUTLINE.md`, `notebooks/colab_train.ipynb`).
  - Executes an end-to-end sample prediction through the real MobileNetV2 model using a synthetic `(1, 224, 224, 3)` RGB tensor, validating the output schema and confidence bounds.

---

## 3. Command-Line Inference Sanity Check

```bash
python scripts/check_inference.py
```
- Loads `models/real_model.keras`, tests real photographic downsampling, and verifies that arbitrary RGBA/RGB inputs are properly converted to shape `(1, 224, 224, 3)` with valid top-k class rankings.

---

## 4. Reproducing Model Training Pipeline

Training can be performed locally or via Google Colab with GPU acceleration.

### 4.1 Step 1: Data Preparation
Download and organize the Kaggle real clothing dataset (`agrigorev/clothing-dataset-full`) into train, validation, and test splits:
```bash
python scripts/prepare_data.py --output-dir data/clothing_prepared
```

### 4.2 Step 2: Two-Stage Transfer Learning
Execute the two-stage training pipeline (Stage 1: frozen backbone head training; Stage 2: unfreezing the top 30 layers with fine-tuning):
```bash
python scripts/train_model.py --data-dir data/clothing_prepared --output-dir . --epochs-head 10 --epochs-finetune 10 --batch-size 32
```
This saves the trained weights to `models/real_model.keras`, `models/class_names.json`, and `models/metadata.json`.

### 4.3 Step 3: Test Split Evaluation
Compute comprehensive classification metrics, confusion matrices, and misclassified example galleries on the held-out test split:
```bash
python scripts/evaluate_model.py --data-dir data/clothing_prepared --output-dir .
```
All evaluation artifacts are automatically written to `results/`.

### 4.4 Google Colab (Recommended for GPU Acceleration)
- Open `notebooks/colab_train.ipynb` on [Google Colab](https://colab.research.google.com).
- Set runtime to **T4 GPU** (`Runtime > Change runtime type > T4 GPU`).
- Run all cells to download data, train MobileNetV2, evaluate test metrics, and export the generated artifacts.
