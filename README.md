# FashionVision v2 — Real Clothing Classification with Transfer Learning

A production-ready machine learning system for classifying real-world clothing photographs using **MobileNetV2 transfer learning**. Features two-stage fine-tuning, real-time inference on 224×224 RGB photos, a reactive **Streamlit web application** with webcam support, and comprehensive evaluation reports.

---

## Features

- **Real Clothing Photo Classification**: Classifies real-world garment photographs (dress, shirt, t-shirt) with confidence scores and ranked probability distributions
- **MobileNetV2 Transfer Learning**: Leverages ImageNet-pretrained features with two-stage fine-tuning (frozen base → selective layer unfreezing)
- **Production-Grade Preprocessing**: EXIF orientation correction, RGBA/grayscale→RGB conversion, and high-quality LANCZOS 224×224 resizing
- **Interactive Streamlit Dashboard (`app.py`)**:
  - Upload photos, use webcam capture, or test with prepared samples
  - Side-by-side original vs preprocessed (224×224 RGB) visualizer
  - Real-time top-k probability rankings with confidence percentages
  - Model performance metrics, confusion matrix, and training curves
  - Downloadable JSON prediction reports with model provenance
- **Robust Data Pipeline**: Kaggle dataset integration, stratified/group-aware splitting, and class imbalance handling
- **Production QA**: Automated unit tests with `pytest` and comprehensive verification script

---

## Architecture Overview

```
+------------------------------------------------------------------------------+
|                          User Input (Real Photo)                             |
|         Upload JPEG/PNG/WebP • Webcam Capture • Test Sample Selector         |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                       Preprocessing Pipeline (224×224 RGB)                   |
|   • EXIF orientation correction (ImageOps.exif_transpose)                    |
|   • RGBA/Grayscale/Palette → RGB conversion                                  |
|   • High-quality LANCZOS downsampling to 224×224                             |
|   • Float32 [0, 255] tensor (batch dimension added)                          |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                       MobileNetV2 Transfer Model                             |
|   • Data Augmentation Layers (RandomFlip, Rotation, Zoom, Contrast)         |
|   • In-Graph Rescaling: [0, 255] → [-1.0, 1.0]                              |
|   • MobileNetV2 Base (ImageNet pretrained, 3.5M parameters)                  |
|     - Stage 1 Training: Frozen (feature extraction only)                     |
|     - Stage 2 Fine-tuning: Top 30 layers unfrozen (LR: 1e-5)                |
|   • GlobalAveragePooling2D                                                   |
|   • Dropout (0.3)                                                            |
|   • Dense (num_classes) + Softmax                                            |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                        Streamlit Web UI (app.py)                             |
|   • Real-time predictions with confidence scores                             |
|   • Top-k class rankings with interactive bar charts                         |
|   • Performance metrics: accuracy, precision, recall, F1                     |
|   • Confusion matrix and training curves visualization                       |
|   • Downloadable JSON audit reports with metadata                            |
+------------------------------------------------------------------------------+
```

---

## Quick Start

Use **Python 3.10 to 3.13**. The application runs on Windows, macOS, and Linux.

### Windows (PowerShell)
```powershell
# Navigate to the repository
cd "ML mini Project"

# Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\Activate.ps1

# Install dependencies
python -m pip install -r requirements.txt

# Run the Streamlit application
python -m streamlit run app.py
```

### macOS / Linux (Bash)
```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open **http://localhost:8501** in your browser to interact with the dashboard.

---

## Training Pipeline

The model is trained in two stages using Google Colab (T4 GPU recommended):

### Stage 1: Head Training (Frozen Base)
- **Duration**: 5-10 epochs
- **Strategy**: MobileNetV2 base frozen, train only classification head
- **Learning Rate**: 1e-3 (Adam optimizer)
- **Purpose**: Stabilize randomly initialized classifier before fine-tuning

### Stage 2: Fine-Tuning (Top 30 Layers)
- **Duration**: 5-10 epochs
- **Strategy**: Unfreeze top 30 MobileNetV2 layers (keep BatchNorm frozen)
- **Learning Rate**: 1e-5 (Adam optimizer)
- **Purpose**: Adapt high-level features to clothing-specific textures/patterns

### Training Commands

```bash
# Prepare dataset (requires Kaggle API credentials)
python scripts/prepare_data.py --output-dir data/clothing_prepared

# Train model (use Google Colab for GPU acceleration)
python scripts/train_model.py \
    --data-dir data/clothing_prepared \
    --epochs-head 10 \
    --epochs-finetune 10 \
    --batch-size 32

# Evaluate on test split
python scripts/evaluate_model.py --data-dir data/clothing_prepared

# Smoke test inference
python scripts/check_inference.py
```

---

## Input Preprocessing

| Input Format | Handling | Output |
| :--- | :--- | :--- |
| **RGB JPEG/PNG/WebP** | Direct resize to 224×224 | `(1, 224, 224, 3)` float32 |
| **RGBA (with alpha)** | Flatten alpha channel → RGB | `(1, 224, 224, 3)` float32 |
| **Grayscale** | Convert to 3-channel RGB | `(1, 224, 224, 3)` float32 |
| **EXIF rotated (smartphone)** | Auto-transpose to correct orientation | `(1, 224, 224, 3)` float32 |
| **Any resolution** | LANCZOS downsample to 224×224 | `(1, 224, 224, 3)` float32 |

**Pixel Range**: [0, 255] → Model's internal Rescaling layer converts to [-1.0, 1.0]

---

## Project Structure

```
FashionVision/
├── .gitattributes              # Git line ending normalization & binary attributes
├── .gitignore                  # Python, venv, cache, and data exclusion rules
├── .streamlit/
│   └── config.toml             # Streamlit theming and server configuration
├── README.md                   # Project documentation (this file)
├── app.py                      # Interactive Streamlit web application
├── requirements.txt            # Core production dependencies
├── requirements-lock.txt       # Pinned dependency versions for reproducibility
├── docs/                       # Technical documentation
│   ├── ARCHITECTURE.md         # System architecture and design decisions
│   ├── EXPERIMENTS.md          # Training experiments and hyperparameter tuning
│   └── REPORT_OUTLINE.md       # Academic report structure
├── models/                     # Trained model artifacts
│   ├── real_model.keras        # MobileNetV2 transfer learning model weights
│   ├── class_names.json        # Class labels (dress, shirt, t-shirt)
│   └── metadata.json           # Training metadata (hyperparams, TF version, etc.)
├── notebooks/                  # Jupyter/Colab training notebooks
│   ├── colab_train_v2.ipynb    # Complete v2 training pipeline for Colab
│   └── ...                     # Legacy notebooks
├── results/                    # Evaluation outputs and visualizations
│   ├── metrics.json            # Test accuracy, precision, recall, F1
│   ├── classification_report.json # Per-class performance breakdown
│   ├── history.json            # Training history (loss & accuracy curves)
│   ├── confusion_matrix.png    # Confusion matrix visualization
│   ├── misclassified_examples.png # Grid of misclassified test images
│   └── training_curves.png     # Training & validation curves plot
├── scripts/                    # Command-line training and evaluation tools
│   ├── prepare_data.py         # Download & prepare clothing dataset from Kaggle
│   ├── train_model.py          # Two-stage transfer learning training script
│   ├── evaluate_model.py       # Test set evaluation with metrics & plots
│   ├── check_inference.py      # Inference smoke test on sample images
│   └── verify_project.py       # Automated project integrity checker
├── src/                        # Core Python package
│   ├── __init__.py             # Package marker
│   ├── data.py                 # Dataset loading and preprocessing utilities
│   ├── model.py                # MobileNetV2 model architecture definition
│   ├── inference.py            # Image preprocessing and prediction functions
│   └── reporting.py            # Metrics computation and plot generation
├── submission/                 # Academic submission materials
│   ├── SETUP.md                # Environment setup guide
│   ├── DEMO_GUIDE.md           # Demo walkthrough for evaluation
│   ├── PROJECT_REPORT.md       # Formal project report
│   └── verification.json       # Verification status
└── tests/                      # Automated test suite
    ├── __init__.py             # Test package marker
    ├── test_preprocessing.py   # Preprocessing pipeline tests
    └── test_inference.py       # Inference function tests
```

---

## Verification & Testing

Run automated tests to verify system integrity:

```bash
# Run unit & integration tests
python -m pytest -v

# Run comprehensive project verification
python scripts/verify_project.py
```

---

## Key Technical Decisions

### Why MobileNetV2?
- **Efficiency**: 3.5M parameters, lightweight enough for CPU inference
- **Performance**: Strong ImageNet pretrained features transfer well to clothing
- **Mobile-Ready**: Designed for mobile/edge deployment scenarios

### Why Two-Stage Training?
- **Stage 1**: Prevents randomly initialized classifier from destroying pretrained features through large gradients
- **Stage 2**: Fine-tunes high-level features to recognize clothing-specific patterns (collars, textures, hemlines)

### Why Keep BatchNorm Frozen?
- BatchNorm layers track running statistics (mean/variance). Updating these on small fine-tuning batches destabilizes learned representations
- Only convolutional/dense weights are fine-tuned in Stage 2

### Why In-Graph Normalization?
- Embedding `Rescaling(1/127.5, -1)` inside the saved model guarantees identical preprocessing between training and inference
- Eliminates common deployment bugs from preprocessing mismatches

---

## Limitations & Future Work

### Current Limitations
- **Limited Classes**: Currently trained on 3 classes (dress, shirt, t-shirt) from small dataset
- **Dataset Size**: Small training set (~40 images) limits generalization
- **Background Sensitivity**: Works best with plain backgrounds; complex scenes may confuse the model
- **Domain Gap**: ImageNet→Clothing domain gap means some low-level features may not be optimal

### Future Improvements
- Expand to 10+ clothing categories (jacket, pants, shoes, accessories)
- Train on larger datasets (DeepFashion, Fashion200K, iMaterialist)
- Add multi-label support (e.g., "long-sleeve shirt with buttons")
- Implement attention mechanisms for fine-grained feature localization
- Add explainability (Grad-CAM visualizations)
- Deploy as REST API or mobile application

---

## Citation & Acknowledgments

**Dataset**: Clothing Dataset by Alexey Grigorev  
Source: [kaggle.com/datasets/agrigorev/clothing-dataset-full](https://www.kaggle.com/datasets/agrigorev/clothing-dataset-full)

**Base Model**: MobileNetV2 (ImageNet pretrained)  
Reference: Sandler et al. (2018). "MobileNetV2: Inverted Residuals and Linear Bottlenecks"

---

## License

This project is for educational purposes. Model weights and code are provided as-is.

For academic submissions, refer to [submission/PROJECT_REPORT.md](submission/PROJECT_REPORT.md) and [submission/DEMO_GUIDE.md](submission/DEMO_GUIDE.md).
