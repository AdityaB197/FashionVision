# FashionVision — Apparel Classification with CNNs & Transfer Learning

An educational machine learning system for classifying clothing articles and fashion apparel. Features a lightweight from-scratch Convolutional Neural Network (CNN) trained on **Fashion-MNIST** (achieving **91.61% test accuracy**), a **MobileNetV2** transfer learning pipeline for real-world clothing photography, a reactive **Streamlit web application**, photographic color-inversion preprocessing, and exportable JSON prediction reports.

---

## Features

- **10-Class Fashion Classification**: Real-time categorization across 10 clothing categories with confidence percentages and top-3 ranked probability distributions.
- **Empirical Accuracy**: **91.61% accuracy** and **91.58% macro F1-score** evaluated on the standard 10,000-image held-out test split.
- **Photographic Inversion Preprocessing**: Instant color inversion (`ImageOps.invert`) adapting natural dark-clothes-on-white-background smartphone photos to Fashion-MNIST's light-on-dark convention.
- **Robust Downsampling & Orientation**: `ImageOps.exif_transpose` orientation correction and 8-lobed `LANCZOS` downsampling to preserve sharp fabric boundaries.
- **Dual Architecture Support**:
  - **FashionVision v1**: Lightweight custom CNN (225k parameters, ~879 KB) trained from scratch.
  - **FashionVision v2**: Deep transfer learning with MobileNetV2 pretrained on ImageNet, fine-tuned in two stages for real-world clothing photos.
- **Reactive Streamlit Dashboard (`app.py`)**:
  - Upload custom JPEG/PNG/WebP images or pick from curated test samples.
  - Side-by-side original vs preprocessed tensor visualizer.
  - Interactive top-3 probability bar charts.
  - Model Performance tab with confusion matrices, training loss/accuracy curves, and misclassified example galleries.
  - Downloadable JSON prediction reports with timestamps, input parameters, and model provenance.
- **Production QA**: 11 unit/integration tests with `pytest` and a 25-point automated verification script (`scripts/verify_project.py`).

---

## Architecture Overview

```
+----------------------------------------------------------------------------------+
|                                User Input Image                                  |
|         (Upload: JPEG/PNG/WebP, or pre-bundled test-set / synthetic sample)      |
+----------------------------------------------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            Preprocessing Pipeline                                |
|   - Grayscale conversion (v1) / 3-Channel RGB (v2)                               |
|   - ImageOps.exif_transpose (fixes smartphone photo rotations)                   |
|   - Optional Inversion (maps dark-clothes-on-white-bg -> white-on-dark)          |
|   - High-quality PIL LANCZOS downsampling (28x28 for v1, 224x224 for v2)         |
+----------------------------------------------------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+---------------------------------------+ +---------------------------------------+
|          v1 CNN Model                 | |      v2 MobileNetV2 Model             |
|  - Conv2D (32, 3x3) + ReLU            | |  - Training Augmentation Layers       |
|  - MaxPooling2D (2x2)                 | |  - In-Graph Rescaling ([-1.0, 1.0])   |
|  - Conv2D (64, 3x3) + ReLU            | |  - MobileNetV2 Backbone (Pretrained)  |
|  - MaxPooling2D (2x2)                 | |    (Stage 1: Frozen; Stage 2: Top 30) |
|  - Dropout (0.25)                     | |  - GlobalAveragePooling2D             |
|  - Flatten                            | |  - Dropout (0.30)                     |
|  - Dense (128) + ReLU                 | |  - Dense (C) + Softmax                |
|  - Dropout (0.50)                     | +---------------------------------------+
|  - Dense (10) + Softmax               |                      |
+---------------------------------------+                      |
                     |                                         |
                     +--------------------+--------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            Streamlit Web UI (app.py)                             |
|   - Real-time top-3 class rankings with confidence percentages                   |
|   - Interactive probability distribution bar charts                              |
|   - Model Performance & Validation audit tab                                     |
|   - Downloadable audit reports (JSON) with full provenance metadata              |
+----------------------------------------------------------------------------------+
```

---

## Quick Start

Use **Python 3.10 to 3.13**. The application runs on Windows, macOS, and Linux.

### Windows (PowerShell)
```powershell
# Clone or navigate to the repository
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

## Supported Inputs & Preprocessing

| Input Type | Supported Models | Preprocessing Transformation | Normalization | Invert Available? |
| :--- | :--- | :--- | :--- | :---: |
| **Grayscale 28×28 scan** | FashionVision v1 | Grayscale Mode "L" -> shape `(1, 28, 28, 1)` | `/ 255.0` -> `[0.0, 1.0]` | Yes |
| **RGB/RGBA smartphone photo** | FashionVision v1 & v2 | EXIF transpose -> RGB/L -> LANCZOS downsampling | In-graph or `/ 255.0` | Yes (Recommended for dark-on-white) |
| **High-resolution clothing photo** | FashionVision v2 | EXIF transpose -> RGB Mode -> `224×224` | In-graph `[-1.0, 1.0]` | No (Natural RGB) |

---

## Measured Experimental Results

All quantitative metrics are evaluated on the standard held-out **10,000-image Fashion-MNIST test split** (1,000 images per class):

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **91.61%** |
| **Macro Precision** | **91.58%** |
| **Macro Recall** | **91.61%** |
| **Macro F1-Score** | **91.58%** |
| **Weighted F1-Score** | **91.58%** |

### Per-Class Evaluation Breakdown

| Category | Precision | Recall | F1-Score | Test Samples | Assessment |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Trouser** | 99.59% | 97.60% | **98.59%** | 1,000 | Near-perfect separation |
| **Sandal** | 98.79% | 98.00% | **98.39%** | 1,000 | High geometric distinction |
| **Bag** | 98.29% | 97.70% | **97.99%** | 1,000 | Distinctive handle/box geometry |
| **Ankle boot** | 96.62% | 97.20% | **96.91%** | 1,000 | Strong shaft/heel features |
| **Sneaker** | 96.22% | 96.60% | **96.41%** | 1,000 | Strong sole/profile features |
| **Dress** | 90.40% | 93.20% | **91.78%** | 1,000 | High contour recognition |
| **Pullover** | 87.10% | 87.10% | **87.10%** | 1,000 | Solid recognition |
| **Coat** | 85.91% | 87.80% | **86.84%** | 1,000 | Solid recognition |
| **T-shirt/top** | 85.26% | 87.90% | **86.56%** | 1,000 | Moderate collar confusion |
| **Shirt** | 77.66% | 73.00% | **75.26%** | 1,000 | Subtle button/collar ambiguity |

See [detailed experiment logs](docs/EXPERIMENTS.md), [system architecture notes](docs/ARCHITECTURE.md), and the [formal project report](submission/PROJECT_REPORT.md).

---

## Project Structure

```
FashionVision/
├── .gitattributes              # Git line ending normalization & binary attributes
├── .gitignore                  # Python, venv, cache, and data exclusion rules
├── .streamlit/
│   └── config.toml             # Streamlit branding, colors, and server settings
├── README.md                   # Comprehensive project documentation
├── app.py                      # Interactive Streamlit dashboard
├── requirements.txt            # Core production dependencies
├── requirements-lock.txt       # Pinned environment reproduction dependencies
├── docs/                       # Technical documentation & experimental logs
│   ├── ARCHITECTURE.md         # Layer-by-layer topologies & data flow
│   ├── EXPERIMENTS.md          # Quantitative evaluations & error diagnosis
│   └── REPORT_OUTLINE.md       # Academic mini-project report syllabus outline
├── models/                     # v1 Trained artifacts (Fashion-MNIST)
│   ├── fashion_cnn.keras       # Trained Keras CNN model weights (91.61% acc)
│   ├── class_names.json        # 10 Fashion-MNIST category labels
│   └── metadata.json           # Training hyperparameters and GPU metadata
├── models_v2/                  # v2 Artifacts (MobileNetV2 real clothing classifier)
├── notebooks/                  # Interactive notebooks
│   ├── colab_training.ipynb    # Google Colab GPU training notebook
│   └── colab_train.ipynb       # Colab training reference
├── results/                    # v1 Evaluation plots and metric files
│   ├── metrics.json            # Overall accuracy, macro/weighted F1
│   ├── classification_report.json # Per-class precision, recall, support
│   ├── history.json            # 20-epoch loss and accuracy curves
│   ├── confusion_matrix.png    # 10x10 normalized confusion matrix plot
│   ├── misclassified_examples.png # 4x4 visual gallery of error cases
│   └── training_curves.png     # Train vs validation loss & accuracy
├── results_v2/                 # v2 Evaluation outputs and training curves
├── scripts/                    # Command-line pipelines & tools
│   ├── train_model.py          # Trains v1 CNN from scratch
│   ├── evaluate_model.py       # Evaluates v1 model on test split
│   ├── check_inference.py      # v1 sanity check on 20 test samples
│   ├── prepare_real_data.py    # v2 Kaggle dataset download & group split
│   ├── train_real.py           # v2 two-stage transfer learning pipeline
│   ├── evaluate_real.py        # v2 evaluation script
│   ├── check_real_inference.py # v2 inference sanity check
│   └── verify_project.py       # Automated 25-point system integrity verifier
├── src/                        # Core Python package
│   ├── __init__.py             # Package marker
│   ├── data.py                 # Fashion-MNIST raw loading & preprocessing
│   ├── model.py                # v1 CNN Keras model definition
│   ├── inference.py            # v1 image preparation & top-k inference
│   ├── reporting.py            # Metrics, plots, and report generation helpers
│   ├── real_data.py            # v2 dataset loader & class weights
│   ├── real_model.py           # v2 MobileNetV2 architecture & unfreezing
│   └── real_inference.py       # v2 photographic image prep & prediction
├── submission/                 # Academic viva and project submission package
│   ├── SETUP.md                # Environment setup & reproduction guide
│   ├── DEMO_GUIDE.md           # 5-minute walkthrough & viva defense Q&A
│   ├── PROJECT_REPORT.md       # Formal academic mini-project report
│   └── verification.json       # System verification status artifact
└── tests/                      # Automated unit and integration test suite
    ├── __init__.py             # Test package marker
    ├── test_preprocessing.py   # 5 preprocessing & boundary tests
    └── test_inference.py       # 6 inference, mock, & integration tests
```

---

## Verification & Testing

Verify that all unit tests and system integrity checks pass:

```bash
# Run 11 unit & integration tests
python -m pytest -v

# Run 25-point repository integrity verifier
python scripts/verify_project.py
```

---

## Demonstration & Academic Defense

For university evaluations, demonstrations, and viva examinations, refer to the [Demonstration & Viva Guide](submission/DEMO_GUIDE.md) and the [Formal Project Report](submission/PROJECT_REPORT.md).
