# Project Report Outline — FashionVision

This outline defines the formal structure of the academic mini-project report for **FashionVision: Apparel Classification using Convolutional Neural Networks and Transfer Learning**.

---

## Section 1: Executive Summary & Abstract
- High-level overview of apparel classification in computer vision and e-commerce.
- Summary of the dual-model methodology (v1 custom CNN on Fashion-MNIST + v2 MobileNetV2 transfer learning on real photographs).
- Summary of empirical outcomes: 91.61% test accuracy on 10,000 Fashion-MNIST benchmark images, interactive Streamlit application, and automated audit reporting.

## Section 2: Introduction & Problem Statement
- Background of automated apparel tagging and inventory management.
- Problem Statement: Automated categorization of clothing items across standardized synthetic scans and high-resolution real photographs.
- Objectives:
  1. Build a high-precision, low-latency CNN classifier for standardized fashion items.
  2. Implement transfer learning to adapt deep ImageNet representations to real clothing photography.
  3. Develop an accessible web interface with live confidence rankings, image transformation visualization, and exportable reports.
  4. Ensure reproducible data splits and pipeline verification tests.

## Section 3: Literature Survey & Theoretical Foundations
- Evolution of computer vision in fashion: From handcrafted HOG/SIFT descriptors to deep CNNs.
- Theoretical foundations of Convolutional Layers: Local receptive fields, weight sharing, and translation invariance.
- Non-linear activations: Rectified Linear Units (ReLU) and vanishing gradient mitigation.
- Spatial dimension reduction: Max Pooling and feature abstraction.
- Regularization techniques: Dropout (Srivastava et al., 2014) and batch normalization.
- Transfer Learning: Feature reuse, domain adaptation, and freezing strategies.

## Section 4: Dataset Specifications & Data Hygiene
- **Dataset 1: Fashion-MNIST (Zalando Research)**
  - 70,000 28×28 grayscale images across 10 balanced categories.
  - Stratified 54k train / 6k validation / 10k test split.
  - Preprocessing pipeline: `[0, 255] -> [0.0, 1.0]` scaling, channel expansion `(N, 28, 28, 1)`, and dark-on-white photographic inversion.
- **Dataset 2: Real Clothing Dataset (`agrigorev/clothing-dataset-full`)**
  - ~5,000 real-world consumer photographs.
  - Data hygiene: Filtering invalid categories (`not sure`, `other`, `skip`) and under-represented classes.
  - Group-aware splitting by contributor `sender_id` (70/15/15) to prevent background data leakage.
  - Image standardization: EXIF orientation correction and LANCZOS downsampling to 224×224 RGB.

## Section 5: System Architecture & Proposed Methodology
- System block diagram and end-to-end data pipeline.
- **FashionVision v1 (Custom CNN)**:
  - Input: `(28, 28, 1)` -> `Conv2D(32, 3x3)` -> `MaxPool(2x2)` -> `Conv2D(64, 3x3)` -> `MaxPool(2x2)` -> `Dropout(0.25)` -> `Dense(128)` -> `Dropout(0.50)` -> `Dense(10, Softmax)`.
  - Total parameters: 225,034.
- **FashionVision v2 (MobileNetV2 Transfer Learning)**:
  - In-graph `Rescaling(1/127.5, offset=-1)` layer for unified serving.
  - Pretrained MobileNetV2 backbone with depthwise separable convolutions.
  - Two-stage training protocol: Head training (Adam lr=1e-3) followed by fine-tuning of top 30 layers with frozen BatchNorm (Adam lr=1e-5).

## Section 6: Experimental Setup & Training Protocols
- Hardware configurations: NVIDIA Tesla T4 GPU (Google Colab) and local CPU inference.
- Software stack: Python 3.13 / 3.11, TensorFlow 2.21, Scikit-learn, PIL, Streamlit, Pytest.
- Convergence tracking: Training curves, validation loss monitoring, and early stopping.

## Section 7: Empirical Results & Discussion
- Overall test metrics: Accuracy (91.61%), Macro Precision (91.58%), Macro Recall (91.61%), Macro F1 (91.58%).
- Per-class performance analysis:
  - High-performing classes: Trouser (98.59% F1), Sandal (98.39% F1), Bag (97.99% F1), Ankle boot (96.91% F1).
  - Challenging classes: Shirt (75.26% F1) vs T-shirt/top (86.56% F1) and Coat (86.84% F1).
- Confusion matrix insights and visual error diagnosis.

## Section 8: Web Application & Production Inference
- Reactive Streamlit UI (`app.py`):
  - Preprocessing preview (original vs preprocessed tensor).
  - Top-3 prediction confidence ranking bar charts.
  - Performance audit tabs with live rendering of confusion matrices and loss curves.
  - Automated JSON prediction report generator with SHA provenance metadata.

## Section 9: Failure Analysis & Limitations
- Morphological ambiguity in 28×28 grayscale images (subtle button/collar features blurred out).
- Real-photo domain shift: Background clutter, multi-garment outfits, lighting variances.
- Disclaimers on model confidence vs real-world ground truth.

## Section 10: Conclusion & Future Scope
- Summary of project achievements.
- Future enhancements: Multi-label classification (e.g., color + pattern + category), bounding box object detection (YOLOv8), and on-device quantization (TFLite).

## Section 11: Viva Voce & Demonstration Guide
- Curated list of typical examiner questions and model architecture defenses.

## Section 12: References
- Formal bibliography of papers and datasets (Fashion-MNIST, MobileNetV2, Adam, Dropout).
