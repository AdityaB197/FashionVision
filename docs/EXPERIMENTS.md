# Experimental Evaluation & Empirical Results — FashionVision

This document details the experimental methodology, quantitative evaluations, per-class error breakdowns, and ablation studies conducted for the **FashionVision** project.

---

## 1. Experiment 1: Fashion-MNIST Baseline CNN (v1)

### 1.1 Experimental Configuration
- **Dataset**: Fashion-MNIST (Zalando Research)
- **Data Partitions**:
  - Training: 54,000 samples (stratified split with seed 42)
  - Validation: 6,000 samples (stratified split with seed 42)
  - Testing: 10,000 samples (held-out benchmark split, 1,000 per class)
- **Optimizer**: Adam (learning rate = 1e-3, beta_1 = 0.9, beta_2 = 0.999)
- **Loss Function**: Sparse Categorical Crossentropy
- **Batch Size**: 128
- **Epochs**: 20 (Best checkpoint saved at epoch 16 based on validation loss)
- **Compute Hardware**: NVIDIA Tesla T4 GPU (Google Colab runtime)
- **Total Training Duration**: 53.7 seconds

### 1.2 Quantitative Benchmark Results (Test Split: 10,000 Images)

| Metric | Score |
| :--- | :--- |
| **Overall Accuracy** | **91.61%** |
| **Macro Precision** | **91.58%** |
| **Macro Recall** | **91.61%** |
| **Macro F1-Score** | **91.58%** |
| **Weighted F1-Score** | **91.58%** |

### 1.3 Per-Class Performance Breakdown

| Class Index | Category | Test Support | Precision | Recall | F1-Score | Status |
| :---: | :--- | :---: | :---: | :---: | :---: | :--- |
| 0 | **T-shirt/top** | 1,000 | 85.26% | 87.90% | 86.56% | Good |
| 1 | **Trouser** | 1,000 | 99.59% | 97.60% | **98.59%** | Exceptional |
| 2 | **Pullover** | 1,000 | 87.10% | 87.10% | 87.10% | Good |
| 3 | **Dress** | 1,000 | 90.40% | 93.20% | 91.78% | High |
| 4 | **Coat** | 1,000 | 85.91% | 87.80% | 86.84% | Good |
| 5 | **Sandal** | 1,000 | 98.79% | 98.00% | **98.39%** | Exceptional |
| 6 | **Shirt** | 1,000 | 77.66% | 73.00% | **75.26%** | Most Challenging |
| 7 | **Sneaker** | 1,000 | 96.22% | 96.60% | 96.41% | High |
| 8 | **Bag** | 1,000 | 98.29% | 97.70% | **97.99%** | Exceptional |
| 9 | **Ankle boot** | 1,000 | 96.62% | 97.20% | 96.91% | High |

### 1.4 Detailed Confusion Matrix Analysis & Error Diagnosis

Analysis of `results/confusion_matrix.png` and `results/misclassified_examples.png` reveals distinct clusters:

1. **Top-Performing Classes (F1 > 96%)**:
   - **Trouser (98.59% F1)**: Distinctive elongated leg silhouette produces minimal false positives.
   - **Footwear (Sandal 98.39%, Ankle boot 96.91%, Sneaker 96.41%)**: Morphologically distinct from apparel. Confusion between Sandal and Sneaker is negligible (open vs closed toe).
   - **Bag (97.99% F1)**: Handles and rectangular geometry provide strong discriminative features.

2. **The "Upper-Body Apparel Cluster" (Shirt, T-shirt, Pullover, Coat)**:
   - **Shirt (75.26% F1)** exhibits the lowest score in the benchmark.
   - 11.2% of true Shirts are misclassified as T-shirt/top, 8.4% as Pullover, and 5.7% as Coat.
   - **Root Cause**: In 28×28 grayscale images, button plackets and collar creases occupy only 1–2 pixels and are blurred by downsampling. The broad torso outline of a Shirt closely mimics a T-shirt or Pullover when button details are absent.

---

## 2. Experiment 2: Transfer Learning on Real Clothing Photos (v2)

### 2.1 Problem Formulation & Dataset Preparation
To address real-world consumer photographs where standard 28×28 Fashion-MNIST models fail due to domain shift:
- **Dataset**: `agrigorev/clothing-dataset-full` (~5,000 high-resolution real clothing images).
- **Data Hygiene & Cleaning**:
  - Filtered non-clothing entries (`not sure`, `other`, `skip`).
  - Filtered under-represented categories (< 100 images).
  - Retained verified categories (e.g., dress, shirt, t-shirt, etc.).
- **Group-Aware Splitting**:
  - Split 70% train / 15% val / 15% test.
  - Grouped by `sender_id` to ensure photos from the same contributor/camera setup never appear in both train and test splits, preventing contributor leakage.
- **Image Standardization**:
  - EXIF orientation transpose applied.
  - LANCZOS downsampling to 224×224 RGB JPEG at 95 quality.

### 2.2 Model Architecture & Two-Stage Training Protocol
- **Backbone**: MobileNetV2 pretrained on ImageNet-1k (frozen during Stage 1).
- **Stage 1 (Head-Only)**:
  - Train newly initialized classification head (GlobalAveragePooling2D + Dropout(0.3) + Dense(num_classes)).
  - Optimizer: Adam (lr = 1e-3).
  - Early stopping on `val_loss` with `patience=4`.
- **Stage 2 (Fine-Tuning)**:
  - Top 30 layers of MobileNetV2 unfrozen; `BatchNormalization` layers kept frozen in inference mode.
  - Optimizer: Adam (conservative lr = 1e-5).
  - Early stopping and model checkpoint restoring best weights.

---

## 3. Comparative Architectural Benchmark

| Dimension | FashionVision v1 (Custom CNN) | FashionVision v2 (MobileNetV2) |
| :--- | :--- | :--- |
| **Target Domain** | Standardized 28×28 Grayscale | Real-World Photographic RGB (224×224) |
| **Pretrained Weights** | None (Trained from scratch) | ImageNet-1k Backbone |
| **Total Parameters** | 225,034 (~879 KB) | ~2.26 Million (~9.2 MB) |
| **Input Channels** | 1 (Grayscale) | 3 (RGB) |
| **In-Graph Preprocessing** | Standalone NumPy (`/ 255.0`) | Built-in `layers.Rescaling(1/127.5, offset=-1)` |
| **Data Augmentation** | None | Flip, Rotation, Zoom, Contrast, Brightness |
| **Primary Test Accuracy** | **91.61%** (Fashion-MNIST) | Transfer fine-tuned on real dataset |
| **Inference Latency (CPU)**| ~1.2 ms / sample | ~24.5 ms / sample |
| **Primary Deployment** | Lightweight educational baseline | Real-photo consumer garment classifier |

---

## 4. Key Takeaways & Practical Insights

1. **Domain Mismatch Requires Inversion**: Standard Fashion-MNIST represents dark backgrounds with white garments. Natural smartphone uploads feature white backgrounds with dark garments. The `invert=True` option in `src/inference.py` restores expected feature polarities.
2. **In-Graph Rescaling Eliminates Serving Errors**: Including `layers.Rescaling` inside the Keras functional model ensures inference accepts raw `[0, 255]` pixel arrays, guaranteeing zero serving-time preprocessing drift.
3. **Group-Aware Splitting Prevents Overoptimistic Metrics**: In user-contributed datasets, contributors shoot multiple garments against identical wall colors and lighting. Splitting by contributor groups prevents models from exploiting background correlations.
