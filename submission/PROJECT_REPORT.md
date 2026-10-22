# FashionVision: Apparel Classification using Convolutional Neural Networks and Transfer Learning

**Academic Mini-Project Report**  
*Department of Computer Science & Engineering / Machine Learning*

---

## Executive Summary & Abstract

Automated visual categorization of apparel is a fundamental challenge in retail automation, visual search, and inventory logistics. This project presents **FashionVision**, an end-to-end machine learning system integrating two complementary modeling strategies:

1. **FashionVision v1 (From-Scratch CNN)**: A lightweight, custom Convolutional Neural Network developed and trained on the 70,000-image **Fashion-MNIST** dataset (28×28 grayscale). The architecture achieves **91.61% overall accuracy** and a **91.58% macro F1-score** across 10 balanced categories on the held-out 10,000-sample test set.
2. **FashionVision v2 (Transfer Learning)**: A deep transfer learning pipeline utilizing **MobileNetV2** pretrained on ImageNet-1k, adapted via a two-stage training strategy (classifier head training followed by fine-tuning of the top 30 layers with frozen batch normalization) on real-world clothing photographs from the Kaggle clothing dataset.

The system is deployed via an interactive, browser-based **Streamlit web application** featuring real-time image transformation previews, dynamic top-3 confidence rankings, color inversion for smartphone photographs, and downloadable structured JSON audit reports.

---

## 1. Introduction & Problem Statement

Modern e-commerce and retail supply chains process millions of apparel images daily. Manual categorization is labor-intensive, slow, and error-prone due to subtle stylistic boundaries. Automated image classification provides rapid, standardized cataloging.

### 1.1 Challenges
- **Low-Resolution Ambiguity**: Under constrained 28×28 resolutions, garments within the upper-body family (Shirts, T-shirts, Pullovers, Coats) exhibit overlapping torso silhouettes where collar and button details are lost.
- **Domain Shift in Real Photography**: Standard benchmarks present standardized white clothing on black backgrounds, whereas consumer smartphone photographs present dark garments against white or cluttered backgrounds with unpredictable lighting and camera orientations.

### 1.2 Project Objectives
1. Implement a clean, reproducible CNN pipeline from scratch in TensorFlow/Keras without data leakage.
2. Formulate a transfer learning strategy to handle real-world photographic inputs.
3. Engineer a user-friendly, responsive Streamlit dashboard for real-time inference and performance audit.
4. Provide comprehensive unit testing, system verification, and documentation.

---

## 2. Dataset Specifications & Data Hygiene

### 2.1 Dataset 1: Fashion-MNIST
- **Total Images**: 70,000 images, 28×28 pixels, 1 grayscale channel.
- **Classes (10 Categories)**: `T-shirt/top`, `Trouser`, `Pullover`, `Dress`, `Coat`, `Sandal`, `Shirt`, `Sneaker`, `Bag`, `Ankle boot`.
- **Partitioning**:
  - Training: 54,000 samples (stratified, seed = 42).
  - Validation: 6,000 samples (stratified, seed = 42).
  - Test: 10,000 samples (held-out benchmark split, exactly 1,000 samples per class).
- **Preprocessing**: Values scaled from `[0, 255]` uint8 to `[0.0, 1.0]` float32 and expanded to tensor shape `(N, 28, 28, 1)`.

### 2.2 Dataset 2: Real Clothing Dataset (`agrigorev/clothing-dataset-full`)
- **Total Images**: ~5,000 high-resolution user-contributed clothing photographs.
- **Data Cleaning**: Filtered non-clothing entries (`not sure`, `other`, `skip`) and categories with fewer than 100 samples.
- **Group-Aware Splitting**: Partitioned 70% train / 15% val / 15% test grouped by contributor `sender_id`. This guarantees that photographs from the same user/camera setup never appear across splits, preventing background leakage.
- **Standardization**: Orientation corrected via EXIF transpose and downsampled to `224×224` RGB JPEG using LANCZOS interpolation.

---

## 3. System Architecture & Methodology

### 3.1 FashionVision v1 Architecture (Custom CNN)
The baseline model processes 28×28 grayscale images through a two-block convolutional feature extractor followed by a dense classification head:

$$\text{Input}(28, 28, 1) \rightarrow \text{Conv2D}(32, 3\times3, \text{ReLU}) \rightarrow \text{MaxPool}(2\times2) \rightarrow \text{Conv2D}(64, 3\times3, \text{ReLU}) \rightarrow \text{MaxPool}(2\times2) \rightarrow \text{Dropout}(0.25) \rightarrow \text{Flatten} \rightarrow \text{Dense}(128, \text{ReLU}) \rightarrow \text{Dropout}(0.50) \rightarrow \text{Dense}(10, \text{Softmax})$$

- **Total Trainable Parameters**: 225,034 (879 KB).
- **Regularization Strategy**: Spatial dropout (0.25) after pooling blocks prevents feature co-adaptation; dense dropout (0.50) prevents dense weight memorization.

### 3.2 FashionVision v2 Architecture (MobileNetV2 Transfer Learning)
For real photographic classification:
- **In-Graph Rescaling**: `layers.Rescaling(scale=1/127.5, offset=-1.0)` is integrated directly inside the saved model, mapping raw `[0, 255]` RGB pixels to `[-1.0, 1.0]` automatically.
- **Data Augmentation**: `RandomFlip`, `RandomRotation(0.1)`, `RandomZoom(0.15)`, `RandomContrast(0.1)`, and `RandomBrightness(0.1)` simulate real-world photographic variability.
- **Two-Stage Training**:
  - *Stage 1*: Backbone frozen, classifier head trained with Adam ($10^{-3}$).
  - *Stage 2*: Top 30 layers unfrozen, batch normalization frozen in inference mode, trained with Adam ($10^{-5}$).

---

## 4. Empirical Evaluation & Results

### 4.1 Fashion-MNIST Benchmark Results (10,000 Test Images)

```
================================================================================
Test Accuracy: 91.61% | Macro Precision: 91.58% | Macro Recall: 91.61% | Macro F1: 91.58%
================================================================================
```

#### Per-Class Performance Table:

| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Trouser** | 99.59% | 97.60% | **98.59%** | 1,000 |
| **Sandal** | 98.79% | 98.00% | **98.39%** | 1,000 |
| **Bag** | 98.29% | 97.70% | **97.99%** | 1,000 |
| **Ankle boot** | 96.62% | 97.20% | **96.91%** | 1,000 |
| **Sneaker** | 96.22% | 96.60% | **96.41%** | 1,000 |
| **Dress** | 90.40% | 93.20% | **91.78%** | 1,000 |
| **Pullover** | 87.10% | 87.10% | **87.10%** | 1,000 |
| **Coat** | 85.91% | 87.80% | **86.84%** | 1,000 |
| **T-shirt/top** | 85.26% | 87.90% | **86.56%** | 1,000 |
| **Shirt** | 77.66% | 73.00% | **75.26%** | 1,000 |

### 4.2 Error Diagnosis & Confusion Analysis
- **Distinctive Geometry Classes**: Trousers, footwear (Sandals, Sneakers, Boots), and Bags achieve F1 scores exceeding 96.4%. Their physical silhouettes create unique spatial activations.
- **The Upper-Body Silhouette Dilemma**: `Shirt` achieves the lowest F1 score (75.26%). Over 11% of true Shirts are misclassified as `T-shirt/top` and 8% as `Pullover`. In a 28×28 grayscale raster, button plackets and collar folds are blurred, leaving torso outlines indistinguishable from plain T-shirts.

---

## 5. Deployment & Web Application

The system includes a reactive dashboard (`app.py`) built in Streamlit:
- **Interactive Inference**: Users can upload arbitrary image formats (JPEG, PNG, WebP) or select bundled samples.
- **Photographic Inversion**: A dedicated toggle flips dark-on-white smartphone photos to white-on-black, harmonizing input polarity with Fashion-MNIST.
- **Confidence Visualization**: Real-time horizontal bar charts illustrate top-3 predicted class likelihoods.
- **Performance Auditing**: Dedicated tabs visualize confusion matrices, training loss/accuracy trajectories, and misclassified example galleries directly from evaluated artifacts.
- **Immutable Report Generation**: Generates compliant, downloadable JSON reports containing input hashes, UTC timestamps, and prediction provenance.

---

## 6. System Verification & Quality Assurance

To ensure software integrity:
- **Pytest Suite (`tests/`)**: 11 unit tests validate tensor shapes, value ranges, boundary mapping, input immutability, color inversion math, mock model ranking, and artifact loading. All 11 tests pass.
- **Verification Script (`scripts/verify_project.py`)**: 25 automated checks verify presence of all 19 mandatory files/directories, documentation integrity, class name alignment, and end-to-end sample inference. All 25 checks pass (0 warnings, 0 failures).

---

## 7. Conclusion & Future Work

FashionVision demonstrates that a carefully regularized, two-layer CNN trained on Fashion-MNIST achieves 91.61% accuracy, while MobileNetV2 transfer learning effectively extends capabilities to real clothing photography.

### Future Work:
1. **Multi-Task Learning**: Extend models to predict garment color, fabric texture, and gender simultaneously.
2. **Object Detection**: Incorporate YOLOv8 to localize and crop individual garments within complex, multi-person lifestyle photography before classification.
3. **Edge Optimization**: Quantize models to TensorFlow Lite (INT8) for edge deployment on mobile devices.
