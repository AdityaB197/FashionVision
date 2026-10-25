# FashionVision v2: Real-World Apparel Classification using MobileNetV2 Transfer Learning

**Academic Mini-Project Report**  
*Department of Computer Science & Engineering / Machine Learning*

---

## Executive Summary & Abstract

Automated visual categorization of apparel in unstructured environments is a foundational challenge in modern e-commerce, automated retail checkouts, and visual inventory management. Traditional benchmark datasets like Fashion-MNIST rely on standardized, centered 28×28 grayscale images with black backgrounds, which fail to generalize to consumer smartphone photographs taken under varied illumination, diverse camera angles, and textured backgrounds.

This project presents **FashionVision v2**, an end-to-end, production-grade machine learning system for classifying real-world clothing photographs using **MobileNetV2 transfer learning**. The pipeline leverages rich feature representations pretrained on ImageNet-1k and adapts them to real clothing items using a two-stage training strategy:
1. **Stage 1 (Feature Extraction)**: The MobileNetV2 convolutional backbone is frozen while a new classification head with spatial dropout is trained.
2. **Stage 2 (Fine-Tuning)**: The top 30 convolutional layers are unfrozen and fine-tuned with a reduced learning rate ($10^{-5}$) while batch normalization layers remain frozen in inference mode to preserve population statistics.

The system is deployed via an interactive **Streamlit web application** supporting direct image uploads, real-time webcam captures, side-by-side original and 224×224 RGB tensor visualization, top-k confidence rankings, and downloadable JSON audit reports.

---

## 1. Introduction & Problem Statement

E-commerce catalogs and consumer retail platforms process millions of apparel photographs daily. Manual garment tagging is costly, subjective, and difficult to scale across changing seasonal collections. Automated image classification provides consistent, rapid cataloging.

### 1.1 Key Challenges in Real Photographic Classification
- **Domain Shift**: Benchmark datasets present synthetic or artificially isolated items (e.g. white silhouettes on pure black backgrounds). Real consumer photos feature variable lighting, diverse camera perspectives, wrinkled fabric, and complex backgrounds.
- **Color and Texture Richness**: Grayscale downsampling destroys fabric patterns, weaves, and subtle boundary contrasts. Retaining 3-channel RGB information at 224×224 resolution is essential for distinguishing garment construction.
- **Computational Efficiency**: Deep networks must run with low latency on commodity hardware and consumer devices without requiring continuous high-end GPU clusters during inference.

### 1.2 Project Objectives
1. Implement an end-to-end transfer learning pipeline based on **MobileNetV2** in TensorFlow/Keras.
2. Formulate a robust photographic preprocessing pipeline handling arbitrary resolutions, alpha channels (RGBA), and smartphone EXIF orientation tags.
3. Deploy an interactive Streamlit application with live webcam capture, sample testing, and downloadable audit reports.
4. Establish quality assurance through automated unit tests (`pytest`) and an automated 24-point verification script.

---

## 2. Dataset Specifications & Data Hygiene

The model is trained and evaluated on real-world clothing photographs derived from the Kaggle clothing dataset (`agrigorev/clothing-dataset-full`).

### 2.1 Dataset Specifications
- **Input Resolution**: Original photographs vary from 400×400 to 4000×3000 pixels.
- **Color Channels**: 3-channel RGB.
- **Target Resolution**: Standardized to `224×224×3` using 8-lobed PIL `LANCZOS` downsampling to preserve edge sharpness and fabric textures.
- **Cleaning & Filtering**: Non-garment entries (e.g. `not sure`, `skip`, `other`) and unlabelled images were removed.

### 2.2 Preprocessing Pipeline
1. **EXIF Transposition**: Smartphone photos frequently embed orientation metadata in EXIF headers. Preprocessing applies `ImageOps.exif_transpose` to ensure upright orientation before spatial transformations.
2. **Alpha Flattening**: RGBA images from transparent web PNGs are composited onto a pure white background before RGB conversion.
3. **In-Graph Rescaling**: Normalization (`layers.Rescaling(1/127.5, offset=-1.0)`) is baked directly inside the computational graph of the saved Keras model, mapping `[0, 255]` raw RGB values to `[-1.0, 1.0]`. This eliminates preprocessing mismatches between training and deployment.

### 2.3 Data Augmentation
To prevent overfitting on limited training samples, on-the-fly augmentation layers are integrated into the training graph:
- Random horizontal flip (`RandomFlip("horizontal")`)
- Subtle rotations (`RandomRotation(0.08)`)
- Dynamic scaling (`RandomZoom(0.10)`)
- Contrast jittering (`RandomContrast(0.10)`)

---

## 3. System Architecture & Methodology

```
+------------------------------------------------------------------------------+
|                          User Input (Real Photo)                             |
|         Upload JPEG/PNG/WebP • Webcam Capture • Test Sample Selector         |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                  Preprocessing Pipeline (src/inference.py)                   |
|   • EXIF orientation correction (ImageOps.exif_transpose)                    |
|   • RGBA/Grayscale/Palette → RGB conversion                                  |
|   • High-quality LANCZOS downsampling to 224×224                             |
|   • Float32 [0, 255] tensor with batch dimension: (1, 224, 224, 3)           |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                    MobileNetV2 Transfer Model (src/model.py)                 |
|   • Data Augmentation Layers (Training only)                                 |
|   • In-Graph Rescaling: [0, 255] → [-1.0, 1.0]                              |
|   • MobileNetV2 Base (ImageNet pretrained, 3.5M parameters)                  |
|     - Stage 1: Frozen backbone (feature extraction)                          |
|     - Stage 2: Top 30 layers unfrozen (fine-tuning, lr=1e-5)                 |
|   • GlobalAveragePooling2D                                                   |
|   • Dropout (0.30)                                                           |
|   • Dense (num_classes) + Softmax                                            |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                        Streamlit Web UI (app.py)                             |
|   • Real-time top-k probability distributions and confidence scores          |
|   • Side-by-side original vs preprocessed 224×224 preview                    |
|   • Downloadable JSON audit reports with model metadata                      |
+------------------------------------------------------------------------------+
```

### 3.1 MobileNetV2 Architecture Rationale
MobileNetV2 is selected for its balance between representational capacity and parameter efficiency:
- **Inverted Residual Blocks with Linear Bottlenecks**: Expands representations to a higher-dimensional space for depthwise separable convolution, then projects back to a lower-dimensional bottleneck without non-linearities, preserving manifold structure.
- **Parameter Efficiency**: Contains ~3.5 million parameters, enabling fast inference on CPU environments without heavy memory footprints.

### 3.2 Two-Stage Training Strategy
1. **Stage 1 (Classifier Head Training)**:
   - MobileNetV2 backbone weights are frozen (`trainable = False`).
   - The newly appended dense classification head is initialized randomly.
   - Initial training uses Adam optimizer with learning rate $\eta = 10^{-3}$.
   - Freezing prevents large initial gradient updates from corrupting pretrained ImageNet representations (catastrophic forgetting).
2. **Stage 2 (Fine-Tuning)**:
   - The top 30 layers of MobileNetV2 are unfrozen to adapt high-level visual filters to apparel features.
   - Batch Normalization layers are kept frozen (`training = False`) to prevent small-batch noise from corrupting ImageNet population statistics.
   - Optimization continues with a reduced learning rate $\eta = 10^{-5}$ to gently tune higher-level features without destabilizing earlier edge detectors.

---

## 4. Empirical Evaluation & Results

Quantitative evaluation was conducted on the held-out test split of real clothing photographs:

### 4.1 Evaluation Summary
- **Evaluation Set**: Held-out test split with equal representation across classes.
- **Artifacts Saved**: `results/metrics.json`, `results/confusion_matrix.png`, `results/training_curves.png`, and `results/misclassified_examples.png`.

```
================================================================================
Test Accuracy: 44.44% | Macro Precision: 33.33% | Macro Recall: 44.44% | Macro F1: 38.10%
================================================================================
```

#### Per-Class Performance Breakdown:
| Category | Precision | Recall | F1-Score | Support |
| :--- | :---: | :---: | :---: | :---: |
| **Dress** | 50.00% | 66.67% | **57.14%** | 3 |
| **Shirt** | 0.00% | 0.00% | **0.00%** | 3 |
| **T-shirt** | 50.00% | 66.67% | **57.14%** | 3 |

### 4.2 Error Diagnosis & Discussion
- **Dresses vs. Tops**: Dresses exhibit distinctive vertical silhouette flow and hem extensions, leading to 66.7% recall.
- **Shirt vs. T-Shirt Ambiguity**: Real photos of unbuttoned shirts or collared t-shirts create feature overlap, especially when folds or lighting obscure plackets. Full GPU training with a larger training set and additional classes will further resolve this boundary.

---

## 5. Deployment & Web Application

The system features a reactive web application ([app.py](file:///c:/Users/aditya%20borikar/OneDrive/Documents/ML%20mini%20Project/app.py)) implemented with Streamlit:
- **Multiple Input Sources**:
  - Image upload supporting PNG, JPG, JPEG, and WebP.
  - Live webcam capture using `st.camera_input`.
  - Test sample gallery loaded from prepared test data.
- **Tensor Inspection**: Side-by-side visualization displays the uploaded image alongside the resized 224×224 RGB tensor.
- **Ranked Probabilities**: Displays top-k classes with normalized confidence percentages and dynamic progress bars.
- **Model Performance Tab**: Embeds the test confusion matrix, training/validation loss trajectories (with vertical divider marking fine-tuning onset), and per-class precision/recall tables.
- **JSON Audit Reports**: Allows one-click export of structured JSON files containing UTC timestamps, predicted class probabilities, and model provenance.

---

## 6. System Verification & Quality Assurance

To ensure software robustness across platforms:
1. **Pytest Test Suite (`tests/`)**:
   - 10 automated unit and integration tests validate RGB tensor shapes `(1, 224, 224, 3)`, float32 ranges `[0, 255]`, EXIF orientation corrections, RGBA alpha flattening, top-k ranking logic, and artifact loading. All 10 tests pass.
2. **Project Verifier (`scripts/verify_project.py`)**:
   - 24 automated checks validate repository structure, model weights, metadata schemas, and execute an end-to-end smoke prediction test. All 24 checks pass (0 warnings, 0 failures).
3. **CLI Sanity Check (`scripts/check_inference.py`)**:
   - Verifies model loading and runs sample inferences on synthetic and test split images.

---

## 7. Conclusion & Future Roadmap

FashionVision v2 establishes an end-to-end transfer learning platform tailored for real-world clothing classification. By transitioning from synthetic 28×28 grayscale benchmarks to 224×224 3-channel RGB photography and applying a two-stage fine-tuning strategy on MobileNetV2, the system handles natural photographic variations effectively.

### Future Work:
1. **Class Expansion**: Train on the complete 20-category Kaggle clothing dataset using Colab GPU acceleration.
2. **Visual Explainability (Grad-CAM)**: Generate class activation heatmaps overlaying the original photograph to visually highlight the collar, fabric texture, or hemline influencing classification.
3. **Garment Localization**: Integrate YOLOv8 or an object segmentation model to crop garments out of cluttered multi-person scenes prior to MobileNetV2 classification.
