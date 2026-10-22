# System Architecture — FashionVision

FashionVision is a production-grade machine learning system designed to classify fashion articles and apparel. The repository integrates two distinct models tailored to different domains:

1. **FashionVision v1**: A custom, lightweight Convolutional Neural Network (CNN) trained from scratch on the standardized **Fashion-MNIST** dataset (28×28 grayscale).
2. **FashionVision v2**: A transfer learning architecture leveraging **MobileNetV2** pretrained on ImageNet, fine-tuned on real-world clothing photographs from the Kaggle clothing dataset.

Both architectures feed into a unified Streamlit web application providing real-time inference, class probability rankings, color-inversion preprocessing, and structured JSON prediction report generation.

---

## 1. High-Level Data Flow

```
+----------------------------------------------------------------------------------+
|                                User Input Image                                  |
|         (Upload: JPEG/PNG/WebP, or pre-bundled test-set / synthetic sample)      |
+----------------------------------------------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            Preprocessing Pipeline                                |
|                                                                                  |
|   v1 (Fashion-MNIST standard):            v2 (Real Clothing Photos):             |
|   - ImageOps.exif_transpose               - ImageOps.exif_transpose              |
|   - Grayscale conversion (Mode "L")       - 3-Channel RGB conversion (Mode "RGB")|
|   - Optional Inversion (dark-on-light)    - High-quality LANCZOS resize (224x224)|
|   - LANCZOS resize to 28x28               - float32 NumPy array in [0.0, 255.0]  |
|   - float32 scaling (/ 255.0) -> [0, 1]   - In-graph Keras Rescaling ([-1, 1])   |
|   - Tensor shape: (1, 28, 28, 1)          - Tensor shape: (1, 224, 224, 3)       |
+----------------------------------------------------------------------------------+
                                          |
                     +--------------------+--------------------+
                     |                                         |
                     v                                         v
+---------------------------------------+ +---------------------------------------+
|          v1 CNN Model                 | |      v2 MobileNetV2 Model             |
|  - Conv2D (32, 3x3) + ReLU            | |  - Training Augmentation Layers       |
|  - MaxPooling2D (2x2)                 | |  - Rescaling(1/127.5, offset=-1)      |
|  - Conv2D (64, 3x3) + ReLU            | |  - MobileNetV2 Base (Pretrained)      |
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
|                               Inference Engine                                   |
|   - Extracts top-k probability distributions (default k=3)                       |
|   - Class index to human-readable label mapping                                  |
|   - Prediction latency and confidence scoring                                    |
|   - Produces JSON-serializable structured dictionary                             |
+----------------------------------------------------------------------------------+
                                          |
                                          v
+----------------------------------------------------------------------------------+
|                            Streamlit Web UI (app.py)                             |
|   - Caching with @st.cache_resource                                              |
|   - Side-by-side original and preprocessed image rendering                       |
|   - Interactive probability distribution bar charts                              |
|   - Downloadable evaluation reports with provenance metadata                     |
+----------------------------------------------------------------------------------+
```

---

## 2. Model Architectures in Detail

### 2.1 FashionVision v1: Custom CNN Architecture

The v1 network strikes an optimal balance between parameter efficiency and representation capacity for 28×28 single-channel images.

```
Layer (type)                  Output Shape         Param #    Details
===================================================================================
image_input (InputLayer)      [(None, 28, 28, 1)]  0          Grayscale tensor
conv2d_1 (Conv2D)             (None, 26, 26, 32)   320        3x3 kernels, ReLU
max_pooling2d_1 (MaxPooling)  (None, 13, 13, 32)   0          2x2 window, stride 2
conv2d_2 (Conv2D)             (None, 11, 11, 64)   18,496     3x3 kernels, ReLU
max_pooling2d_2 (MaxPooling)  (None, 5, 5, 64)     0          2x2 window, stride 2
dropout_1 (Dropout)           (None, 5, 5, 64)     0          Rate = 0.25
flatten (Flatten)             (None, 1600)         0          1600 feature units
dense_1 (Dense)               (None, 128)          204,928    ReLU activation
dropout_2 (Dropout)           (None, 128)          0          Rate = 0.50
predictions (Dense)           (None, 10)           1,290      Softmax activation
===================================================================================
Total params: 225,034 (879.04 KB)
Trainable params: 225,034
Non-trainable params: 0
```

#### Key Design Decisions:
- **Hierarchical Feature Learning**: The first `Conv2D(32)` captures low-level primitives (edges, boundaries, simple textures). The second `Conv2D(64)` combines these into higher-level fashion silhouettes (sleeves, straps, soles, collars).
- **Two-Stage Regularization**:
  - `Dropout(0.25)` after the spatial pooling blocks prevents co-adaptation of spatial feature maps.
  - `Dropout(0.50)` before the final classification head prevents dense neurons from memorizing individual training sample quirks.
- **Output Softmax**: 10 output units correspond to the standard 10 Fashion-MNIST categories, producing a valid probability distribution that sums to 1.0.

---

### 2.2 FashionVision v2: Transfer Learning with MobileNetV2

Real clothing photographs require substantially richer semantic understanding than 28×28 grayscale digits or scans. MobileNetV2 provides inverted residual blocks with linear bottlenecks, pretrained on 1.4 million images from ImageNet-1k.

```
+--------------------------------------------------------------------+
| Input: (224, 224, 3), float32 in [0, 255]                          |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| Data Augmentations (active during training only):                  |
| - RandomFlip("horizontal")                                         |
| - RandomRotation(0.1)                                              |
| - RandomZoom(0.15)                                                 |
| - RandomContrast(0.1)                                              |
| - RandomBrightness(0.1)                                            |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| Rescaling(scale=1/127.5, offset=-1) -> maps [0, 255] to [-1.0, 1.0]|
| (Internal Keras layer ensuring identical train/inference pipeline) |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| MobileNetV2 Backbone (include_top=False, weights="imagenet")       |
| - Stage 1 (Head training): All backbone layers frozen (trainable=F)|
| - Stage 2 (Fine-tuning): Top 30 layers unfrozen (trainable=True),  |
|   with BatchNormalization layers kept frozen (inference mode).     |
+--------------------------------------------------------------------+
                                  |
                                  v
+--------------------------------------------------------------------+
| Classification Head:                                               |
| - GlobalAveragePooling2D                                           |
| - Dropout(0.30)                                                    |
| - Dense(num_classes, activation="softmax")                         |
+--------------------------------------------------------------------+
```

#### Why In-Graph Rescaling?
Placing `layers.Rescaling(scale=1/127.5, offset=-1.0)` inside the model functional graph guarantees that any downstream consumer (Streamlit UI, Python script, or mobile deployment) feeds raw `[0, 255]` pixel values. This completely eliminates the classic production bug where training used `preprocess_input` but inference forgot or applied incorrect normalization.

#### Two-Stage Transfer Learning Strategy:
1. **Stage 1 (Head Only, Adam lr=1e-3)**:
   The newly initialized classification head has random weights. If gradients backpropagated through MobileNetV2 immediately, massive early updates would shatter the delicate, pretrained visual filters. Freezing the base allows the classification head to settle.
2. **Stage 2 (Fine-Tuning, Adam lr=1e-5)**:
   Unfreezing the top 30 convolutional layers allows the network to adapt high-level visual features (fabric folds, collars, shoe laces, pockets) to clothing photography. A 100x lower learning rate prevents catastrophic forgetting.
3. **Freezing Batch Normalization**:
   Updating running mean and variance statistics on small fine-tuning batches destabilizes representations. Keeping `BatchNormalization` in inference mode preserves batch statistics from ImageNet.

---

## 3. Preprocessing and Photographic Inversion

Fashion-MNIST images are standardized as **light articles on black backgrounds** (`pixel = 0` is background, `pixel = 255` is foreground). In contrast, consumer smartphone photographs and online catalogs typically feature **dark articles on light/white backgrounds**.

To address this domain mismatch:
- `prepare_image(pil_image, invert=False)` in `src/inference.py` offers an `invert` parameter:
  ```python
  if invert:
      img = ImageOps.invert(img)
  ```
- Downsampling uses `Image.Resampling.LANCZOS`, which calculates an 8-lobed sinc interpolation kernel. This avoids anti-aliasing artifacts and preserves sharp clothing edges better than nearest-neighbor or bilinear interpolation.
- Orientation normalization uses `ImageOps.exif_transpose`, automatically rotating smartphone photos that encode rotation in EXIF metadata rather than pixel grids.

---

## 4. UI and Web Application Architecture

`app.py` is structured as a reactive single-page dashboard:
1. **Model & Metadata Caching**:
   `@st.cache_resource` loads `load_artifacts()` once per server lifecycle, eliminating redundant disk I/O and graph compilation on each Streamlit rerun.
2. **Tabbed Workflow**:
   - **Tab 1: Single Prediction & Analysis**: Upload user images or pick curated samples, view live side-by-side transformed images, inspect top-3 probability bars, and download structured JSON audit reports.
   - **Tab 2: Model Performance & Validation**: Inspect saved confusion matrices, training loss/accuracy curves, classification reports, and misclassified example grids directly from `results/`.
   - **Tab 3: Architecture & Model Specs**: Review layer topology, training metadata, hardware environments, and dataset properties.
