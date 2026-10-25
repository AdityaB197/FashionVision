# System Architecture — FashionVision v2

FashionVision v2 is a production-grade machine learning system for classifying real-world clothing photographs using **MobileNetV2 transfer learning**. The system leverages ImageNet-pretrained features, adapted through two-stage fine-tuning on real apparel photos from the Kaggle clothing dataset.

The architecture integrates preprocessing, model inference, and a Streamlit web application providing real-time predictions, performance metrics, and structured JSON reports.

---

## 1. High-Level Data Flow

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
|   • Data Augmentation Layers (RandomFlip, Rotation, Zoom, Contrast, Bright) |
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
|                        Inference Engine (src/inference.py)                   |
|   • predict_top_k_real(): Extract top-k probability distributions            |
|   • Class index → human-readable label mapping                               |
|   • JSON-serializable structured output with confidence scores               |
+------------------------------------------------------------------------------+
                                      |
                                      v
+------------------------------------------------------------------------------+
|                        Streamlit Web UI (app.py)                             |
|   • @st.cache_resource for model loading                                     |
|   • Side-by-side: original photo + preprocessed 224×224 RGB tensor           |
|   • Interactive top-k probability bar charts                                 |
|   • Performance metrics tab: confusion matrix, training curves               |
|   • Downloadable JSON audit reports with model metadata                      |
+------------------------------------------------------------------------------+
```

---

## 2. MobileNetV2 Transfer Learning Architecture

### 2.1 Model Structure (src/model.py)

The v2 network uses MobileNetV2 as a pretrained feature extractor, topped with a custom classification head:

```
Layer Type                      Output Shape           Param #    Details
====================================================================================
image_input (InputLayer)        (None, 224, 224, 3)    0          RGB tensor [0, 255]

--- Data Augmentation Block ---
aug_flip (RandomFlip)           (None, 224, 224, 3)    0          Horizontal flip
aug_rotation (RandomRotation)   (None, 224, 224, 3)    0          ±36° rotation
aug_zoom (RandomZoom)           (None, 224, 224, 3)    0          ±15% zoom
aug_contrast (RandomContrast)   (None, 224, 224, 3)    0          ±10% contrast
aug_brightness (RandomBrightness)(None, 224, 224, 3)   0          ±10% brightness

input_rescaling (Rescaling)     (None, 224, 224, 3)    0          [0,255]→[-1,1]

--- MobileNetV2 Base (Pretrained on ImageNet) ---
mobilenetv2_base (Functional)   (None, 7, 7, 1280)     ~3.5M      Frozen in Stage 1
                                                                   Top 30 unfrozen in Stage 2

--- Classification Head ---
global_avg_pool (GlobalAvgPool) (None, 1280)           0          Spatial pooling
head_dropout (Dropout)          (None, 1280)           0          Rate = 0.3
predictions (Dense)             (None, num_classes)    1280*C+C   Softmax, float32
====================================================================================
Total params: ~3.5M + classification head
Trainable (Stage 1): ~1.3K (classification head only)
Trainable (Stage 2): ~1.1M (top 30 layers + classification head)
```

### 2.2 Key Design Decisions

#### Why MobileNetV2?
- **Efficiency**: 3.5M parameters, designed for mobile/edge deployment
- **Transfer Quality**: Strong ImageNet features transfer well to clothing domain
- **Depthwise Separable Convolutions**: Efficient spatial-channel factorization
- **Inverted Residuals**: Linear bottlenecks preserve information flow

#### Why In-Graph Rescaling?
```python
layers.Rescaling(scale=1/127.5, offset=-1.0, name="input_rescaling")
```
- Embedding normalization **inside the saved model** guarantees identical preprocessing between training and production inference
- Eliminates common deployment bugs from preprocessing mismatches
- Accepts standard [0, 255] RGB pixels, converts internally to [-1, 1] (MobileNetV2's expected range)

#### Why Data Augmentation Layers?
Unlike Fashion-MNIST's uniform scans, real clothing photos vary wildly in:
- **Lighting conditions**: Indoor vs outdoor, shadows, highlights
- **Camera angles**: Straight-on vs angled, zoom levels
- **Backgrounds**: Plain vs cluttered, various colors

Augmentation forces the model to learn **invariant clothing features** (textures, patterns, shapes) rather than memorizing backgrounds or lighting.
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
