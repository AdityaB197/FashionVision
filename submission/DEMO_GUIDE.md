# Demonstration & Viva Voce Guide — FashionVision v2

This guide provides a structured presentation roadmap for showcasing the **FashionVision v2** project during project demonstrations, reviews, and university viva examinations.

---

## 1. Live Demonstration Walkthrough (5-Minute Script)

### Step 1: Launch and Dashboard Overview
- **Action**: Launch the dashboard with `python -m streamlit run app.py` and open `http://localhost:8501`.
- **Talking Point**:
  > *"FashionVision v2 is a deep transfer learning apparel classification platform utilizing MobileNetV2 pretrained on ImageNet-1k, fine-tuned specifically for real-world clothing photographs. The interface is built with Streamlit and includes real-time webcam capture, multi-source input support, high-resolution 224×224 RGB tensor visualization, top-k rankings, and auditable JSON prediction reports."*

### Step 2: Testing with Real Photos
- **Action**: In the sidebar, select **"Sample from test set"** (if available) or **"Upload a photo"** and upload a dress or shirt photo.
- **Talking Point**:
  > *"Unlike benchmark models trained on 28×28 grayscale silhouettes, FashionVision v2 operates directly on full 3-channel RGB photography. The model identifies complex textures, cuts, and silhouettes in natural lighting conditions."*

### Step 3: Webcam Capture Feature
- **Action**: In the sidebar, select **"Take a photo (webcam)"**. Take a snapshot or hold up a garment to the camera.
- **Talking Point**:
  > *"The Streamlit application includes live webcam capture via `st.camera_input`. The captured frame is automatically stripped of alpha channels, normalized for EXIF rotation, and sized for inference in real time."*

### Step 4: Preprocessing & Model View Inspection
- **Action**: Point to the side-by-side display: Original Input on the left, Model View (224×224 RGB) on the right.
- **Talking Point**:
  > *"Here we see our preprocessing pipeline in action. Using high-quality 8-lobed Lanczos downsampling, garment boundaries and fabric textures are preserved without introducing interpolation artifacts."*

### Step 5: Ranked Softmax Confidence Distribution
- **Action**: Show the dynamic horizontal progress bars displaying top-k predictions.
- **Talking Point**:
  > *"The dense classification head outputs normalized softmax probabilities. Notice the ranked distribution: even in ambiguous cases, the second- and third-place categories provide context on morphological similarity."*

### Step 6: Auditable Report Export
- **Action**: Scroll to the bottom of the Prediction tab and click **"Download prediction report (JSON)"**. Open the downloaded file.
- **Talking Point**:
  > *"Every inference output can be exported as an immutable JSON audit log containing UTC timestamps, predicted class probabilities, input dimensions, and model provenance metadata."*

### Step 7: Empirical Audit Tab
- **Action**: Click on the **"Model performance"** and **"Training curves"** tabs.
- **Talking Point**:
  > *"Our evaluation tab visualizes the held-out test split metrics, confusion matrix, and training curves. Notice the red dashed line on the training curves indicating the onset of Stage 2 fine-tuning, demonstrating how unfreezing the top 30 layers accelerates loss convergence."*

---

## 2. Top Examiner Viva Questions & Model Defenses

### Q1: Why did you choose MobileNetV2 transfer learning instead of training a CNN from scratch?
**Answer**:
> *"Training a deep convolutional network from scratch on real-world photography requires millions of labeled images to learn generalized visual filters (edges, textures, shapes). By utilizing MobileNetV2 pretrained on ImageNet-1k (1.4 million images), we leverage robust feature extractors already capable of distinguishing complex visual patterns. We only need to train a new classification head and fine-tune higher-level layers, achieving superior generalization with substantially less training data and compute time."*

### Q2: Why use a two-stage training strategy (freeze head, then fine-tune)?
**Answer**:
> *"MobileNetV2's weights were pretrained on ImageNet and already contain sophisticated visual representations. When we append a new, randomly initialized dense classification head, initial training produces large gradient errors. If we backpropagated these large gradients through MobileNetV2 immediately, they would destroy the pretrained weights (catastrophic forgetting). In Stage 1, we freeze the backbone and train only the head with $\eta = 10^{-3}$. In Stage 2, once the head is stable, we unfreeze the top 30 layers and use a 100× lower learning rate ($10^{-5}$) to gently adapt higher-level filters without destabilizing low-level edge detectors."*

### Q3: Why must BatchNormalization layers remain frozen during transfer learning fine-tuning?
**Answer**:
> *"Batch Normalization layers maintain running statistics (mean and variance) of feature activations. In fine-tuning, batch sizes are relatively small. If BatchNorm layers were allowed to update their running statistics on these small batches, the sudden variance shifts would destabilize the entire network. Freezing BatchNorm in inference mode preserves the stable ImageNet population statistics."*

### Q4: Why put `layers.Rescaling(1/127.5, offset=-1)` inside the Keras model rather than in Python scripts?
**Answer**:
> *"Putting normalization inside the saved model guarantees that training and serving preprocessing are strictly identical (preventing training-serving skew). If normalization were done purely in Python scripts, any client (Streamlit app, mobile app, microservice) that forgot to divide by 127.5 and subtract 1 would experience silent, catastrophic accuracy degradation. With in-graph rescaling, the model safely accepts raw 0–255 RGB pixels everywhere."*

### Q5: Why is 224×224 RGB necessary for real clothing photos instead of 28×28 grayscale?
**Answer**:
> *"Real apparel relies heavily on color, fabric texture, button plackets, collar cuts, and sleeve contours. Downscaling a real photo to 28×28 grayscale destroys these discriminative cues, collapsing shirts, jackets, and t-shirts into indistinct blobs. 224×224 3-channel RGB preserves the visual details required for deep feature extractors like MobileNetV2 to separate subtle apparel styles."*

### Q6: How does your preprocessing handle smartphone photos taken at different angles?
**Answer**:
> *"Modern smartphones store camera orientation in EXIF metadata tags. If an image is read without orientation correction, it may appear rotated by 90 or 180 degrees, confusing the model. In `src/inference.py`, we apply `ImageOps.exif_transpose` before resizing, guaranteeing that all images are oriented upright before being fed into the neural network."*

### Q7: How does the system handle transparent PNG images?
**Answer**:
> *"Transparent PNGs contain an alpha channel (RGBA). Simply discarding the alpha channel can convert transparent areas into black pixels, introducing severe artificial edges. Our pipeline detects the alpha channel and composites the image onto a clean white background using Pillow before converting to 3-channel RGB."*
