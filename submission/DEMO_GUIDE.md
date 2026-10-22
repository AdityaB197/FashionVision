# Demonstration & Viva Voce Guide — FashionVision

This guide provides a structured presentation roadmap for showcasing the **FashionVision** project during project demonstrations, reviews, and university viva examinations.

---

## 1. Live Demonstration Walkthrough (5-Minute Script)

### Step 1: Launch and Dashboard Overview
- **Action**: Launch the dashboard with `streamlit run app.py` and open `http://localhost:8501`.
- **Talking Point**:
  > *"FashionVision is a deep learning apparel classification platform with two complementary models: a lightweight from-scratch CNN achieving 91.61% accuracy on the 10-class Fashion-MNIST benchmark, and a MobileNetV2 transfer learning pipeline for real-world clothing photography. The interface is built with Streamlit and includes real-time preprocessing visualization, top-k rankings, and auditable JSON prediction reports."*

### Step 2: High-Confidence Benchmark Classification
- **Action**: In the sidebar, select **"Choose a sample image"** and pick **Trouser** or **Ankle boot**.
- **Talking Point**:
  > *"When presented with distinct silhouette garments like Trousers or Ankle boots, the CNN identifies spatial contours with near 100% confidence. Notice the dynamic bar chart below, displaying the top-3 predicted classes with their softmax probability distribution."*

### Step 3: Real Photograph Domain Shift & Invert Option
- **Action**: Upload a real smartphone photo of a dark garment on a white desk/background. Initially keep **"Invert colors"** unchecked, then check **"Invert colors (for dark clothes on light backgrounds)"**.
- **Talking Point**:
  > *"Fashion-MNIST images are standardized as light apparel on black backgrounds (pixel 0 is background, 255 is foreground). Ordinary consumer photos feature dark clothes on white backgrounds. Without preprocessing, a white background overwhelms the convolutional filters. By applying `ImageOps.invert`, we map the background back to dark, aligning the input with the model's training distribution."*

### Step 4: The "Shirt vs T-shirt" Morphological Ambiguity
- **Action**: Select the **Shirt** sample. Observe the top-3 distribution (Shirt as #1, with T-shirt/top or Coat as close contenders).
- **Talking Point**:
  > *"Upper-body garments share broad rectangular torso silhouettes. At 28×28 resolution, discriminative details like collar folds and button plackets occupy only a few blurred pixels. This reflects a genuine computer vision challenge that our confusion matrix quantifies."*

### Step 5: Auditable Report Export
- **Action**: Scroll to the bottom of Tab 1 and click **"Download Prediction Report (JSON)"**. Open the downloaded JSON file.
- **Talking Point**:
  > *"Every inference output can be downloaded as an immutable audit record containing UTC timestamps, model architecture parameters, input source hash, preprocessing options, and academic disclaimers."*

### Step 6: Empirical Audit Tab
- **Action**: Click on **"Model Performance & Validation"** tab.
- **Talking Point**:
  > *"All metrics are computed on the held-out 10,000-image test set. The training curves show rapid convergence by epoch 16 with early stopping. The confusion matrix highlights that while Trousers (98.6% F1) and Footwear (>96% F1) perform exceptionally, Shirt (75.3% F1) exhibits expected overlap with T-shirt and Coat."*

---

## 2. Top Examiner Viva Questions & Model Defenses

### Q1: Why did you choose a Convolutional Neural Network (CNN) instead of a standard Multi-Layer Perceptron (Dense / MLP)?
**Answer**:
> *"A standard MLP flattens the image into a 784-dimensional vector, completely discarding spatial 2D relationships between adjacent pixels. Furthermore, an MLP lacks translation invariance—a shoe in the top-left corner would activate completely different weights than a shoe in the bottom-right. CNNs utilize small 3×3 shared convolutional filters that slide across the entire image, detecting local features (edges, corners, curves) regardless of spatial position, while requiring drastically fewer trainable parameters (225k vs millions)."*

### Q2: Why did you choose 32 filters in the first Conv2D layer and 64 filters in the second?
**Answer**:
> *"Hierarchical representation learning. The initial layer with 32 filters operates directly on raw pixels to capture basic edge orientations and gradients. The subsequent layer with 64 filters combines these simple edge activations from the previous layer into richer, higher-level spatial patterns such as hemlines, waistbands, sleeves, and collars. Doubling filter depth as spatial resolution halves via pooling is standard deep learning practice to maintain representational capacity."*

### Q3: Why is Dropout applied twice, and why are the rates different (0.25 vs 0.50)?
**Answer**:
> *"Dropout randomly zeroes out a fraction of neurons during training to prevent co-adaptation. In the convolutional feature extraction stage, adjacent feature map pixels are strongly correlated; dropping 25% of activations is sufficient to prevent overfitting without losing spatial continuity. In the dense classification head (128 neurons), the risk of overfitting is much higher because dense layers contain over 90% of the model's total parameters (204,928 weights). A higher dropout rate of 50% forces redundant, robust feature representations before the final softmax decision."*

### Q4: Why did Shirt achieve only 75.26% F1-score compared to Trouser at 98.59%?
**Answer**:
> *"Morphological ambiguity under low resolution. In a 28×28 grayscale representation, a button placket or collar crease spans only 1–2 pixels, which often blend into noise during downsampling. Consequently, Shirts, T-shirts, and Pullovers exhibit nearly identical torso silhouettes. Trousers, sandals, and bags, on the other hand, possess distinct aspect ratios and geometric boundaries (two leg bifurcations, open straps, handles) that produce unambiguous convolutional activations."*

### Q5: In FashionVision v2, why use a two-stage training strategy (freeze head, then fine-tune)?
**Answer**:
> *"MobileNetV2's weights were pretrained on ImageNet and already contain sophisticated visual filters. When we append a new, randomly initialized dense classification head, initial training produces large gradient errors. If we backpropagated these large gradients through MobileNetV2 immediately, they would destroy the pretrained weights (catastrophic forgetting). In Stage 1, we freeze the backbone and train only the head with lr=1e-3. In Stage 2, once the head is stable, we unfreeze the top 30 layers and use a 100x lower learning rate (1e-5) to gently adapt higher-level filters without destabilizing low-level edge detectors."*

### Q6: Why must BatchNormalization layers remain frozen during transfer learning fine-tuning?
**Answer**:
> *"Batch Normalization layers maintain running statistics (mean and variance) of the activations. Our fine-tuning batches are relatively small and specific to clothing. If BatchNorm layers were allowed to update their running statistics on these small batches, the sudden variance shifts would destabilize the entire network. Freezing BatchNorm preserves the stable ImageNet population statistics."*

### Q7: Why put `layers.Rescaling(1/127.5, offset=-1)` inside the Keras model in v2 rather than in Python preprocessing?
**Answer**:
> *"Putting normalization inside the saved model guarantees that training and serving preprocessing are strictly identical. If normalization were done purely in Python scripts, any client (Streamlit app, mobile app, microservice) that forgot to divide by 127.5 and subtract 1 would experience silent, catastrophic accuracy degradation. With in-graph rescaling, the model safely accepts raw 0–255 RGB pixels everywhere."*
