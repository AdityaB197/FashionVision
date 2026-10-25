"""FashionVision v2 — Streamlit Web Application.

Real clothing photo classification powered by MobileNetV2 transfer learning.
Upload photos, use your webcam, or test with prepared samples.
"""

import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import streamlit as st

# Add project root to sys.path to import from src modules
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.inference import load_real_artifacts, prepare_real_image, predict_top_k_real
from src.reporting import build_prediction_report

# Page configuration
st.set_page_config(
    page_title="FashionVision v2",
    page_icon="👗",
    layout="wide",
)

st.title("FashionVision v2")
st.markdown("### Real Clothing Photo Classification with MobileNetV2 Transfer Learning")

st.info(
    "🎯 **How it works:** This model uses MobileNetV2 pretrained on ImageNet, fine-tuned on real clothing photographs. "
    "It works best with centered garments on plain backgrounds. High confidence does not guarantee correctness."
)

# Caching is crucial in Streamlit:
# @st.cache_resource keeps the heavy TensorFlow model loaded in memory across user interactions.
@st.cache_resource
def cached_load_artifacts():
    return load_real_artifacts()

# Load model artifacts with user-friendly error handling
try:
    model, class_names, metadata = cached_load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Ensure that 'real_model.keras', 'class_names.json', and 'metadata.json' exist in the models/ directory.")
    st.stop()

# Load sample test images if available
sample_images = []
test_dir = Path("tmp_real/test")
if test_dir.exists():
    for class_dir in sorted(test_dir.iterdir()):
        if class_dir.is_dir():
            for img_path in sorted(class_dir.glob("*.jpg"))[:3]:  # Max 3 per class
                sample_images.append({
                    "path": img_path,
                    "class": class_dir.name,
                    "display": f"{class_dir.name}/{img_path.name}"
                })

# Sidebar: Controls & Input Selection
st.sidebar.header("Input Controls")

input_options = ["Upload a photo"]
if sample_images:
    input_options.insert(0, "Sample from test set")
input_options.append("Take a photo (webcam)")

input_source_choice = st.sidebar.radio(
    "Choose input source:",
    options=input_options,
    index=0,
)

pil_image = None
input_source_str = ""
true_label_name = None

if input_source_choice == "Sample from test set" and sample_images:
    sample_options = [s["display"] for s in sample_images]
    selected_sample = st.sidebar.selectbox(
        "Select a test sample:",
        options=sample_options,
        index=0,
    )
    sample_idx = sample_options.index(selected_sample)
    sample_data = sample_images[sample_idx]
    
    try:
        pil_image = Image.open(sample_data["path"])
        input_source_str = f"Test sample: {sample_data['display']}"
        true_label_name = sample_data["class"]
        st.sidebar.info(f"Ground Truth: **{true_label_name}**")
    except Exception as e:
        st.error(f"Could not open sample image: {e}")
        st.stop()

elif input_source_choice == "Take a photo (webcam)":
    st.sidebar.info("📸 Use the camera input below to capture a photo")
    camera_photo = st.camera_input("Take a picture of clothing")
    
    if camera_photo is not None:
        try:
            pil_image = Image.open(camera_photo)
            input_source_str = "webcam capture"
        except Exception as e:
            st.error(f"Could not process camera image: {e}")
            st.stop()
    else:
        st.info("Please take a photo using your webcam or switch to upload mode.")
        st.stop()

else:  # Upload a photo
    uploaded_file = st.sidebar.file_uploader(
        "Upload a clothing photo:",
        type=["png", "jpg", "jpeg", "webp"],
        help="Upload a photo of clothing. Works best with centered garments on plain backgrounds.",
    )
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            input_source_str = f"uploaded file: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Could not open uploaded image: {e}")
            st.stop()
    else:
        st.info("Please upload a photo from the sidebar or switch to another input mode.")
        st.stop()

# Tabs for structured navigation
tab_pred, tab_perf, tab_curves = st.tabs([
    "Prediction",
    "Model Performance",
    "Training Curves",
])

# ----------------- TAB 1: Prediction -----------------
with tab_pred:
    # Preprocess image: EXIF correction, RGB conversion, 224x224 resize
    prep_tensor = prepare_real_image(pil_image)
    pred_res = predict_top_k_real(model, prep_tensor, class_names, k=min(3, len(class_names)))

    pred_class = pred_res["predicted_class"]
    confidence = pred_res["confidence"]

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Original Input**")
        # Display original with reasonable size constraint
        display_img = pil_image.copy()
        display_img.thumbnail((400, 400))
        st.image(display_img, caption=input_source_str)

    with col_img2:
        st.markdown("**Model View (224×224 RGB)**")
        # Show the preprocessed tensor as the model sees it
        preview_224 = Image.fromarray(prep_tensor[0].astype(np.uint8))
        st.image(preview_224, caption="224×224 RGB input tensor", width=224)

    st.markdown("---")
    st.subheader(f"Prediction: **{pred_class}** ({confidence * 100:.1f}%)")

    if true_label_name is not None:
        if pred_class.lower() == true_label_name.lower():
            st.success(f"✓ Matches ground truth: **{true_label_name}**")
        else:
            st.warning(f"✗ Does not match ground truth: Expected **{true_label_name}**, got **{pred_class}**")

    st.markdown(f"#### Top-{len(pred_res['top_k'])} Confidence Scores")
    for item in pred_res["top_k"]:
        c_label, c_bar = st.columns([1, 4])
        with c_label:
            st.write(f"**{item['class_name']}**")
        with c_bar:
            st.progress(float(item["score"]), text=f"{item['score'] * 100:.2f}%")

    # Load test metrics summary if results/metrics.json is available
    test_metrics_data = None
    metrics_path = Path("results/metrics.json")
    if metrics_path.exists():
        try:
            with open(metrics_path, "r", encoding="utf-8") as f:
                test_metrics_data = json.load(f)
        except Exception:
            test_metrics_data = None

    report = build_prediction_report(
        prediction=pred_res,
        metadata=metadata,
        input_source=input_source_str,
        test_metrics=test_metrics_data,
    )

    st.markdown("---")
    st.download_button(
        label="📥 Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="fashionvision_v2_prediction.json",
        mime="application/json",
    )

# ----------------- TAB 2: Model Performance -----------------
with tab_perf:
    st.subheader("Test Set Evaluation Results")
    st.caption("Metrics computed on the held-out test split (never seen during training).")

    metrics_file = Path("results/metrics.json")
    if metrics_file.exists():
        with open(metrics_file, "r", encoding="utf-8") as f:
            metrics_json = json.load(f)

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Test Accuracy", f"{metrics_json['accuracy'] * 100:.2f}%")
        m2.metric("Macro Precision", f"{metrics_json['macro_precision'] * 100:.2f}%")
        m3.metric("Macro Recall", f"{metrics_json['macro_recall'] * 100:.2f}%")
        m4.metric("Macro F1", f"{metrics_json['macro_f1'] * 100:.2f}%")

        st.markdown("#### Per-Class Performance")
        per_class_table = []
        for cname, stats in metrics_json.get("per_class", {}).items():
            per_class_table.append({
                "Class": cname,
                "Precision": f"{stats['precision'] * 100:.2f}%",
                "Recall": f"{stats['recall'] * 100:.2f}%",
                "F1-Score": f"{stats['f1'] * 100:.2f}%",
                "Support": stats["support"],
            })
        st.dataframe(per_class_table, use_container_width=True)
    else:
        st.info("results/metrics.json not found. Run scripts/evaluate_model.py to produce evaluation metrics.")

    cm_file = Path("results/confusion_matrix.png")
    if cm_file.exists():
        st.markdown("#### Confusion Matrix")
        st.image(str(cm_file), caption="Confusion Matrix on Test Set", use_container_width=True)
    else:
        st.info("results/confusion_matrix.png not found. Run scripts/evaluate_model.py to produce the plot.")

    mc_file = Path("results/misclassified_examples.png")
    if mc_file.exists():
        st.markdown("#### Misclassified Test Examples")
        st.image(str(mc_file), caption="Sample of misclassified test images", use_container_width=True)
    else:
        st.info("results/misclassified_examples.png not found.")

# ----------------- TAB 3: Training Curves -----------------
with tab_curves:
    st.subheader("Training Progress")
    st.caption("Two-stage transfer learning: Stage 1 (frozen base) + Stage 2 (fine-tuned top layers)")

    tc_file = Path("results/training_curves.png")
    if tc_file.exists():
        st.image(str(tc_file), caption="Training & Validation Curves (Accuracy & Loss)", use_container_width=True)
    else:
        st.info("results/training_curves.png not found. Run scripts/train_model.py to generate training curves.")

    # Show training history details if available
    history_file = Path("results/history.json")
    if history_file.exists():
        try:
            with open(history_file, "r", encoding="utf-8") as f:
                history_data = json.load(f)
            
            st.markdown("#### Training Summary")
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("**Stage 1 (Head Training)**")
                if "accuracy" in history_data and len(history_data["accuracy"]) > 0:
                    st.metric("Final Train Accuracy", f"{history_data['accuracy'][-1] * 100:.2f}%")
                if "val_accuracy" in history_data and len(history_data["val_accuracy"]) > 0:
                    st.metric("Final Val Accuracy", f"{history_data['val_accuracy'][-1] * 100:.2f}%")
            
            with col2:
                st.markdown("**Stage 2 (Fine-tuning)**")
                # Note: This assumes history contains both stages concatenated
                # Actual implementation may vary based on how history is saved
                st.caption("See curves plot for detailed progression")
                
        except Exception:
            pass

st.sidebar.markdown("---")
st.sidebar.markdown("**Model Info**")
st.sidebar.caption(f"Architecture: {metadata.get('base_model_name', 'MobileNetV2')}")
st.sidebar.caption(f"Classes: {len(class_names)}")
st.sidebar.caption(f"Input size: 224×224 RGB")
