"""FashionVision — Streamlit Web Application.

A lightweight, interactive clothing classifier demonstrating CNN inference
on Fashion-MNIST standardized images.
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

from src.data import load_fashion_mnist
from src.inference import load_artifacts, prepare_image, predict_top3
from src.reporting import build_prediction_report

# Page configuration
st.set_page_config(
    page_title="FashionVision",
    page_icon="👗",
    layout="wide",
)

st.title("FashionVision")
st.markdown("### CNN-based Fashion-MNIST Clothing Classifier")

# Clear disclaimer right below title as requested
st.warning(
    "⚠️ **Disclaimer:** This model was trained on standardized 28x28 grayscale Fashion-MNIST images "
    "and is **NOT** reliable on arbitrary real-world clothing photos."
)

# Caching is crucial in Streamlit:
# @st.cache_resource keeps the heavy TensorFlow model loaded in memory across user interactions.
@st.cache_resource
def cached_load_artifacts():
    return load_artifacts()

# @st.cache_data caches static data (like the 10,000 test images) so we avoid re-reading on every rerun.
@st.cache_data
def cached_load_test_data():
    (_, _), (_, _), (x_test, y_test) = load_fashion_mnist()
    return x_test, y_test

# Load model artifacts with user-friendly error handling
try:
    model, class_names, metadata = cached_load_artifacts()
except Exception as e:
    st.error(f"Error loading model artifacts: {e}")
    st.info("Ensure that 'fashion_cnn.keras', 'class_names.json', and 'metadata.json' exist in the models/ directory.")
    st.stop()

# Load test dataset with error handling
try:
    x_test, y_test = cached_load_test_data()
except Exception as e:
    st.error(f"Error loading Fashion-MNIST test dataset: {e}")
    st.stop()

# Sidebar: Controls & Input Selection
st.sidebar.header("Input Controls")

# Input selection radio: default to test set sample so the app immediately renders an active prediction
input_source_choice = st.sidebar.radio(
    "Choose input source:",
    options=["Sample from Fashion-MNIST test set", "Upload an image"],
    index=0,
)

# Why invert exists: Fashion-MNIST images have white/light items on black background.
# Real-world photos or sketches typically have dark clothes on a light background.
invert = st.sidebar.checkbox(
    "Invert colors",
    value=False,
    help=(
        "Fashion-MNIST items are light objects on a black background; "
        "ordinary photos are usually dark objects on a light background, so turn this on for those."
    ),
)

pil_image = None
input_source_str = ""
true_label_idx = None
true_label_name = None

if input_source_choice == "Sample from Fashion-MNIST test set":
    sample_index = st.sidebar.number_input(
        "Select test sample index (0 - 9999):",
        min_value=0,
        max_value=9999,
        value=0,
        step=1,
    )
    raw_array = x_test[sample_index]
    true_label_idx = int(y_test[sample_index])
    true_label_name = class_names[true_label_idx]
    pil_image = Image.fromarray(raw_array)
    input_source_str = f"Fashion-MNIST test sample #{sample_index}"
    st.sidebar.info(f"Ground Truth: **{true_label_name}** (Class #{true_label_idx})")

else:  # Upload an image
    uploaded_file = st.sidebar.file_uploader(
        "Upload a clothing image:",
        type=["png", "jpg", "jpeg"],
        help="Upload an image to classify. Works best with centered clothing items.",
    )
    if uploaded_file is not None:
        try:
            pil_image = Image.open(uploaded_file)
            input_source_str = f"uploaded file: {uploaded_file.name}"
        except Exception as e:
            st.error(f"Could not open uploaded image: {e}")
            st.stop()
    else:
        st.info("Please upload an image from the sidebar or switch to the sample mode to test predictions.")
        st.stop()

# Tabs for structured navigation
tab_pred, tab_perf, tab_curves, tab_misclass = st.tabs([
    "Prediction",
    "Model performance",
    "Training curves",
    "Misclassified examples",
])

# ----------------- TAB 1: Prediction -----------------
with tab_pred:
    # Shared preprocessing ensures image sizing and normalization strictly match training
    prep_tensor = prepare_image(pil_image, invert=invert)
    pred_res = predict_top3(model, prep_tensor, class_names, k=3)

    pred_class = pred_res["predicted_class"]
    confidence = pred_res["confidence"]

    col_img1, col_img2 = st.columns(2)
    with col_img1:
        st.markdown("**Original Input**")
        st.image(pil_image, caption=input_source_str, width=200)

    with col_img2:
        st.markdown("**Model View (28x28)**")
        # Upscale 28x28 image using nearest-neighbor so pixels stay sharp for inspection
        preview_28x28 = Image.fromarray(
            (prep_tensor[0, :, :, 0] * 255.0).astype(np.uint8)
        ).resize((200, 200), Image.Resampling.NEAREST)
        st.image(preview_28x28, caption="28x28 grayscale input tensor", width=200)

    st.markdown("---")
    st.subheader(f"Prediction: **{pred_class}** ({confidence * 100:.1f}%)")

    if true_label_name is not None:
        if pred_res["predicted_index"] == true_label_idx:
            st.success(f"Matches ground truth: **{true_label_name}**")
        else:
            st.warning(f"Does not match ground truth: Expected **{true_label_name}**, got **{pred_class}**")

    st.markdown("#### Top-3 Confidence Scores")
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
        invert=invert,
        test_metrics=test_metrics_data,
    )

    st.markdown("---")
    st.download_button(
        label="Download prediction report (JSON)",
        data=json.dumps(report, indent=2),
        file_name="prediction_report.json",
        mime="application/json",
    )

# ----------------- TAB 2: Model performance -----------------
with tab_perf:
    st.subheader("Official 10,000-Image Test Set Results")
    st.caption("Computed during evaluation on Google Colab with test set.")

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

# ----------------- TAB 3: Training curves -----------------
with tab_curves:
    st.subheader("Training and Validation Curves")
    st.caption("Generated across training epochs on Google Colab.")

    tc_file = Path("results/training_curves.png")
    if tc_file.exists():
        st.image(str(tc_file), caption="Training & Validation Curves (Accuracy & Loss)", use_container_width=True)
    else:
        st.info("results/training_curves.png not found. Run scripts/evaluate_model.py to generate this plot.")

# ----------------- TAB 4: Misclassified examples -----------------
with tab_misclass:
    st.subheader("Misclassified Test Examples")
    st.caption("Sample of test images where model prediction differed from ground truth.")

    mc_file = Path("results/misclassified_examples.png")
    if mc_file.exists():
        st.image(str(mc_file), caption="Misclassified Fashion-MNIST Test Samples", use_container_width=True)
    else:
        st.info("results/misclassified_examples.png not found. Run scripts/evaluate_model.py to generate this plot.")
