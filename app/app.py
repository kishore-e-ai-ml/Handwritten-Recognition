"""
Streamlit Web Application for Handwritten Character Recognition.
Supports both MNIST digit recognition and EMNIST character recognition.
Features: drawing canvas, file upload, top-3 predictions, saliency map explainability.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import streamlit as st
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Streamlit page config (must be first st call)
st.set_page_config(
    page_title="Handwritten Character Recognition",
    page_icon="✍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.predict import Predictor
from src.explainability import compute_saliency_map
from src.preprocessing import preprocess_image_for_model

# ─── CSS styling ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
.main-header {
    font-size: 2.2rem; font-weight: 800;
    background: linear-gradient(135deg, #1e3c72, #2a5298);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem;
}
.sub-header {
    font-size: 1rem; color: #555; margin-bottom: 1.5rem;
}
.result-box {
    background: linear-gradient(135deg, #f8f9fa, #e9ecef);
    border-left: 5px solid #2a5298;
    border-radius: 8px; padding: 1rem 1.5rem; margin: 0.5rem 0;
}
.confidence-high { color: #28a745; font-weight: 700; }
.confidence-med  { color: #ffc107; font-weight: 700; }
.confidence-low  { color: #dc3545; font-weight: 700; }
.metric-label { font-size: 0.85rem; color: #888; text-transform: uppercase; }
.big-char { font-size: 5rem; font-weight: 900; text-align: center; color: #1e3c72; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown("## ⚙️ Settings")

dataset_mode = st.sidebar.radio(
    "Recognition Mode",
    ["MNIST – Digits (0–9)", "EMNIST – Characters (0–9, A–Z, a–z)"],
    index=0
)
dataset_type = "mnist" if "MNIST" in dataset_mode and "EMNIST" not in dataset_mode else "emnist"

input_mode = st.sidebar.radio(
    "Input Method",
    ["🖊️ Draw on Canvas", "📁 Upload Image"]
)

show_saliency = st.sidebar.checkbox("Show Saliency Explainability", value=True)

# ─── Model loading (cached) ──────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_predictor(dtype: str) -> Predictor:
    return Predictor(dataset_type=dtype)

predictor = load_predictor(dataset_type)

# ─── Main header ─────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">✍️ Handwritten Character Recognition</p>', unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">Using Convolutional Neural Networks &nbsp;|&nbsp; '
    f'Mode: <strong>{"MNIST Digits (0–9)" if dataset_type == "mnist" else "EMNIST Characters (47 classes)"}</strong></p>',
    unsafe_allow_html=True
)

col_input, col_result = st.columns([1, 1], gap="large")

# ─── Input column ─────────────────────────────────────────────────────────────
with col_input:
    st.markdown("### Input")
    image_array = None

    if "🖊️ Draw on Canvas" in input_mode:
        try:
            from streamlit_drawable_canvas import st_canvas
            canvas_result = st_canvas(
                fill_color="rgba(255, 255, 255, 0.0)",
                stroke_width=18,
                stroke_color="#FFFFFF",
                background_color="#000000",
                height=280,
                width=280,
                drawing_mode="freedraw",
                key=f"canvas_{dataset_type}",
                return_image_data=True,
            )
            if canvas_result.image_data is not None:
                drawn = canvas_result.image_data
                _, processed_drawing = preprocess_image_for_model(drawn, dataset_type)
                if processed_drawing.max() > 0.05:
                    image_array = drawn
                    st.caption("✅ Drawing detected – click **Recognize** below.")
                else:
                    st.caption("👆 Draw a character above, then click Recognize.")
        except ImportError:
            st.warning("Canvas library not available. Please use Upload mode.")
            input_mode = "📁 Upload Image"

    if "📁 Upload Image" in input_mode:
        uploaded = st.file_uploader(
            "Upload a handwritten image (PNG, JPG, BMP)",
            type=["png", "jpg", "jpeg", "bmp"]
        )
        if uploaded:
            pil_img = Image.open(uploaded)
            st.image(pil_img, caption="Uploaded Image", width=200)
            image_array = pil_img

    st.markdown("---")
    recognize_btn = st.button("🔍 Recognize Character", type="primary", use_container_width=True)

# ─── Results column ──────────────────────────────────────────────────────────
with col_result:
    st.markdown("### Prediction")

    if recognize_btn:
        if image_array is None:
            st.error("⚠️ Please draw a character or upload an image first.")
        else:
            with st.spinner("Recognizing…"):
                try:
                    result = predictor.predict(image_array, top_k=3)

                    conf = result['confidence']
                    conf_class = "confidence-high" if conf >= 80 else ("confidence-med" if conf >= 50 else "confidence-low")
                    conf_emoji = "🟢" if conf >= 80 else ("🟡" if conf >= 50 else "🔴")

                    # Main prediction
                    st.markdown(f'<p class="big-char">{result["label"]}</p>', unsafe_allow_html=True)
                    st.markdown(f"""
                    <div class="result-box">
                      <div class="metric-label">Predicted Character</div>
                      <div style="font-size:1.8rem;font-weight:800;color:#1e3c72">{result["label"]}</div>
                      <div class="metric-label" style="margin-top:0.5rem">Confidence</div>
                      <div class="{conf_class}" style="font-size:1.5rem">{conf_emoji} {conf:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                    if conf < 50:
                        st.warning("⚠️ Low confidence prediction – the model is uncertain. Try a cleaner drawing.")

                    # Top-3 bar chart
                    st.markdown("#### Top-3 Predictions")
                    labels = [t[0] for t in result['top_k']]
                    probs  = [t[1] for t in result['top_k']]

                    fig_bar, ax = plt.subplots(figsize=(4, 2.2))
                    colors = ['#1e3c72', '#5b8dd9', '#aec6f0']
                    bars = ax.barh(labels[::-1], probs[::-1], color=colors[::-1], edgecolor='white')
                    ax.set_xlim(0, 105)
                    for bar, prob in zip(bars, probs[::-1]):
                        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                                f"{prob:.1f}%", va='center', fontsize=9, fontweight='bold')
                    ax.set_xlabel("Probability (%)")
                    ax.set_title("Top-3 Class Probabilities")
                    ax.spines[['top', 'right']].set_visible(False)
                    plt.tight_layout()
                    st.pyplot(fig_bar, use_container_width=True)
                    plt.close()

                    # Preprocessed image
                    st.markdown("#### Preprocessed Input (28×28)")
                    st.image(
                        result['preprocessed_image'],
                        caption="Normalized 28×28 image fed to CNN",
                        width=112,
                        clamp=True
                    )

                    # Saliency map
                    if show_saliency:
                        st.markdown("#### Saliency Map (Explainability)")
                        import torch
                        tensor_input, _ = preprocess_image_for_model(image_array, dataset_type)
                        tensor_input = tensor_input.to(predictor.device)
                        saliency = compute_saliency_map(predictor.model, tensor_input, target_class=result['class_index'])

                        orig_np = result['preprocessed_image']
                        hot = plt.colormaps['hot'](saliency)[:, :, :3]
                        orig_rgb = np.stack([orig_np] * 3, axis=-1)
                        overlay = np.clip(0.5 * orig_rgb + 0.5 * hot, 0, 1)

                        fig_sal, axes = plt.subplots(1, 3, figsize=(8, 2.5))
                        axes[0].imshow(orig_np, cmap='gray'); axes[0].set_title('Input'); axes[0].axis('off')
                        axes[1].imshow(saliency, cmap='hot'); axes[1].set_title('Saliency'); axes[1].axis('off')
                        axes[2].imshow(overlay); axes[2].set_title('Overlay'); axes[2].axis('off')
                        fig_sal.suptitle("Which pixels influenced the prediction?", fontsize=10)
                        plt.tight_layout()
                        st.pyplot(fig_sal, use_container_width=True)
                        plt.close()

                except Exception as e:
                    st.error(f"❌ Prediction failed: {e}")
    else:
        st.info("👈 Draw or upload an image, then click **Recognize Character**.")

# ─── Architecture info ─────────────────────────────────────────────────────────
with st.expander("📐 CNN Architecture & Model Info"):
    st.markdown(f"""
**Dataset Mode:** `{"MNIST" if dataset_type == "mnist" else "EMNIST Balanced"}`
**Classes:** `{"10 (digits 0–9)" if dataset_type == "mnist" else "47 (digits 0–9, uppercase A–Z, select lowercase)"}`
**Model Parameters:** `{"468,458" if dataset_type == "mnist" else "473,231"}`
**Input Shape:** `(1, 1, 28, 28)` – Batch × Channel × Height × Width

```
Input (1×28×28)
  → Conv Block 1: Conv2D(32) → BN → ReLU → Conv2D(32) → ReLU → MaxPool2D → Dropout(0.25)
  → Conv Block 2: Conv2D(64) → BN → ReLU → Conv2D(64) → ReLU → MaxPool2D → Dropout(0.25)
  → Flatten → Dense(128) → BN → ReLU → Dropout(0.5)
  → Output: Dense({10 if dataset_type == "mnist" else 47}, Softmax)
```

**Training:**
- Optimizer: Adam (lr=0.001, weight_decay=1e-4)
- Loss: Cross-Entropy
- Callbacks: EarlyStopping (patience=4), ReduceLROnPlateau, ModelCheckpoint
- Augmentation: Random Rotation ±10°, Affine Translate ±8%, Scale ±5%
""")
