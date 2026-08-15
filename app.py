import os

import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "BreastInsight.h5")
CLASS_NAMES = ["benign", "malignant", "normal"]

st.set_page_config(page_title="BreastInsight", page_icon="🩻", layout="centered")


@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)


model = load_model()

st.title("🩻 BreastInsight")
st.markdown(
    "Classifies breast ultrasound images as **normal**, **benign**, or **malignant**. "
    "Trained on the [BUSI dataset](https://www.kaggle.com/datasets/aryashah2k/breast-ultrasound-images-dataset)."
)

st.error(
    "⚠️ **Not for clinical use.** This is a research/educational demo, not a diagnostic tool. "
    "It reaches 69.2% validation accuracy and misses over half of malignant cases in testing "
    "(malignant recall 0.47) — see **Model Performance** below before reading anything into "
    "a single prediction.",
    icon="⚠️",
)

uploaded = st.file_uploader("Upload a breast ultrasound image", type=["png", "jpg", "jpeg"])

if uploaded is not None:
    img = Image.open(uploaded).convert("RGB").resize((224, 224))
    st.image(img, caption="Uploaded image", width=300)

    # Model's first layer is Rescaling(1./255) -- feed raw 0-255 pixels, do not pre-normalize.
    x = np.expand_dims(np.array(img), axis=0)
    pred = model.predict(x, verbose=0)[0]
    probs = tf.nn.softmax(pred).numpy()
    top = int(np.argmax(probs))

    st.subheader(f"Prediction: **{CLASS_NAMES[top]}**")
    for name, p in sorted(zip(CLASS_NAMES, probs), key=lambda t: -t[1]):
        st.write(f"{name}: {p:.1%}")
        st.progress(float(p))

st.divider()
st.subheader("Model Performance")
st.markdown(
    """
Evaluated on a held-out validation set (156 images, 20% split):

| Metric | Value |
|---|---|
| Overall accuracy | 69.2% |
| Benign recall | 0.91 |
| Malignant recall | 0.47 |
| Normal recall | 0.14 |

Small, imbalanced training set (780 images: 437 benign / 210 malignant / 133 normal) — the
model is reliable on benign cases but misses over half of malignant cases and struggles badly
on normal scans. See the [repo](https://github.com/aljuhaeda/BreastInsight) for the full
data-integrity fix history behind these numbers.
"""
)
