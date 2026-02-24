import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Eye Disease Detection",
    page_icon="E",
    layout="wide"
)

# ---------- LOAD MODEL ----------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("best_model.h5")

model = load_model()
class_names = ['cataract', 'diabetic_retinopathy', 'glaucoma', 'normal']

# ---------- CUSTOM CSS ----------
st.markdown("""
<style>
.main-title {
    font-size: 150px;
    font-weight: bold;
    color: #2E86C1;
}
.subtitle {
    font-size: 18px;
    color: gray;
}
.card {
    padding: 20px;
    border-radius: 12px;
    background-color: #f5f7fa;
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.footer {
    text-align: center;
    color: gray;
    font-size: 14px;
}
</style>
""", unsafe_allow_html=True)

# ---------- HEADER ----------
# ---- CSS FIRST ----
st.markdown("""
<style>
.main-title {
    font-size: 60px;
    font-weight: bold;
    color: #2E86C1;
}
.subtitle {
    font-size: 28px;
    color: gray;
}
</style>
""", unsafe_allow_html=True)

# ---- THEN TITLE ----
st.markdown('<p class="main-title">Explainable Eye Disease Detection System</p>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">CNN‑based Retinal Disease Classification</p>', unsafe_allow_html=True)

st.divider()

# ---------- SIDEBAR ----------
st.sidebar.title("🩺 About System")
st.sidebar.info(
    "This AI system detects eye diseases from retinal images "
    "using Deep Learning (CNN). Explainability with Grad‑CAM "
    "will be integrated in future versions."
)

st.sidebar.markdown("### 📌 Features")
st.sidebar.write("✔ Disease Classification")
st.sidebar.write("✔ Confidence Score")
st.sidebar.write("✔ Multi‑class Prediction")
st.sidebar.write("🚧 Grad‑CAM Visualization (Coming Soon)")

# ---------- FILE UPLOADER ----------
uploaded_file = st.file_uploader(
    "📤 Upload Retinal Image",
    type=["jpg", "jpeg", "png"]
)

# ---------- PREDICTION FUNCTION ----------
def predict_image(img):
    img = img.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array, verbose=0)
    idx = np.argmax(predictions[0])
    confidence = predictions[0][idx] * 100

    return class_names[idx], confidence, predictions[0]

# ---------- MAIN CONTENT ----------
if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Uploaded Image")
        st.image(image, use_container_width=True)

    with col2:
        st.markdown("### Prediction Panel")

        if st.button("Analyze Image"):

            disease, confidence, probs = predict_image(image)

            if disease == "normal":
                st.success(f"Prediction: {disease.upper()}")
            else:
                st.error(f"Prediction: {disease.upper()}")

            st.metric("Confidence Score", f"{confidence:.2f}%")

            st.markdown("#### Class Probabilities")
            for i, name in enumerate(class_names):
                st.progress(float(probs[i]))
                st.write(f"{name}: {probs[i]*100:.2f}%")

# ---------- GRAD-CAM COMING SOON ----------
st.divider()
st.markdown("## 🔬 Explainability Module")

st.info("🚧 Grad‑CAM Visualization — Coming Soon")

st.write("""
Future version will highlight disease‑affected regions in the retina
using **Grad‑CAM heatmaps** for explainable AI.
""")

# ---------- FOOTER ----------
st.divider()
st.markdown("""
<div style='text-align:center; padding:10px; color:#7f8c8d; font-size:13px;'>
© 2026 Eye Disease Detection AI System<br>
Deep Learning • Explainable AI • Medical Imaging
</div>
""", unsafe_allow_html=True)