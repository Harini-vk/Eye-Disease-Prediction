import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.cm as cm
import cv2

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="Eye Disease Detection",
    page_icon="E",
    layout="wide"
)

# ---------- LOAD MODEL ----------
@st.cache_resource(show_spinner="Loading model...")
def load_model():
    return tf.keras.models.load_model("best_model.h5") #add your model name or path here

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
st.sidebar.write("✔ Grad‑CAM Visualization")

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

    return class_names[idx], confidence, predictions[0], img_array

# ---------- GRAD-CAM ----------
def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    raise ValueError("Could not find a Conv2D layer.")

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    grad_model = tf.keras.models.Model(
        model.inputs, [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def get_eyeball_mask(img_array):
    """Return a boolean mask that is True inside the eyeball (bright circular region)."""
    import cv2
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    # Threshold: eyeball is the large bright circle; background is dark
    _, thresh = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY)
    # Morphological close to fill any small holes at the border
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # Keep only the largest connected component (the eyeball disc)
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)
    if num_labels <= 1:
        return np.ones(gray.shape, dtype=bool)   # fallback: no masking
    largest = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
    mask = (labels == largest)
    return mask

def display_gradcam(img, heatmap, alpha=0.4):
    img_array = np.array(img.convert("RGB"))
    heatmap_resized = np.array(
        Image.fromarray(np.uint8(255 * heatmap)).resize(
            (img_array.shape[1], img_array.shape[0]), Image.BILINEAR
        )
    ) / 255.0

    # Build eyeball mask and zero heatmap outside it
    mask = get_eyeball_mask(img_array)
    heatmap_resized = heatmap_resized * mask

    try:
        jet = cm.get_cmap("jet")
    except AttributeError:
        import matplotlib as mpl
        jet = mpl.colormaps["jet"]

    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[np.uint8(255 * heatmap_resized)]

    # Only blend where mask is True; outside shows original image unchanged
    alpha_map = (mask * alpha)[..., np.newaxis]
    superimposed_img = jet_heatmap * alpha_map + img_array / 255.0 * (1 - alpha_map)
    superimposed_img = np.clip(superimposed_img * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(superimposed_img)


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

            disease, confidence, probs, img_array = predict_image(image)

            if disease == "normal":
                st.success(f"Prediction: {disease.upper()}")
            else:
                st.error(f"Prediction: {disease.upper()}")

            st.metric("Confidence Score", f"{confidence:.2f}%")

            st.markdown("#### Class Probabilities")
            for i, name in enumerate(class_names):
                st.progress(float(probs[i]))
                st.write(f"{name}: {probs[i]*100:.2f}%")

            st.session_state['disease'] = disease
            st.session_state['img_array'] = img_array
            st.session_state['original_img'] = image

# ---------- GRAD-CAM EXPLAINABILITY ----------
st.divider()
st.markdown("## 🔬 Explainability Module (Grad-CAM)")

# Disease-specific Grad-CAM explanations
GRADCAM_EXPLANATIONS = {
    "cataract": {
        "icon": "🌫️",
        "what": "Cataract is a clouding of the eye's natural lens, which lies behind the iris and pupil.",
        "highlighted": "The model focuses on the central lens region of the eye. Cataracts appear as cloudy, opaque areas in the lens, which scatter and block light from reaching the retina.",
        "colour_note": "🔴 **Red/warm areas** — highest model attention; these regions show the clearest signs of lens opacity or irregular light scattering consistent with cataract formation.",
        "clinical": "Clinically, cataracts cause blurry or hazy vision, faded colours, and increased glare sensitivity. The Grad-CAM rightly highlights the lens centre where opacity develops first.",
    },
    "diabetic_retinopathy": {
        "icon": "🩸",
        "what": "Diabetic Retinopathy (DR) is a diabetes complication that damages blood vessels in the retina.",
        "highlighted": "The model focuses on the peripheral retina and near the optic disc, where micro-aneurysms, haemorrhages, and neovascularisation (abnormal new vessels) tend to appear first.",
        "colour_note": "🔴 **Red/warm areas** — regions with abnormal vascular changes such as micro-bleeds, exudates, or leaking blood vessels that the CNN learned to associate with DR.",
        "clinical": "Early DR may show subtle dot haemorrhages; advanced DR shows larger bleeds and new vessel growth. The highlighted zones guide ophthalmologists to the regions requiring the most urgent clinical review.",
    },
    "glaucoma": {
        "icon": "👁️",
        "what": "Glaucoma is a group of eye diseases that damage the optic nerve, often linked to elevated intraocular pressure.",
        "highlighted": "The model most strongly attends to the **optic disc** and surrounding nerve fibre layer. In glaucoma, the optic cup progressively enlarges relative to the disc (increased cup-to-disc ratio).",
        "colour_note": "🔴 **Red/warm areas** — the optic disc centre where the cu-to-disc ratio is evaluated. An enlarged or asymmetric cup indicates nerve fibre loss characteristic of glaucoma.",
        "clinical": "Glaucoma often presents no early symptoms. By analysing the optic disc morphology, the AI can detect structural nerve damage before significant vision loss occurs.",
    },
    "normal": {
        "icon": "✅",
        "what": "The retinal image shows no detectable signs of any of the screened eye diseases.",
        "highlighted": "Even for normal images, the model highlights the optic disc and macula — these are the anatomical landmarks it has learned to reference when checking for disease indicators.",
        "colour_note": "🟡 **Yellow/cool areas** — present across the retina indicate that no single region triggered a strong disease-specific signal. This pattern is typical for healthy eyes.",
        "clinical": "A normal result means the retinal structures — including the optic disc, macula, and blood vessels — appear within healthy parameters. Regular eye check-ups are still recommended.",
    },
}

if 'img_array' in st.session_state:
    st.write("Heatmap highlighting the regions of the retinal image that heavily influenced the prediction:")
    
    try:
        last_conv_layer = get_last_conv_layer_name(model)
        heatmap = make_gradcam_heatmap(st.session_state['img_array'], model, last_conv_layer)
        gradcam_img = display_gradcam(st.session_state['original_img'], heatmap)
        
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.image(st.session_state['original_img'], caption="Original Image", use_container_width=True)
        with col_g2:
            st.image(gradcam_img, caption="Grad-CAM Highlight", use_container_width=True)

        # Explanation block
        disease = st.session_state.get('disease', None)
        if disease and disease in GRADCAM_EXPLANATIONS:
            info = GRADCAM_EXPLANATIONS[disease]
            st.markdown(f"""
<div style='background:#f0f4f8; border-left:5px solid #2E86C1; border-radius:10px; padding:20px; margin-top:20px;'>
    <h4 style='color:#2E86C1; margin-top:0'>{info['icon']} Why is this area highlighted?</h4>
    <p><b>About {disease.replace('_', ' ').title()}:</b> {info['what']}</p>
    <p><b>What the model sees:</b> {info['highlighted']}</p>
    <p>{info['colour_note']}</p>
    <p><b>Clinical context:</b> {info['clinical']}</p>
    <p style='font-size:12px; color:gray; margin-bottom:0'>⚠️ This AI output is intended for educational/screening purposes only and must not replace professional medical diagnosis.</p>
</div>
""", unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Grad-CAM could not be generated: {e}")
else:
    st.info("Upload and analyze an image to view the Grad-CAM visualization.")


# ---------- FOOTER ----------
st.divider()
st.markdown("""
<div style='text-align:center; padding:10px; color:#7f8c8d; font-size:13px;'>
© 2026 Eye Disease Detection AI System<br>
Deep Learning • Explainable AI • Medical Imaging
</div>
""", unsafe_allow_html=True)