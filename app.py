"""
Plant Disease Detector - Streamlit app
Run:  streamlit run app.py
"""
import json
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image

IMG_SIZE = (224, 224)

# General treatment tips matched by keyword in the disease name.
# These are basic guidance only - confirm with a local agriculture expert.
TREATMENTS = {
    "healthy": "The leaf looks healthy. Keep up regular watering, sunlight and monitoring.",
    "late_blight": "Remove and destroy infected leaves. Avoid overhead watering and use a recommended fungicide.",
    "early_blight": "Remove infected leaves, rotate crops, and apply a suitable fungicide.",
    "bacterial_spot": "Avoid working with wet plants, remove infected leaves, and use copper-based sprays.",
    "leaf_mold": "Improve air circulation, reduce humidity, and apply a fungicide if severe.",
    "septoria": "Remove infected leaves, mulch the soil, and apply a suitable fungicide.",
    "spider_mites": "Spray leaves with water or insecticidal soap; use a miticide if heavy.",
    "target_spot": "Remove infected leaves, improve airflow, and use a recommended fungicide.",
    "mosaic_virus": "No cure. Remove and destroy infected plants and control insects that spread it.",
    "yellow_leaf_curl": "No cure. Remove infected plants, control whiteflies, use resistant varieties.",
    "powdery_mildew": "Improve airflow, avoid overcrowding, and apply sulfur or a suitable fungicide.",
    "rust": "Remove infected leaves and apply a suitable fungicide.",
    "scab": "Remove fallen leaves, prune for airflow, and use a fungicide in the early season.",
    "rot": "Remove infected parts, improve drainage and airflow, and apply a fungicide.",
    "leaf_scorch": "Remove infected leaves, water at the base, and avoid overcrowding.",
    "leaf_spot": "Remove infected leaves, avoid overhead watering, and use a suitable fungicide.",
}
DEFAULT_TIP = "Remove affected leaves, isolate the plant if possible, and consult a local agriculture expert."


@st.cache_resource
def load():
    model = tf.keras.models.load_model("plant_model.keras")
    names = json.load(open("class_names.json"))
    return model, names


def treatment_for(label: str) -> str:
    key = label.lower()
    for k, tip in TREATMENTS.items():
        if k in key:
            return tip
    return DEFAULT_TIP


def pretty(label: str):
    plant, _, disease = label.partition("___")
    return plant.replace("_", " "), disease.replace("_", " ")


st.set_page_config(page_title="Plant Disease Detector", page_icon="🌿")
st.title("🌿 Plant Disease Detector")
st.write("Upload a clear photo of a single leaf to detect its disease.")

model, class_names = load()
file = st.file_uploader("Leaf image", type=["jpg", "jpeg", "png"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, caption="Uploaded leaf", use_container_width=True)

    arr = np.expand_dims(np.array(img.resize(IMG_SIZE), dtype="float32"), 0)
    probs = model.predict(arr)[0]  # preprocessing is inside the model
    top3 = np.argsort(probs)[::-1][:3]

    label = class_names[top3[0]]
    plant, disease = pretty(label)
    conf = probs[top3[0]] * 100

    st.subheader(f"{plant}: {disease}")
    st.metric("Confidence", f"{conf:.1f}%")
    if conf < 60:
        st.warning("Low confidence. Try a clearer, closer photo of one leaf.")

    st.info("**Suggested treatment:** " + treatment_for(label))

    with st.expander("Top 3 predictions"):
        for i in top3:
            p, d = pretty(class_names[i])
            st.write(f"{p} - {d}: {probs[i] * 100:.1f}%")

st.caption("For learning purposes. Confirm with an agriculture expert before treating crops.")
