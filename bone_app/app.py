import streamlit as st
import torch
import cv2
import numpy as np
from model import BoneResNet

# =====================
# App Config
# =====================
st.set_page_config(
    page_title="Bone Fracture Detection",
    layout="centered"
)

st.title("🦴 Bone Fracture Detection")
st.write("Upload an X-ray image to predict if there is a fracture.")

# =====================
# Device
# =====================
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# Preprocessing (CLAHE)
# =====================
def equalize(img):
    img = img.astype("uint8")
    clahe = cv2.createCLAHE(tileGridSize=(8,8))
    img = clahe.apply(img)
    img = img / 255.0
    return img

# =====================
# Load Model
# =====================
@st.cache_resource
def load_model():
    model = BoneResNet().to(device)
    model.load_state_dict(
        torch.load("best_model.pth", map_location=device)
    )
    model.eval()
    return model

model = load_model()

# =====================
# Image Upload
# =====================
uploaded_file = st.file_uploader(
    "Upload X-ray Image",
    type=["jpg", "png", "jpeg"]
)

if uploaded_file is not None:
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)

    st.image(img, caption="Uploaded Image", channels="GRAY")

    # Preprocess
    img = cv2.resize(img, (224,224))
    img = equalize(img)
    img = np.expand_dims(img, axis=(0,1))
    img = torch.tensor(img, dtype=torch.float32).to(device)

    # Predict
    with torch.no_grad():
        logit = model(img)
        prob = torch.sigmoid(logit).item()   # 👈 مهم

    if prob < 0.5:
        st.error(f"🟥 Fractured\n\nConfidence: {(1-prob)*100:.2f}%")
    else:
        st.success(f"🟩 Not Fractured\n\nConfidence: {prob*100:.2f}%")

