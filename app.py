import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Page config
st.set_page_config(
    page_title="Bone Fracture Detection",
    page_icon="🦴",
    layout="wide"
)

# Load model (cached so it only loads once)
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()

CLASS_NAMES = {
    0: "Elbow Fracture",
    1: "Finger Fracture",
    2: "Forearm Fracture",
    3: "Humerus Fracture",
    4: "Shoulder Fracture",
    5: "Wrist Fracture",
    6: "Hand Fracture",
    7: "Hip Fracture",
    8: "Leg Fracture",
}

# Header
st.title("🦴 Vision in the Wild — Bone Fracture Detection")
st.markdown("**AI-powered X-Ray Screening System | YOLOv8l | 9 Anatomical Classes**")
st.markdown("Upload an X-Ray image to receive instant fracture localization and classification.")
st.divider()

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    confidence = st.slider("Confidence Threshold (%)", 10, 90, 25, 5)
    st.divider()
    st.markdown("### 📋 Supported Classes")
    for k, v in CLASS_NAMES.items():
        st.markdown(f"**{k}:** {v}")
    st.divider()
    st.caption("Model trained on 15,000+ X-rays across 3 datasets. Achieved 51.5% mAP50.")

# Main layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("📤 Upload X-Ray")
    uploaded_file = st.file_uploader("Choose an X-Ray image", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        image_np = np.array(image)
        st.image(image, caption="Uploaded X-Ray", use_column_width=True)
        analyze = st.button("🔍 Analyze X-Ray", type="primary", use_container_width=True)
    else:
        analyze = False

with col2:
    st.subheader("📊 Detection Results")

    if uploaded_file and analyze:
        with st.spinner("Analyzing X-Ray..."):
            results = model.predict(image_np, conf=confidence / 100)
            result_img = results[0].plot()
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

        st.image(result_img, caption="Detection Output", use_column_width=True)

        st.divider()
        st.subheader("🩻 Diagnostic Report")

        if len(results[0].boxes) == 0:
            st.success(f"✅ No fractures detected above {confidence}% confidence threshold.")
        else:
            st.warning(f"⚠️ {len(results[0].boxes)} potential fracture(s) detected:")
            for box in results[0].boxes:
                class_id   = int(box.cls[0])
                class_name = CLASS_NAMES.get(class_id, f"Class {class_id}")
                conf_score = float(box.conf[0]) * 100
                st.markdown(f"- **{class_name}** — Confidence: `{conf_score:.1f}%`")

        st.caption("⚕️ This is an AI screening tool. Always consult a qualified radiologist for clinical diagnosis.")
    elif not uploaded_file:
        st.info("👈 Upload an X-Ray image on the left to get started.")
