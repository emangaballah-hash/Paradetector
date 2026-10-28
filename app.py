import streamlit as st
import gdown
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ============================================================
# 1. Page Configuration
# ============================================================
st.set_page_config(page_title="ParaSherlock", page_icon="🕵️‍♀️", layout="centered")

# ============================================================
# 2. Model Loading
# ============================================================
@st.cache_resource
def load_yolo_model():
    yolo_id = "1kE10xBaywLF7nZolKJSdOT1_OO4gkEP1"
    if not os.path.exists("best.pt"):
        gdown.download(f"https://drive.google.com/uc?id={yolo_id}", "best.pt", quiet=False)
    return YOLO("best.pt")

with st.spinner("Initializing ParaSherlock (Loading YOLO Model)..."):
    yolo_model = load_yolo_model()

# Class mapping
class_names = {0: 'Giardia', 1: 'Amoeba', 2: 'Ascaris'}

# ============================================================
# 3. User Interface
# ============================================================
st.title("🕵️‍♀️ ParaSherlock: Digital Parasite Detective")
st.markdown("""
**Welcome to ParaSherlock.** Upload a stool smear image for automated detection and intensity estimation of intestinal parasites.
""")

uploaded_file = st.file_uploader("Upload Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Adjust Detection Sensitivity", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

# ============================================================
# 4. Inference Logic
# ============================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Running Detection Pipeline..."):
            
            # YOLO Detection
            results = yolo_model.predict(source=image_np, conf=conf_threshold, verbose=False)
            result = results[0]
            
            counts = {0: 0, 1: 0, 2: 0}
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                counts[cls_id] += 1
            
            # ============================================================
            # 5. Display Results
            # ============================================================
            st.success("Analysis Complete!")
            
            st.subheader("📊 Intensity Report")
            total = 0
            for cid, count in counts.items():
                if count > 0:
                    st.write(f"- **{class_names[cid]}**: {count} detected")
                    total += count
            
            if total == 0:
                st.info("✅ No parasites detected in this image.")
            else:
                st.info(f"🔬 **Total Parasites Detected**: {total}")
            
            # Display Annotated Image
            st.subheader("🖼️ Detection Localization")
            res_plotted = result.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Detected Parasites", use_column_width=True)
