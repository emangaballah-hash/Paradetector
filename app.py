import streamlit as st
import gdown
import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
import tensorflow as tf

# ============================================================
# 1. Page Configuration
# ============================================================
st.set_page_config(page_title="ParaSherlock", page_icon="🕵️‍♀️", layout="centered")

# ============================================================
# 2. Model Loading
# ============================================================
@st.cache_resource
def load_models():
    # YOLO
    yolo_id = "1kE10xBaywLF7nZolKJSdOT1_OO4gkEP1"
    if not os.path.exists("best.pt"):
        gdown.download(f"https://drive.google.com/uc?id={yolo_id}", "best.pt", quiet=False)
    yolo_model = YOLO("best.pt")
    
    # Secondary
    sec_id = "13445L-CX3F2HR7IFjll4KHqGrt66lAf-"
    if not os.path.exists("secondary_model.keras"):
        gdown.download(f"https://drive.google.com/uc?id={sec_id}", "secondary_model.keras", quiet=False)
    sec_model = tf.keras.models.load_model("secondary_model.keras")
    
    return yolo_model, sec_model

with st.spinner("Initializing ParaSherlock..."):
    yolo_model, sec_model = load_models()

class_names = {0: 'Giardia', 1: 'Amoeba', 2: 'Ascaris'}

# ============================================================
# 3. User Interface
# ============================================================
st.title("🕵️‍♀️ ParaSherlock: Digital Parasite Detective")
st.markdown("""
Upload an image for automated detection, counting, and morphological assessment.
""")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Detection Sensitivity", 0.1, 0.9, 0.25, 0.05)

# ============================================================
# 4. Inference
# ============================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Running Two-Stage Pipeline..."):
            results = yolo_model.predict(source=image_np, conf=conf_threshold, verbose=False)
            result = results[0]
            
            counts = {0: 0, 1: 0, 2: 0}
            detailed_results = []
            
            for box in result.boxes:
                cls_id = int(box.cls[0])
                counts[cls_id] += 1
                
                # Crop
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
                crop = image_np[y1:y2, x1:x2]
                
                # Secondary Model
                morph_conf = 0.0
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    crop_resized = cv2.resize(crop, (224, 224))
                    crop_array = np.expand_dims(crop_resized, axis=0)
                    preds = sec_model.predict(crop_array, verbose=0)
                    morph_conf = np.max(preds[0]) * 100
                
                detailed_results.append({
                    "Detected Species (YOLO)": class_names.get(cls_id, "Unknown"),
                    "Morphological Confidence": f"{morph_conf:.1f}%"
                })
            
            # Display
            st.success("Analysis Complete!")
            st.subheader("📊 Intensity & Morphological Report")
            
            total = sum(counts.values())
            if total == 0:
                st.info("No parasites detected.")
            else:
                st.write(f"🔬 **Total Parasites Detected**: {total}")
                for cid, count in counts.items():
                    if count > 0:
                        st.write(f"- **{class_names[cid]}**: {count}")
                
                st.dataframe(detailed_results, use_container_width=True)
            
            # Image
            res_plotted = result.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Detection Localization", use_column_width=True)
