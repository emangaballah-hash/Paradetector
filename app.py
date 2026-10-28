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
# 2. Model Loading (Cached for performance)
# ============================================================
@st.cache_resource
def load_models():
    # --- Download and Load YOLOv8 Model ---
    yolo_id = "1kE10xBaywLF7nZolKJSdOT1_OO4gkEP1"
    if not os.path.exists("best.pt"):
        gdown.download(f"https://drive.google.com/uc?id={yolo_id}", "best.pt", quiet=False)
    yolo_model = YOLO("best.pt")
    
    # --- Download and Load Secondary Morphology Classifier ---
    sec_id = "13445L-CX3F2HR7IFjll4KHqGrt66lAf-"
    if not os.path.exists("secondary_model.keras"):
        gdown.download(f"https://drive.google.com/uc?id={sec_id}", "secondary_model.keras", quiet=False)
    sec_model = tf.keras.models.load_model("secondary_model.keras")
    
    return yolo_model, sec_model

with st.spinner("Initializing ParaSherlock Models (Loading from Google Drive)..."):
    yolo_model, sec_model = load_models()

# Class mapping
class_names = {0: 'Giardia', 1: 'Amoeba', 2: 'Ascaris'}

# ============================================================
# 3. User Interface
# ============================================================
st.title("🕵️‍♀️ ParaSherlock: AI Parasite Detective")
st.markdown("""
**Welcome to ParaSherlock.** Upload a stool smear image for automated detection, counting, and morphological profiling of intestinal parasites.
*   **Stage 1 (Detection):** YOLOv8 localizes and counts parasites.
*   **Stage 2 (Morphology):** EfficientNetB0 analyzes the microscopic shape of each detected parasite.
""")

uploaded_file = st.file_uploader("Upload Image (JPG, JPEG, PNG)", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("Adjust Detection Sensitivity (Confidence Threshold)", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

# ============================================================
# 4. Inference Logic
# ============================================================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    if st.button("🔍 Analyze Image", type="primary"):
        with st.spinner("Running Two-Stage Pipeline (Detection + Morphology)..."):
            
            # Stage 1: YOLO Detection
            results = yolo_model.predict(source=image_np, conf=conf_threshold, verbose=False)
            result = results[0]
            
            counts = {0: 0, 1: 0, 2: 0}
            morphology_results = []
            
            # Process each detected bounding box
            for box in result.boxes:
                cls_id = int(box.cls[0])
                counts[cls_id] += 1
                
                # Crop parasite for Stage 2
                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(image_np.shape[1], x2), min(image_np.shape[0], y2)
                
                crop = image_np[y1:y2, x1:x2]
                
                if crop.size > 0 and crop.shape[0] > 10 and crop.shape[1] > 10:
                    # Preprocess crop for EfficientNet
                    crop_resized = cv2.resize(crop, (224, 224))
                    crop_array = np.expand_dims(crop_resized, axis=0)
                    
                    # Stage 2: Morphology Prediction
                    preds = sec_model.predict(crop_array, verbose=0)
                    pred_class = np.argmax(preds[0])
                    pred_conf = np.max(preds[0])
                    
                    morphology_results.append({
                        "Detected Class": class_names.get(cls_id, "Unknown"),
                        "Morphology Profile": class_names.get(pred_class, "Unknown"),
                        "Shape Confidence": f"{pred_conf*100:.1f}%"
                    })
            
            # ============================================================
            # 5. Display Results
            # ============================================================
            st.success("Analysis Complete!")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Intensity Report")
                total = 0
                for cid, count in counts.items():
                    if count > 0:
                        st.write(f"- **{class_names[cid]}**: {count} detected")
                        total += count
                st.info(f"🔬 **Total Parasites Detected**: {total}")
                
            with col2:
                st.subheader("🔬 Morphological Profiling")
                if len(morphology_results) > 0:
                    st.dataframe(morphology_results, use_container_width=True)
                else:
                    st.write("No detailed morphology available.")
            
            # Display Annotated Image
            st.subheader("🖼️ Detection Localization")
            res_plotted = result.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="Detected Parasites", use_column_width=True)
