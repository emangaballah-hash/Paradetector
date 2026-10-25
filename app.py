import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# تحميل النموذج
@st.cache_resource
def load_model():
    return YOLO("best.pt")

model = load_model()
class_names = {0: 'Giardia', 1: 'Amoeba', 2: 'Ascaris'}

st.title("🦠 AI Parasite Detector")
st.write("ارفعي صورة مسحة برازية، وسيقوم الذكاء الاصطناعي بالكشف عن الطفيليات وتقدير شدة العدوى.")

uploaded_file = st.file_uploader("ارفعي صورة المسحة هنا", type=["jpg", "jpeg", "png"])
conf_threshold = st.slider("حساسية الكشف (Confidence)", min_value=0.1, max_value=0.9, value=0.25, step=0.05)

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    
    if st.button("🔍 فحص الصورة"):
        with st.spinner("جاري التحليل..."):
            results = model.predict(source=image_np, conf=conf_threshold)
            result = results[0]
            
            counts = {0: 0, 1: 0, 2: 0}
            for box in result.boxes:
                cls_id = int(box.cls[0])
                counts[cls_id] += 1
            
            st.subheader("📊 تقدير شدة العدوى:")
            total = 0
            for cls_id, count in counts.items():
                if count > 0:
                    st.write(f"🔹 **{class_names[cls_id]}**: {count} طفيلي")
                    total += count
            
            if total == 0:
                st.success("✅ لم يتم العثور على طفيليات (أو الصورة غير واضحة).")
            else:
                st.info(f"🔬 **الإجمالي**: {total} طفيليات")
            
            res_plotted = result.plot()
            res_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
            st.image(res_rgb, caption="النتيجة (الصناديق المحيطة)", use_column_width=True)
