import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# โหลดโมเดลที่ฝึกไว้
model = tf.keras.models.load_model("./models/vegetable_cnn_model.h5")

# หมวดหมู่ของผัก 15 ชนิด
class_names = [ "Capsicum","Cauliflower", "Radish"]

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="Vegetable Image Classification", layout="wide")

# ส่วนหัวของหน้า
st.title("🥦 ระบบจำแนกภาพผักด้วย CNN")
st.write("📌 อัปโหลดรูปภาพและให้โมเดลช่วยทำนาย!")

# แสดงรายการของหมวดหมู่ผัก 15 ชนิด
st.subheader("📋 รายการหมวดหมู่ที่สามารถจำแนกได้")

st.markdown("""
✅ **หมวดหมู่ของผักที่สามารถจำแนกได้**:
- 🥬 **Cauliflower** (กะหล่ำปลี)
- 🫑 **Capsicum** (มะเขือเทศ)
- 🍠 **Radish** (พริกหยวก)
""")

# แสดงตัวอย่างของผักแต่ละประเภท
st.subheader("🔍 ตัวอย่างหมวดหมู่ของผักที่สามารถจำแนกได้")
sample_images = {
    "🫑 Capsicum": "./image/NN/capsicum.jpg",
    "🍠 Radish": "./image/NN/radish.jpg",
    "🥬 Cauliflower": "./image/NN/cauliflower.jpg",
}

col1, col2, col3, col4 = st.columns(4)
for col, (label, img_path) in zip([col1, col2, col3, col4], sample_images.items()):
    col.image(img_path, caption=label,width=180)

st.markdown("---")  # เส้นคั่น

# คำแนะนำเกี่ยวกับการอัปโหลดรูปภาพ
st.subheader("📤 วิธีการอัปโหลดรูปภาพ")
st.write("""
✅ **เลือกรูปภาพที่ต้องการ** โดยสามารถเป็นรูปที่มีอยู่แล้วในอุปกรณ์ของคุณ  
✅ **ประเภทของรูปภาพ**: ไฟล์ต้องเป็น **JPG, JPEG หรือ PNG**  
✅ **ภาพควรมีความชัดเจน** และควรเป็นภาพที่ตรงกับหมวดหมู่ของผักที่รองรับ  
""")

# อัปโหลดภาพ
uploaded_file = st.file_uploader("📤 กรุณาอัปโหลดภาพของคุณที่นี่...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # แสดงภาพที่อัปโหลด
    image = Image.open(uploaded_file)
    st.image(image, caption="📸 ภาพที่อัปโหลด", width=200)

    # แปลงภาพให้ตรงกับรูปแบบที่โมเดลใช้
    img_array = np.array(image.resize((150, 150))) / 255.0  # ปรับขนาดและ Normalize
    img_array = np.expand_dims(img_array, axis=0)  # เพิ่มมิติให้เหมาะกับโมเดล

    # ทำการทำนาย
    prediction = model.predict(img_array)
    predicted_class = class_names[np.argmax(prediction)]
    confidence_scores = prediction[0] * 100  # คำนวณเปอร์เซ็นต์ความมั่นใจ
    
    # แสดงผลลัพธ์
    st.subheader(f"🔍 **ผลลัพธ์: {predicted_class}**")

    # แสดงกราฟแท่งความน่าจะเป็นของแต่ละคลาส
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.barh(class_names, confidence_scores, color="lightgreen")
    ax.set_xlim(0, 100)  # กำหนดขอบเขตของแกน X
    ax.set_xlabel("Confidence (%)")
    ax.set_title("Class Probability Distribution")
    st.pyplot(fig)

st.markdown("---")  # เส้นคั่น
st.write("🎯 **หมายเหตุ**: โมเดลถูกฝึกด้วยชุดข้อมูลผัก 15 ชนิด และอาจมีข้อจำกัดในการทำนายภาพที่แตกต่างจากภาพฝึก")
