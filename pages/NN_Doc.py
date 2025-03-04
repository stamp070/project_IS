import streamlit as st

st.set_page_config(page_title="NN document", layout="wide")

# Title Section
st.title("🧠 Neural Network Model Documentation (CIFAR-10)")
st.write("รายละเอียดโมเดล **CNN** ของเราที่ได้รับการบนชุดข้อมูล **CIFAR-10**")
st.write("---")

st.header("1️⃣ Dataset Overview 📊")

st.write("""
ชุดข้อมูล **CIFAR-10** ประกอบด้วย 60,000 ภาพสี ที่มีขนาด 32x32 พิกเซล แบ่งออกเป็น **10 หมวดหมู่** ได้แก่:
- ✈️ **เครื่องบิน (Airplane)**
- 🚗 **รถยนต์ (Automobile)**
- 🐦 **นก (Bird)**
- 🐱 **แมว (Cat)**
- 🦌 **กวาง (Deer)**
- 🐶 **สุนัข (Dog)**
- 🐸 **กบ (Frog)**
- 🐴 **ม้า (Horse)**
- 🚢 **เรือ (Ship)**
- 🚚 **รถบรรทุก (Truck)**
""")

st.image("./image/NN/1.png", caption="📸 Sample Images from CIFAR-10")
st.write("---")

st.header("2️⃣ Model Architecture 🏗️")
st.write("""
โมเดล **Convolutional Neural Network (CNN)** ของเรามีองค์ประกอบดังนี้:
- 3 ชั้น Convolutional Layers (Conv2D) ที่ใช้ฟิลเตอร์ขนาดต่างๆ กัน
- MaxPooling layers เพื่อลดขนาดของภาพ
- Flatten & Dense layers สำหรับการจำแนกหมวดหมู่
- Softmax activation สำหรับการคาดการณ์แบบ Multi-class
""")

st.image("./image/NN/2.png", caption="🛠️ Model Summary")
st.write("---")

st.header("3️⃣ Model Training ⚡")

st.write("""
โมเดลได้รับการฝึกโดยใช้:
- **ตัวปรับค่า (Optimizer)**: Adam
- **ฟังก์ชันค่าความสูญเสีย (Loss Function)**: Sparse Categorical Crossentropy
- **ตัวชี้วัดประสิทธิภาพ (Metric)**: Accuracy
- **รอบการฝึก (Epochs)**: 20"
""")

st.image("./image/NN/3.png", caption="📈 Training Process")
st.write("---")

st.header("4️⃣ Model Performance 📊")
st.write("ทำการวัดค่า Acuracy & Confusion Matrix ของโมเดล")
st.image("./image/NN/4.png", caption="📈 Accuracy & Loss Graphs")
st.image("./image/NN/5.png", caption="📈 Model Confusion Matrix")

col1, col2 = st.columns(2)

with col1:
    st.subheader("✅ Accuracy & Loss Graphs")
    st.image("./image/NN/graph.png", caption="📊 Training & Validation Metrics")

with col2:
    st.subheader("🟦 Confusion Matrix")
    st.image("./image/NN/Confusion matrix.png", caption="📊 Model Confusion Matrix")
st.write("---")

st.header("5️⃣ Precision, Recall, and F1 Score 📏")

st.image("./image/NN/precision.png", caption="📊 Report")

st.write("""
✅ หมวดที่ทำงานได้ดี:

- Automobile 🚗, Ship 🚢, Truck 🚚 และ Horse 🐴 เป็นหมวดที่โมเดลสามารถแยกแยะได้ดีที่สุด

⚠️ หมวดที่มีปัญหา:

- Cat 🐱 และ Dog 🐶 มีค่าความแม่นยำต่ำ อาจเกิดจากภาพมีลักษณะคล้ายกัน ทำให้โมเดลสับสน
Bird 🐦 มีค่า Recall สูงแต่ Precision ต่ำ หมายถึงโมเดลอาจมีการทำนายผิดพลาดมาก
""")
st.write("---")

st.subheader("📌 Deploy on Streamlit")
st.write("บันทึกโมเดลและโหลดเข้า Streamlit")
st.image("./image/NN/6.png")
st.write("---")

st.success("🎯 Thank you for exploring our Neural Network documentation!")
