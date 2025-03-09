import streamlit as st
import matplotlib.pyplot as plt

# ตั้งค่าหน้าเว็บให้แสดงผลเต็มจอและกำหนด Title
st.set_page_config(page_title="Neural Network Model Explaination", layout="wide")

# แสดงหัวข้อหลักของเอกสาร
st.title("🧠 Neural Network Model Explaination")

url = "https://www.kaggle.com/datasets/misrakahmed/vegetable-image-dataset/data"

st.write("ผมได้นำข้อมูลมาจาก [vetgetable image dataset](%s) จำนวน 15 ชนิด และได้เลือกมา 3 ชนิดเพื่อความแม่นยำของ Model มากยิ่งขึ้น" % url)
col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    ✅ **หมวดหมู่ของผักทั้งหมดได้**:
    - 🥕 **Carrot** (แครอท)
    - 🍈 **Papaya** (มะละกอ)
    - 🍠 **Radish** (หัวไชเท้า)
    - 🥬 **Cabbage** (กะหล่ำปลี)
    - 🍅 **Tomato** (มะเขือเทศ)
    - 🌸 **Cauliflower** (กะหล่ำดอก)
    - 🫑 **Capsicum** (พริกหยวก)
    - 🍀 **Bitter Gourd** (มะระ)
    - 🥦 **Broccoli** (บรอกโคลี)
    - 🥔 **Potato** (มันฝรั่ง)
    - 🥒 **Cucumber** (แตงกวา)
    - 🌿 **Bean** (ถั่ว)
    - 🍆 **Brinjal** (มะเขือม่วง)
    - 🥒 **Bottle Gourd** (บวบ)
    - 🎃 **Pumpkin** (ฟักทอง)
    """)
with col2:
    st.markdown("""
    ✅ **หมวดหมู่ของผักเราสามารถเลือกได้**:
    - 🥬 **Cauliflower** (กะหล่ำปลี)
    - 🫑 **Capsicum** (มะเขือเทศ)
    - 🍠 **Radish** (พริกหยวก)
    """)
st.markdown("---")

# การเตรียมข้อมูล
st.header("1️⃣ การเตรียมข้อมูล (Data Preparation)")
st.write("""
การเตรียมข้อมูลเป็นขั้นตอนสำคัญก่อนที่จะนำไปใช้ Train โมเดล เพื่อให้สามารถเรียนรู้และทำนายผลได้อย่างแม่นยำ โดยมีขั้นตอนดังนี้:

**1. เลือกพืชที่มีความแตกต่างกันชัดเจน**  
- คัดเลือกเฉพาะพืชที่มีลักษณะเด่นชัดและไม่คล้ายกัน เพื่อลดปัญหาความซับซ้อนของการเรียนรู้โมเดล  
- ตัวอย่างพืชที่เลือก: **Tomato, Radish, Cauliflower**  

**2. สร้างโฟลเดอร์สำหรับเก็บข้อมูลที่เลือก**  
- กำหนดเส้นทางของไฟล์ที่ต้องการคัดลอก  
- ตรวจสอบว่าโฟลเดอร์ปลายทางมีอยู่หรือไม่ หากไม่มีให้สร้างขึ้นมา  

**3. คัดลอกเฉพาะโฟลเดอร์ของพืชที่ต้องการ**  
- ทำการคัดลอกข้อมูลเฉพาะโฟลเดอร์ที่เลือกไปยังโฟลเดอร์ใหม่ เพื่อใช้เป็น Training Set  

**4. แสดงตัวอย่างภาพของข้อมูลที่เลือก**  
- โหลดภาพตัวอย่างจากโฟลเดอร์ที่เลือก และแสดงผลในรูปแบบของ Subplot  
- ปรับแต่งภาพให้แสดงชื่อหมวดหมู่ของพืชที่เลือกไว้  
""")
st.image("./image/NN/1.png", caption="🔹 โครงสร้างข้อมูลที่ใช้ในโมเดล")
st.image("./image/NN/2.png", caption="🔹 ตัวอย่างภาพจาก Dataset",width=600)
st.markdown("---")

# ตรวจสอบภาพที่อาจเสียหาย
st.header("2️⃣ ตรวจสอบภาพที่อาจเสียหาย")
st.write("กระบวนการนี้ใช้ตรวจสอบภาพที่อาจมีปัญหาในการโหลดหรือใช้งาน")
st.image("./image/NN/3.png", caption="🔹 ตรวจสอบไฟล์ภาพ")
st.markdown("---")

# การเตรียมข้อมูลก่อนนำไปใช้เทรนโมเดล
st.header("3️⃣ การเตรียมข้อมูลสำหรับ Image Classification")
st.write("ใช้ **Image Data Generator** เพื่อโหลดและปรับแต่งภาพสำหรับการเทรนโมเดล")
st.write("""
- **Normalization**: ใช้ `rescale=1.0/255.0` เพื่อปรับค่าพิกเซลของภาพจากช่วง **0-255** ให้อยู่ในช่วง **0-1**  
- **Data Augmentation**: ใช้เทคนิคการหมุน, การเลื่อน, การขยาย และการพลิกภาพแนวนอน เพื่อเพิ่มความหลากหลายของข้อมูลฝึกสอน 
- **โหลดข้อมูลจากโฟลเดอร์, ปรับขนาดรูปให้เป็น 150x150 px,  โหลดรูปภาพทีละ 32 รูป**
""")
st.image("./image/NN/4.png", caption="🔹 กระบวนการโหลดข้อมูลภาพ")
st.markdown("---")

st.header("4️⃣ Model Architecture")
st.write("""
### **รายละเอียดของโมเดล CNN**
🔹 **Convolutional Layers**
- **Conv2D (32 filters, 3x3 kernel, ReLU activation)** → ใช้ 32 ฟิลเตอร์เพื่อดึงคุณสมบัติจากภาพ
- **MaxPooling2D (2x2)** → ลดขนาดของ Feature Maps เพื่อลดภาระการคำนวณ
- **Conv2D (64 filters, 3x3 kernel, ReLU activation)** → ใช้ 64 ฟิลเตอร์เพื่อเพิ่มความสามารถในการดึงคุณสมบัติเชิงลึก
- **MaxPooling2D (2x2)** → ลดขนาดของข้อมูลอีกครั้งเพื่อรักษาคุณสมบัติสำคัญ

🔹 **Fully Connected Layers**
- **Flatten Layer** → แปลงข้อมูลจาก 2D Feature Maps ให้เป็น 1D Vector สำหรับการจำแนก
- **Dense (48 neurons, ReLU activation)** → Fully Connected Layer ที่ช่วยให้โมเดลเรียนรู้รูปแบบของข้อมูล
- **Dense (num_classes, Softmax activation)** → เลเยอร์สุดท้ายสำหรับการจำแนกภาพออกเป็นจำนวนคลาสที่ต้องการ
""")
st.image("./image/NN/5.png", caption="🔹 การสร้างโมเดล")
st.markdown("---")

# การฝึกโมเดล (Training)
st.header("5️⃣ กระบวนการฝึกโมเดล (Model Training)")
st.write("เราใช้โมเดล CNN ที่ออกแบบมาให้เรียนรู้จากข้อมูลภาพ และฝึกด้วย **Adam Optimizer**")
st.image("./image/NN/6.png", caption="🔹 กระบวนการเทรนโมเดล")
st.markdown("---")


st.header("6️⃣ Model Performance 📊")
st.write("ทำการวัดค่า Acuracy ของโมเดล")
st.image("./image/NN/7.png", caption="🔹 📈 Accuracy & Loss codes")
st.write("### **✅ Accuracy & Loss Graphs**")
st.image("./image/NN/8.png", caption="🔹 📈 Accuracy & Loss Graphs")
st.markdown("---")

st.header("📌 Deploy on Streamlit")

st.write("บันทึกโมเดลและโหลดเข้า Streamlit")
st.image("./image/NN/9.png")
st.markdown("---")
