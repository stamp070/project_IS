import streamlit as st

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="💡 AI Model Showcase", layout="wide")

# สไตล์สำหรับเนื้อหาที่อยู่กึ่งกลาง
def center_text(text, size=24, color="black"):
    st.markdown(f"<h{size} style='text-align: center; color: {color};'>{text}</h{size}>", unsafe_allow_html=True)

# ส่วนหัวของหน้า

### 🔷 **ข้อมูลของนักศึกษา**
st.header("📌 ข้อมูลนักศึกษา")

st.subheader("👨‍🎓 นักศึกษา")
st.write("**ชื่อ:** นาย วรธน มีมูล")
st.write("**รหัสนักศึกษา:** 6604062630498")
st.write("**Section:** 3")
st.write("**มหาวิทยาลัย:** มหาวิทยาลัยเทคโนโลยีพระจอมเกล้าพระนครเหนือ")
st.write("**สาขาวิชา:** วิทยาการคอมพิวเตอร์")



st.markdown("---")

### 🔷 **โมเดลที่พัฒนา**
st.header("📊 โมเดลที่พัฒนา")

# จัด layout เป็น 2 คอลัมน์
col1, col2 = st.columns(2)

with col1:
    st.subheader("📌 Model 1: Image Classification (CNN)")
    st.write("""
    - **Dataset:** Vetgetable Image 🥦
    - **ประเภทข้อมูล:** เลือก 3 รูปภาพจาก 15 หมวดหมู่ (Capsicum, Cauliflower, Radish)  
    - **อัลกอริทึม:** Convolutional Neural Network (CNN)  
    - **เป้าหมาย:** ทำนายภาพจากหมวดหมู่ Vetgetable Image
    - **ผลลัพธ์:** แสดงผลการจำแนกหมวดหมู่พร้อมความมั่นใจ  
    """)
    if st.button("🔍 ทดสอบโมเดล Image Classification"):
        st.switch_page("pages/NN.py")

with col2:
    st.subheader("📌 Model 2: Diamond Price Prediction")
    st.write("""
    - **Dataset:** ข้อมูลราคาเพชร 💎  
    - **ประเภทข้อมูล:** คุณสมบัติของเพชร (น้ำหนัก, สี, ความสะอาด ฯลฯ)  
    - **อัลกอริทึม:** Random Forest Regressor & Support Vector Regression (SVR)  
    - **เป้าหมาย:** ทำนายราคาของเพชร  
    - **ผลลัพธ์:** แสดงราคาที่ทำนาย พร้อมค่าความผิดพลาดของโมเดล  
    """)
    if st.button("💎 ทดสอบโมเดล Diamond Price Prediction"):
        st.switch_page("pages/ML.py")

st.markdown("---")

### 🔷 **แนะนำการใช้งาน**
st.header("📖 วิธีใช้งานโมเดล")

st.write("""
1️⃣ **เลือกโมเดลที่ต้องการทดสอบ**  
2️⃣ **CNN Image Classification:** อัปโหลดรูปภาพ แล้วระบบจะจำแนกหมวดหมู่ให้  
3️⃣ **Diamond Price Prediction:** ป้อนคุณสมบัติของเพชร ระบบจะทำนายราคาให้  
4️⃣ **ดูผลลัพธ์ของโมเดล**  
""")

st.success("🎯 พร้อมเริ่มใช้งานแล้ว! เลือกโมเดลที่ต้องการทดสอบด้านบน 👆")
