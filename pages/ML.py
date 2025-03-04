import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# ตั้งค่าหน้าเว็บ
st.set_page_config(page_title="💎 Diamond Price Prediction", layout="wide")

# โหลดโมเดลที่ฝึกไว้แล้ว
svm_model = joblib.load("./models/diamond_price_model_svr.pkl")
rf_model = joblib.load("./models/diamond_price_model_rf.pkl")

# โหลด dataset เพื่อนำ feature names มาใช้งาน
df = ["carat","cut","color","clarity","depth","table","price","x","y","z"]
df = pd.DataFrame(df)

# กำหนด Feature ที่ใช้ในโมเดล
feature_columns = ["carat", "cut", "color", "clarity", "table", "x", "y", "z"]

# 🎨 สไตล์ UI
st.markdown("<h1 style='text-align: center;'>💎 Diamond Price Prediction 💎</h1>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align: center;'>🔹 ทำนายราคาของเพชรตามคุณสมบัติที่กำหนด 🔹</h4>", unsafe_allow_html=True)
st.write("---")

# 📌 คำอธิบายเกี่ยวกับแต่ละ Feature
st.subheader("🔍 คำอธิบายคุณสมบัติของเพชร")
st.markdown("""
- **💠 Carat**: น้ำหนักของเพชร (ยิ่งเยอะ ราคายิ่งสูง)
- **🔷 Cut**: การเจียระไนของเพชร (ระดับคุณภาพ: Ideal > Premium > Very Good > Good > Fair)
- **🌈 Color**: สีของเพชร (D คือใสที่สุด → J คือเหลืองที่สุด)
- **🔬 Clarity**: ความสะอาดของเพชร (เช่น IF คือไม่มีตำหนิเลย ส่วน I1 คือมีตำหนิเยอะ)
- **📏 Table**: ขนาดสัดส่วนของด้านบนเพชร (%)
- **📏 X (mm)**: ความยาวของเพชรในมิลลิเมตร
- **📏 Y (mm)**: ความกว้างของเพชรในมิลลิเมตร
- **📏 Z (mm)**: ความลึกของเพชรในมิลลิเมตร
""")

st.write("---")

# 📌 ฟังก์ชันสุ่มข้อมูลเพชร
def generate_random_diamond():
    return {
        "carat": np.random.uniform(0.5, 1.3),
        "cut": np.random.choice(["Ideal", "Premium", "Very Good", "Good", "Fair"], p=[0.40, 0.25, 0.22, 0.10, 0.03]),
        "color": np.random.choice(["G", "E", "F", "H", "I", "D", "J"], p=[0.20, 0.18, 0.17, 0.15, 0.12, 0.10, 0.08]),
        "clarity": np.random.choice(["SI1", "VS2", "SI2", "VS1", "VVS2", "VVS1", "IF", "I1"], p=[0.28, 0.27, 0.18, 0.15, 0.07, 0.035, 0.01, 0.005]),
        "table": np.random.uniform(56, 59),
        "x": np.random.uniform(4.75, 6.5),
        "y": np.random.uniform(4.7, 6.5),
        "z": np.random.uniform(2.8, 4)
    }

# กำหนดค่าเริ่มต้นของเพชรแบบสุ่ม
if "random_diamond" not in st.session_state:
    st.session_state.random_diamond = generate_random_diamond()

if st.button("🔀 สุ่มข้อมูลเพชร"):
    st.session_state.random_diamond = generate_random_diamond()

random_data = st.session_state.random_diamond

# 📌 ช่องกรอกข้อมูล
st.subheader("🎯 ป้อนค่าคุณสมบัติของเพชร")
col1, col2 = st.columns(2)
with col1:
    carat = st.number_input("💠 Carat", min_value=0.2, max_value=5.0, step=0.01, value=random_data["carat"])
    cut = st.selectbox("🔷 Cut", ["Fair", "Good", "Very Good", "Premium", "Ideal"], index=["Fair", "Good", "Very Good", "Premium", "Ideal"].index(random_data["cut"]))
    color = st.selectbox("🌈 Color", ["D", "E", "F", "G", "H", "I", "J"], index=["D", "E", "F", "G", "H", "I", "J"].index(random_data["color"]))
    clarity = st.selectbox("🔬 Clarity", ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"], index=["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"].index(random_data["clarity"]))

with col2:
    table = st.number_input("📏 Table (%)", min_value=50.0, max_value=70.0, step=0.1, value=random_data["table"])
    x = st.number_input("📏 X (mm)", min_value=3.0, max_value=10.0, step=0.1, value=random_data["x"])
    y = st.number_input("📏 Y (mm)", min_value=3.0, max_value=10.0, step=0.1, value=random_data["y"])
    z = st.number_input("📏 Z (mm)", min_value=2.0, max_value=6.0, step=0.1, value=random_data["z"])

# 📌 ทำนายราคาเพชร
if st.button("📊 Predict Price"):
    input_df = pd.DataFrame([[carat, cut, color, clarity, table, x, y, z]], columns=feature_columns)
    try:
        svr_pred = svm_model.predict(input_df)[0]  
        rf_pred = rf_model.predict(input_df)[0]  

        st.session_state.svm_prediction = svr_pred
        st.session_state.rf_prediction = rf_pred
        st.success("✅ ทำนายราคาสำเร็จ! **:red[ ** หมายเหตุ:ค่าที่แสดงเป็นเพียงการพยากรณ์จากโมเดล AI เท่านั้น ราคาจริงอาจแตกต่างออกไป ขึ้นอยู่กับปัจจัยอื่นๆ* ]**")
    except Exception as e:
        st.error(f"❌ Error during prediction: {e}")

# 📌 แสดงผลการพยากรณ์
if "svm_prediction" in st.session_state and "rf_prediction" in st.session_state:
    st.subheader("💰 Predicted Diamond Prices")
    st.write(f"🔵 **SVR Model Prediction:** ${st.session_state.svm_prediction:,.2f}")
    st.write(f"🟢 **Random Forest Model Prediction:** ${st.session_state.rf_prediction:,.2f}")

# 📊 Performance Comparison
st.subheader("📊 Model Performance Comparison")
performance_data = {"Model": ["SVM", "Random Forest"], "R²": [0.9126, 0.9788], "MAE": [466.328, 219.954], "RMSE": [765.652, 377.127]}
performance_df = pd.DataFrame(performance_data).set_index("Model")

fig, ax = plt.subplots(figsize=(5,3))
fig.set_size_inches(30, 10)
performance_df[["MAE", "RMSE"]].plot(kind="bar", ax=ax, color=["#FF5733", "#33FF57"])
st.pyplot(fig)
