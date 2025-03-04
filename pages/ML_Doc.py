import streamlit as st

st.set_page_config(page_title="ML document", layout="wide")

# ส่วนหัวของหน้า
st.title("📌 Machine Learning Code Explanation")
st.write("\n💡 รายละเอียดและกระบวนการของโค้ดสำหรับพยากรณ์ราคาเพชรด้วย Machine Learning")

# แบ่ง Section
st.header("1️⃣ Data Exploration & Cleaning 🧼")

st.subheader("🔍 ตรวจสอบข้อมูลเบื้องต้น")
st.write("โหลดข้อมูลและตรวจสอบตัวอย่างข้อมูลจากไฟล์ Diamonds_Prices.csv")
st.code("""
df = pd.read_csv("Diamonds_Prices.csv")
df.head()
df.tail()
""", language="python")

# แสดงภาพตัวอย่าง
st.image("./image/ML/1.png", caption="🔹 ตัวอย่างข้อมูล (Head & Tail)")

st.subheader("🛠 ตรวจสอบค่าที่หายไป (Missing Values)")
st.code("""
df.isnull().sum()
""", language="python")
st.image("./image/ML/2.png", caption="🔹 ไม่มีค่า Missing Values")
st.write("---")

st.header("2️⃣ Data Cleaning & Preprocessing 🧽")
st.subheader("🗑 ลบคอลัมน์ที่ไม่จำเป็น")
st.write("ลบคอลัมน์ Unnamed: 0 ที่ไม่มีความจำเป็นในโมเดล")
st.code("df = df.drop(columns='Unnamed: 0', errors='ignore')", language="python")
st.image("./image/ML/3.png", caption="🔹 ลบคอลัมน์ Unnamed: 0")

st.subheader("📊 ตรวจสอบข้อมูลหมวดหมู่")
st.write("แสดงการกระจายตัวของข้อมูลประเภทข้อความ (Categorical Features)")
st.code("""
text_features = ["cut", "clarity", "color"]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, feature in enumerate(text_features):
    df[feature].value_counts().plot(kind='bar', ax=axes[i])
    axes[i].set_title(f"Distribution of {feature}")
    axes[i].set_xlabel(feature)
    axes[i].set_ylabel("Count")
plt.tight_layout()
plt.show()
""", language="python")
st.image("./image/ML/4.png", caption="🔹 การกระจายตัวของข้อมูลหมวดหมู่")

st.subheader("📊 ตรวจสอบข้อมูลตัวเลข")
st.write("ใช้ describe() เพื่อตรวจสอบค่าทางสถิติเบื้องต้นของข้อมูลตัวเลข")
st.code("df.describe()", language="python")
st.image("./image/ML/5.png", caption="🔹 ค่าทางสถิติของข้อมูลตัวเลข")

st.subheader("📌 Correlation Analysis")
st.write("ตรวจสอบความสัมพันธ์ของตัวแปรโดยใช้ Heatmap")
st.code("""
sns.heatmap(df.corr(numeric_only=True), annot=True")
""", language="python")
st.image("./image/ML/6.png", caption="🔹 Correlation Heatmap")

st.subheader("📌 ลบคอลัมน์ที่ไม่สำคัญ")
st.write("จากการวิเคราะห์พบว่า Depth มีความสัมพันธ์ต่ำกับ Price สามารถลบออกได้")
st.code("df = df.drop(columns=['depth'])", language="python")
st.image("./image/ML/7.png", caption="🔹 ลบคอลัมน์ Depth ออกจาก Dataframe")
st.write("---")


st.header("3️⃣ Feature Engineering & Preprocessing 🛠")
st.subheader("📌 จัดการ Outliers ด้วย IQR Method")
st.code("""
def no_outliers(column):
    if column.dtype in ['float64', 'int64']:
        q1, q3 = column.quantile([0.25, 0.75])
        lower_limit = q1 - 1.5 * (q3 - q1)
        upper_limit = q3 + 1.5 * (q3 - q1)
        column = column.clip(lower=lower_limit, upper=upper_limit)
    return column
df = no_outliers(df)
""", language="python")
st.image("./image/ML/8.png", caption="🔹 ลบคอลัมน์ Depth ออกจาก Dataframe")

st.subheader("📌 Scaling & Encoding")
st.code("""
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer

num_features = ["carat", "table", "x", "y", "z"]
cat_features = ["cut", "color", "clarity"]

preprocessor = make_column_transformer(
    (StandardScaler(), num_features),
    (OneHotEncoder(handle_unknown="ignore"), cat_features),
    remainder="drop"
)
""", language="python")
st.image("./image/ML/9.png", caption="🔹 Scaling & Encoding")
st.write("---")

st.header("4️⃣ Model Training & Evaluation 🎯")

st.subheader("📌 แบ่งข้อมูลออกเป็นชุดฝึกสอนและทดสอบ")
st.write("""
ก่อนการฝึกสอนโมเดล เราแบ่งข้อมูลออกเป็น **ชุดฝึกสอน (Training set)** และ **ชุดทดสอบ (Testing set)** 
เพื่อให้มั่นใจว่าโมเดลสามารถทำนายข้อมูลใหม่ได้อย่างมีประสิทธิภาพ
""")
st.code("""
from sklearn.model_selection import train_test_split
X = df.[feature]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
""", language="python")

st.subheader("📌 การฝึกสอนโมเดล")
st.write("""
เราใช้โมเดล **Random Forest Regressor** และ **Support Vector Regressor (SVR)** 
เพื่อนำมาเปรียบเทียบประสิทธิภาพในการพยากรณ์ราคาเพชร
- **Random Forest Regressor**: เป็นโมเดลที่ใช้การรวมผลจากหลายต้นไม้การตัดสินใจ (Decision Trees) 
เพื่อเพิ่มความแม่นยำและลดปัญหา Overfitting
- **Support Vector Regressor (SVR)**: ใช้หลักการของ **Support Vector Machine (SVM)** 
โดยการหาฟังก์ชันที่เหมาะสมที่สุดในการพยากรณ์ค่าเป้าหมาย โดยให้ความสำคัญกับค่าที่ใกล้เส้นคาดการณ์มากที่สุด
""")
st.code("""
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

# ฝึกสอนโมเดล Random Forest
rf_model = RandomForestRegressor(n_estimators=300, max_depth=15, min_samples_split=5, min_samples_leaf=2, random_state=42)
rf_model.fit(X_train, y_train)

# ฝึกสอนโมเดล SVR
svr_model = SVR(kernel='rbf', C=100, gamma=0.01, epsilon=0.1, max_iter=10000)
svr_model.fit(X_train, y_train)
""", language="python")

# แสดงรูปภาพของโค้ดที่ใช้ในการฝึกสอนโมเดล
st.image("./image/ML/10.png", caption="🔹 การสร้างและฝึกสอนโมเดล Random Forest")
st.image("./image/ML/11.png", caption="🔹 การสร้างและฝึกสอนโมเดล Random SVR")


st.subheader("📌 การประเมินผลของโมเดล")
st.write("""
หลังจากฝึกสอนโมเดลแล้ว เราต้องประเมินผลเพื่อดูว่าโมเดลสามารถพยากรณ์ข้อมูลใหม่ได้ดีเพียงใด โดยใช้เมตริกสำคัญดังนี้:
- **R² (Coefficient of Determination)**: วัดว่าสัดส่วนของค่าที่พยากรณ์สามารถอธิบายค่าจริงได้ดีแค่ไหน (ค่าที่ใกล้ 1 มากที่สุดแสดงถึงความแม่นยำสูง)
- **MAE (Mean Absolute Error)**: ค่าความคลาดเคลื่อนเฉลี่ยระหว่างค่าจริงกับค่าที่โมเดลพยากรณ์ได้
- **RMSE (Root Mean Squared Error)**: ใช้วัดค่าความคลาดเคลื่อนโดยให้ความสำคัญกับค่าผิดพลาดที่สูงกว่าค่าเฉลี่ย
""")
st.code("""
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    return {"R²": r2, "MAE": mae, "RMSE": rmse}

rf_metrics = evaluate_model(rf_model, X_test, y_test)
svr_metrics = evaluate_model(svr_model, X_test, y_test)
rf_metrics, svr_metrics
""", language="python")

# แสดงรูปภาพผลลัพธ์การประเมินโมเดล
st.image("./image/ML/12.png", caption="🔹 ผลลัพธ์การประเมินโมเดล Random Forest และ SVR")

st.write("📊 **สรุปผลการประเมินโมเดล**")
st.write("""
- **Random Forest**
  - R²: **0.97**
  - MAE: **219.95**
  - RMSE: **377.12**
- **SVR**
  - R²: **0.91**
  - MAE: **466.32**
  - RMSE: **765.65**
""")
st.write("""
จากผลลัพธ์ โมเดล **Random Forest** มีค่าความแม่นยำที่ดีกว่า **SVR** โดยมีค่า **R² สูงกว่า** 
และ **MAE & RMSE ต่ำกว่า** แสดงให้เห็นว่า Random Forest สามารถพยากรณ์ราคาเพชรได้ดีกว่าในกรณีนี้
""")
st.write("---")

# -------------------------------------------
# 5️⃣ Model Testing with Random Data 🔬
# -------------------------------------------
st.header("5️⃣ Model Testing with Random Data 🔬")

st.subheader("📌 Generating Random Diamond Data")
st.write("เพื่อประเมินประสิทธิภาพของโมเดล จึงสร้างฟังก์ชั่นสุ่มคุณสมบัติเพชรและคาดการณ์ราคา")

st.code("""
import numpy as np

cut_options = ["Fair", "Good", "Very Good", "Premium", "Ideal"]
color_options = ["D", "E", "F", "G", "H", "I", "J"]
clarity_options = ["I1", "SI2", "SI1", "VS2", "VS1", "VVS2", "VVS1", "IF"]

def generate_random_diamond():
    return {
        "carat": np.random.uniform(0.5, 1.3),
        "cut": np.random.choice(cut_options, p=[0.10, 0.15, 0.25, 0.30, 0.20]),
        "color": np.random.choice(color_options, p=[0.15, 0.15, 0.15, 0.15, 0.13, 0.12, 0.05]),
        "clarity": np.random.choice(clarity_options, p=[0.08, 0.10, 0.15, 0.15, 0.18, 0.15, 0.12, 0.07]),
        "table": np.random.uniform(56, 59),
        "x": np.random.uniform(4.5, 6.5),
        "y": np.random.uniform(4.7, 6.5),
        "z": np.random.uniform(2.8, 4)
    }

# Generate 5 random samples
random_diamonds = [generate_random_diamond() for _ in range(5)]
random_diamonds = pd.DataFrame(random_diamonds)
random_test_rf = rf_pipeline.predict(random_diamonds)
random_test_svr = svr_pipeline.predict(random_diamonds)
        
# Print        
print(random_test_rf)
print(random_test_svr)
""", language="python")

st.image("./image/ML/13.png", caption="🔹 Sample of Randomly Generated Diamond Data")

st.subheader("📌 Deploy on Streamlit")
st.write("บันทึกโมเดลและโหลดเข้า Streamlit เพื่อใช้ทำนายราคาเพชรแบบ Interactive")
st.code("""
import joblib
joblib.dump(rf_model, "rf_model.pkl")
joblib.dump(svr_model, "svr_model.pkl")
""", language="python")
st.write("---")
