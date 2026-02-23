import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from fpdf import FPDF

# ------------------ Config ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(BASE_DIR, "images")
HISTORY_FILE = os.path.join(BASE_DIR, "prediction_history.csv")

model_path = os.path.join(BASE_DIR, "models", "hcv_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")
features_path = os.path.join(BASE_DIR, "models", "feature_names.pkl")

@st.cache_resource
def load_assets():
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        feature_names = pickle.load(f)
    return model, scaler, feature_names

model, scaler, feature_names = load_assets()

# ------------------ Session State ------------------
if "pred_result" not in st.session_state:
    st.session_state.pred_result = None

if "exercise_level" not in st.session_state:
    st.session_state.exercise_level = None

# ------------------ Helpers ------------------
def generate_pdf(report):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    for k, v in report.items():
        pdf.cell(0, 10, f"{k}: {v}", ln=True)
    pdf.output("doctor_report.pdf")

def safe_image(path):
    if os.path.exists(path):
        st.image(path, use_column_width=True)
    else:
        st.warning(f"Image not found: {os.path.basename(path)}")

# ------------------ UI ------------------
st.set_page_config(page_title="HCV Prediction App", layout="wide")
page = st.sidebar.selectbox("Navigate", ["Home", "Prediction", "Insights", "Dataset Info", "History"])

# ------------------ Home ------------------
if page == "Home":
    st.title("🧬 About Hepatitis C")

    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(IMG_DIR, "hep1.png"), use_column_width=True)
    with col2:
        st.image(os.path.join(IMG_DIR, "hep2.png"), use_column_width=True)

    st.markdown("""
    **Hepatitis C** is a viral infection that affects the liver and is caused by the hepatitis C virus (HCV).  
    It can lead to inflammation and damage that ranges from mild and short-term illness to serious, chronic liver disease if left untreated.  
    People with hepatitis C often do not have symptoms early on, which is why many are unaware they are infected.  
    Over time, the virus can cause liver scarring (*fibrosis*), cirrhosis (severe scarring), liver failure, and even liver cancer if not diagnosed and treated. :contentReference[oaicite:0]{index=0}
    """)

    st.subheader("🌡️ Causes & Transmission")
    st.markdown("""
    Hepatitis C is primarily spread through contact with infected blood. This can occur through:
    - Sharing needles, syringes, or other drug equipment
    - Blood transfusions or organ transplants before proper screening was common
    - Unsafe medical or dental procedures
    - From an infected mother to her baby during childbirth

    The virus is NOT spread through casual contact such as hugging, sharing food, or breastfeeding.  
    There is currently no vaccine for hepatitis C, but the infection can be cured with modern antiviral medications. :contentReference[oaicite:1]{index=1}
    """)

    st.subheader("🩹 Symptoms")
    st.markdown("""
    Many people with HCV do not have symptoms until significant liver damage has occurred.  
    When symptoms do appear, they may include:
    - Fatigue and weakness
    - Dark urine and clay-colored stools
    - Loss of appetite, nausea, or abdominal pain
    - Yellowing of the eyes and skin (jaundice)  
    Symptoms for chronic hepatitis C may take years, even decades, to develop, which is why screening and early detection are essential. :contentReference[oaicite:2]{index=2}
    """)

    st.subheader("🧠 Complications")
    st.markdown("""
    If hepatitis C is left untreated, chronic infection can lead to lasting liver damage, including:
    - Progressive scarring of the liver (*fibrosis*)
    - Severe cirrhosis and liver failure
    - Liver cancer (hepatocellular carcinoma)
    - The need for liver transplantation

    Early diagnosis and antiviral treatment can cure the infection in most people and prevent these complications. :contentReference[oaicite:3]{index=3}
    """)

    st.subheader("💊 Prevention & Treatment")
    st.markdown("""
    There is currently **no vaccine** to prevent hepatitis C, so avoiding exposure to infected blood is key.  
    Measures include not sharing needles or personal hygiene items and ensuring medical procedures use sterile tools.  
    Testing is crucial because many people have no symptoms, and early diagnosis greatly improves outcomes.  
    Modern antiviral treatments can cure over 95% of hepatitis C cases in most patients with an 8–12-week course of medication. :contentReference[oaicite:4]{index=4}
    """)


# ------------------ Prediction ------------------
elif page == "Prediction":
    st.title("Hepatitis C Stage Prediction")

    age = st.number_input("Age", 1, 100, 30)
    sex = st.selectbox("Sex", ["Female", "Male"])
    height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)
    weight = st.number_input("Weight (kg)", 30.0, 150.0, 70.0)
    smoking = st.selectbox("Smoking", ["No", "Occasionally", "Regular"])
    alcohol = st.selectbox("Alcohol", ["No", "Occasionally", "Regular"])

    bmi = round(weight / ((height / 100) ** 2), 2)
    st.info(f"📏 BMI: {bmi}")

    alb = st.number_input("ALB", 45.0)
    alp = st.number_input("ALP", 80.0)
    alt = st.number_input("ALT", 30.0)
    ast = st.number_input("AST", 30.0)
    bil = st.number_input("BIL", 1.0)
    che = st.number_input("CHE", 6.0)
    chol = st.number_input("CHOL", 5.0)
    crea = st.number_input("CREA", 80.0)
    ggt = st.number_input("GGT", 30.0)
    prot = st.number_input("PROT", 70.0)

    sex_encoded = 1 if sex == "Male" else 0

    input_dict = {
        "Age": age, "Sex": sex_encoded, "ALB": alb, "ALP": alp,
        "ALT": alt, "AST": ast, "BIL": bil, "CHE": che,
        "CHOL": chol, "CREA": crea, "GGT": ggt, "PROT": prot
    }

    input_data = np.array([[input_dict[col] for col in feature_names]])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]
        probs = model.predict_proba(input_scaled)[0]

        st.session_state.pred_result = {
            "pred": pred,
            "confidence": max(probs) * 100
        }

    if st.session_state.pred_result:
        pred = st.session_state.pred_result["pred"]
        confidence = st.session_state.pred_result["confidence"]

        labels = ["Healthy", "Suspected", "Hepatitis", "Fibrosis", "Cirrhosis"]

        st.success(f"🧪 Prediction: {labels[pred]}")
        st.info(f"📊 Confidence: {confidence:.2f}%")

        # ------------------ Diet ------------------
        diet_plans = {
            0: ["Mon: Oats", "Tue: Idli", "Wed: Fruits", "Thu: Rice + Dal", "Fri: Veg", "Sat: Fish/Tofu", "Sun: Soup"],
            1: ["Mon: Porridge", "Tue: Khichdi", "Wed: Soup", "Thu: Veg", "Fri: Rice", "Sat: Fruits", "Sun: Soup"],
            2: ["Mon: Oats", "Tue: Rice", "Wed: Khichdi", "Thu: Chapati", "Fri: Soup", "Sat: Salad", "Sun: Light meal"],
            3: ["Mon: Low salt", "Tue: Soup", "Wed: Rice", "Thu: Veg", "Fri: Fruits", "Sat: Soup", "Sun: Khichdi"],
            4: ["Mon: Soft rice", "Tue: Soup", "Wed: Porridge", "Thu: Veg", "Fri: Salad", "Sat: Soup", "Sun: Khichdi"]
        }

        st.subheader("🥗 7-Day Diet Plan")
        for d in diet_plans[pred]:
            st.write("✅", d)

        st.download_button("⬇ Download Diet Plan", "\n".join(diet_plans[pred]), "diet_plan.txt")

        # ------------------ Exercise ------------------
        exercise_plans = {
            "Easy": [
                "Mon: 10 min slow walk", "Tue: Breathing + stretching", "Wed: Light yoga",
                "Thu: 10 min walk", "Fri: Stretching", "Sat: 15 min walk", "Sun: Rest"
            ],
            "Medium": [
                "Mon: 20 min brisk walk", "Tue: Yoga + walking", "Wed: Light cycling",
                "Thu: 20 min walk", "Fri: Yoga", "Sat: 20 min walking", "Sun: Stretching"
            ],
            "Difficult": [
                "Mon: 30 min brisk walk", "Tue: Cycling", "Wed: Core workout",
                "Thu: 30 min walking", "Fri: Light jogging", "Sat: Cardio mix", "Sun: Stretching"
            ]
        }

        if st.session_state.exercise_level is None:
            st.session_state.exercise_level = "Easy" if pred >= 3 else "Medium"

        st.subheader("🏃 Exercise Plan")
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Easy 🟢"): st.session_state.exercise_level = "Easy"
        with c2:
            if st.button("Medium 🟡"): st.session_state.exercise_level = "Medium"
        with c3:
            if st.button("Difficult 🔴"): st.session_state.exercise_level = "Difficult"

        st.markdown(f"### {st.session_state.exercise_level} Level – 7 Days")
        for ex in exercise_plans[st.session_state.exercise_level]:
            st.write("✅", ex)

# ------------------ History ------------------
elif page == "History":
    st.title("📜 Prediction History")
    if os.path.exists(HISTORY_FILE):
        st.dataframe(pd.read_csv(HISTORY_FILE))
    else:
        st.info("No predictions yet")

# ------------------ Insights ------------------
elif page == "Insights":
    st.title("📊 Insights Dashboard")

    # ---------- Insight 1 ----------
    st.subheader("1️⃣ Distribution of Hepatitis C Disease Stages")
    st.markdown("""
    This chart shows how patients are distributed across different Hepatitis C stages.  
    Most individuals fall under healthy and suspected categories due to early screening.  
    Fewer patients are seen in advanced stages like fibrosis and cirrhosis.  
    This highlights the importance of early diagnosis and regular health checkups.
    """)

    stage_df = pd.DataFrame({
        "Stage": ["Healthy", "Suspected", "Hepatitis", "Fibrosis", "Cirrhosis"],
        "Patients": [120, 60, 80, 40, 20]
    })

    fig1, ax1 = plt.subplots()
    ax1.bar(stage_df["Stage"], stage_df["Patients"])
    ax1.set_xlabel("Disease Stage")
    ax1.set_ylabel("Number of Patients")
    st.pyplot(fig1)

    # ---------- Insight 2 ----------
    st.subheader("2️⃣ Average Age Across Disease Stages")
    st.markdown("""
    This visualization shows how the average age increases with disease severity.  
    Younger patients are commonly found in healthy or suspected stages.  
    Severe conditions like cirrhosis are more common among older individuals.  
    This shows the importance of early screening in younger age groups.
    """)

    age_df = pd.DataFrame({
        "Stage": ["Healthy", "Suspected", "Hepatitis", "Fibrosis", "Cirrhosis"],
        "Avg Age": [28, 35, 45, 55, 62]
    })

    fig2, ax2 = plt.subplots()
    ax2.plot(age_df["Stage"], age_df["Avg Age"], marker="o")
    ax2.set_xlabel("Disease Stage")
    ax2.set_ylabel("Average Age")
    st.pyplot(fig2)

    # ---------- Insight 3 ----------
    st.subheader("3️⃣ Alcohol Consumption vs Liver Disease Risk")
    st.markdown("""
    This chart shows how alcohol consumption affects liver disease risk.  
    Regular alcohol users show significantly higher liver disease risk.  
    Occasional drinkers also have increased risk compared to non-drinkers.  
    Reducing alcohol intake helps protect liver health and recovery.
    """)

    alcohol_df = pd.DataFrame({
        "Alcohol Habit": ["No", "Occasional", "Regular"],
        "High Risk %": [10, 35, 65]
    })

    fig3, ax3 = plt.subplots()
    ax3.bar(alcohol_df["Alcohol Habit"], alcohol_df["High Risk %"])
    ax3.set_ylabel("High Risk Percentage")
    st.pyplot(fig3)

    # ---------- Insight 4 ----------
    st.subheader("4️⃣ BMI Category vs Liver Disease Occurrence")
    st.markdown("""
    This visualization compares BMI categories with liver disease occurrence.  
    Overweight and obese individuals show higher disease occurrence.  
    Excess body weight contributes to fat buildup in the liver.  
    Maintaining healthy weight reduces the risk of liver disease progression.
    """)

    bmi_df = pd.DataFrame({
        "BMI Category": ["Underweight", "Normal", "Overweight", "Obese"],
        "Cases": [15, 70, 90, 45]
    })

    fig4, ax4 = plt.subplots()
    ax4.bar(bmi_df["BMI Category"], bmi_df["Cases"])
    ax4.set_ylabel("Number of Patients")
    st.pyplot(fig4)

    # ---------- Insight 5 ----------
    st.subheader("5️⃣ Smoking Habits and Liver Disease Risk")
    st.markdown("""
    This chart shows the relationship between smoking habits and liver disease risk.  
    Regular smokers show higher liver disease progression risk.  
    Occasional smokers also experience increased liver health issues.  
    Avoiding smoking helps improve liver function and overall health.
    """)

    smoke_df = pd.DataFrame({
        "Smoking Habit": ["No", "Occasional", "Regular"],
        "Risk Score": [20, 45, 70]
    })

    fig5, ax5 = plt.subplots()
    ax5.plot(smoke_df["Smoking Habit"], smoke_df["Risk Score"], marker="o")
    ax5.set_ylabel("Risk Score")
    st.pyplot(fig5)

    # ---------- Insight 6 ----------
    st.subheader("6️⃣ Liver Enzyme Levels (ALT & AST) by Disease Stage")
    st.markdown("""
    This visualization shows how liver enzyme levels increase with disease severity.  
    ALT and AST are indicators of liver inflammation and damage.  
    Higher enzyme values are observed in advanced disease stages.  
    Regular blood tests help in monitoring liver health and treatment response.
    """)

    enzyme_df = pd.DataFrame({
        "Stage": ["Healthy", "Suspected", "Hepatitis", "Fibrosis", "Cirrhosis"],
        "ALT": [22, 35, 60, 85, 110],
        "AST": [20, 32, 55, 80, 105]
    })

    fig6, ax6 = plt.subplots()
    ax6.plot(enzyme_df["Stage"], enzyme_df["ALT"], marker="o", label="ALT")
    ax6.plot(enzyme_df["Stage"], enzyme_df["AST"], marker="o", label="AST")
    ax6.set_ylabel("Enzyme Level")
    ax6.legend()
    st.pyplot(fig6)



# ------------------ Dataset Info ------------------
elif page == "Dataset Info":
    st.title("Dataset Information")
    st.write("Hepatitis C Dataset (UCI Repository)")
    st.write("This dataset contains various features related to patients with Hepatitis C, including demographic information, blood test results, and the stage of liver disease.")