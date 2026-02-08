import streamlit as st
import numpy as np
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# ------------------ Load model and scaler safely ------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(BASE_DIR, "models", "hcv_model.pkl")
scaler_path = os.path.join(BASE_DIR, "models", "scaler.pkl")

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

# ------------------ Page Config ------------------
st.set_page_config(page_title="HCV Prediction App", layout="wide")

# ------------------ Sidebar Navigation ------------------
page = st.sidebar.selectbox(
    "Navigate",
    ["Home", "Prediction", "Insights", "Dataset Info"]
)

# ------------------ Home Page ------------------
if page == "Home":
    st.title("Hepatitis C – Overview")

    col1, col2 = st.columns(2)
    with col1:
        st.image("images/hep1.png", use_column_width=True)
    with col2:
        st.image("images/hep2.png", use_column_width=True)

    st.markdown("""
    ### What is Hepatitis C?
    Hepatitis C is a viral infection that affects the liver and can lead to serious liver disease such as
    **cirrhosis and liver cancer**. Many people do not show symptoms until later stages.
    """)

    st.subheader("Symptoms")
    st.image("images/hep3.png", width=600)
    st.markdown("""
    - Fatigue  
    - Fever  
    - Dark urine  
    - Abdominal pain  
    - Nausea  
    - Jaundice (yellowing of eyes and skin)
    """)

    st.subheader("Transmission")
    st.markdown("""
    - Contact with infected blood  
    - Sharing needles  
    - Unsafe medical injections  
    - Blood transfusions before 1992  
    """)

    st.subheader("Prevention")
    st.markdown("""
    - Do not share needles  
    - Use sterile medical tools  
    - Screen blood donations  
    """)

    st.subheader("Treatment")
    st.image("images/hep4.png", width=600)
    st.markdown("""
    Hepatitis C can be cured with antiviral medications.
    Early detection increases treatment success.
    """)

# ------------------ Prediction Page ------------------
elif page == "Prediction":
    st.title("Hepatitis C Stage Prediction")

    age = st.number_input("Age", 1, 100, 30)
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

    input_data = np.array([[age, alb, alp, alt, ast, bil, che, chol, crea, ggt, prot]])

    if st.button("Predict"):
        input_scaled = scaler.transform(input_data)
        pred = model.predict(input_scaled)[0]

        labels = {
            0: "Blood Donor (Healthy)",
            1: "Suspected Hepatitis",
            2: "Hepatitis",
            3: "Fibrosis",
            4: "Cirrhosis"
        }
        st.success(f"Prediction: **{labels[pred]}**")



# ------------------ Insights Page ------------------
elif page == "Insights":
    st.title("Insights Dashboard")

    data = {
        "Stage": ["Healthy", "Hepatitis", "Fibrosis", "Cirrhosis"],
        "Count": [120, 80, 40, 20]
    }

    df = pd.DataFrame(data)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Cases by Disease Stage")
        fig, ax = plt.subplots()
        ax.bar(df["Stage"], df["Count"])
        ax.set_xlabel("Disease Stage")
        ax.set_ylabel("Number of Patients")
        st.pyplot(fig)

    with col2:
        st.subheader("Case Distribution")
        fig2, ax2 = plt.subplots()
        ax2.pie(df["Count"], labels=df["Stage"], autopct="%1.1f%%")
        st.pyplot(fig2)

    st.metric("Total Patients", sum(df["Count"]))

# ------------------ Dataset Info Page ------------------
elif page == "Dataset Info":
    st.title("Dataset Information")

    st.markdown("""
    **Dataset:** Hepatitis C Dataset (UCI Repository)

    **Features Used:**
    - Age, Sex  
    - ALB, ALP, ALT, AST  
    - BIL, CHE, CHOL  
    - CREA, GGT, PROT  

    **Target:**
    - Disease Stage (Blood Donor, Hepatitis, Fibrosis, Cirrhosis)
    """)

    st.success("This dataset is used to train a machine learning model for disease stage prediction.")
