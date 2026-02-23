# preprocess.py
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    # 🔥 DROP unwanted index column if present
    if "Unnamed: 0" in df.columns:
        df.drop(columns=["Unnamed: 0"], inplace=True)

    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    df["Sex"] = df["Sex"].map({"f": 0, "m": 1}).fillna(df["Sex"])
    df["Category"] = df["Category"].astype("category").cat.codes

    X = df.drop("Category", axis=1)
    y = df["Category"]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    feature_names = list(X.columns)

    return X_scaled, y, scaler, feature_names
