import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

def load_and_preprocess(csv_path):
    df = pd.read_csv(csv_path)

    #  Remove accidental index column
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]

    # Handle missing values & duplicates
    df.fillna(df.mean(numeric_only=True), inplace=True)
    df.drop_duplicates(inplace=True)

    # Encode categorical columns
    le_sex = LabelEncoder()
    le_category = LabelEncoder()

    df["Sex"] = le_sex.fit_transform(df["Sex"])
    df["Category"] = le_category.fit_transform(df["Category"])

    # Split features & target
    X = df.drop("Category", axis=1)
    y = df["Category"]

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    #  Return feature names also
    return X_scaled, y, scaler, list(X.columns)
