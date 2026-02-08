from preprocess import load_and_preprocess
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os
import pandas as pd

os.makedirs("models", exist_ok=True)

# Load & preprocess
X, y, scaler, feature_names = load_and_preprocess("hcvdat0.csv")

# Convert to DataFrame to preserve feature order
X = pd.DataFrame(X, columns=feature_names)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train model
model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
print("Model Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Save model artifacts
with open("models/hcv_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("models/feature_names.pkl", "wb") as f:
    pickle.dump(feature_names, f)

print("\nModel, scaler, and feature names saved successfully!")
print("Features used:", feature_names)
