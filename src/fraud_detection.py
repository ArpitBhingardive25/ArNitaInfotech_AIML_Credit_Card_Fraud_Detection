# ==========================================
# CREDIT CARD FRAUD DETECTION - TRAINING FILE
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from imblearn.over_sampling import SMOTE


# ==========================================
# LOAD DATASET
# ==========================================

print("\n📂 Loading Dataset...")
data = pd.read_csv("data/fraudTest.csv")

print("✅ Dataset Loaded Successfully")
print("Dataset Shape:", data.shape)
print(data.head())


# ==========================================
# DROP UNNECESSARY COLUMNS
# ==========================================

drop_cols = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last']

for col in drop_cols:
    if col in data.columns:
        data.drop(col, axis=1, inplace=True)


# ==========================================
# ENCODE CATEGORICAL COLUMNS
# ==========================================

encoder = LabelEncoder()

for col in data.select_dtypes(include=['object']).columns:
    data[col] = encoder.fit_transform(data[col])


# ==========================================
# SPLIT FEATURES & TARGET
# ==========================================

X = data.drop('is_fraud', axis=1)
y = data['is_fraud']


# ==========================================
# HANDLE CLASS IMBALANCE
# ==========================================

print("\n⚖ Applying SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)


# ==========================================
# TRAIN TEST SPLIT
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42
)


# ==========================================
# FEATURE SCALING
# ==========================================

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ==========================================
# TRAIN MODELS
# ==========================================

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100)
}

print("\n🚀 Training Models...")

for name, model in models.items():

    print("\n==============================")
    print(f"Model: {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


# ==========================================
# SAVE FINAL MODEL + SCALER
# ==========================================

print("\n💾 Saving Final Model...")

final_model = RandomForestClassifier(n_estimators=100)
final_model.fit(X_train, y_train)

joblib.dump(final_model, "models/fraud_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print("✅ Model Saved Successfully!")
print("✅ Scaler Saved Successfully!")
print("\n🎉 TRAINING COMPLETE!")
