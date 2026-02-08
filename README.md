# 💳 Credit Card Fraud Detection System
## 📌 Project Overview

This project is a Machine Learning based Credit Card Fraud Detection System that identifies whether a transaction is fraudulent or legitimate. The system is trained on a real-world dataset and deployed as an interactive Streamlit Web Application where users can upload transaction data and get fraud predictions instantly.

## 🎯 Objectives

- Detect fraudulent credit card transactions
- Handle imbalanced transaction data using SMOTE
- Compare multiple machine learning models
- Deploy trained model using Streamlit
- Provide real-time fraud prediction from uploaded CSV data

## 📊 Dataset

Source: Kaggle Fraud Detection Dataset
File Used: fraudTest.csv

## Dataset Contains:

- Transaction details
- Merchant information
- Location data
- Transaction time data
- Fraud Label (is_fraud)

## 🛠️ Technologies Used
### Programming Language

Python

### Libraries

- Pandas
- NumPy
- Scikit-learn
- Matplotlib
- Seaborn
- Imbalanced-learn (SMOTE)
- Joblib
- Streamlit

## 🧠 Machine Learning Models Used
Model	Purpose
Logistic Regression	Baseline Model
Decision Tree	Rule-based Classification
Random Forest	Final Selected Model

✅ Random Forest gave best performance and was used for deployment.

🔄 Project Workflow
1️⃣ Data Preprocessing

Removed unnecessary columns

Encoded categorical variables

Feature scaling using StandardScaler

2️⃣ Handling Imbalanced Data

Used SMOTE (Synthetic Minority Oversampling Technique)

3️⃣ Model Training

Logistic Regression

Decision Tree

Random Forest

4️⃣ Model Evaluation

Accuracy Score

Classification Report

Confusion Matrix Visualization

5️⃣ Model Deployment

Saved Model using Joblib

Saved Scaler for consistent predictions

Built Streamlit Web App

🌐 Streamlit Web Application Features

✅ Upload Transaction CSV
✅ Automatic Data Preprocessing
✅ Fraud Prediction
✅ Fraud Probability Score
✅ Fraud Count Summary

📂 Project Structure
credit_card_fraud_detection/
│
├── data/
│   └── fraudTest.csv
│
├── models/
│   ├── fraud_model.pkl
│   └── scaler.pkl
│
├── src/
│   ├── fraud_detection.py
│   └── app.py
│
├── requirements.txt
└── README.md

▶️ How to Run This Project
🔹 Step 1 — Install Dependencies
pip install -r requirements.txt

🔹 Step 2 — Train Model
python src/fraud_detection.py


This will:

Train ML models

Save model file

Save scaler file

🔹 Step 3 — Run Web App
python -m streamlit run src/app.py


Then open:
```
http://localhost:8501
```
## 📈 Sample Output

Fraud Prediction (0 = Legit, 1 = Fraud)
Fraud Probability Score
Fraud Transaction Count

## 🔑 Key Learnings

Handling Imbalanced Data in ML
Model Training & Evaluation
ML Model Serialization
Building ML Web Apps using Streamlit
Debugging Real Deployment Issues

## 🚀 Future Improvements

Add Dashboard Visualizations
Deploy on Cloud (Streamlit Cloud / AWS)
Add Real-time Transaction Prediction
Add User Authentication

👨‍💻 Author
Arpit Bhingardive

# ArNitaInfotech_AIML_credit_card_fraud_detection

