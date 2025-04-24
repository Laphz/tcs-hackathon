# 💳 Credit Risk Prediction App

A Streamlit-based web application that predicts **credit risk** (High or Low) based on user inputs using a trained **Random Forest Classifier** on the German Credit dataset. This app is designed to assist financial institutions in assessing the creditworthiness of loan applicants.

---

## 📊 Features

- Interactive Streamlit UI
- Real-time prediction using trained model
- Input forms for dynamic feature entry
- Scalable and modular code structure
- Encoders and scaler used during training applied consistently in inference

---

## 🧠 Model Details

- **Model**: Random Forest Classifier
- **Target**: `HighCreditRisk` (1: High, 0: Low)
- **Data Source**: UCI German Credit Data
- **Preprocessing**:
  - Label Encoding for categorical variables
  - Feature Scaling using StandardScaler
- **Trained on**: Features like Age, Job, Housing, Purpose, etc.

---
