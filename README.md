# Machine Learning–Based Prediction of Customer Churn Using Demographic, Financial, and Behavioral Features

## 📌 Project Overview
Customer churn is a major challenge in service-oriented industries such as banking, where losing customers directly impacts revenue and long-term growth.  
This project focuses on building an **end-to-end machine learning pipeline** to predict customer churn and explain **why customers are likely to leave** using **Explainable AI (XAI)** techniques.

The solution not only predicts churn but also provides **interpretable insights** that can help organizations design proactive and personalized retention strategies.

---

## 🎯 Problem Statement
Traditional churn analysis methods are reactive, rely on manual reporting, and fail to identify complex behavioral patterns among customers.  
There is a need for an automated, data-driven, and interpretable solution that can:
- Predict customers likely to churn  
- Explain the key factors contributing to churn  

---

## 🎯 Objectives
- Build a machine learning model to predict customer churn  
- Compare multiple ML models and select the best-performing one  
- Perform data preprocessing and feature engineering  
- Interpret model predictions using Explainable AI (SHAP)  

---

## 📂 Dataset
- **Dataset:** Banking Customer Churn Dataset  
- **Target Variable:** `Exited` (1 = Churn, 0 = Not Churn)  
- **Key Features:**  
  - Age  
  - Credit Score  
  - Balance  
  - Tenure  
  - Number of Products  
  - Customer Activity Status  
  - Estimated Salary  

---

## ⚙️ Methodology
1. **Data Preprocessing**
   - Missing value imputation  
   - Outlier handling using IQR capping  
   - Feature scaling and encoding  

2. **Feature Engineering**
   - Balance per product  
   - Log-transformed numerical features  
   - Correlation-based feature reduction  

3. **Model Training**
   - Logistic Regression  
   - Random Forest Classifier  
   - XGBoost Classifier  

4. **Model Selection**
   - Hyperparameter tuning using RandomizedSearchCV  
   - Model comparison using cross-validated ROC-AUC  

5. **Explainability**
   - Global and local interpretation using SHAP  

---

## 🤖 Machine Learning Models Used
- **Logistic Regression** – Baseline, interpretable model  
- **Random Forest** – Best-performing model (ROC-AUC ≈ 0.86)  
- **XGBoost** – Gradient boosting model for structured data  

---

## 📊 Results
- **Best Model:** Random Forest Classifier  
- **ROC-AUC (Test):** ~0.86  
- **Key Churn Drivers Identified:**  
  - Age  
  - Credit Score  
  - Number of Products  
  - Customer Activity Status  

SHAP analysis provided both **global feature importance** and **individual customer-level explanations**.

---

## 🧠 Explainable AI (SHAP)
- **Global Explainability:** Identifies overall churn-driving factors  
- **Local Explainability:** Explains why a specific customer is predicted to churn  

This improves trust, transparency, and business usability of the model.

---

## 🛠 Technologies Used
- **Programming Language:** Python  
- **Libraries:** Pandas, NumPy, Scikit-learn, XGBoost, SHAP  
- **Visualization:** Matplotlib, Seaborn  
- **Evaluation Metrics:** ROC-AUC, Precision, Recall, F1-score  
- **Environment:** Jupyter Notebook  

---

## 🚀 Future Scope
- Threshold optimization based on business requirements  
- Deployment using Streamlit or similar web frameworks  
- Integration with real-time customer data for continuous churn monitoring  

---

## 📌 Conclusion
This project demonstrates how machine learning combined with explainable AI can effectively predict customer churn while providing actionable insights.  
The approach enables data-driven, transparent, and proactive customer retention strategies in the banking sector.

---

## 👤 Author
**Rahul Pathania**

---

## ⭐ If you find this project useful
Feel free to ⭐ the repository and share feedback!
