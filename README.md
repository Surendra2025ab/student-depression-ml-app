# 🎓 Student Depression Prediction using Machine Learning

---

## 1. Problem Statement

Mental health issues among students are increasing globally due to academic pressure, financial stress, lifestyle imbalance, and personal challenges. Early detection of depression can help institutions and individuals take timely preventive measures.

This project aims to build and compare multiple **Machine Learning classification models** to predict whether a student is likely to be depressed (`1`) or not depressed (`0`) based on demographic, academic, and psychological factors.

An interactive web-based application has been developed using **Streamlit** to allow users to upload test data and evaluate model performance dynamically.

---

## 2. Dataset Description

**Dataset Name:** Student Depression Dataset  
**Source:** Kaggle (https://www.kaggle.com/datasets/hopesb/student-depression-dataset)  

### Target Variable:
- `Depression`
  - `0` → Not Depressed  
  - `1` → Depressed  

### Features Include:

- Gender  
- Age  
- City  
- Profession  
- Academic Pressure  
- Work Pressure  
- CGPA  
- Study Satisfaction  
- Job Satisfaction  
- Sleep Duration  
- Dietary Habits  
- Degree  
- Suicidal Thoughts  
- Work/Study Hours  
- Financial Stress  
- Family History of Mental Illness  

---

## 3. Machine Learning Models Used

The following six classification models were trained and evaluated:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbors (kNN)**
4. **Naive Bayes (Gaussian)**
5. **Random Forest (Ensemble Model)**
6. **XGBoost (Ensemble Model)**

---

## 4. Model Comparison Table

The models were evaluated using:

- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

> ℹ️ The values below were obtained by running the Streamlit application on the provided test dataset. Results may vary slightly depending on the dataset used for evaluation.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.8450 | 0.9181 | 0.8568 | 0.8825 | 0.8695 | 0.6787 |
| Decision Tree | 0.7724 | 0.7660 | 0.8072 | 0.8032 | 0.8052 | 0.5315 |
| kNN | 0.7283 | 0.7869 | 0.7349 | 0.8387 | 0.7834 | 0.4300
| Naive Bayes | 0.7923 | 0.8744 | 0.7832 | 0.8923 | 0.8342 | 0.5679 |
| Random Forest (Ensemble) | 0.8419 | 0.9133 | 0.8521 | 0.8834 | 0.8675 | 0.6725 |
| XGBoost (Ensemble) | 0.8367 | 0.9092 | 0.8502 | 0.8755 | 0.8627 | 0.6620 |

---

## Observations on Model Performance

| ML Model Name | Observation about Model Performance |
|---------------|--------------------------------------|
| **Logistic Regression** | Achieved the highest overall performance (Accuracy: 84.50%, AUC: 0.9181). Maintained strong balance between precision and recall, resulting in the highest MCC (0.6787). Indicates that the dataset is largely linearly separable. |
| **Decision Tree** | Moderate performance (Accuracy: 77.24%, AUC: 0.7660). Balanced precision and recall but lower MCC (0.5315) suggests weaker generalization. Likely affected by high variance typical of single-tree models. |
| **k-Nearest Neighbors (kNN)** | Lowest accuracy (72.83%) among all models. Higher recall but lower precision indicates more false positives. Lowest MCC (0.4300) suggests reduced stability in classification performance. |
| **Naive Bayes** | Good recall (0.8923) indicating strong ability to detect positive cases. Moderate accuracy (79.23%) and MCC (0.5679). Performance may be limited by the independence assumption between features. |
| **Random Forest (Ensemble)** | Strong ensemble performance (Accuracy: 84.19%, AUC: 0.9133). Balanced precision and recall with high MCC (0.6725). Reduced variance compared to Decision Tree and handled nonlinear patterns effectively. |
| **XGBoost (Ensemble)** | High performance (Accuracy: 83.67%, AUC: 0.9092). Balanced metrics with strong F1 score and MCC (0.6620). Effectively captured complex relationships, slightly below Logistic Regression and Random Forest. |

---

## Streamlit Application Features

The deployed Streamlit app allows users to:

- Download a sample test dataset  
- Upload test data (CSV format)  
- Select a machine learning model  
- View evaluation metrics  
- View confusion matrix  
- View classification report
- Download predicted results on uploaded test dataset (CSV file)  

---

## Requirements

- Python  
- Scikit-learn  
- XGBoost  
- Pandas  
- Matplotlib  
- Seaborn  
- Streamlit  

---

## Project Structure

```
student-depression-ml-app/
│
├── app.py
├── requirements.txt
├── README.md
└── models/
    ├── StudentDepression_training.ipynb
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── logistic.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

---

## Conclusion

This project demonstrates a complete end-to-end Machine Learning workflow:

- Data preprocessing  
- Model training  
- Model comparison  
- Performance evaluation  
- Deployment using Streamlit Community Cloud  

---

## 👨‍💻 Author: Surendra Kumar Sharma

Developed as part of a Machine Learning deployment project.
