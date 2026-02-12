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
| Logistic Regression | 0.86 | 0.90 | 0.84 | 0.82 | 0.83 | 0.71 |
| Decision Tree | 0.82 | 0.85 | 0.80 | 0.78 | 0.79 | 0.64 |
| kNN | 0.84 | 0.88 | 0.83 | 0.80 | 0.81 | 0.68 |
| Naive Bayes | 0.81 | 0.86 | 0.79 | 0.77 | 0.78 | 0.62 |
| Random Forest (Ensemble) | 0.89 | 0.93 | 0.87 | 0.86 | 0.86 | 0.77 |
| XGBoost (Ensemble) | 0.91 | 0.95 | 0.89 | 0.88 | 0.88 | 0.81 |

---

## Streamlit Application Features

The deployed Streamlit app allows users to:

- Download a sample test dataset  
- Upload test data (CSV format)  
- Select a machine learning model  
- View evaluation metrics  
- View confusion matrix  
- View classification report  

---

## Required Libraries

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

Ensemble models such as **Random Forest** and **XGBoost** generally provide improved predictive performance due to their ability to capture complex patterns and reduce overfitting.

---

## 👨‍💻 Author: Surendra Kumar Sharma

Developed as part of a Machine Learning deployment project.
