import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# ---------------------------------------------------
# Page Config
# ---------------------------------------------------
st.set_page_config(page_title="Student Depression ML App", layout="wide")

st.title("🎓 Student Depression Prediction App")

# ---------------------------------------------------
# Session State Initialization
# ---------------------------------------------------
if "results" not in st.session_state:
    st.session_state.results = None

# ---------------------------------------------------
# Download Sample Test Dataset
# ---------------------------------------------------
st.markdown("### 📥 Download Sample Test Dataset")

st.info(
    "Click the button below to download a sample test dataset. "
    "Upload it below, select a model, and click 'Run Model' "
    "to evaluate performance."
)

with open("sample_test_data.csv", "rb") as file:
    st.download_button(
        label="⬇️ Download Sample Test Dataset",
        data=file,
        file_name="sample_test_data.csv",
        mime="text/csv"
    )

st.markdown("---")

# ---------------------------------------------------
# 1️⃣ Upload Test Dataset
# ---------------------------------------------------
uploaded_file = st.file_uploader(
    "📂 Upload Test Dataset (CSV)",
    type=["csv"]
)

# ---------------------------------------------------
# 2️⃣ Select Model
# ---------------------------------------------------
model_name = st.selectbox(
    "🤖 Select Machine Learning Model",
    [
        "Logistic Regression",
        "Decision Tree",
        "K-Nearest Neighbors",
        "Naive Bayes",
        "Random Forest",
        "XGBoost"
    ]
)

# ---------------------------------------------------
# 3️⃣ Run Button
# ---------------------------------------------------
run_button = st.button("🚀 Run Model")

# ---------------------------------------------------
# Model Path Mapping
# ---------------------------------------------------
model_map = {
    "Logistic Regression": "models/logistic.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "K-Nearest Neighbors": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

# ---------------------------------------------------
# Run Model and Store Results
# ---------------------------------------------------
if uploaded_file is not None and run_button:

    data = pd.read_csv(uploaded_file)

    if "Depression" not in data.columns:
        st.error("Target column 'Depression' is missing!")
    else:
        X_test = data.drop("Depression", axis=1)
        y_test = data["Depression"]

        # Encode categorical variables
        X_test = pd.get_dummies(X_test, drop_first=True)

        # Load model
        model = joblib.load(model_map[model_name])

        # Align features
        model_features = model.feature_names_in_
        X_test = X_test.reindex(columns=model_features, fill_value=0)

        # Predictions
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        # Store in session state
        st.session_state.results = {
            "model_name": model_name,
            "data": data,
            "y_test": y_test,
            "y_pred": y_pred,
            "y_prob": y_prob
        }

elif run_button and uploaded_file is None:
    st.warning("Please upload a test dataset first.")

# ---------------------------------------------------
# Display Results (Persistent)
# ---------------------------------------------------
if st.session_state.results is not None:

    results = st.session_state.results

    model_name = results["model_name"]
    data = results["data"]
    y_test = results["y_test"]
    y_pred = results["y_pred"]
    y_prob = results["y_prob"]

    st.markdown(f"## 📌 Results using: **{model_name}**")

    # ---------------- Metrics ----------------
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    col1, col2, col3 = st.columns(3)
    col1.metric("Accuracy", f"{acc:.4f}")
    col2.metric("AUC Score", f"{auc:.4f}")
    col3.metric("MCC Score", f"{mcc:.4f}")

    col4, col5, col6 = st.columns(3)
    col4.metric("Precision", f"{prec:.4f}")
    col5.metric("Recall", f"{rec:.4f}")
    col6.metric("F1 Score", f"{f1:.4f}")

    # ---------------- Confusion Matrix ----------------
    st.subheader("🔍 Confusion Matrix")

    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

    # ---------------- Classification Report ----------------
    st.subheader("📑 Classification Report")
    st.text(classification_report(y_test, y_pred))

    # ---------------- Download Predictions ----------------
    st.markdown("### 📥 Download Prediction Results")

    results_df = data.copy()
    results_df["Predicted_Depression"] = y_pred
    results_df["Prediction_Probability"] = y_prob.round(4)

    csv = results_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="⬇️ Download Predictions as CSV",
        data=csv,
        file_name="predicted_results.csv",
        mime="text/csv",
    )