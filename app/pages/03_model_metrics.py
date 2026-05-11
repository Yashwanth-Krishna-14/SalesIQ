"""
Model metrics page: comparison table, confusion matrix, feature importance.
"""

import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image

st.set_page_config(page_title="Model Metrics", layout="wide")
st.title("📈 Model Performance")

# Dummy metrics for demonstration (replace with actual logged values)
metrics_data = {
    "Model": ["Random Forest", "XGBoost"],
    "Accuracy": [0.87, 0.89],
    "F1 Score": [0.86, 0.88],
    "Precision": [0.85, 0.88],
    "Recall": [0.87, 0.89],
    "AUC": [0.92, 0.94]
}
df_metrics = pd.DataFrame(metrics_data)
st.subheader("Model Comparison")
st.dataframe(df_metrics, use_container_width=True)

# Confusion matrix images
st.subheader("Confusion Matrices")
col1, col2 = st.columns(2)
cm_rf = Path("mlflow/mlartifacts/confusion_matrix_rf.png")
cm_xgb = Path("mlflow/mlartifacts/confusion_matrix_xgb.png")
if cm_rf.exists():
    with col1:
        st.image(Image.open(cm_rf), caption="Random Forest", use_column_width=True)
if cm_xgb.exists():
    with col2:
        st.image(Image.open(cm_xgb), caption="XGBoost", use_column_width=True)

# Feature importance
st.subheader("Feature Importance")
fi_path = Path("mlflow/mlartifacts/feature_importance.png")
if fi_path.exists():
    st.image(Image.open(fi_path), caption="Feature Importance (XGBoost)", use_column_width=True)