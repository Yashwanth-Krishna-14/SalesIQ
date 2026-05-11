"""
Main Streamlit multi-page app entry point with sidebar navigation.
Caches data and models in session state.
"""

import streamlit as st
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(page_title="SalesIQ", layout="wide")

# Load data and models into session state
@st.cache_resource
def load_processed_data():
    path = Path("data/processed/leads_processed.csv")
    if path.exists():
        return pd.read_csv(path)
    else:
        st.warning("Processed data not found. Please run preprocessing first.")
        return None

@st.cache_resource
def load_models():
    from src.models.classifier import load_model
    models = {}
    rf_path = Path("models/rf_model.pkl")
    xgb_path = Path("models/xgb_model.pkl")
    if rf_path.exists():
        models['rf'] = load_model(rf_path)
    if xgb_path.exists():
        models['xgb'] = load_model(xgb_path)
    return models

if "df" not in st.session_state:
    st.session_state.df = load_processed_data()
if "models" not in st.session_state:
    st.session_state.models = load_models()

# Sidebar navigation
st.sidebar.title("🏆 SalesIQ")
st.sidebar.markdown("AI-powered lead scoring & forecasting")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to", [
    "Lead Scores", "Segments", "Model Metrics", "LLM Insights"
])

if page == "Lead Scores":
    st.switch_page("app/pages/01_lead_scores.py")
elif page == "Segments":
    st.switch_page("app/pages/02_segments.py")
elif page == "Model Metrics":
    st.switch_page("app/pages/03_model_metrics.py")
elif page == "LLM Insights":
    st.switch_page("app/pages/04_llm_insights.py")