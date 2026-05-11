"""
Lead scores page: distribution chart and filterable table.
"""

import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Lead Scores", layout="wide")
st.title("📊 Lead Scores")

df = st.session_state.get('df')
if df is None:
    st.error("No data loaded. Please run preprocessing first.")
    st.stop()

# Distribution bar chart
label_counts = df['label'].value_counts().reset_index()
label_counts.columns = ['Label', 'Count']
fig = px.bar(label_counts, x='Label', y='Count', title="Lead Score Distribution (Hot/Warm/Cold)", color='Label')
st.plotly_chart(fig, use_container_width=True)

# Filters
col1, col2 = st.columns(2)
with col1:
    industries = ['All'] + sorted(df['industry'].unique())
    selected_industry = st.selectbox("Filter by Industry", industries)
with col2:
    regions = ['All'] + sorted(df['region'].unique())
    selected_region = st.selectbox("Filter by Region", regions)

filtered_df = df.copy()
if selected_industry != 'All':
    filtered_df = filtered_df[filtered_df['industry'] == selected_industry]
if selected_region != 'All':
    filtered_df = filtered_df[filtered_df['region'] == selected_region]

st.subheader("Lead Data")
st.dataframe(filtered_df[['lead_id', 'company_name', 'industry', 'region', 'engagement_score', 'label']], use_container_width=True)