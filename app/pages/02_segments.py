"""
Segments page: KMeans cluster scatter plot, persona distribution, and average metrics.
"""

import streamlit as st
import plotly.express as px
import pandas as pd

st.set_page_config(page_title="Segments", layout="wide")
st.title("👥 Customer Segments")

df = st.session_state.get('df')
if df is None or 'persona' not in df.columns:
    st.error("No persona data found. Please run clustering first.")
    st.stop()

# Scatter plot: engagement_score vs email_opens, color by persona
fig = px.scatter(df, x='engagement_score', y='email_opens', color='persona',
                 title="Customer Segments (KMeans Clusters)",
                 hover_data=['company_name', 'industry'])
st.plotly_chart(fig, use_container_width=True)

# Pie chart of personas
persona_counts = df['persona'].value_counts().reset_index()
persona_counts.columns = ['Persona', 'Count']
fig_pie = px.pie(persona_counts, values='Count', names='Persona', title="Persona Distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# Average metrics per persona
avg_metrics = df.groupby('persona').agg({
    'engagement_score': 'mean',
    'email_opens': 'mean',
    'annual_revenue': 'mean',
    'days_since_contact': 'mean'
}).round(2).reset_index()
st.subheader("Average Metrics by Persona")
st.dataframe(avg_metrics, use_container_width=True)