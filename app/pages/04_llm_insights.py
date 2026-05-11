"""
LLM Insights page: AI-generated recommendations per persona with regenerate button.
"""

import streamlit as st
from src.llm.insight_generator import generate_all_insights

st.set_page_config(page_title="LLM Insights", layout="wide")
st.title("🤖 AI-Powered Sales Insights")

df = st.session_state.get('df')
if df is None or 'persona' not in df.columns:
    st.error("No persona data. Run clustering first.")
    st.stop()

# Store insights in session state to avoid recomputing every time
if 'insights' not in st.session_state:
    st.session_state.insights = generate_all_insights(df)

if st.button("🔄 Regenerate Insights"):
    with st.spinner("Calling GPT..."):
        st.session_state.insights = generate_all_insights(df)
    st.success("Insights regenerated!")

for persona, insight in st.session_state.insights.items():
    with st.expander(f"✨ {persona} Segment", expanded=True):
        st.write(insight)