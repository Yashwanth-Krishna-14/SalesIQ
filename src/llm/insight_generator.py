"""
Generate AI insights for each customer segment using LangChain.
"""

from typing import Dict
import pandas as pd
from .chain import build_chain

def generate_segment_insight(segment_name: str, stats: Dict) -> str:
    """Call LLM chain to generate insight for a segment."""
    chain = build_chain()
    stats_str = ", ".join(f"{k}: {v}" for k, v in stats.items())
    response = chain.invoke({"segment_name": segment_name, "stats": stats_str})
    return response['text']

def generate_all_insights(df_with_personas: pd.DataFrame) -> Dict[str, str]:
    """Generate insights for each unique persona."""
    insights = {}
    for persona in df_with_personas['persona'].unique():
        subset = df_with_personas[df_with_personas['persona'] == persona]
        stats = {
            'count': len(subset),
            'avg_engagement': subset['engagement_score'].mean(),
            'avg_email_opens': subset['email_opens'].mean(),
            'top_industry': subset['industry'].mode()[0] if not subset['industry'].mode().empty else 'N/A',
            'avg_revenue': subset['annual_revenue'].mean()
        }
        insight = generate_segment_insight(persona, stats)
        insights[persona] = insight
    return insights