"""
Unit tests for LLM insight generation with mocked OpenAI.
"""

import pytest
from unittest.mock import patch
from src.llm.insight_generator import generate_segment_insight

@patch('src.llm.chain.ChatOpenAI')
def test_generate_segment_insight_returns_string(mock_openai):
    # Mock the LLM response
    mock_llm_instance = mock_openai.return_value
    mock_llm_instance.invoke.return_value = {"text": "Mocked insight: focus on high-engagement leads."}
    
    # Patch the build_chain inside insight_generator
    with patch('src.llm.insight_generator.build_chain') as mock_build:
        mock_chain = mock_build.return_value
        mock_chain.invoke.return_value = {"text": "Mocked insight: focus on high-engagement leads."}
        
        stats = {'count': 100, 'avg_engagement': 75.0, 'avg_email_opens': 5, 'top_industry': 'Tech', 'avg_revenue': 500000}
        result = generate_segment_insight("Champion", stats)
        assert isinstance(result, str)
        assert len(result) > 0