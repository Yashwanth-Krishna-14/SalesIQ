"""
Unit tests for data preprocessing functions.
"""

import pytest
import pandas as pd
import numpy as np
from src.data.loader import validate_schema
from src.data.preprocessor import encode_features

def test_validate_schema_pass():
    df = pd.DataFrame({
        'lead_id': [1], 'company_name': ['A'], 'industry': ['Tech'],
        'company_size': ['S'], 'engagement_score': [0.5], 'email_opens': [1],
        'website_visits': [2], 'deal_stage': ['Lead'], 'lead_source': ['Web'],
        'region': ['NA'], 'annual_revenue': [1000], 'days_since_contact': [10],
        'label': ['Hot']
    })
    # Should not raise
    validate_schema(df)

def test_validate_schema_fail():
    df = pd.DataFrame({'lead_id': [1], 'company_name': ['A']})
    with pytest.raises(ValueError, match="Missing required columns"):
        validate_schema(df)

def test_encode_features_output_shape():
    df = pd.DataFrame({
        'lead_id': [1,2], 'company_name': ['A','B'], 'industry': ['Tech','Fin'],
        'company_size': ['S','M'], 'engagement_score': [0.5,0.7], 'email_opens': [1,2],
        'website_visits': [2,3], 'deal_stage': ['Lead','Qualified'], 'lead_source': ['Web','Ref'],
        'region': ['NA','EU'], 'annual_revenue': [1000,2000], 'days_since_contact': [10,20],
        'label': ['Hot','Warm']
    })
    df_encoded, _ = encode_features(df)
    assert df_encoded.shape[0] == 2
    assert 'industry' in df_encoded.columns
    assert df_encoded['industry'].dtype == np.int32 or df_encoded['industry'].dtype == np.int64