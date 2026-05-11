"""
Functions to load and validate raw CRM data.
"""

import pandas as pd
from pathlib import Path
from typing import List

REQUIRED_COLUMNS = [
    'lead_id', 'company_name', 'industry', 'company_size', 'engagement_score',
    'email_opens', 'website_visits', 'deal_stage', 'lead_source', 'region',
    'annual_revenue', 'days_since_contact', 'label'
]

def load_raw_data(path: str | Path) -> pd.DataFrame:
    """Load CSV from given path and return DataFrame."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    df = pd.read_csv(path)
    return df

def validate_schema(df: pd.DataFrame, required_columns: List[str] = None) -> None:
    """
    Check that all required columns exist in DataFrame.
    Raises ValueError if any column missing.
    """
    required = required_columns or REQUIRED_COLUMNS
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")