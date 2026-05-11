"""
Preprocessing: encoding, scaling, and train-test split.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple

def encode_features(df: pd.DataFrame, target_col: str = 'label') -> Tuple[pd.DataFrame, dict]:
    """
    Label encode categorical features and scale numeric features.
    Returns transformed DataFrame and a dict of fitted scalers/encoders.
    """
    df_processed = df.copy()
    cat_cols = ['industry', 'company_size', 'deal_stage', 'lead_source', 'region']
    num_cols = ['engagement_score', 'email_opens', 'website_visits', 'annual_revenue', 'days_since_contact']
    
    encoders = {}
    for col in cat_cols:
        le = LabelEncoder()
        df_processed[col] = le.fit_transform(df_processed[col])
        encoders[col] = le
    
    scaler = StandardScaler()
    df_processed[num_cols] = scaler.fit_transform(df_processed[num_cols])
    encoders['scaler'] = scaler
    
    # Keep target as is (we'll encode separately if needed)
    return df_processed, encoders

def split_data(df: pd.DataFrame, target_col: str = 'label', test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Split features and target into train/test sets."""
    X = df.drop(columns=[target_col])
    y = df[target_col]
    # Encode target labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=test_size, random_state=random_state)
    return X_train, X_test, y_train, y_test, le

def save_processed(df: pd.DataFrame, path: str | Path) -> None:
    """Save processed DataFrame to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)