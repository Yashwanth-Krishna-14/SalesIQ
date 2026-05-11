"""
Random Forest and XGBoost classifiers with save/load utilities.
"""

import pickle
from pathlib import Path
from typing import Any, Dict
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

def train_random_forest(X_train, y_train, params: Dict[str, Any] = None) -> RandomForestClassifier:
    """Train Random Forest classifier with optional hyperparameters."""
    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'random_state': 42
    }
    if params:
        default_params.update(params)
    model = RandomForestClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def train_xgboost(X_train, y_train, params: Dict[str, Any] = None) -> xgb.XGBClassifier:
    """Train XGBoost classifier with optional hyperparameters."""
    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss'
    }
    if params:
        default_params.update(params)
    model = xgb.XGBClassifier(**default_params)
    model.fit(X_train, y_train)
    return model

def save_model(model: Any, path: str | Path) -> None:
    """Save model as pickle file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)

def load_model(path: str | Path) -> Any:
    """Load pickle model."""
    with open(path, 'rb') as f:
        return pickle.load(f)