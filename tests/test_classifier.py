"""
Unit tests for classifier training and evaluation.
"""

import pytest
import numpy as np
from sklearn.datasets import make_classification
from src.models.classifier import train_random_forest, train_xgboost
from src.models.evaluator import evaluate_classifier

@pytest.fixture
def data():
    X, y = make_classification(n_samples=100, n_features=5, n_classes=3, n_informative=3, random_state=42)
    return X, y

def test_train_random_forest_returns_fitted(data):
    X, y = data
    model = train_random_forest(X, y)
    assert hasattr(model, "predict")
    model.predict(X)  # should not raise

def test_evaluate_classifier_returns_required_keys(data):
    X, y = data
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    model = train_random_forest(X_train, y_train)
    metrics = evaluate_classifier(model, X_test, y_test)
    required = {'accuracy', 'f1', 'precision', 'recall', 'auc'}
    assert required.issubset(metrics.keys())
    assert all(isinstance(v, float) for v in metrics.values())