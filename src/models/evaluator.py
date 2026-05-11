"""
Evaluation metrics and plotting for classification models.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, confusion_matrix
from pathlib import Path
from typing import Dict

def evaluate_classifier(model, X_test, y_test) -> Dict[str, float]:
    """Return dict of accuracy, f1, precision, recall, AUC (ovr)."""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    
    # For multi-class, use 'ovr' and average='weighted'
    auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred, average='weighted'),
        'precision': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'auc': auc
    }
    return metrics

def plot_confusion_matrix(model, X_test, y_test, save_path: str | Path) -> None:
    """Plot and save confusion matrix as PNG."""
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

def plot_feature_importance(model, feature_names: list, save_path: str | Path) -> None:
    """Plot feature importance (for tree-based models)."""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        raise AttributeError("Model does not have feature_importances_ attribute")
    
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10,6))
    plt.title('Feature Importance')
    plt.bar(range(len(importances)), importances[indices], align='center')
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path)
    plt.close()