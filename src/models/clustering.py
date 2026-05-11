"""
KMeans clustering for customer segmentation and persona mapping.
"""

import pickle
from pathlib import Path
import pandas as pd
from sklearn.cluster import KMeans

def train_kmeans(X, n_clusters: int = 3, random_state: int = 42) -> KMeans:
    """Fit KMeans model on feature matrix X."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init='auto')
    kmeans.fit(X)
    return kmeans

def assign_personas(df: pd.DataFrame, model: KMeans, feature_cols: list) -> pd.DataFrame:
    """
    Add a 'persona' column to DataFrame based on cluster labels.
    Mapping: Cluster 0 = Champion, Cluster 1 = Prospect, Cluster 2 = At-Risk
    """
    clusters = model.predict(df[feature_cols])
    persona_map = {0: 'Champion', 1: 'Prospect', 2: 'At-Risk'}
    df['persona'] = [persona_map[c] for c in clusters]
    return df

def save_cluster_model(model: KMeans, path: str | Path) -> None:
    """Save KMeans model as pickle."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(model, f)