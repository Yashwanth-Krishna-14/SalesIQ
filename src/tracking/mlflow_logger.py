"""
MLflow logging utilities for experiment tracking.
"""

import mlflow
from typing import Dict, Any
from pathlib import Path

def start_run(run_name: str, experiment_name: str = "SalesIQ") -> None:
    """Start an MLflow run; create experiment if not exists."""
    mlflow.set_experiment(experiment_name)
    mlflow.start_run(run_name=run_name)

def log_params(params: Dict[str, Any]) -> None:
    """Log parameters to current run."""
    mlflow.log_params(params)

def log_metrics(metrics: Dict[str, float]) -> None:
    """Log metrics to current run."""
    mlflow.log_metrics(metrics)

def log_artifact(path: str | Path) -> None:
    """Log a file artifact."""
    mlflow.log_artifact(str(path))

def end_run() -> None:
    """End the current MLflow run."""
    mlflow.end_run()