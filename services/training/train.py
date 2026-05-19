"""
Training Service — MLOps PoC
Handles reproducible model training for both use cases.
Uses MLflow for experiment tracking, model versioning, and registry.
"""
import os
import json
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

FRAUD_FEATURES = [
    "amount", "hour_of_day", "day_of_week",
    "distance_from_home_km", "is_foreign",
    "previous_30d_avg", "velocity_1h",
]

CHURN_FEATURES = [
    "tenure_days", "monthly_charges", "total_charges",
    "num_products", "support_tickets_90d",
    "last_login_days_ago", "nps_score",
]


def _generate_fraud_data(n_samples: int = 10_000) -> pd.DataFrame:
    """
    Synthetic fraud dataset.
    In production: replace with DVC-tracked dataset pull.
    """
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "amount": rng.lognormal(4, 1.5, n_samples),
        "hour_of_day": rng.integers(0, 24, n_samples),
        "day_of_week": rng.integers(0, 7, n_samples),
        "distance_from_home_km": rng.exponential(20, n_samples),
        "is_foreign": rng.integers(0, 2, n_samples),
        "previous_30d_avg": rng.lognormal(4, 1, n_samples),
        "velocity_1h": rng.integers(0, 10, n_samples),
    })
    # Label: fraud when amount > avg * 3 and distance > 100km
    df["label"] = (
        (df["amount"] > df["previous_30d_avg"] * 3) &
        (df["distance_from_home_km"] > 100)
    ).astype(int)
    return df


def _generate_churn_data(n_samples: int = 10_000) -> pd.DataFrame:
    """Synthetic churn dataset."""
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "tenure_days": rng.integers(0, 3650, n_samples),
        "monthly_charges": rng.uniform(20, 200, n_samples),
        "total_charges": rng.uniform(0, 10000, n_samples),
        "num_products": rng.integers(1, 6, n_samples),
        "support_tickets_90d": rng.integers(0, 20, n_samples),
        "last_login_days_ago": rng.integers(0, 365, n_samples),
        "nps_score": rng.uniform(0, 10, n_samples),
    })
    # Label: churn when low tenure + high support tickets + low NPS
    df["label"] = (
        (df["tenure_days"] < 365) &
        (df["support_tickets_90d"] > 5) &
        (df["nps_score"] < 5)
    ).astype(int)
    return df


def train_model(use_case: str, params: dict | None = None) -> str:
    """
    Train model for a given use case.
    Returns: MLflow run_id
    """
    params = params or {
        "n_estimators": 200,
        "max_depth": 4,
        "learning_rate": 0.05,
        "subsample": 0.8,
    }

    experiment_name = f"mlops-poc-{use_case}"
    mlflow.set_experiment(experiment_name)

    if use_case == "fraud_detection":
        df = _generate_fraud_data()
        features = FRAUD_FEATURES
    elif use_case == "churn_prediction":
        df = _generate_churn_data()
        features = CHURN_FEATURES
    else:
        raise ValueError(f"Unknown use case: {use_case}")

    X = df[features]
    y = df["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save reference stats for drift monitoring
    reference_stats = X_train.describe().to_dict()

    with mlflow.start_run() as run:
        mlflow.log_params(params)
        mlflow.log_param("use_case", use_case)
        mlflow.log_param("n_train_samples", len(X_train))
        mlflow.log_param("n_test_samples", len(X_test))
        mlflow.log_param("features", features)
        mlflow.log_param("train_date", datetime.utcnow().isoformat())

        pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("model", GradientBoostingClassifier(**params, random_state=42)),
        ])
        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "f1": f1_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_prob),
        }
        mlflow.log_metrics(metrics)

        # Save reference stats as artifact for drift monitor
        stats_path = f"/tmp/{use_case}_reference_stats.json"
        with open(stats_path, "w") as f:
            json.dump(reference_stats, f)
        mlflow.log_artifact(stats_path, "reference")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        mlflow.sklearn.log_model(
            pipeline,
            "model",
            registered_model_name=f"poc-{use_case}",
        )

        print(f"[{use_case}] Run {run.info.run_id} | F1={metrics['f1']:.4f} | AUC={metrics['roc_auc']:.4f}")
        return run.info.run_id


if __name__ == "__main__":
    import sys
    use_case = sys.argv[1] if len(sys.argv) > 1 else "fraud_detection"
    run_id = train_model(use_case)
    print(f"Training complete. Run ID: {run_id}")
