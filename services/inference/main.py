"""
Inference Service — MLOps PoC
Serves fraud detection and churn prediction models via FastAPI.
Includes: model caching, Prometheus metrics, health checks, async prediction.
"""
import os
import time
import mlflow
import mlflow.sklearn
import numpy as np
from contextlib import asynccontextmanager
from datetime import datetime
from typing import List

from fastapi import FastAPI, HTTPException, BackgroundTasks
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app

# Add shared to path — in production use a proper package
import sys
sys.path.insert(0, "/app/shared")
from schemas.events import (
    TransactionFeatures, FraudPrediction,
    CustomerFeatures, ChurnPrediction,
)

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
FRAUD_MODEL_ALIAS = os.getenv("FRAUD_MODEL_ALIAS", "champion")
CHURN_MODEL_ALIAS = os.getenv("CHURN_MODEL_ALIAS", "champion")

mlflow.set_tracking_uri(MLFLOW_URI)

# ─── Prometheus Metrics ───────────────────────────────────────────────────────

PREDICTION_COUNTER = Counter(
    "mlops_predictions_total",
    "Total predictions made",
    ["use_case", "result"],
)
PREDICTION_LATENCY = Histogram(
    "mlops_prediction_latency_seconds",
    "Prediction latency",
    ["use_case"],
    buckets=[0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0],
)
FRAUD_SCORE_GAUGE = Gauge("mlops_fraud_score_last", "Last fraud score returned")
CHURN_SCORE_GAUGE = Gauge("mlops_churn_score_last", "Last churn probability returned")
MODEL_VERSION_GAUGE = Gauge(
    "mlops_model_version_info",
    "Loaded model version",
    ["use_case", "version"],
)

# ─── Model Cache ─────────────────────────────────────────────────────────────

models: dict = {}
model_versions: dict = {}


def load_model(use_case: str, alias: str) -> tuple:
    """Load model from MLflow registry using alias (champion/challenger)."""
    model_uri = f"models:/poc-{use_case}@{alias}"
    try:
        model = mlflow.sklearn.load_model(model_uri)
        client = mlflow.MlflowClient()
        version_info = client.get_model_version_by_alias(f"poc-{use_case}", alias)
        version = version_info.version
        print(f"Loaded {use_case} model version {version} (alias: {alias})")
        return model, version
    except Exception as e:
        print(f"Warning: Could not load {use_case} from registry ({e}). Using mock.")
        return _get_mock_model(use_case), "mock-v0"


def _get_mock_model(use_case: str):
    """Fallback mock model for PoC demo without MLflow running."""
    class MockModel:
        def predict_proba(self, X):
            rng = np.random.default_rng(int(time.time() * 1000) % 2**31)
            probs = rng.uniform(0.01, 0.99, (len(X), 2))
            probs = probs / probs.sum(axis=1, keepdims=True)
            return probs
        def predict(self, X):
            return (self.predict_proba(X)[:, 1] > 0.5).astype(int)
    return MockModel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load models
    models["fraud"], model_versions["fraud"] = load_model("fraud_detection", FRAUD_MODEL_ALIAS)
    models["churn"], model_versions["churn"] = load_model("churn_prediction", CHURN_MODEL_ALIAS)

    MODEL_VERSION_GAUGE.labels(use_case="fraud", version=model_versions["fraud"]).set(1)
    MODEL_VERSION_GAUGE.labels(use_case="churn", version=model_versions["churn"]).set(1)

    yield

    # Shutdown: cleanup
    models.clear()


app = FastAPI(
    title="MLOps PoC — Inference Service",
    version="1.0.0",
    description="Real-time model serving for Fraud Detection and Churn Prediction",
    lifespan=lifespan,
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ─── UC-01: Fraud Detection ───────────────────────────────────────────────────

FRAUD_FEATURE_ORDER = [
    "amount", "hour_of_day", "day_of_week",
    "distance_from_home_km", "is_foreign",
    "previous_30d_avg", "velocity_1h",
]


@app.post("/predict/fraud", response_model=FraudPrediction, tags=["UC-01 Fraud"])
async def predict_fraud(
    transaction: TransactionFeatures,
    background_tasks: BackgroundTasks,
) -> FraudPrediction:
    """
    UC-01: Predict fraud probability for a single transaction.
    Returns score in < 100ms for real-time payment approval decisions.
    """
    start = time.perf_counter()

    features = [[
        transaction.amount,
        transaction.hour_of_day,
        transaction.day_of_week,
        transaction.distance_from_home_km,
        int(transaction.is_foreign),
        transaction.previous_30d_avg,
        transaction.velocity_1h,
    ]]

    try:
        fraud_score = float(models["fraud"].predict_proba(features)[0][1])
        is_fraud = fraud_score > 0.5
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    latency_ms = (time.perf_counter() - start) * 1000

    # Metrics
    PREDICTION_COUNTER.labels(
        use_case="fraud",
        result="fraud" if is_fraud else "legit",
    ).inc()
    PREDICTION_LATENCY.labels(use_case="fraud").observe(latency_ms / 1000)
    FRAUD_SCORE_GAUGE.set(fraud_score)

    # Async: log to drift monitor queue
    background_tasks.add_task(
        _log_prediction_for_drift,
        "fraud_detection",
        features[0],
        FRAUD_FEATURE_ORDER,
    )

    return FraudPrediction(
        transaction_id=transaction.transaction_id,
        fraud_score=round(fraud_score, 4),
        is_fraud=is_fraud,
        model_version=model_versions["fraud"],
        latency_ms=round(latency_ms, 2),
    )


@app.post("/predict/fraud/batch", response_model=List[FraudPrediction], tags=["UC-01 Fraud"])
async def predict_fraud_batch(transactions: List[TransactionFeatures]) -> List[FraudPrediction]:
    """Batch fraud prediction — up to 500 transactions per call."""
    if len(transactions) > 500:
        raise HTTPException(status_code=422, detail="Max batch size is 500")

    results = []
    for tx in transactions:
        result = await predict_fraud(tx, BackgroundTasks())
        results.append(result)
    return results


# ─── UC-02: Churn Prediction ──────────────────────────────────────────────────

CHURN_FEATURE_ORDER = [
    "tenure_days", "monthly_charges", "total_charges",
    "num_products", "support_tickets_90d",
    "last_login_days_ago", "nps_score",
]

CHURN_SEGMENTS = {
    (0.7, 1.0): ("high_risk", "Immediate retention call + 30% discount offer"),
    (0.4, 0.7): ("medium_risk", "Personalized email campaign + loyalty points"),
    (0.0, 0.4): ("low_risk", "Standard NPS survey"),
}


def _get_churn_segment(prob: float) -> tuple[str, str]:
    for (low, high), (segment, action) in CHURN_SEGMENTS.items():
        if low <= prob <= high:
            return segment, action
    return "low_risk", "No action required"


@app.post("/predict/churn", response_model=ChurnPrediction, tags=["UC-02 Churn"])
async def predict_churn(
    customer: CustomerFeatures,
    background_tasks: BackgroundTasks,
) -> ChurnPrediction:
    """
    UC-02: Predict churn probability for a customer.
    Returns segment + recommended CRM action.
    """
    features = [[
        customer.tenure_days,
        customer.monthly_charges,
        customer.total_charges,
        customer.num_products,
        customer.support_tickets_90d,
        customer.last_login_days_ago,
        customer.nps_score or 5.0,
    ]]

    try:
        churn_prob = float(models["churn"].predict_proba(features)[0][1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    segment, action = _get_churn_segment(churn_prob)

    PREDICTION_COUNTER.labels(use_case="churn", result=segment).inc()
    PREDICTION_LATENCY.labels(use_case="churn").observe(0)
    CHURN_SCORE_GAUGE.set(churn_prob)

    background_tasks.add_task(
        _log_prediction_for_drift,
        "churn_prediction",
        features[0],
        CHURN_FEATURE_ORDER,
    )

    return ChurnPrediction(
        customer_id=customer.customer_id,
        churn_probability=round(churn_prob, 4),
        churn_segment=segment,
        recommended_action=action,
        model_version=model_versions["churn"],
    )


# ─── Health & Management ──────────────────────────────────────────────────────

@app.get("/health", tags=["Ops"])
async def health():
    return {
        "status": "healthy",
        "models_loaded": list(models.keys()),
        "versions": model_versions,
        "timestamp": datetime.utcnow().isoformat(),
    }


@app.post("/models/reload", tags=["Ops"])
async def reload_models():
    """Hot-reload models from registry without downtime."""
    models["fraud"], model_versions["fraud"] = load_model("fraud_detection", FRAUD_MODEL_ALIAS)
    models["churn"], model_versions["churn"] = load_model("churn_prediction", CHURN_MODEL_ALIAS)
    return {"status": "reloaded", "versions": model_versions}


# ─── Background Tasks ─────────────────────────────────────────────────────────

async def _log_prediction_for_drift(use_case: str, features: list, feature_names: list):
    """
    Async background task: sends feature values to drift monitor.
    In production: publish to Kafka/SQS topic.
    """
    import httpx
    payload = dict(zip(feature_names, features))
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            await client.post(
                f"http://drift-monitor:8001/ingest/{use_case}",
                json=payload,
            )
    except Exception:
        pass  # Non-blocking: drift monitor failure should not affect predictions
