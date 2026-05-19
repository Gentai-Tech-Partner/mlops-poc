"""
Drift Monitor Service — MLOps PoC
Detects statistical data drift comparing production traffic vs training distribution.
Uses Population Stability Index (PSI) + KS test.
Triggers retraining alert via webhook when drift threshold is exceeded.
"""
import os
import json
import asyncio
import httpx
import numpy as np
from collections import deque
from datetime import datetime
from scipy import stats
from fastapi import FastAPI, HTTPException
from prometheus_client import Gauge, Counter, make_asgi_app

import sys
sys.path.insert(0, "/app/shared")
from schemas.events import DriftReport, ModelHealthStatus, UseCase

DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
WINDOW_SIZE = int(os.getenv("WINDOW_SIZE", "500"))
RETRAINING_WEBHOOK = os.getenv("RETRAINING_WEBHOOK", "http://training-service:8002/retrain")

app = FastAPI(
    title="MLOps PoC — Drift Monitor",
    version="1.0.0",
    description="Statistical drift detection for production ML models",
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

# ─── Prometheus Metrics ───────────────────────────────────────────────────────

DRIFT_SCORE_GAUGE = Gauge(
    "mlops_drift_score",
    "Current drift score per feature",
    ["use_case", "feature"],
)
DRIFT_ALERT_COUNTER = Counter(
    "mlops_drift_alerts_total",
    "Number of drift alerts triggered",
    ["use_case"],
)
RETRAINING_TRIGGER_COUNTER = Counter(
    "mlops_retraining_triggers_total",
    "Number of retraining jobs triggered",
    ["use_case"],
)

# ─── In-memory sliding windows ────────────────────────────────────────────────
# In production: replace with Redis streams or Kafka consumer

production_windows: dict[str, dict[str, deque]] = {
    "fraud_detection": {},
    "churn_prediction": {},
}

# Reference stats loaded from MLflow artifacts at startup
reference_stats: dict[str, dict] = {}

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


def _load_reference_stats():
    """Load reference statistics from MLflow artifacts or fallback to defaults."""
    import mlflow
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000"))

    for use_case, features in [
        ("fraud_detection", FRAUD_FEATURES),
        ("churn_prediction", CHURN_FEATURES),
    ]:
        try:
            client = mlflow.MlflowClient()
            model_version = client.get_model_version_by_alias(f"poc-{use_case}", "champion")
            run_id = model_version.run_id
            local_path = mlflow.artifacts.download_artifacts(
                run_id=run_id,
                artifact_path=f"reference/{use_case}_reference_stats.json",
            )
            with open(local_path) as f:
                reference_stats[use_case] = json.load(f)
            print(f"Loaded reference stats for {use_case}")
        except Exception as e:
            print(f"Using synthetic reference stats for {use_case}: {e}")
            # Fallback: synthetic reference means for demo
            reference_stats[use_case] = {
                f: {"mean": 100.0, "std": 50.0} for f in features
            }

        # Initialize sliding windows
        for feature in features:
            production_windows[use_case][feature] = deque(maxlen=WINDOW_SIZE)


@app.on_event("startup")
async def startup():
    _load_reference_stats()
    # Start background drift analysis loop
    asyncio.create_task(_drift_analysis_loop())


# ─── Ingestion endpoint ───────────────────────────────────────────────────────

@app.post("/ingest/{use_case}", tags=["Ingestion"])
async def ingest_prediction(use_case: str, features: dict):
    """
    Receive feature values from inference service (fire-and-forget).
    Adds to sliding window for drift analysis.
    """
    if use_case not in production_windows:
        raise HTTPException(status_code=404, detail=f"Unknown use case: {use_case}")

    for feature, value in features.items():
        if feature in production_windows[use_case]:
            try:
                production_windows[use_case][feature].append(float(value))
            except (TypeError, ValueError):
                pass

    return {"status": "ingested"}


# ─── Drift Analysis ───────────────────────────────────────────────────────────

def _compute_psi(reference: np.ndarray, production: np.ndarray, bins: int = 10) -> float:
    """
    Population Stability Index.
    PSI < 0.1: no drift | 0.1-0.2: slight drift | > 0.2: significant drift
    """
    ref_hist, bin_edges = np.histogram(reference, bins=bins)
    prod_hist, _ = np.histogram(production, bins=bin_edges)

    ref_pct = ref_hist / len(reference) + 1e-6
    prod_pct = prod_hist / len(production) + 1e-6

    psi = np.sum((prod_pct - ref_pct) * np.log(prod_pct / ref_pct))
    return float(psi)


def _analyze_drift(use_case: str) -> list[DriftReport]:
    """Run drift analysis for all features of a use case."""
    reports = []
    windows = production_windows[use_case]
    ref_stats = reference_stats.get(use_case, {})

    for feature, window in windows.items():
        if len(window) < 50:  # Need minimum samples
            continue

        production_data = np.array(list(window))
        ref_mean = ref_stats.get(feature, {}).get("mean", 100.0)
        ref_std = ref_stats.get(feature, {}).get("std", 50.0)

        # Generate reference sample from stored distribution
        rng = np.random.default_rng(42)
        reference_sample = rng.normal(ref_mean, ref_std, 1000)

        # KS Test
        ks_stat, p_value = stats.ks_2samp(reference_sample, production_data)

        # PSI
        psi = _compute_psi(reference_sample, production_data)

        is_drifted = psi > DRIFT_THRESHOLD or p_value < 0.05

        report = DriftReport(
            use_case=use_case,
            feature_name=feature,
            drift_score=round(psi, 4),
            threshold=DRIFT_THRESHOLD,
            is_drifted=is_drifted,
            reference_mean=round(ref_mean, 4),
            current_mean=round(float(np.mean(production_data)), 4),
            p_value=round(float(p_value), 4),
        )
        reports.append(report)

        # Update Prometheus
        DRIFT_SCORE_GAUGE.labels(use_case=use_case, feature=feature).set(psi)

        if is_drifted:
            DRIFT_ALERT_COUNTER.labels(use_case=use_case).inc()
            print(f"DRIFT ALERT [{use_case}] Feature '{feature}': PSI={psi:.4f}")

    return reports


async def _trigger_retraining(use_case: str, reason: str):
    """Call training service to kick off a retraining job."""
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            response = await client.post(
                RETRAINING_WEBHOOK,
                json={"use_case": use_case, "reason": reason, "triggered_at": datetime.utcnow().isoformat()},
            )
        if response.status_code == 200:
            RETRAINING_TRIGGER_COUNTER.labels(use_case=use_case).inc()
            print(f"Retraining triggered for {use_case}: {reason}")
    except Exception as e:
        print(f"Failed to trigger retraining for {use_case}: {e}")


async def _drift_analysis_loop():
    """Background task: runs drift analysis every 60 seconds."""
    while True:
        await asyncio.sleep(60)
        for use_case in production_windows:
            try:
                reports = _analyze_drift(use_case)
                drifted_features = [r.feature_name for r in reports if r.is_drifted]

                # Trigger retraining if more than 30% of features are drifted
                if reports and len(drifted_features) / len(reports) > 0.3:
                    reason = f"Drift detected in features: {drifted_features}"
                    await _trigger_retraining(use_case, reason)

            except Exception as e:
                print(f"Drift analysis error for {use_case}: {e}")


# ─── Query endpoints ──────────────────────────────────────────────────────────

@app.get("/drift/{use_case}", response_model=list[DriftReport], tags=["Analysis"])
async def get_drift_report(use_case: str):
    """Get current drift report for a use case."""
    if use_case not in production_windows:
        raise HTTPException(status_code=404, detail=f"Unknown use case: {use_case}")
    return _analyze_drift(use_case)


@app.get("/health", tags=["Ops"])
async def health():
    window_sizes = {
        uc: {f: len(w) for f, w in features.items()}
        for uc, features in production_windows.items()
    }
    return {
        "status": "healthy",
        "window_sizes": window_sizes,
        "drift_threshold": DRIFT_THRESHOLD,
        "timestamp": datetime.utcnow().isoformat(),
    }
