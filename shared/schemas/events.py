"""
Shared Pydantic schemas used across all microservices.
Defines the contracts between services.
"""
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum


class UseCase(str, Enum):
    FRAUD = "fraud_detection"
    CHURN = "churn_prediction"


# ─── UC-01: Fraud Detection ───────────────────────────────────────────────────

class TransactionFeatures(BaseModel):
    transaction_id: str
    amount: float = Field(..., gt=0)
    merchant_category: str
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    distance_from_home_km: float = Field(..., ge=0)
    is_foreign: bool
    previous_30d_avg: float
    velocity_1h: int = Field(..., ge=0, description="Transactions in last 1h")


class FraudPrediction(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0, le=1)
    is_fraud: bool
    model_version: str
    latency_ms: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── UC-02: Churn Prediction ──────────────────────────────────────────────────

class CustomerFeatures(BaseModel):
    customer_id: str
    tenure_days: int = Field(..., ge=0)
    monthly_charges: float = Field(..., ge=0)
    total_charges: float = Field(..., ge=0)
    num_products: int = Field(..., ge=1)
    support_tickets_90d: int = Field(..., ge=0)
    last_login_days_ago: int = Field(..., ge=0)
    nps_score: Optional[float] = Field(None, ge=0, le=10)
    contract_type: str  # "monthly" | "annual" | "two_year"


class ChurnPrediction(BaseModel):
    customer_id: str
    churn_probability: float = Field(..., ge=0, le=1)
    churn_segment: str  # "high_risk" | "medium_risk" | "low_risk"
    recommended_action: str
    model_version: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ─── Drift & Monitoring ───────────────────────────────────────────────────────

class DriftReport(BaseModel):
    use_case: UseCase
    feature_name: str
    drift_score: float
    threshold: float
    is_drifted: bool
    reference_mean: float
    current_mean: float
    p_value: float
    timestamp: datetime = Field(default_factory=datetime.utcnow)


class ModelHealthStatus(BaseModel):
    use_case: UseCase
    model_version: str
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]
    avg_latency_ms: float
    requests_per_minute: float
    drift_detected: bool
    needs_retraining: bool
    timestamp: datetime = Field(default_factory=datetime.utcnow)
