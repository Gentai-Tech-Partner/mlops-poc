"""
Smoke tests — run after each deployment to validate both use cases end-to-end.
Can be run locally: python scripts/smoke_test.py
"""
import httpx
import sys
import os

BASE_URL = os.getenv("INFERENCE_URL", "http://localhost:8000")


def test_health():
    r = httpx.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    print(f"[OK] Health check: {data}")


def test_fraud_prediction():
    payload = {
        "transaction_id": "smoke-test-001",
        "amount": 9500.0,
        "merchant_category": "electronics",
        "hour_of_day": 3,
        "day_of_week": 6,
        "distance_from_home_km": 450.0,
        "is_foreign": True,
        "previous_30d_avg": 120.0,
        "velocity_1h": 4,
    }
    r = httpx.post(f"{BASE_URL}/predict/fraud", json=payload, timeout=5.0)
    assert r.status_code == 200
    data = r.json()
    assert "fraud_score" in data
    assert 0 <= data["fraud_score"] <= 1
    assert data["latency_ms"] < 200  # SLA: < 200ms
    print(f"[OK] UC-01 Fraud: score={data['fraud_score']}, fraud={data['is_fraud']}, latency={data['latency_ms']}ms")


def test_churn_prediction():
    payload = {
        "customer_id": "smoke-test-cust-001",
        "tenure_days": 90,
        "monthly_charges": 89.99,
        "total_charges": 810.0,
        "num_products": 1,
        "support_tickets_90d": 8,
        "last_login_days_ago": 45,
        "nps_score": 3.5,
        "contract_type": "monthly",
    }
    r = httpx.post(f"{BASE_URL}/predict/churn", json=payload, timeout=5.0)
    assert r.status_code == 200
    data = r.json()
    assert "churn_probability" in data
    assert "churn_segment" in data
    assert "recommended_action" in data
    assert 0 <= data["churn_probability"] <= 1
    print(f"[OK] UC-02 Churn: prob={data['churn_probability']}, segment={data['churn_segment']}")
    print(f"     Action: {data['recommended_action']}")


def test_batch_fraud():
    transactions = [
        {
            "transaction_id": f"batch-{i}",
            "amount": float(100 * i),
            "merchant_category": "food",
            "hour_of_day": 12,
            "day_of_week": 1,
            "distance_from_home_km": float(i * 10),
            "is_foreign": False,
            "previous_30d_avg": 150.0,
            "velocity_1h": 1,
        }
        for i in range(1, 6)
    ]
    r = httpx.post(f"{BASE_URL}/predict/fraud/batch", json=transactions, timeout=10.0)
    assert r.status_code == 200
    data = r.json()
    assert len(data) == 5
    print(f"[OK] UC-01 Batch: {len(data)} predictions returned")


if __name__ == "__main__":
    print(f"\n=== Smoke Tests — {BASE_URL} ===\n")
    failures = []
    for test in [test_health, test_fraud_prediction, test_churn_prediction, test_batch_fraud]:
        try:
            test()
        except Exception as e:
            failures.append((test.__name__, str(e)))
            print(f"[FAIL] {test.__name__}: {e}")

    print(f"\n{'='*40}")
    if failures:
        print(f"FAILED: {len(failures)} test(s)")
        sys.exit(1)
    else:
        print("ALL TESTS PASSED")
        sys.exit(0)
