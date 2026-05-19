"""
Model quality gate — runs in CI after training.
Fails the pipeline if model metrics don't meet minimum thresholds.
"""
import os
import sys
import mlflow

MLFLOW_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

THRESHOLDS = {
    "fraud_detection": {"f1": 0.70, "roc_auc": 0.80},
    "churn_prediction": {"f1": 0.65, "roc_auc": 0.75},
}


def validate_use_case(use_case: str) -> bool:
    client = mlflow.MlflowClient()
    experiment = client.get_experiment_by_name(f"mlops-poc-{use_case}")
    if not experiment:
        print(f"[FAIL] No experiment found for {use_case}")
        return False

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["start_time DESC"],
        max_results=1,
    )
    if not runs:
        print(f"[FAIL] No runs found for {use_case}")
        return False

    run = runs[0]
    metrics = run.data.metrics
    thresholds = THRESHOLDS[use_case]
    passed = True

    for metric, min_value in thresholds.items():
        actual = metrics.get(metric, 0)
        status = "OK" if actual >= min_value else "FAIL"
        if actual < min_value:
            passed = False
        print(f"  [{status}] {use_case} / {metric}: {actual:.4f} (min: {min_value})")

    return passed


if __name__ == "__main__":
    print("\n=== Model Quality Gate ===\n")
    all_passed = True

    for use_case in THRESHOLDS:
        print(f"Validating {use_case}...")
        if not validate_use_case(use_case):
            all_passed = False

    print()
    if all_passed:
        print("Quality gate PASSED. Models meet minimum thresholds.")
        sys.exit(0)
    else:
        print("Quality gate FAILED. Pipeline blocked.")
        sys.exit(1)
