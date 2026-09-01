"""Comprehensive API test suite for RecoverAI FastAPI application."""

import pytest
from fastapi.testclient import TestClient

from backend.app.integrations.razorpay import RazorpayTestClient
from backend.app.main import app
from backend.app.services import RecoveryService


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolate_razorpay_client():
    """Ensure API tests run in deterministic mock mode without triggering external HTTP requests."""
    service = RecoveryService.get_instance()
    original_client = service.razorpay_client
    service.razorpay_client = RazorpayTestClient(key_id="", key_secret="")
    yield
    service.razorpay_client = original_client



def test_health_check(client: TestClient):
    """Verify health endpoint."""
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_predict_endpoint(client: TestClient):
    """Verify predict endpoint returns 5 candidate action probabilities."""
    payload = {
        "recovery_case_id": "CASE-TEST-PREDICT-01",
        "amount_at_risk": 1500.0,
        "failure_reason": "BANK_TIMEOUT",
        "payment_method": "UPI",
        "retry_count": 0,
    }
    resp = client.post("/api/v1/recovery/predict", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["recovery_case_id"] == "CASE-TEST-PREDICT-01"
    assert "probabilities" in data
    assert len(data["probabilities"]) == 5
    assert "RETRY" in data["probabilities"]
    assert "top_action" in data
    assert data["top_probability"] >= 0.0


def test_recommend_endpoint_dry_run(client: TestClient):
    """Verify recommend endpoint applies policy and explainability without external execution."""
    payload = {
        "recovery_case_id": "CASE-TEST-REC-01",
        "amount_at_risk": 750.0,
        "failure_reason": "INSUFFICIENT_FUNDS",
        "payment_method": "CARD",
        "retry_count": 0,
    }
    resp = client.post("/api/v1/recovery/recommend", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert "model_recommendation" in data
    assert "policy_decision" in data
    assert "explanation" in data
    assert data["execution"]["status"] == "DRY_RUN"


def test_execute_endpoint_test_mode(client: TestClient):
    """Verify execute endpoint with execute=True dispatches via Razorpay Test Mode."""
    payload = {
        "case_data": {
            "recovery_case_id": "CASE-TEST-EXEC-01",
            "amount_at_risk": 1250.0,
            "failure_reason": "GATEWAY_TIMEOUT",
            "payment_method": "NETBANKING",
            "retry_count": 0,
        },
        "execute": True,
        "idempotency_key": "idem_test_exec_001",
    }
    resp = client.post("/api/v1/recovery/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()
    assert data["execution"]["status"] == "SUCCESS"
    assert data["execution"]["test_mode"] is True


def test_idempotent_duplicate_prevention(client: TestClient):
    """Verify duplicate idempotency keys return IDEMPOTENT_SKIPPED."""
    payload = {
        "case_data": {
            "recovery_case_id": "CASE-TEST-IDEM-01",
            "amount_at_risk": 3000.0,
            "failure_reason": "NETWORK_ERROR",
            "payment_method": "UPI",
        },
        "execute": True,
        "idempotency_key": "idem_duplicate_test_token_99",
    }
    first_resp = client.post("/api/v1/recovery/execute", json=payload)
    assert first_resp.status_code == 200
    assert first_resp.json()["execution"]["status"] == "SUCCESS"

    second_resp = client.post("/api/v1/recovery/execute", json=payload)
    assert second_resp.status_code == 200
    assert second_resp.json()["execution"]["status"] == "IDEMPOTENT_SKIPPED"


def test_simulate_gateway_failure_handled(client: TestClient):
    """Verify simulated gateway failure returns graceful FAILED status without 500 crash."""
    payload = {
        "case_data": {
            "recovery_case_id": "CASE-TEST-FAIL-01",
            "amount_at_risk": 5000.0,
            "failure_reason": "BANK_TIMEOUT",
            "payment_method": "UPI",
        },
        "execute": True,
        "simulate_failure": True,
    }
    resp = client.post("/api/v1/recovery/execute", json=payload)
    assert resp.status_code == 200
    assert resp.json()["execution"]["status"] == "FAILED"


def test_audit_trail_endpoint(client: TestClient):
    """Verify audit trail returns historical chronological events."""
    case_id = "CASE-TEST-AUDIT-01"
    payload = {
        "case_data": {
            "recovery_case_id": case_id,
            "amount_at_risk": 2100.0,
            "failure_reason": "TEMPORARY_BANK_ERROR",
            "payment_method": "UPI",
        },
        "execute": False,
    }
    _ = client.post("/api/v1/recovery/execute", json=payload)

    audit_resp = client.get(f"/api/v1/recovery/{case_id}/audit")
    assert audit_resp.status_code == 200
    audit_data = audit_resp.json()
    assert audit_data["recovery_case_id"] == case_id
    assert audit_data["event_count"] >= 3
    event_types = [e["event_type"] for e in audit_data["audit_trail"]]
    assert "CASE_RECEIVED" in event_types
    assert "ML_RANKING_EVALUATED" in event_types
    assert "POLICY_EVALUATED" in event_types


def test_metrics_endpoints(client: TestClient):
    """Verify metrics endpoints return synthetic KPI summary and breakdowns."""
    resp_sum = client.get("/api/v1/metrics/summary")
    assert resp_sum.status_code == 200
    assert resp_sum.json()["is_synthetic"] is True
    assert "kpis" in resp_sum.json()
    assert resp_sum.json()["kpis"]["total_opportunities"] == 3000

    resp_act = client.get("/api/v1/metrics/actions")
    assert resp_act.status_code == 200
    assert "ml_action_distribution" in resp_act.json()

    resp_rec = client.get("/api/v1/metrics/recovery")
    assert resp_rec.status_code == 200
    assert "by_failure_reason" in resp_rec.json()
