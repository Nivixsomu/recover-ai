"""End-to-End integration tests for RecoverAI lifecycle."""

import pytest
from fastapi.testclient import TestClient

from backend.app.integrations.razorpay import RazorpayTestClient
from backend.app.main import app
from backend.app.services import AuditService, RecoveryService


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture(autouse=True)
def isolate_razorpay_client():
    """Ensure e2e tests run in deterministic mock mode without triggering external HTTP requests."""
    service = RecoveryService.get_instance()
    original_client = service.razorpay_client
    service.razorpay_client = RazorpayTestClient(key_id="", key_secret="")
    yield
    service.razorpay_client = original_client



def test_e2e_normal_successful_flow(client: TestClient):
    """E2E Flow 1: Full lifecycle for standard transient payment failure."""
    case_id = "E2E-CASE-NORMAL-01"
    payload = {
        "case_data": {
            "recovery_case_id": case_id,
            "amount_at_risk": 2000.0,
            "failure_reason": "BANK_TIMEOUT",
            "payment_method": "UPI",
            "customer_segment": "NORMAL",
            "retry_count": 0,
            "cooldown_eligible": True,
        },
        "execute": True,
        "idempotency_key": f"idem_{case_id}",
    }

    resp = client.post("/api/v1/recovery/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["recovery_case_id"] == case_id
    assert data["selected_action"] in ["RETRY", "PAYMENT_LINK", "REMINDER", "HUMAN_REVIEW", "NO_ACTION"]
    assert data["execution"]["status"] == "SUCCESS"
    assert data["execution"]["test_mode"] is True

    # Check persistence and audit trail
    audit_resp = client.get(f"/api/v1/recovery/{case_id}/audit")
    assert audit_resp.status_code == 200
    audit_events = audit_resp.json()["audit_trail"]
    event_types = [e["event_type"] for e in audit_events]

    assert "CASE_RECEIVED" in event_types
    assert "ML_RANKING_EVALUATED" in event_types
    assert "POLICY_EVALUATED" in event_types
    assert "ACTION_EXECUTED" in event_types


def test_e2e_policy_blocked_retry_fallback(client: TestClient):
    """E2E Flow 2: PolicyEngine blocks RETRY when retry_count >= 2 and falls back to PAYMENT_LINK."""
    case_id = "E2E-CASE-RETRY-BLOCKED-02"
    payload = {
        "case_data": {
            "recovery_case_id": case_id,
            "amount_at_risk": 1500.0,
            "failure_reason": "GATEWAY_TIMEOUT",
            "payment_method": "NETBANKING",
            "retry_count": 2,  # Limit reached
            "cooldown_eligible": True,
        },
        "execute": True,
        "idempotency_key": f"idem_{case_id}",
    }

    resp = client.post("/api/v1/recovery/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    policy = data["policy_decision"]
    assert "RETRY" in policy["blocked_actions"]
    assert "ERR_RETRY_LIMIT_EXCEEDED" in policy["blocked_actions"]["RETRY"]
    assert data["selected_action"] != "RETRY"
    assert data["execution"]["status"] == "SUCCESS"


def test_e2e_high_value_human_review_escalation(client: TestClient):
    """E2E Flow 3: Large amount (₹75k) escalates to HUMAN_REVIEW."""
    case_id = "E2E-CASE-HIGH-VALUE-03"
    payload = {
        "case_data": {
            "recovery_case_id": case_id,
            "amount_at_risk": 75000.0,
            "failure_reason": "PAYMENT_METHOD_DECLINED",
            "payment_method": "CARD",
            "retry_count": 0,
        },
        "execute": True,
        "idempotency_key": f"idem_{case_id}",
    }

    resp = client.post("/api/v1/recovery/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["selected_action"] == "HUMAN_REVIEW"
    assert data["execution"]["reference_id"].startswith("TICKET-")


def test_e2e_idempotent_duplicate_prevention(client: TestClient):
    """E2E Flow 4: Repeat execution with same token is safely skipped."""
    case_id = "E2E-CASE-IDEM-04"
    key = "idem_e2e_token_9999"
    payload = {
        "case_data": {
            "recovery_case_id": case_id,
            "amount_at_risk": 3200.0,
            "failure_reason": "NETWORK_ERROR",
            "payment_method": "UPI",
        },
        "execute": True,
        "idempotency_key": key,
    }

    first = client.post("/api/v1/recovery/execute", json=payload)
    assert first.status_code == 200
    assert first.json()["execution"]["status"] == "SUCCESS"

    second = client.post("/api/v1/recovery/execute", json=payload)
    assert second.status_code == 200
    assert second.json()["execution"]["status"] == "IDEMPOTENT_SKIPPED"


def test_e2e_dry_run_mode(client: TestClient):
    """E2E Flow 5: execute=False produces full analysis without calling external execution."""
    case_id = "E2E-CASE-DRYRUN-05"
    payload = {
        "case_data": {
            "recovery_case_id": case_id,
            "amount_at_risk": 1800.0,
            "failure_reason": "INSUFFICIENT_FUNDS",
            "payment_method": "CARD",
        },
        "execute": False,
    }

    resp = client.post("/api/v1/recovery/execute", json=payload)
    assert resp.status_code == 200
    data = resp.json()

    assert data["execution"]["status"] == "DRY_RUN"
    assert "dry_run" in data["execution"]["reference_id"]
