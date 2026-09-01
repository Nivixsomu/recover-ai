"""Tests for Razorpay Test Mode adapter, idempotency, and error handling."""

import pytest

from backend.app.integrations.razorpay import (
    RazorpayActionExecutionResult,
    RazorpayInvalidKeyModeError,
    RazorpayOrderRequest,
    RazorpayPaymentLinkRequest,
    RazorpayTestClient,
)


@pytest.fixture
def client() -> RazorpayTestClient:
    """Explicitly isolated mock test client (empty credentials) for offline unit test execution."""
    return RazorpayTestClient(key_id="", key_secret="")




def test_order_creation_test_mode(client: RazorpayTestClient):
    """Verify test order creation returns valid order metadata."""
    req = RazorpayOrderRequest(amount=1200.0, receipt="rcpt_test_001")
    resp = client.create_order(req)

    assert resp.order_id.startswith("order_")
    assert resp.amount == 1200.0
    assert resp.currency == "INR"
    assert resp.test_mode is True


def test_payment_link_creation_test_mode(client: RazorpayTestClient):
    """Verify test payment link generation returns short URL."""
    req = RazorpayPaymentLinkRequest(amount=850.0, description="Test Recovery Link")
    resp = client.create_payment_link(req)

    assert resp.link_id.startswith("plink_")
    assert resp.amount == 850.0
    assert "rzp.io" in resp.short_url
    assert resp.test_mode is True


def test_execute_all_actions(client: RazorpayTestClient):
    """Verify execution adapter handles all 5 recovery actions safely."""
    actions = ["RETRY", "PAYMENT_LINK", "REMINDER", "HUMAN_REVIEW", "NO_ACTION"]

    for act in actions:
        result = client.execute_recovery_action(
            action=act,
            recovery_case_id=f"CASE-TEST-{act}",
            amount=500.0,
        )
        assert isinstance(result, RazorpayActionExecutionResult)
        assert result.status == "SUCCESS"
        assert result.action == act
        assert result.test_mode is True
        assert result.reference_id is not None


def test_dry_run_mode(client: RazorpayTestClient):
    """Verify dry_run=True returns DRY_RUN status without mutating external state."""
    result = client.execute_recovery_action(
        action="RETRY",
        recovery_case_id="CASE-DRY-RUN-1",
        amount=1000.0,
        dry_run=True,
    )
    assert result.status == "DRY_RUN"
    assert "Dry run" in result.message


def test_idempotency_prevents_duplicate_execution(client: RazorpayTestClient):
    """Verify duplicate idempotency keys block repeat action dispatch."""
    key = "idem_key_unique_12345"

    first = client.execute_recovery_action(
        action="PAYMENT_LINK",
        recovery_case_id="CASE-IDEM-1",
        amount=2500.0,
        idempotency_key=key,
    )
    assert first.status == "SUCCESS"

    second = client.execute_recovery_action(
        action="PAYMENT_LINK",
        recovery_case_id="CASE-IDEM-1",
        amount=2500.0,
        idempotency_key=key,
    )
    assert second.status == "IDEMPOTENT_SKIPPED"
    assert "Duplicate execution blocked" in second.message


def test_simulated_failure_handled_gracefully(client: RazorpayTestClient):
    """Verify simulated network/gateway failures return FAILED status without crashing."""
    result = client.execute_recovery_action(
        action="RETRY",
        recovery_case_id="CASE-FAIL-1",
        amount=1500.0,
        simulate_failure=True,
    )
    assert result.status == "FAILED"
    assert "timeout" in result.message.lower()


def test_prefix_guard_blocks_live_keys():
    """Verify that attempting to configure a live key (rzp_live_*) raises a safety exception."""
    with pytest.raises(RazorpayInvalidKeyModeError) as exc_info:
        RazorpayTestClient(key_id="rzp_live_unauthorized12345", key_secret="secret_xyz")

    assert "STRICT SAFETY VIOLATION" in str(exc_info.value.message)
    assert exc_info.value.error_code == "ERR_RAZORPAY_INVALID_KEY_MODE"


def test_prefix_guard_blocks_arbitrary_non_test_keys():
    """Verify that arbitrary keys not starting with rzp_test_ are rejected."""
    with pytest.raises(RazorpayInvalidKeyModeError):
        RazorpayTestClient(key_id="invalid_prefix_key", key_secret="secret_xyz")


def test_prefix_guard_allows_valid_test_key_format():
    """Verify that valid test keys (rzp_test_*) pass validation successfully."""
    client = RazorpayTestClient(key_id="rzp_test_samplekey12345", key_secret="sample_secret_abc")
    assert client.has_credentials is True
    # Should not raise
    client.validate_test_mode_key()

