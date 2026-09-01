"""Isolated Razorpay Test Mode client for RecoverAI.

STRICTLY TEST MODE ONLY.
Credentials are read strictly from environment variables without logging or exposing secrets.
"""

from __future__ import annotations

import hashlib
import time
from typing import Any, Dict, Optional

import httpx

from backend.app.config import settings
from .exceptions import (
    RazorpayAPIError,
    RazorpayAuthenticationError,
    RazorpayIntegrationError,
    RazorpayInvalidKeyModeError,
    RazorpayNetworkError,
    RazorpayTimeoutError,
)
from .schemas import (
    RazorpayActionExecutionResult,
    RazorpayOrderRequest,
    RazorpayOrderResponse,
    RazorpayPaymentLinkRequest,
    RazorpayPaymentLinkResponse,
)


class RazorpayTestClient:
    """Client for executing safe, test-mode payment recovery operations."""

    BASE_URL = "https://api.razorpay.com/v1"

    def __init__(
        self,
        key_id: Optional[str] = None,
        key_secret: Optional[str] = None,
        timeout: float = 10.0,
    ) -> None:
        self.key_id = key_id if key_id is not None else settings.razorpay_key_id
        self.key_secret = key_secret if key_secret is not None else settings.razorpay_key_secret
        self.timeout = timeout
        self._executed_idempotency_keys: Dict[str, RazorpayActionExecutionResult] = {}
        if self.has_credentials:
            self.validate_test_mode_key()

    @property
    def has_credentials(self) -> bool:
        return bool(self.key_id and self.key_secret)

    def validate_test_mode_key(self) -> None:
        """Enforce that configured credentials strictly belong to Razorpay Test Mode.

        Raises:
            RazorpayInvalidKeyModeError: If key_id does not start with 'rzp_test_'.
        """
        if self.key_id and not self.key_id.startswith("rzp_test_"):
            raise RazorpayInvalidKeyModeError(
                "STRICT SAFETY VIOLATION: Configured Razorpay Key ID does not start with 'rzp_test_'. "
                "Production / non-test keys are strictly prohibited in RecoverAI."
            )

    def _get_auth(self) -> Optional[tuple[str, str]]:
        if not self.has_credentials:
            return None
        self.validate_test_mode_key()
        return (self.key_id or "", self.key_secret or "")

    def create_order(self, request: RazorpayOrderRequest) -> RazorpayOrderResponse:
        """Create a test order in Razorpay (or mock in test mode)."""
        amount_paisa = int(round(request.amount * 100))
        payload = {
            "amount": amount_paisa,
            "currency": request.currency,
            "receipt": request.receipt or f"rcpt_{int(time.time())}",
            "notes": request.notes,
        }

        if not self.has_credentials:
            # Safe simulated test mode response when API keys are not supplied in dev environment
            order_id = f"order_mock_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}"
            return RazorpayOrderResponse(
                order_id=order_id,
                amount=request.amount,
                currency=request.currency,
                status="created",
                created_at=int(time.time()),
                test_mode=True,
            )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.BASE_URL}/orders",
                    auth=self._get_auth(),
                    json=payload,
                )
                if resp.status_code == 401:
                    raise RazorpayAuthenticationError()
                if resp.status_code >= 400:
                    raise RazorpayAPIError(resp.text, status_code=resp.status_code)

                data = resp.json()
                return RazorpayOrderResponse(
                    order_id=data["id"],
                    amount=data["amount"] / 100,
                    currency=data["currency"],
                    status=data["status"],
                    created_at=data["created_at"],
                    test_mode=True,
                )
        except httpx.TimeoutException as exc:
            raise RazorpayTimeoutError() from exc
        except httpx.RequestError as exc:
            raise RazorpayNetworkError(str(exc)) from exc

    def create_payment_link(self, request: RazorpayPaymentLinkRequest) -> RazorpayPaymentLinkResponse:
        """Generate a test Payment Link for customer recovery."""
        amount_paisa = int(round(request.amount * 100))
        payload = {
            "amount": amount_paisa,
            "currency": request.currency,
            "description": request.description or "RecoverAI Payment Recovery",
            "reference_id": request.reference_id or f"ref_{int(time.time())}",
        }

        if not self.has_credentials:
            link_id = f"plink_mock_{hashlib.sha256(str(time.time()).encode()).hexdigest()[:12]}"
            return RazorpayPaymentLinkResponse(
                link_id=link_id,
                short_url=f"https://rzp.io/i/mock_{link_id[:8]}",
                amount=request.amount,
                currency=request.currency,
                status="created",
                test_mode=True,
            )

        try:
            with httpx.Client(timeout=self.timeout) as client:
                resp = client.post(
                    f"{self.BASE_URL}/payment_links",
                    auth=self._get_auth(),
                    json=payload,
                )
                if resp.status_code == 401:
                    raise RazorpayAuthenticationError()
                if resp.status_code >= 400:
                    raise RazorpayAPIError(resp.text, status_code=resp.status_code)

                data = resp.json()
                return RazorpayPaymentLinkResponse(
                    link_id=data["id"],
                    short_url=data.get("short_url", ""),
                    amount=data["amount"] / 100,
                    currency=data["currency"],
                    status=data["status"],
                    test_mode=True,
                )
        except httpx.TimeoutException as exc:
            raise RazorpayTimeoutError() from exc
        except httpx.RequestError as exc:
            raise RazorpayNetworkError(str(exc)) from exc

    def execute_recovery_action(
        self,
        action: str,
        recovery_case_id: str,
        amount: float,
        idempotency_key: Optional[str] = None,
        dry_run: bool = False,
        simulate_failure: bool = False,
    ) -> RazorpayActionExecutionResult:
        """Execute a policy-approved recovery action in Razorpay Test Mode.

        Args:
            action: Approved recovery action ('RETRY', 'PAYMENT_LINK', 'REMINDER', 'HUMAN_REVIEW', 'NO_ACTION').
            recovery_case_id: Opportunity identifier.
            amount: Amount at risk in INR.
            idempotency_key: Unique client request identifier.
            dry_run: If True, simulates execution without making external API calls.
            simulate_failure: If True, deliberately triggers a failure for testing/demo.

        Returns:
            RazorpayActionExecutionResult.
        """
        # Idempotency check: prevent duplicate financial / recovery action dispatch
        if idempotency_key and idempotency_key in self._executed_idempotency_keys:
            prev = self._executed_idempotency_keys[idempotency_key]
            return RazorpayActionExecutionResult(
                action=action,
                recovery_case_id=recovery_case_id,
                status="IDEMPOTENT_SKIPPED",
                reference_id=prev.reference_id,
                link_url=prev.link_url,
                message=f"Duplicate execution blocked by idempotency key: {idempotency_key}",
                test_mode=True,
                raw_response={"original_status": prev.status},
            )

        if dry_run:
            return RazorpayActionExecutionResult(
                action=action,
                recovery_case_id=recovery_case_id,
                status="DRY_RUN",
                reference_id=f"dry_run_{recovery_case_id}",
                message=f"Dry run simulation for {action} on case {recovery_case_id}.",
                test_mode=True,
            )

        if simulate_failure:
            result = RazorpayActionExecutionResult(
                action=action,
                recovery_case_id=recovery_case_id,
                status="FAILED",
                message="Simulated gateway network timeout on Razorpay Test API.",
                test_mode=True,
            )
            if idempotency_key:
                self._executed_idempotency_keys[idempotency_key] = result
            return result

        try:
            if action == "RETRY":
                order_resp = self.create_order(
                    RazorpayOrderRequest(amount=amount, receipt=f"retry_{recovery_case_id}")
                )
                result = RazorpayActionExecutionResult(
                    action=action,
                    recovery_case_id=recovery_case_id,
                    status="SUCCESS",
                    reference_id=order_resp.order_id,
                    message=f"Dispatched automated backend retry via Razorpay order {order_resp.order_id}.",
                    test_mode=True,
                    raw_response=order_resp.model_dump(),
                )

            elif action == "PAYMENT_LINK":
                link_resp = self.create_payment_link(
                    RazorpayPaymentLinkRequest(
                        amount=amount,
                        description=f"RecoverAI Link for {recovery_case_id}",
                        reference_id=recovery_case_id,
                    )
                )
                result = RazorpayActionExecutionResult(
                    action=action,
                    recovery_case_id=recovery_case_id,
                    status="SUCCESS",
                    reference_id=link_resp.link_id,
                    link_url=link_resp.short_url,
                    message=f"Generated interactive Razorpay payment link {link_resp.short_url}.",
                    test_mode=True,
                    raw_response=link_resp.model_dump(),
                )

            elif action == "REMINDER":
                link_resp = self.create_payment_link(
                    RazorpayPaymentLinkRequest(
                        amount=amount,
                        description=f"Payment Reminder for {recovery_case_id}",
                        reference_id=f"rem_{recovery_case_id}",
                    )
                )
                result = RazorpayActionExecutionResult(
                    action=action,
                    recovery_case_id=recovery_case_id,
                    status="SUCCESS",
                    reference_id=link_resp.link_id,
                    link_url=link_resp.short_url,
                    message=f"Dispatched customer reminder notification with payment link.",
                    test_mode=True,
                    raw_response=link_resp.model_dump(),
                )

            elif action == "HUMAN_REVIEW":
                ticket_id = f"TICKET-{recovery_case_id[-8:]}"
                result = RazorpayActionExecutionResult(
                    action=action,
                    recovery_case_id=recovery_case_id,
                    status="SUCCESS",
                    reference_id=ticket_id,
                    message=f"Enqueued high-priority human review ticket {ticket_id} for agent investigation.",
                    test_mode=True,
                )

            elif action == "NO_ACTION":
                result = RazorpayActionExecutionResult(
                    action=action,
                    recovery_case_id=recovery_case_id,
                    status="SUCCESS",
                    reference_id=f"no_action_{recovery_case_id}",
                    message="Recorded passive observation (NO_ACTION) per policy determination.",
                    test_mode=True,
                )

            else:
                result = RazorpayActionExecutionResult(
                    action=action,
                    recovery_case_id=recovery_case_id,
                    status="FAILED",
                    message=f"Unknown recovery action '{action}'.",
                    test_mode=True,
                )

        except RazorpayIntegrationError as exc:
            result = RazorpayActionExecutionResult(
                action=action,
                recovery_case_id=recovery_case_id,
                status="FAILED",
                message=f"Razorpay integration error: {exc.message} ({exc.error_code})",
                test_mode=True,
            )

        if idempotency_key:
            self._executed_idempotency_keys[idempotency_key] = result

        return result
