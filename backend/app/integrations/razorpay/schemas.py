"""Data schemas for Razorpay Test Mode integration."""

from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import BaseModel, Field


class RazorpayOrderRequest(BaseModel):
    """Request payload to create a test order."""

    amount: float = Field(..., gt=0, description="Amount in INR")
    currency: str = Field(default="INR")
    receipt: Optional[str] = None
    notes: Dict[str, Any] = Field(default_factory=dict)


class RazorpayOrderResponse(BaseModel):
    """Response returned from order creation."""

    order_id: str
    amount: float
    currency: str
    status: str
    created_at: int
    test_mode: bool = True


class RazorpayPaymentLinkRequest(BaseModel):
    """Request payload to generate a payment link."""

    amount: float = Field(..., gt=0, description="Amount in INR")
    currency: str = Field(default="INR")
    description: Optional[str] = None
    customer_id: Optional[str] = None
    reference_id: Optional[str] = None


class RazorpayPaymentLinkResponse(BaseModel):
    """Response returned from payment link generation."""

    link_id: str
    short_url: str
    amount: float
    currency: str
    status: str
    test_mode: bool = True


class RazorpayActionExecutionResult(BaseModel):
    """Immutable execution outcome from Razorpay Test Mode adapter."""

    action: str
    recovery_case_id: str
    status: str  # 'SUCCESS', 'FAILED', 'DRY_RUN', 'IDEMPOTENT_SKIPPED'
    reference_id: Optional[str] = None
    link_url: Optional[str] = None
    message: str
    test_mode: bool = True
    raw_response: Optional[Dict[str, Any]] = None
