"""Pydantic API schemas for RecoverAI endpoints."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RecoveryCaseInput(BaseModel):
    """Payload representing an in-flight payment failure opportunity."""

    recovery_case_id: Optional[str] = None
    payment_id: Optional[str] = None
    customer_id: Optional[str] = None
    amount_at_risk: float = Field(..., gt=0, description="Amount of failed transaction in INR")
    failure_reason: str = Field(..., description="Root cause failure category")
    payment_method: str = Field(..., description="Payment rail (UPI, CARD, NETBANKING, WALLET)")
    customer_segment: Optional[str] = "NORMAL"
    retry_count: int = Field(default=0, ge=0)
    attempt_count: int = Field(default=1, ge=1)
    historical_success_rate: float = Field(default=0.70, ge=0.0, le=1.0)
    customer_transaction_frequency: float = Field(default=3.5, ge=0.0)
    historical_transaction_count: int = Field(default=10, ge=0)
    historical_average_amount: Optional[float] = None
    customer_lifetime_value: Optional[float] = None
    previous_failure_count: int = Field(default=0, ge=0)
    previous_success_count: int = Field(default=7, ge=0)
    time_since_previous_payment_hours: float = Field(default=24.0, ge=0.0)
    hour_of_day: int = Field(default=14, ge=0, le=23)
    day_of_week: int = Field(default=2, ge=0, le=6)
    cooldown_eligible: bool = True
    amount_limit_eligible: Optional[bool] = None
    human_review_required: bool = False


class RecoveryExecuteRequest(BaseModel):
    """Request payload for recovery recommendation or execution."""

    case_data: RecoveryCaseInput
    execute: bool = Field(default=False, description="Set True to dispatch via Razorpay Test Mode; False for Dry Run")
    idempotency_key: Optional[str] = Field(default=None, description="Unique client idempotency token")
    simulate_failure: bool = Field(default=False, description="Set True to test graceful gateway error handling")


class RecoveryPredictResponse(BaseModel):
    """Raw model inference response."""

    recovery_case_id: str
    model_version: str
    probabilities: Dict[str, float]
    top_action: str
    top_probability: float
    expected_recovery_value: float


class RecoveryDecisionResponse(BaseModel):
    """Comprehensive recovery recommendation and execution response."""

    recovery_case_id: str
    amount_at_risk: float
    failure_reason: str
    payment_method: str
    model_recommendation: Dict[str, Any]
    policy_decision: Dict[str, Any]
    selected_action: str
    explanation: Dict[str, Any]
    execution: Dict[str, Any]
    audit_trail_url: str
