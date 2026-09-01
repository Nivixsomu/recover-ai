"""Recovery opportunity prediction, policy filtering, execution, and audit API router."""

from __future__ import annotations

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query

from backend.app.schemas.recovery import (
    RecoveryCaseInput,
    RecoveryDecisionResponse,
    RecoveryExecuteRequest,
    RecoveryPredictResponse,
)
from backend.app.services import AuditService, RecoveryService

router = APIRouter(prefix="/api/v1/recovery", tags=["Recovery"])


@router.post("/predict", response_model=RecoveryPredictResponse)
def predict_recovery_action(payload: RecoveryCaseInput) -> Dict[str, Any]:
    """Extract pre-intervention features and predict raw recovery probabilities for all 5 candidate actions."""
    try:
        service = RecoveryService.get_instance()
        return service.predict(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(exc)}") from exc


@router.post("/recommend", response_model=RecoveryDecisionResponse)
def recommend_recovery_action(payload: RecoveryCaseInput) -> Dict[str, Any]:
    """Run full ML ranking, PolicyEngine evaluation, and explainability without external execution (Dry Run)."""
    try:
        service = RecoveryService.get_instance()
        return service.recommend(payload.model_dump())
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(exc)}") from exc


@router.post("/execute", response_model=RecoveryDecisionResponse)
def execute_recovery_action(payload: RecoveryExecuteRequest) -> Dict[str, Any]:
    """Execute complete recovery lifecycle: ML Ranking -> PolicyEngine -> Audit Log -> Razorpay Test Mode."""
    try:
        service = RecoveryService.get_instance()
        return service.process_and_execute(
            case_input=payload.case_data.model_dump(),
            execute=payload.execute,
            idempotency_key=payload.idempotency_key,
            simulate_failure=payload.simulate_failure,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Execution error: {str(exc)}") from exc


@router.get("/cases")
def list_recovery_cases(limit: int = Query(50, ge=1, le=200), offset: int = Query(0, ge=0)) -> List[Dict[str, Any]]:
    """List historical recovery cases processed by RecoverAI."""
    audit_service = AuditService()
    return audit_service.list_cases(limit=limit, offset=offset)


@router.get("/{recovery_case_id}")
def get_recovery_case(recovery_case_id: str) -> Dict[str, Any]:
    """Retrieve details for a specific recovery opportunity."""
    audit_service = AuditService()
    case = audit_service.get_case(recovery_case_id)
    if not case:
        raise HTTPException(status_code=404, detail=f"Recovery case '{recovery_case_id}' not found.")
    return case


@router.get("/{recovery_case_id}/audit")
def get_recovery_case_audit(recovery_case_id: str) -> Dict[str, Any]:
    """Retrieve the immutable chronological audit trail for a recovery opportunity."""
    audit_service = AuditService()
    trail = audit_service.get_case_audit_trail(recovery_case_id)
    if not trail:
        raise HTTPException(status_code=404, detail=f"No audit events found for case '{recovery_case_id}'.")
    return {
        "recovery_case_id": recovery_case_id,
        "event_count": len(trail),
        "audit_trail": trail,
    }
