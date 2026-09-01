"""Metrics, KPI aggregates, and evaluation API router."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict
from fastapi import APIRouter, HTTPException

PROJECT_ROOT = Path(__file__).resolve().parents[3]
EVAL_SUMMARY_PATH = PROJECT_ROOT / "ml" / "models" / "ml_evaluation_summary.json"

router = APIRouter(prefix="/api/v1/metrics", tags=["Metrics"])


def load_eval_summary() -> Dict[str, Any]:
    if not EVAL_SUMMARY_PATH.exists():
        raise HTTPException(status_code=503, detail="ML evaluation summary artifact not found.")
    with open(EVAL_SUMMARY_PATH, "r") as f:
        return json.load(f)


@router.get("/summary")
def get_metrics_summary() -> Dict[str, Any]:
    """Retrieve top-level business KPIs, benchmark comparison, and revenue lift (Synthetic Test Cohort)."""
    data = load_eval_summary()
    bus = data.get("phase_6g_business_evaluation_test", {})

    return {
        "dataset_name": "recovery-simulator-v2",
        "seed": 20260401,
        "is_synthetic": True,
        "disclaimer": "All recovery performance and lift statistics reflect controlled synthetic simulation dynamics.",
        "kpis": {
            "total_opportunities": bus.get("total_opportunities", 3000),
            "total_amount_at_risk": bus.get("total_amount_at_risk", 6073213.32),
            "baseline_recovered_revenue": bus.get("baseline_policy", {}).get("recovered_revenue", 3422581.38),
            "baseline_recovery_rate": bus.get("baseline_policy", {}).get("recovery_rate", 0.5530),
            "ml_recovered_revenue": bus.get("ml_policy", {}).get("recovered_revenue", 3531916.62),
            "ml_recovery_rate": bus.get("ml_policy", {}).get("recovery_rate", 0.5687),
            "oracle_potential_revenue": bus.get("oracle", {}).get("potential_recovered_revenue", 4246431.60),
            "oracle_potential_rate": bus.get("oracle", {}).get("potential_recovery_rate", 0.6917),
            "absolute_revenue_lift": bus.get("revenue_lift", {}).get("absolute_inr", 109335.24),
            "relative_revenue_lift_percent": bus.get("revenue_lift", {}).get("relative_percentage", 3.19),
        },
    }


@router.get("/actions")
def get_action_metrics() -> Dict[str, Any]:
    """Retrieve action distribution comparisons between Baseline and ML Policy."""
    data = load_eval_summary()
    bus = data.get("phase_6g_business_evaluation_test", {})

    return {
        "baseline_action_distribution": bus.get("baseline_policy", {}).get("action_distribution", {}),
        "ml_action_distribution": bus.get("ml_policy", {}).get("action_distribution", {}),
    }


@router.get("/recovery")
def get_recovery_subgroups() -> Dict[str, Any]:
    """Retrieve subgroup recovery rate breakdowns by failure reason, rail, segment, and amount band."""
    data = load_eval_summary()
    bus = data.get("phase_6g_business_evaluation_test", {})
    return bus.get("subgroup_recovery_rates", {})
