"""Measure latency across ML inference, action ranking, policy engine, DB, and API."""

from __future__ import annotations

from pathlib import Path
import sys
import time
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.app.main import app
from backend.app.services import RecoveryService


def benchmark(num_runs: int = 100) -> dict:
    service = RecoveryService.get_instance()
    client = TestClient(app)

    raw_case = {
        "recovery_case_id": "BENCH-CASE-001",
        "amount_at_risk": 2500.0,
        "failure_reason": "BANK_TIMEOUT",
        "payment_method": "UPI",
        "retry_count": 0,
        "cooldown_eligible": True,
    }
    case = service._normalize_case_input(raw_case)

    # 1. Action Ranking Latency
    ranking_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = service.ranker.rank_case(case)
        ranking_times.append((time.perf_counter() - t0) * 1000)

    # 2. Policy Engine Latency
    ranking = service.ranker.rank_case(case)
    policy_times = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = service.policy_engine.evaluate_ranking(case, ranking)
        policy_times.append((time.perf_counter() - t0) * 1000)

    # 3. Full Recovery Service Processing (Inference + Policy + Audit DB)
    service_times = []
    for idx in range(num_runs):
        case_iter = raw_case.copy()
        case_iter["recovery_case_id"] = f"BENCH-{idx}"
        t0 = time.perf_counter()
        _ = service.process_and_execute(case_iter, execute=False)
        service_times.append((time.perf_counter() - t0) * 1000)

    # 4. End-to-End API HTTP Request Latency
    api_times = []
    for idx in range(num_runs):
        payload = {
            "case_data": {
                "recovery_case_id": f"BENCH-API-{idx}",
                "amount_at_risk": 2500.0,
                "failure_reason": "BANK_TIMEOUT",
                "payment_method": "UPI",
            },
            "execute": False,
        }
        t0 = time.perf_counter()
        resp = client.post("/api/v1/recovery/execute", json=payload)
        api_times.append((time.perf_counter() - t0) * 1000)

    results = {
        "num_iterations": num_runs,
        "action_ranking_ms": {
            "mean": float(np.mean(ranking_times)),
            "p50": float(np.percentile(ranking_times, 50)),
            "p95": float(np.percentile(ranking_times, 95)),
            "p99": float(np.percentile(ranking_times, 99)),
        },
        "policy_engine_ms": {
            "mean": float(np.mean(policy_times)),
            "p50": float(np.percentile(policy_times, 50)),
            "p95": float(np.percentile(policy_times, 95)),
            "p99": float(np.percentile(policy_times, 99)),
        },
        "recovery_service_db_ms": {
            "mean": float(np.mean(service_times)),
            "p50": float(np.percentile(service_times, 50)),
            "p95": float(np.percentile(service_times, 95)),
            "p99": float(np.percentile(service_times, 99)),
        },
        "api_endpoint_e2e_ms": {
            "mean": float(np.mean(api_times)),
            "p50": float(np.percentile(api_times, 50)),
            "p95": float(np.percentile(api_times, 95)),
            "p99": float(np.percentile(api_times, 99)),
        },
    }

    print("==================================================")
    print("RECOVERAI — PHASE 16: PERFORMANCE BENCHMARK")
    print("==================================================")
    print(f"Action Ranking (5 candidates): Mean = {results['action_ranking_ms']['mean']:.2f} ms | P95 = {results['action_ranking_ms']['p95']:.2f} ms")
    print(f"Policy Engine Checks:         Mean = {results['policy_engine_ms']['mean']:.2f} ms | P95 = {results['policy_engine_ms']['p95']:.2f} ms")
    print(f"Recovery Service + SQLite:    Mean = {results['recovery_service_db_ms']['mean']:.2f} ms | P95 = {results['recovery_service_db_ms']['p95']:.2f} ms")
    print(f"FastAPI E2E Endpoint:         Mean = {results['api_endpoint_e2e_ms']['mean']:.2f} ms | P95 = {results['api_endpoint_e2e_ms']['p95']:.2f} ms")
    print("==================================================")

    return results


if __name__ == "__main__":
    benchmark()
