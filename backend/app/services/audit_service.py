"""Immutable audit logging and historical metric retrieval service."""

from __future__ import annotations

import json
from typing import Any, Dict, List, Optional

from backend.app.db.session import get_db_connection


class AuditService:
    """Provides structured persistence for cases, predictions, policy checks, executions, and audit logs."""

    def __init__(self) -> None:
        pass

    def save_case(self, case_dict: Dict[str, Any]) -> None:
        """Upsert recovery case context in database."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO recovery_cases (
                recovery_case_id, payment_id, customer_id, amount_at_risk,
                failure_reason, payment_method, customer_segment, retry_count,
                historical_success_rate, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(recovery_case_id) DO UPDATE SET
                status = excluded.status
            """,
            (
                case_dict.get("recovery_case_id"),
                case_dict.get("payment_id"),
                case_dict.get("customer_id"),
                float(case_dict.get("amount_at_risk", case_dict.get("amount", 0.0))),
                case_dict.get("failure_reason"),
                case_dict.get("payment_method"),
                case_dict.get("customer_segment"),
                int(case_dict.get("retry_count", 0)),
                float(case_dict.get("historical_success_rate", 0.5)),
                case_dict.get("status", "PROCESSED"),
            ),
        )
        conn.commit()
        conn.close()

    def record_audit_event(
        self,
        recovery_case_id: str,
        event_type: str,
        payload: Dict[str, Any],
    ) -> None:
        """Append an immutable audit entry."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO audit_events (recovery_case_id, event_type, payload_json)
            VALUES (?, ?, ?)
            """,
            (recovery_case_id, event_type, json.dumps(payload)),
        )
        conn.commit()
        conn.close()

    def save_prediction(
        self,
        recovery_case_id: str,
        model_version: str,
        probabilities: Dict[str, float],
        rankings: List[Dict[str, Any]],
        top_action: str,
        top_probability: float,
        top_expected_value: float,
    ) -> None:
        """Record model inference results."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO predictions (
                recovery_case_id, model_version, probabilities_json, rankings_json,
                top_action, top_probability, top_expected_value
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recovery_case_id,
                model_version,
                json.dumps(probabilities),
                json.dumps(rankings),
                top_action,
                top_probability,
                top_expected_value,
            ),
        )
        conn.commit()
        conn.close()

    def save_policy_decision(
        self,
        recovery_case_id: str,
        original_action: str,
        selected_action: str,
        is_approved: bool,
        fallback_occurred: bool,
        blocked_actions: Dict[str, str],
        policy_reasons: List[str],
    ) -> None:
        """Record PolicyEngine evaluation and safety decision."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO policy_decisions (
                recovery_case_id, original_action, selected_action, is_approved,
                fallback_occurred, blocked_actions_json, policy_reasons_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                recovery_case_id,
                original_action,
                selected_action,
                int(is_approved),
                int(fallback_occurred),
                json.dumps(blocked_actions),
                json.dumps(policy_reasons),
            ),
        )
        conn.commit()
        conn.close()

    def save_execution(
        self,
        recovery_case_id: str,
        action: str,
        status: str,
        reference_id: Optional[str],
        link_url: Optional[str],
        message: str,
        idempotency_key: Optional[str],
    ) -> None:
        """Record action execution result."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO executions (
                recovery_case_id, action, status, reference_id, link_url,
                message, idempotency_key
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(idempotency_key) DO UPDATE SET
                status = excluded.status,
                message = excluded.message
            """,
            (
                recovery_case_id,
                action,
                status,
                reference_id,
                link_url,
                message,
                idempotency_key,
            ),
        )
        conn.commit()
        conn.close()

    def get_case(self, recovery_case_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve single recovery case record."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM recovery_cases WHERE recovery_case_id = ?", (recovery_case_id,))
        row = cursor.fetchone()
        conn.close()
        return dict(row) if row else None

    def list_cases(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """List recovery cases with latest decisions."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT c.*, p.selected_action, p.is_approved, e.status as execution_status
            FROM recovery_cases c
            LEFT JOIN policy_decisions p ON c.recovery_case_id = p.recovery_case_id
            LEFT JOIN executions e ON c.recovery_case_id = e.recovery_case_id
            ORDER BY c.created_at DESC
            LIMIT ? OFFSET ?
            """,
            (limit, offset),
        )
        rows = [dict(r) for r in cursor.fetchall()]
        conn.close()
        return rows

    def get_case_audit_trail(self, recovery_case_id: str) -> List[Dict[str, Any]]:
        """Retrieve chronological immutable audit events for a recovery case."""
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT event_type, payload_json, created_at
            FROM audit_events
            WHERE recovery_case_id = ?
            ORDER BY id ASC
            """,
            (recovery_case_id,),
        )
        events = [
            {
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
                "timestamp": row["created_at"],
            }
            for row in cursor.fetchall()
        ]
        conn.close()
        return events
