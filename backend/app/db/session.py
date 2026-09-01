"""SQLite database connection and session management."""

from __future__ import annotations

from pathlib import Path
import sqlite3
from typing import Generator

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DB_PATH = PROJECT_ROOT / "recover_ai.db"

_db_initialized = False


def init_db() -> None:
    """Initialize database tables and indexes."""
    global _db_initialized
    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    cursor = conn.cursor()

    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS recovery_cases (
        recovery_case_id TEXT PRIMARY KEY,
        payment_id TEXT,
        customer_id TEXT,
        amount_at_risk REAL NOT NULL,
        failure_reason TEXT NOT NULL,
        payment_method TEXT NOT NULL,
        customer_segment TEXT,
        retry_count INTEGER DEFAULT 0,
        historical_success_rate REAL DEFAULT 0.5,
        status TEXT DEFAULT 'PENDING',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recovery_case_id TEXT NOT NULL,
        model_version TEXT NOT NULL,
        probabilities_json TEXT NOT NULL,
        rankings_json TEXT NOT NULL,
        top_action TEXT NOT NULL,
        top_probability REAL NOT NULL,
        top_expected_value REAL NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recovery_case_id) REFERENCES recovery_cases(recovery_case_id)
    );

    CREATE TABLE IF NOT EXISTS policy_decisions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recovery_case_id TEXT NOT NULL,
        original_action TEXT NOT NULL,
        selected_action TEXT NOT NULL,
        is_approved BOOLEAN NOT NULL,
        fallback_occurred BOOLEAN NOT NULL,
        blocked_actions_json TEXT NOT NULL,
        policy_reasons_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recovery_case_id) REFERENCES recovery_cases(recovery_case_id)
    );

    CREATE TABLE IF NOT EXISTS executions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recovery_case_id TEXT NOT NULL,
        action TEXT NOT NULL,
        status TEXT NOT NULL,
        reference_id TEXT,
        link_url TEXT,
        message TEXT,
        idempotency_key TEXT UNIQUE,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (recovery_case_id) REFERENCES recovery_cases(recovery_case_id)
    );

    CREATE TABLE IF NOT EXISTS audit_events (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        recovery_case_id TEXT NOT NULL,
        event_type TEXT NOT NULL,
        payload_json TEXT NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );

    CREATE INDEX IF NOT EXISTS idx_audit_case_id ON audit_events(recovery_case_id);
    CREATE INDEX IF NOT EXISTS idx_exec_idem_key ON executions(idempotency_key);
    """)

    conn.commit()
    conn.close()
    _db_initialized = True


def get_db_connection() -> sqlite3.Connection:
    """Create and return a configured SQLite connection, ensuring tables exist."""
    global _db_initialized
    if not _db_initialized or not DB_PATH.exists():
        init_db()

    conn = sqlite3.connect(str(DB_PATH), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn
