"""
Task state normalization and reconciliation utilities.

These helpers provide a single source of truth for merging Celery runtime
state with persisted DB state. They are intentionally small and pure so both
API routes and services can reuse the same logic.
"""

from typing import Optional, Tuple


TERMINAL_STATES = {"SUCCESS", "FAILURE", "REVOKED"}
ACTIVE_STATES = {"PENDING", "STARTED", "RETRY", "PROCESSING"}

_DB_TO_CANONICAL = {
    "pending": "PENDING",
    "started": "STARTED",
    "retry": "RETRY",
    "success": "SUCCESS",
    "failure": "FAILURE",
    "revoked": "REVOKED",
}


def normalize_db_status(db_status: Optional[str]) -> Optional[str]:
    """
    Normalize persisted DB status values to Celery-compatible uppercase states.
    """
    if not db_status:
        return None
    value = str(db_status).strip().lower()
    return _DB_TO_CANONICAL.get(value, value.upper())


def reconcile_task_state(celery_state: str, db_status: Optional[str]) -> Tuple[str, str]:
    """
    Reconcile Celery runtime state and DB state.

    Returns:
        tuple[state, source]
        - state: reconciled uppercase state
        - source: "celery", "db", or "reconciled"
    """
    runtime = (celery_state or "PENDING").upper()
    persisted = normalize_db_status(db_status)

    if not persisted:
        return runtime, "celery"

    # Terminal DB states should win over transient runtime ambiguity.
    if persisted in TERMINAL_STATES and runtime not in TERMINAL_STATES:
        return persisted, "db"

    # Redis/Celery result backend can forget intermediate state; keep DB active states.
    if runtime == "PENDING" and persisted in {"STARTED", "RETRY"}:
        return persisted, "db"

    # If Celery says STARTED but DB says RETRY, preserve RETRY semantics.
    if runtime == "STARTED" and persisted == "RETRY":
        return "RETRY", "db"

    # If both terminal but conflict, prefer SUCCESS over FAILURE/REVOKED,
    # then FAILURE over REVOKED.
    if runtime in TERMINAL_STATES and persisted in TERMINAL_STATES and runtime != persisted:
        if "SUCCESS" in {runtime, persisted}:
            return "SUCCESS", "reconciled"
        if "FAILURE" in {runtime, persisted}:
            return "FAILURE", "reconciled"
        return "REVOKED", "reconciled"

    return runtime, "celery"
