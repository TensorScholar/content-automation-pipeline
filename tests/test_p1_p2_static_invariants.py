"""Deterministic tests for the repaired P1/P2 metrics-route static invariant."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "maintenance" / "p1_p2_static_invariants.py"

spec = importlib.util.spec_from_file_location("p1_p2_static_invariants", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(module)

VALID_API_ROUTE = """
from fastapi import FastAPI

app = FastAPI()


def internal_metrics():
    return "ok"


app.add_api_route("/metrics", internal_metrics, methods=["GET"])
"""

VALID_DECORATOR_ROUTE = """
from fastapi import FastAPI

app = FastAPI()


@app.get("/metrics")
def internal_metrics():
    return "ok"
"""

AGGREGATING_HANDLER = """
from fastapi import FastAPI

app = FastAPI()


def internal_metrics():
    return get_summary(123)


app.add_api_route("/metrics", internal_metrics, methods=["GET"])
"""

NO_ROUTE = """
from fastapi import FastAPI

app = FastAPI()


def internal_metrics():
    return "ok"
"""


def test_metrics_route_accepts_cache_only_handler_in_any_supported_form() -> None:
    module.validate_metrics_route_no_db_aggregation(VALID_API_ROUTE)
    module.validate_metrics_route_no_db_aggregation(VALID_DECORATOR_ROUTE)


def test_metrics_route_rejects_aggregating_handler() -> None:
    with pytest.raises(AssertionError):
        module.validate_metrics_route_no_db_aggregation(AGGREGATING_HANDLER)


def test_metrics_route_rejects_missing_registration() -> None:
    with pytest.raises(AssertionError):
        module.validate_metrics_route_no_db_aggregation(NO_ROUTE)


def test_metrics_route_invariant_holds_for_current_application() -> None:
    current = (ROOT / "api" / "main.py").read_text(encoding="utf-8")
    module.validate_metrics_route_no_db_aggregation(current)