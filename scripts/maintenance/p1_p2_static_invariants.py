#!/usr/bin/env python3
"""Fail-closed static invariants for Smarlux P1/P2 launch quality."""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def metrics_handler_name(module_source: str) -> str | None:
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_api_route"
            and len(node.args) >= 2
            and isinstance(node.args[0], ast.Constant)
            and node.args[0].value == "/metrics"
            and isinstance(node.args[1], ast.Name)
        ):
            return node.args[1].id
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            for decorator in node.decorator_list:
                if (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "get"
                    and len(decorator.args) == 1
                    and isinstance(decorator.args[0], ast.Constant)
                    and decorator.args[0].value == "/metrics"
                ):
                    return node.name
    return None


def function_calls(module_source: str, function_name: str, callee: str) -> bool:
    tree = ast.parse(module_source)
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != function_name:
            continue
        for sub in ast.walk(node):
            if not isinstance(sub, ast.Call):
                continue
            sub_func = sub.func
            if isinstance(sub_func, ast.Name) and sub_func.id == callee:
                return True
            if isinstance(sub_func, ast.Attribute) and sub_func.attr == callee:
                return True
    return False


def validate_metrics_route_no_db_aggregation(module_source: str) -> None:
    handler = metrics_handler_name(module_source)
    if handler is None:
        raise AssertionError("Prometheus metrics route registration is missing")
    if function_calls(module_source, handler, "get_summary"):
        raise AssertionError("Prometheus scrape must not trigger database aggregation")


def main() -> int:
    seo = read("services/seo_intelligence_service.py")
    require('ENGINE_VERSION = "seo-intelligence-v2.0"' in seo, "SEO engine version is missing")
    for guardrail in (
        '"uses_llm": False',
        '"performs_network_requests": False',
        '"rewrites_content": False',
        '"publishes_content": False',
        '"explanation_available": True',
    ):
        require(guardrail in seo, f"SEO intelligence guardrail missing: {guardrail}")
    require("MAX_SNAPSHOTS = 2_000" in seo, "SEO snapshot work is not bounded")
    require("MAX_OPPORTUNITIES = 250" in seo, "SEO opportunity work is not bounded")
    require("priority_score" in seo and "confidence" in seo, "SEO scoring is not explainable")
    require(
        not re.search(r"^(?:from|import)\s+(?:httpx|requests)\b", seo, re.M),
        "SEO engine must not import external HTTP clients",
    )

    project_routes = read("api/routes/projects.py")
    require('/{project_id:uuid}/seo-intelligence' in project_routes, "SEO intelligence route is missing")
    require("get_current_active_user" in project_routes, "SEO intelligence route lacks authentication")

    operations = read("services/integration_operations_service.py")
    require("asyncio.gather" in operations, "Integration summaries must be queried concurrently")
    require("recent_limit=10" in operations, "Operational failure output is not bounded")
    require("bounded_lookback" in operations, "Operational summary lookback is not bounded before querying")
    require("_store_snapshot_safely" in operations, "Snapshot writes must be failure-isolated")
    require("INTEGRATION_OPERATIONS_SNAPSHOT_TTL_SECONDS = 900" in operations, "Snapshot TTL is not bounded")
    require("project_id is None" in operations, "Project-scoped views must not overwrite global metrics")

    dependencies = read("api/dependencies.py")
    require("cache=get_redis()" in dependencies, "Integration operations service lacks shared Redis cache")
    require("metrics=get_metrics()" not in re.search(
        r"def get_integration_operations_service.*?\n\n", dependencies, re.S
    ).group(0), "Integration operations must not depend on process-local metrics")

    metrics_renderer = read("infrastructure/integration_metrics.py")
    for metric in (
        "integration_snapshot_available",
        "integration_snapshot_age_seconds",
        "integration_durable_active_items",
        "integration_durable_stale_items",
        "integration_durable_health",
        "integration_durable_recent_total",
        "integration_durable_recent_succeeded",
        "integration_durable_recent_failed",
        "integration_durable_failure_rate",
        "integration_durable_p95_duration_seconds",
    ):
        require(metric in metrics_renderer, f"Durable snapshot metric missing: {metric}")
    for forbidden_label in ("project_id", "site_url", "task_id", "error_category", "user_id"):
        require(forbidden_label not in metrics_renderer, f"High-cardinality metric label risk: {forbidden_label}")
    require("INTEGRATION_NAMES" in metrics_renderer, "Integration labels are not allow-listed")
    require("HEALTH_STATES" in metrics_renderer, "Health labels are not allow-listed")

    system_routes = read("api/routes/system.py")
    require('/integrations/operations' in system_routes, "Integration operations route is missing")
    require("get_current_superuser" in system_routes, "Integration operations route is not privileged")

    tasks = read("orchestration/tasks.py")
    celery = read("orchestration/celery_app.py")
    require(
        "refresh_integration_operational_metrics" in tasks
        and "refresh_integration_operational_metrics" in celery,
        "Central durable integration snapshot refresh is incomplete",
    )
    main_api = read("api/main.py")
    require("render_integration_snapshot_metrics" in main_api, "Metrics endpoint does not export durable snapshot")
    require("get_cached_snapshot" in main_api, "Metrics endpoint is not cache-only")
    require('decode("utf-8", errors="replace")' in main_api, "Prometheus byte payload is not normalized safely")
    validate_metrics_route_no_db_aggregation(main_api)
    require(
        "_refresh_integration_metrics" not in main_api,
        "Integration polling must not be duplicated in every API worker",
    )

    api_client = read("frontend/src/lib/api.ts")
    require('method === "GET"' in api_client, "Client retries are not restricted to GET")
    require("Math.min(maxRetries, 3)" in api_client, "Client retries are not bounded")
    require("REQUEST_DEDUP_TTL" in api_client, "GET request de-duplication is missing")
    require("Retry-After" in api_client, "Retry-After is not respected")
    require("X-Request-ID" in api_client, "Request correlation ID is not surfaced")

    projects_ui = read("frontend/src/components/panels/projects-panel.tsx")
    monitoring_ui = read("frontend/src/components/panels/monitoring-panel.tsx")
    require("/seo-intelligence" in projects_ui, "SEO intelligence UI integration is missing")
    require("/system/integrations/operations" in monitoring_ui, "Operations UI integration is missing")
    require("Promise.allSettled" in monitoring_ui, "Monitoring partial-failure isolation is missing")
    require('aria-live="polite"' in monitoring_ui, "Operations status is not announced accessibly")

    alerts = read("monitoring/alert_rules.yml")
    for alert in (
        "IntegrationSnapshotUnavailable",
        "IntegrationOperationalSnapshotStale",
        "IntegrationStaleWork",
        "IntegrationHealthCritical",
        "IntegrationFailureRateHigh",
    ):
        require(alert in alerts, f"Alert missing: {alert}")
    require("integration_durable_" in alerts, "Alerts do not use durable snapshot metrics")
    require("integration_operations_total" not in alerts, "Alerts still depend on worker-local counters")

    dashboard = read("grafana/dashboards/content-automation-dashboard.json")
    require("Integration Outcomes (24h)" in dashboard, "Integration outcomes dashboard panel is missing")
    require("Integration Operation Latency p95 (24h)" in dashboard, "Integration latency panel is missing")
    require("integration_durable_" in dashboard, "Dashboard does not use durable snapshot metrics")

    repositories = read("knowledge/publishing_repository.py") + read(
        "knowledge/search_console_repository.py"
    )
    require(repositories.count("percentile_cont(0.95)") >= 2, "Durable p95 latency aggregation is incomplete")

    print("P1_P2_STATIC_INVARIANTS_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError, AttributeError) as exc:
        print(f"P1_P2_STATIC_INVARIANTS_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
