#!/usr/bin/env python3
"""Fail-closed static invariants for the Smarlux P0 release candidate."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPECTED_HEAD = "20260801_001"
READONLY_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"


def read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> int:
    service = read("services/search_console_service.py")
    google_auth_urls = set(
        re.findall(r"https://www\.googleapis\.com/auth/[A-Za-z0-9._/-]+", service)
    )
    require(
        google_auth_urls == {READONLY_SCOPE},
        f"Search Console scopes must be exactly read-only; found {sorted(google_auth_urls)}",
    )
    require("include_granted_scopes" not in service, "Incremental OAuth authorization is forbidden for strict read-only access")
    require("granted_scopes != {READONLY_SCOPE}" in service, "Granted OAuth scopes are not validated exactly")

    main_api = read("api/main.py")
    require("search_console" in main_api and "include_router" in main_api, "Search Console router is not registered")

    celery = read("orchestration/celery_app.py")
    compose = read("docker-compose.yml") + "\n" + read("docker-compose.prod.yml")
    require('Queue("integrations"' in celery, "Celery integrations queue is not declared")
    require("integrations" in compose, "Celery integrations queue is not consumed by Compose workers")
    require("content_automation.sync_all_search_console" in celery, "Search Console periodic task is missing")
    require("reconcile_wordpress_publishes" in celery, "WordPress reconciliation schedule is missing")

    schema = read("infrastructure/schema.py")
    require("uq_publishing_success_idempotency" in schema, "WordPress success idempotency index is missing")
    require("search_console_sync_runs" in schema, "Search Console durable sync table is missing")

    publishing_repository = read("knowledge/publishing_repository.py")
    require(
        "publishing_attempts_table.c.task_id == task_id" in publishing_repository,
        "WordPress attempt transitions are not protected by task ownership",
    )
    search_repository = read("knowledge/search_console_repository.py")
    require(
        "search_console_sync_runs_table.c.task_id == task_id" in search_repository,
        "Search Console sync transitions are not protected by task ownership",
    )

    distributor = read("execution/distributer.py")
    require("_verify_wordpress_post" in distributor, "WordPress read-after-write verification is missing")
    require("publish_posts" in distributor, "WordPress publish capability gate is missing")
    require(
        "_validate_wordpress_network_target" in distributor and "address.is_global" in distributor,
        "WordPress outbound target DNS/public-network validation is missing",
    )

    tasks = read("orchestration/tasks.py")
    integration_runner = tasks.split("def _run_integration_async", 1)[1].split("@app.task", 1)[0]
    require("loop.run_until_complete(coro)" in integration_runner, "Integration task runner is incomplete")
    require("loop.close()" not in integration_runner, "Integration tasks must reuse the worker event loop")

    migration = read("alembic/versions/20260801_add_p0_integrations.py")
    require(f'revision = "{EXPECTED_HEAD}"' in migration, "P0 migration revision is unexpected")

    prod_compose = read("docker-compose.prod.yml")
    return_lines = [
        line.strip()
        for line in prod_compose.splitlines()
        if "GOOGLE_SEARCH_CONSOLE_FRONTEND_RETURN_URL" in line
    ]
    require(return_lines, "Search Console frontend return URL is not exposed to production services")
    require(
        all("localhost" not in line and "127.0.0.1" not in line for line in return_lines),
        "Production Search Console return URL must not default to localhost",
    )

    print("P0_STATIC_INVARIANTS_PASS")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (AssertionError, OSError) as exc:
        print(f"P0_STATIC_INVARIANTS_FAIL: {exc}", file=sys.stderr)
        raise SystemExit(1)
