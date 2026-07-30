import os
import time
import uuid
from collections.abc import Callable, Iterator

import httpx
import pytest
import redis


LIVE_API_BASE_URL = os.getenv("LIVE_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
LIVE_REDIS_URL = os.getenv("LIVE_REDIS_URL", "redis://127.0.0.1:6379/0")


def _require_isolated_live_runtime() -> None:
    if os.getenv("LIVE_ISOLATED_RUNTIME") != "1":
        pytest.fail(
            "Live integration tests require LIVE_ISOLATED_RUNTIME=1 and disposable "
            "PostgreSQL/Redis services."
        )


def _clear_live_rate_limits() -> None:
    _require_isolated_live_runtime()
    client = redis.from_url(LIVE_REDIS_URL)
    try:
        keys = list(client.scan_iter(match="rate_limit:*"))
        if keys:
            client.delete(*keys)
    finally:
        client.close()


def _task_ids(snapshot: dict | None, *, scheduled: bool = False) -> set[str]:
    task_ids: set[str] = set()
    for tasks in (snapshot or {}).values():
        for item in tasks:
            task = item.get("request", item) if scheduled else item
            task_id = task.get("id")
            if task_id:
                task_ids.add(task_id)
    return task_ids


def _clear_live_task_queue() -> None:
    _require_isolated_live_runtime()

    from orchestration.celery_app import app

    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        app.control.purge()
        inspector = app.control.inspect(timeout=1)
        active_ids = _task_ids(inspector.active())
        reserved_ids = _task_ids(inspector.reserved())
        scheduled_ids = _task_ids(inspector.scheduled(), scheduled=True)

        if reserved_ids or scheduled_ids:
            app.control.revoke(list(reserved_ids | scheduled_ids))
        if active_ids:
            app.control.revoke(
                list(active_ids),
                terminate=True,
                signal="SIGTERM",
            )
        # Celery keeps revoked ETA/retry tasks visible in `scheduled` until
        # their ETA. They are isolated once revoked, so only running or
        # immediately reservable work must drain before the next test.
        if not active_ids and not reserved_ids:
            app.control.purge()
            return

        time.sleep(0.25)

    pytest.fail("Disposable live-runtime task queue did not become idle.")


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-live",
        action="store_true",
        default=False,
        help="Run tests that require a configured live API and worker runtime.",
    )
    parser.addoption(
        "--run-chaos",
        action="store_true",
        default=False,
        help="Run destructive chaos and fault-injection tests.",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    run_live = config.getoption("--run-live")
    run_chaos = config.getoption("--run-chaos")
    skip_live = pytest.mark.skip(reason="requires --run-live and a configured live runtime")
    skip_chaos = pytest.mark.skip(reason="requires explicit --run-chaos opt-in")

    for item in items:
        if "live" in item.keywords and not run_live:
            item.add_marker(skip_live)
        if "chaos" in item.keywords and not run_chaos:
            item.add_marker(skip_chaos)


@pytest.fixture(scope="session")
def live_auth_token() -> str:
    _require_isolated_live_runtime()
    email = os.getenv("LIVE_MANAGER_EMAIL")
    password = os.getenv("LIVE_MANAGER_PASSWORD")
    if not email or not password:
        pytest.fail("LIVE_MANAGER_EMAIL and LIVE_MANAGER_PASSWORD are required.")

    _clear_live_rate_limits()
    response = httpx.post(
        f"{LIVE_API_BASE_URL}/auth/token",
        data={"username": email, "password": password},
        timeout=30,
    )
    assert response.status_code == 200, (
        f"Live authentication failed with HTTP {response.status_code}."
    )
    return response.json()["access_token"]


@pytest.fixture
def live_auth_headers(live_auth_token: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {live_auth_token}"}


@pytest.fixture(autouse=True)
def isolate_live_runtime_state(request: pytest.FixtureRequest) -> Iterator[None]:
    if "live" not in request.keywords:
        yield
        return

    _clear_live_rate_limits()
    _clear_live_task_queue()
    yield
    _clear_live_task_queue()
    _clear_live_rate_limits()


@pytest.fixture
def reset_live_rate_limits() -> Callable[[], None]:
    return _clear_live_rate_limits


@pytest.fixture
def live_project_factory(
    live_auth_headers: dict[str, str],
) -> Callable[[], str]:
    def create_project() -> str:
        _require_isolated_live_runtime()
        response = httpx.post(
            f"{LIVE_API_BASE_URL}/projects",
            headers=live_auth_headers,
            json={
                "name": f"Isolated Live Test {uuid.uuid4().hex[:10]}",
                "description": "Disposable project for isolated local live tests.",
            },
            timeout=30,
        )
        assert response.status_code == 201, (
            f"Disposable project creation failed with HTTP {response.status_code}."
        )
        return response.json()["id"]

    return create_project


@pytest.fixture
def live_project_id(live_project_factory: Callable[[], str]) -> str:
    return live_project_factory()
