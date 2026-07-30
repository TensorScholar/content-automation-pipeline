"""
Property-Based Tests for Task Reliability
==========================================

Purpose: Find edge cases and mission-critical failures in task processing
NOT just making tests pass, but finding root causes and fixing them.

Test Categories:
1. Task Lifecycle (creation → processing → completion)
2. Worker Failure Scenarios
3. Queue/Broker Issues
4. Serialization/Deserialization
5. Timeout/Retry Logic
6. Concurrent Task Handling
"""

import asyncio
import os
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pytest
import redis
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = [pytest.mark.integration, pytest.mark.live, pytest.mark.slow]

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_BASE = os.getenv("LIVE_API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
REDIS_URL = os.getenv("LIVE_REDIS_URL", "redis://127.0.0.1:6379/0")
MANAGER_EMAIL = os.getenv("LIVE_MANAGER_EMAIL")
MANAGER_PASSWORD = os.getenv("LIVE_MANAGER_PASSWORD")


def _live_credentials() -> dict[str, str]:
    if not MANAGER_EMAIL or not MANAGER_PASSWORD:
        pytest.fail(
            "Set LIVE_MANAGER_EMAIL and LIVE_MANAGER_PASSWORD when running with --run-live."
        )
    return {"username": MANAGER_EMAIL, "password": MANAGER_PASSWORD}


# ==============================================================================
# PROPERTY-BASED TESTS (Edge Case Discovery)
# ==============================================================================

class TestTaskLifecycle:
    """
    Property: Every created task MUST transition through valid states
    and reach a terminal state (SUCCESS/FAILURE) within timeout.
    """

    def test_task_never_stuck_in_pending(self, live_auth_token, live_project_id):
        """
        CRITICAL: Tasks should NEVER remain in PENDING for > 30 seconds
        If a task is PENDING for > 30s, worker is not processing it.

        Root causes to detect:
        - Worker offline
        - Queue routing issue
        - Task not registered
        """
        # Create a simple task
        response = httpx.post(
            f"{API_BASE}/content/generate/async",
            headers={"Authorization": f"Bearer {live_auth_token}"},
            json={
                "project_id": live_project_id,
                "topic": "Test Task Lifecycle",
                "primary_keyword": "test",
                "word_count_range": "800-1000",
            },
            timeout=10
        )

        assert response.status_code == 202, \
            f"Task creation failed: HTTP {response.status_code}"

        task_id = response.json()["task_id"]

        # Monitor task state for 30 seconds
        start_time = time.time()
        max_wait = 30

        while time.time() - start_time < max_wait:
            task_response = httpx.get(
                f"{API_BASE}/content/task/{task_id}",
                headers={"Authorization": f"Bearer {live_auth_token}"},
                timeout=5
            )

            if task_response.status_code == 200:
                state = task_response.json()["state"]

                # Task should move out of PENDING within 30s
                if state != "PENDING":
                    print(f"✓ Task moved to {state} in {time.time() - start_time:.1f}s")
                    return  # PASS

            time.sleep(2)

        # FAIL: Task stuck in PENDING
        pytest.fail(
            f"CRITICAL: Task {task_id} stuck in PENDING for > 30s\n"
            f"Root cause: Worker not processing tasks\n"
            f"Check: 1) Worker online? 2) Correct queue? 3) Task registered?"
        )

    def test_task_reaches_terminal_state(self, live_auth_token, live_project_id):
        """
        Property: Every task MUST reach SUCCESS or FAILURE (never stuck)

        Edge cases:
        - Infinite loops
        - Deadlocks
        - Resource starvation
        """
        # Create task
        response = httpx.post(
            f"{API_BASE}/content/generate/async",
            headers={"Authorization": f"Bearer {live_auth_token}"},
            json={
                "project_id": live_project_id,
                "topic": "Terminal State Test",
                "primary_keyword": "test",
                "word_count_range": "800-1000",
            },
            timeout=10
        )

        assert response.status_code == 202, \
            f"Task creation failed: HTTP {response.status_code}"

        task_id = response.json()["task_id"]

        # Wait for terminal state (max 5 minutes)
        terminal_states = {"SUCCESS", "FAILURE", "REVOKED"}
        start = time.time()

        while time.time() - start < 300:  # 5 min timeout
            resp = httpx.get(
                f"{API_BASE}/content/task/{task_id}",
                headers={"Authorization": f"Bearer {live_auth_token}"},
                timeout=5
            )

            if resp.status_code == 200:
                state = resp.json()["state"]
                if state in terminal_states:
                    print(f"✓ Task reached {state} in {time.time() - start:.1f}s")
                    return  # PASS

            time.sleep(5)

        pytest.fail(
            f"CRITICAL: Task {task_id} never reached terminal state\n"
            f"Last known state: {resp.json()['state']}\n"
            f"Root cause: Task processing hung or infinite loop"
        )

    @given(
        word_count=st.integers(min_value=800, max_value=3500),
        keyword_count=st.integers(min_value=0, max_value=20)
    )
    @settings(
        max_examples=10,
        deadline=timedelta(seconds=60),
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_task_handles_various_inputs(
        self,
        live_auth_token,
        live_project_id,
        reset_live_rate_limits,
        word_count,
        keyword_count,
    ):
        """
        Property-based: Tasks should handle ANY valid input without crashing

        Finds edge cases like:
        - Very large word counts
        - Empty keywords
        - Special characters
        """
        keywords = [f"keyword_{i}" for i in range(keyword_count)]

        reset_live_rate_limits()
        response = httpx.post(
            f"{API_BASE}/content/generate/async",
            headers={"Authorization": f"Bearer {live_auth_token}"},
            json={
                "project_id": live_project_id,
                "topic": f"Property Test {word_count}w {uuid.uuid4().hex[:8]}",
                "primary_keyword": keywords[0] if keywords else None,
                "secondary_keywords": keywords[1:],
                "word_count_range": f"{word_count}-{word_count}",
            },
            timeout=30,
        )

        assert response.status_code == 202, \
            f"Unexpected status {response.status_code} for input: word_count={word_count}, keywords={keyword_count}"


# ==============================================================================
# WORKER HEALTH TESTS
# ==============================================================================

class TestWorkerHealth:
    """
    Tests for detecting worker failures before they cause stuck tasks
    """

    def test_workers_are_online(self):
        """Workers must be online and responding"""
        from orchestration.celery_app import app

        inspect = app.control.inspect(timeout=5)
        active_workers = inspect.active()

        assert active_workers is not None, \
            "CRITICAL: No workers responding. Start workers with: celery -A orchestration.celery_app worker"

        assert len(active_workers) > 0, \
            "CRITICAL: No active workers found"

        print(f"✓ {len(active_workers)} worker(s) online: {list(active_workers.keys())}")

    def test_workers_have_tasks_registered(self):
        """Workers must have content generation tasks registered"""
        from orchestration.celery_app import app

        inspect = app.control.inspect(timeout=5)
        registered = inspect.registered()

        assert registered is not None, "Cannot get registered tasks"

        # Check if any worker has content generation tasks
        all_tasks = []
        for worker, tasks in registered.items():
            all_tasks.extend(tasks)

        required_tasks = [
            'orchestration.tasks.generate_content_task',
        ]

        for task in required_tasks:
            assert task in all_tasks, \
                f"CRITICAL: Task '{task}' not registered in any worker\n" \
                f"Available tasks: {all_tasks}"

        print(f"✓ All required tasks registered")

    def test_redis_connection(self):
        """Redis broker must be accessible"""
        try:
            r = redis.from_url(REDIS_URL)
            r.ping()
            print("✓ Redis connection healthy")
        except redis.ConnectionError as e:
            pytest.fail(f"CRITICAL: Cannot connect to Redis broker\n{e}")


# ==============================================================================
# QUEUE & SERIALIZATION TESTS
# ==============================================================================

class TestQueueReliability:
    """
    Tests for queue routing and serialization issues
    """

    def test_task_serialization(self, live_project_id):
        """Tasks must serialize/deserialize correctly"""
        from orchestration.celery_app import app

        # Try to send a task
        try:
            from orchestration.tasks import generate_content_task

            # This should not raise serialization errors
            result = generate_content_task.apply_async(
                kwargs={
                    "project_id": live_project_id,
                    "topic": "Serialization Test",
                    "primary_keyword": "test",
                    "word_count_range": "800-1000",
                },
                countdown=9999  # Don't actually run it
            )

            # Revoke it immediately
            result.revoke()

            print("✓ Task serialization working")

        except Exception as e:
            pytest.fail(f"CRITICAL: Task serialization failed\n{e}")

    def test_queue_not_overflowing(self):
        """Queue should not have excessive pending tasks"""
        r = redis.from_url(REDIS_URL)

        # Check default queue length
        queue_length = r.llen('celery')

        # If > 100 tasks pending, something is wrong
        assert queue_length < 100, \
            f"WARNING: {queue_length} tasks in queue (> 100). Workers may be overwhelmed or stuck."

        print(f"✓ Queue length: {queue_length} tasks")


# ==============================================================================
# CONCURRENCY & RACE CONDITION TESTS
# ==============================================================================

class TestConcurrentTasks:
    """
    Tests for race conditions and concurrent task handling
    """

    def test_multiple_tasks_simultaneously(
        self,
        live_auth_token,
        live_project_id,
        reset_live_rate_limits,
    ):
        """System should handle multiple concurrent tasks"""
        # Create 5 tasks at once
        task_ids = []
        failures = []

        for i in range(5):
            try:
                reset_live_rate_limits()
                response = httpx.post(
                    f"{API_BASE}/content/generate/async",
                    headers={"Authorization": f"Bearer {live_auth_token}"},
                    json={
                        "project_id": live_project_id,
                        "topic": f"Concurrent Test {i}",
                        "primary_keyword": f"test{i}",
                        "word_count_range": "800-1000",
                    },
                    timeout=30
                )

                if response.status_code == 202:
                    task_ids.append(response.json()["task_id"])
                else:
                    failures.append(f"Task {i}: HTTP {response.status_code}")
            except httpx.ReadTimeout as exc:
                failures.append(f"Task {i}: {exc.__class__.__name__}")

        assert len(task_ids) == 5, \
            f"Only {len(task_ids)}/5 concurrent tasks were accepted: {failures}"

        pending = set(task_ids)
        deadline = time.time() + 120
        while pending and time.time() < deadline:
            for task_id in list(pending):
                resp = httpx.get(
                    f"{API_BASE}/content/task/{task_id}",
                    headers={"Authorization": f"Bearer {live_auth_token}"},
                    timeout=10
                )
                assert resp.status_code == 200, \
                    f"Task status failed with HTTP {resp.status_code}"
                if resp.json()["state"] != "PENDING":
                    pending.remove(task_id)
            if pending:
                time.sleep(2)

        assert not pending, \
            f"Tasks stuck in PENDING after 120s: {sorted(pending)}\n" \
            f"This indicates a concurrency or worker capacity issue"

        print(f"✓ {len(task_ids)} concurrent tasks handled correctly")


# ==============================================================================
# USAGE
# ==============================================================================

"""
Run these tests to find edge cases and mission-critical issues:

# Run all reliability tests
pytest tests/test_task_reliability.py -v

# Run only critical path tests
pytest tests/test_task_reliability.py::TestTaskLifecycle -v

# Run property-based tests (edge case discovery)
pytest tests/test_task_reliability.py::TestTaskLifecycle::test_task_handles_various_inputs -v

# Run with coverage
pytest tests/test_task_reliability.py --cov=orchestration --cov-report=html

# Run continuously (catch intermittent issues)
pytest tests/test_task_reliability.py --count=10 -v
"""
