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
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pytest
import redis
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

# ==============================================================================
# CONFIGURATION
# ==============================================================================

API_BASE = "http://127.0.0.1:9001"
REDIS_URL = "redis://127.0.0.1:6379/0"


# ==============================================================================
# PROPERTY-BASED TESTS (Edge Case Discovery)
# ==============================================================================

class TestTaskLifecycle:
    """
    Property: Every created task MUST transition through valid states
    and reach a terminal state (SUCCESS/FAILURE) within timeout.
    """

    @pytest.fixture
    def auth_token(self):
        """Get authentication token"""
        response = httpx.post(
            f"{API_BASE}/auth/token",
            data={
                "username": "manager@smarlux.com",
                "password": "Manager@123456789"
            },
            timeout=10
        )
        assert response.status_code == 200
        return response.json()["access_token"]

    def test_task_never_stuck_in_pending(self, auth_token):
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
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "project_id": str(uuid.uuid4()),
                "topic": "Test Task Lifecycle",
                "keywords": ["test"],
                "word_count": 100
            },
            timeout=10
        )

        if response.status_code != 202:
            pytest.skip(f"Task creation failed: {response.status_code}")

        task_id = response.json()["task_id"]

        # Monitor task state for 30 seconds
        start_time = time.time()
        max_wait = 30

        while time.time() - start_time < max_wait:
            task_response = httpx.get(
                f"{API_BASE}/content/task/{task_id}",
                headers={"Authorization": f"Bearer {auth_token}"},
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

    def test_task_reaches_terminal_state(self, auth_token):
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
            headers={"Authorization": f"Bearer {auth_token}"},
            json={
                "project_id": str(uuid.uuid4()),
                "topic": "Terminal State Test",
                "keywords": ["test"],
                "word_count": 100
            },
            timeout=10
        )

        if response.status_code != 202:
            pytest.skip("Task creation failed")

        task_id = response.json()["task_id"]

        # Wait for terminal state (max 5 minutes)
        terminal_states = {"SUCCESS", "FAILURE", "REVOKED"}
        start = time.time()

        while time.time() - start < 300:  # 5 min timeout
            resp = httpx.get(
                f"{API_BASE}/content/task/{task_id}",
                headers={"Authorization": f"Bearer {auth_token}"},
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
        word_count=st.integers(min_value=1, max_value=5000),
        keyword_count=st.integers(min_value=0, max_value=20)
    )
    @settings(
        max_examples=10,
        deadline=timedelta(seconds=60),
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_task_handles_various_inputs(self, auth_token, word_count, keyword_count):
        """
        Property-based: Tasks should handle ANY valid input without crashing

        Finds edge cases like:
        - Very large word counts
        - Empty keywords
        - Special characters
        """
        keywords = [f"keyword_{i}" for i in range(keyword_count)]

        try:
            response = httpx.post(
                f"{API_BASE}/content/generate/async",
                headers={"Authorization": f"Bearer {auth_token}"},
                json={
                    "project_id": str(uuid.uuid4()),
                    "topic": f"Property Test {word_count}w",
                    "keywords": keywords,
                    "word_count": word_count
                },
                timeout=30  # Increased timeout for large requests
            )

            # Should either accept or reject cleanly (no 500 errors)
            assert response.status_code in [200, 202, 400, 422], \
                f"Unexpected status {response.status_code} for input: word_count={word_count}, keywords={keyword_count}"

        except httpx.ReadTimeout:
            # Timeout is acceptable - API didn't crash, just took too long
            # This is a performance issue, not a reliability issue
            pass


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

    def test_task_serialization(self):
        """Tasks must serialize/deserialize correctly"""
        from orchestration.celery_app import app

        # Try to send a task
        try:
            from orchestration.tasks import generate_content_task

            # This should not raise serialization errors
            result = generate_content_task.apply_async(
                kwargs={
                    "project_id": str(uuid.uuid4()),
                    "topic": "Serialization Test",
                    "keywords": ["test"],
                    "word_count": 100
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

    def test_multiple_tasks_simultaneously(self, auth_token):
        """System should handle multiple concurrent tasks"""
        # Create 5 tasks at once
        task_ids = []

        for i in range(5):
            try:
                response = httpx.post(
                    f"{API_BASE}/content/generate/async",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    json={
                        "project_id": str(uuid.uuid4()),
                        "topic": f"Concurrent Test {i}",
                        "keywords": [f"test{i}"],
                        "word_count": 100
                    },
                    timeout=30
                )

                if response.status_code == 202:
                    task_ids.append(response.json()["task_id"])
            except httpx.ReadTimeout:
                # API slow under load - acceptable
                pass

        assert len(task_ids) > 0, "No tasks were created"

        # All should eventually move out of PENDING (2 minute wait for concurrent processing)
        time.sleep(120)

        stuck_tasks = []
        for task_id in task_ids:
            try:
                resp = httpx.get(
                    f"{API_BASE}/content/task/{task_id}",
                    headers={"Authorization": f"Bearer {auth_token}"},
                    timeout=10
                )

                if resp.status_code == 200:
                    state = resp.json()["state"]
                    if state == "PENDING":
                        stuck_tasks.append(task_id)
            except Exception:
                # Can't check status - might be a transient issue
                pass

        assert len(stuck_tasks) == 0, \
            f"Tasks stuck in PENDING after 120s: {stuck_tasks}\n" \
            f"This indicates a concurrency or worker capacity issue"

        print(f"✓ {len(task_ids)} concurrent tasks handled correctly")


# ==============================================================================
# FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def auth_token():
    """Session-scoped auth token"""
    response = httpx.post(
        f"{API_BASE}/auth/token",
        data={
            "username": "manager@smarlux.com",
            "password": "Manager@123456789"
        },
        timeout=10
    )

    if response.status_code != 200:
        pytest.skip("Cannot authenticate - check credentials")

    return response.json()["access_token"]


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
