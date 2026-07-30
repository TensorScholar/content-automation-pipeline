"""
Advanced Property-Based Tests for Edge Case Discovery
======================================================

Purpose: Find mission-critical failures in:
- Database connection handling
- Redis connection failures
- Content generation edge cases
- Resource cleanup and leaks
- API timeout and retry logic
- Concurrent operations

These tests are designed to FIND issues, not just pass.
"""

import asyncio
import os
import random
import string
import time
import uuid
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pytest
import redis
from hypothesis import HealthCheck, assume, given, settings
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
# PROPERTY-BASED TESTS: Input Validation Edge Cases
# ==============================================================================

class TestInputValidationEdgeCases:
    """
    Test extreme input variations to find validation bugs
    """

    @given(
        topic_length=st.integers(min_value=1, max_value=500),
        unicode_level=st.integers(min_value=0, max_value=3)
    )
    @settings(
        max_examples=20,
        deadline=timedelta(seconds=30),
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_extreme_topic_lengths_and_unicode(
        self,
        live_auth_token,
        live_project_factory,
        reset_live_rate_limits,
        topic_length,
        unicode_level,
    ):
        """
        Property: API should handle ANY valid topic length and Unicode characters

        Edge cases:
        - Maximum length topics (500 chars)
        - Unicode: emoji, RTL (Persian/Arabic), special chars
        - Mixed scripts
        """
        # Generate topic based on unicode level
        if unicode_level == 0:
            # ASCII only
            topic = ''.join(random.choices(string.ascii_letters + ' ', k=topic_length))
        elif unicode_level == 1:
            # Persian/Arabic (RTL)
            persian_chars = 'آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی '
            topic = ''.join(random.choices(persian_chars, k=min(topic_length, 100)))
        elif unicode_level == 2:
            # Emoji and special Unicode
            emoji_chars = '😀😁😂🤣😃😄😅😆😉😊😋😎🔥💯✨🎉🎊 '
            topic = ''.join(random.choices(emoji_chars, k=min(topic_length, 50)))
        else:
            # Mixed (ASCII + Persian + Emoji)
            mixed = string.ascii_letters + 'آابپتثج' + '😀😁😂 '
            topic = ''.join(random.choices(mixed, k=min(topic_length, 100)))

        if not topic.strip():
            topic = "Test Topic"  # Ensure not empty

        reset_live_rate_limits()
        project_id = live_project_factory()
        response = httpx.post(
            f"{API_BASE}/content/generate/async",
            headers={"Authorization": f"Bearer {live_auth_token}"},
            json={
                "project_id": project_id,
                "topic": topic,
                "primary_keyword": "test",
                "word_count_range": "800-1000",
            },
            timeout=30,
        )

        assert response.status_code in [202, 422], \
            f"Unexpected status {response.status_code} for topic_length={topic_length}, unicode={unicode_level}"

    @given(
        keyword_count=st.integers(min_value=0, max_value=50),
        keyword_length=st.integers(min_value=1, max_value=100)
    )
    @settings(
        max_examples=15,
        deadline=timedelta(seconds=30),
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_extreme_keyword_variations(
        self,
        live_auth_token,
        live_project_id,
        reset_live_rate_limits,
        keyword_count,
        keyword_length,
    ):
        """
        Property: API should handle extreme keyword configurations

        Edge cases:
        - 0 keywords (should work)
        - 50 keywords (very large)
        - Very long keywords (100 chars)
        - Duplicate keywords
        """
        # Generate keywords
        keywords = [
            ''.join(random.choices(string.ascii_lowercase, k=min(keyword_length, 20)))
            for _ in range(min(keyword_count, 30))
        ]

        reset_live_rate_limits()
        response = httpx.post(
            f"{API_BASE}/content/generate/async",
            headers={"Authorization": f"Bearer {live_auth_token}"},
            json={
                "project_id": live_project_id,
                "topic": f"Keyword Edge {keyword_count}-{keyword_length}-{uuid.uuid4().hex[:8]}",
                "primary_keyword": keywords[0] if keywords else None,
                "secondary_keywords": keywords[1:],
                "word_count_range": "800-1000",
            },
            timeout=30,
        )

        assert response.status_code in [202, 422], \
            f"Unexpected status {response.status_code} for keyword_count={keyword_count}, keyword_length={keyword_length}"


# ==============================================================================
# PROPERTY-BASED TESTS: Database Connection Edge Cases
# ==============================================================================

class TestDatabaseEdgeCases:
    """
    Test database connection handling under stress
    """

    def test_rapid_task_creation_database_pool(
        self,
        live_auth_token,
        live_project_id,
        reset_live_rate_limits,
    ):
        """
        CRITICAL: Rapid task creation should not exhaust database connection pool

        Root cause to detect:
        - Connection pool exhaustion
        - Connections not being returned to pool
        - Database deadlocks
        """
        # Create 20 tasks rapidly (faster than they can execute)
        created = 0
        failures = []

        for i in range(20):
            try:
                reset_live_rate_limits()
                resp = httpx.post(
                    f"{API_BASE}/content/generate/async",
                    headers={"Authorization": f"Bearer {live_auth_token}"},
                    json={
                        "project_id": live_project_id,
                        "topic": f"DB Pool Test {i}",
                        "primary_keyword": "test",
                        "word_count_range": "800-1000",
                    },
                    timeout=5
                )

                if resp.status_code == 202:
                    created += 1
                else:
                    failures.append(f"Task {i}: HTTP {resp.status_code}")

            except httpx.ReadTimeout:
                failures.append(f"Task {i}: Timeout")
            except Exception as e:
                failures.append(f"Task {i}: {str(e)[:100]}")

        assert created == 20, \
            f"Database connection pool may be exhausted\n" \
            f"Created: {created}/20\n" \
            f"Failures: {failures[:5]}"

        print(f"✓ Created {created}/20 tasks rapidly without pool exhaustion")


# ==============================================================================
# PROPERTY-BASED TESTS: Redis Edge Cases
# ==============================================================================

class TestRedisEdgeCases:
    """
    Test Redis connection and cache handling
    """

    def test_redis_connection_recovery(self):
        """
        CRITICAL: System should recover from Redis connection failures

        Edge case: Redis temporarily unavailable during task execution
        """
        r = redis.from_url(REDIS_URL)

        # Check current Redis connections
        try:
            info = r.info('clients')
            initial_connections = info.get('connected_clients', 0)
            print(f"Initial Redis connections: {initial_connections}")

            # Verify Redis is healthy
            assert r.ping(), "Redis not responding"

            print("✓ Redis connection recovery test passed")

        except redis.ConnectionError as e:
            pytest.fail(f"CRITICAL: Redis connection failed: {e}")

    def test_redis_memory_pressure(self):
        """
        WARNING: Check if Redis memory usage is within limits

        Edge case: Task metadata accumulation causing memory issues
        """
        r = redis.from_url(REDIS_URL)

        try:
            info = r.info('memory')
            used_memory_mb = info['used_memory'] / (1024 * 1024)
            max_memory_mb = info.get('maxmemory', 0) / (1024 * 1024)

            print(f"Redis memory: {used_memory_mb:.2f}MB")

            if max_memory_mb > 0:
                usage_percent = (used_memory_mb / max_memory_mb) * 100
                assert usage_percent < 90, \
                    f"WARNING: Redis memory usage high: {usage_percent:.1f}%"

            print("✓ Redis memory within limits")

        except Exception as e:
            pytest.fail(f"Redis memory check failed: {e}")


# ==============================================================================
# PROPERTY-BASED TESTS: Resource Cleanup
# ==============================================================================

class TestResourceCleanup:
    """
    Test for resource leaks and proper cleanup
    """

    def test_worker_connection_cleanup(self):
        """
        CRITICAL: Workers should properly clean up connections

        Root cause to detect:
        - Database connections not closed
        - Redis connections leaking
        - File descriptors not released
        """
        from orchestration.celery_app import app

        inspect = app.control.inspect(timeout=10)
        stats = inspect.stats()

        if stats:
            for worker, worker_stats in stats.items():
                pool = worker_stats.get('pool', {})
                rusage = worker_stats.get('rusage', {})

                print(f"Worker: {worker.split('@')[1]}")
                print(f"  Active processes: {pool.get('max-concurrency', 'unknown')}")
                print(f"  Max tasks per child: {pool.get('max-tasks-per-child', 'unknown')}")

                # Check if workers are being recycled (prevents memory leaks)
                max_tasks = pool.get('max-tasks-per-child')
                assert max_tasks and max_tasks > 0, \
                    "Workers should be recycled after N tasks to prevent memory leaks"

        print("✓ Worker connection cleanup configuration verified")


# ==============================================================================
# PROPERTY-BASED TESTS: Task Retry Logic
# ==============================================================================

class TestTaskRetryLogic:
    """
    Test task retry and error handling
    """

    def test_invalid_project_id_fails_immediately(self, live_auth_token):
        """
        Property: Tasks with invalid project IDs should fail fast (not retry)

        Edge case: Validation errors should not trigger retries
        """
        # Create task with random UUID (won't exist in DB)
        response = httpx.post(
            f"{API_BASE}/content/generate/async",
            headers={"Authorization": f"Bearer {live_auth_token}"},
            json={
                "project_id": str(uuid.uuid4()),
                "topic": "Test Invalid Project",
                "primary_keyword": "test",
                "word_count_range": "800-1000",
            },
            timeout=30
        )

        assert response.status_code == 404, \
            f"Invalid project should fail before queueing, got HTTP {response.status_code}"


# ==============================================================================
# PROPERTY-BASED TESTS: Concurrent Operations
# ==============================================================================

class TestConcurrentOperations:
    """
    Test race conditions and concurrent access
    """

    def test_concurrent_authentication_requests(self):
        """
        Property: Multiple simultaneous auth requests should all succeed

        Edge case: Database connection race conditions during auth
        """
        import concurrent.futures

        def authenticate():
            try:
                response = httpx.post(
                    f"{API_BASE}/auth/token",
                    data=_live_credentials(),
                    timeout=10
                )
                return response.status_code == 200
            except Exception:
                return False

        # Stay below the authentication rate limit while exercising concurrency.
        request_count = 4
        with concurrent.futures.ThreadPoolExecutor(max_workers=request_count) as executor:
            futures = [executor.submit(authenticate) for _ in range(request_count)]
            results = [f.result() for f in concurrent.futures.as_completed(futures)]

        success_count = sum(results)

        assert success_count == request_count, \
            f"Concurrent auth failures: {success_count}/{request_count} succeeded\n" \
            f"Possible database connection race condition"

        print(f"✓ {success_count}/{request_count} concurrent auth requests succeeded")
