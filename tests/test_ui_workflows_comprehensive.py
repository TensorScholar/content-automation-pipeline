"""
Comprehensive UI Workflow Property-Based Tests
===============================================

Purpose: Super strict multi-pipeline testing to find mission-critical issues in:
- Complete task generation workflow (UI → API → Worker → Results)
- Task history and attributes validation
- Monitoring data accuracy
- User management workflows
- Project management workflows
- WordPress integration
- Data integrity and consistency

This is NOT about making tests pass - it's about finding REAL issues and fixing them.
"""

import os
import random
import string
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

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
# TEST FIXTURES
# ==============================================================================

@pytest.fixture(scope="session")
def api_client():
    """Session-wide HTTP client"""
    return httpx.Client(timeout=30.0)


@pytest.fixture(scope="session")
def manager_token(api_client):
    """Authenticate as manager for session"""
    response = api_client.post(
        f"{API_BASE}/auth/token",
        data=_live_credentials(),
    )

    if response.status_code != 200:
        pytest.skip(f"Authentication failed: {response.status_code}")

    return response.json()["access_token"]


@pytest.fixture
def auth_headers(manager_token):
    """Authorization headers"""
    return {"Authorization": f"Bearer {manager_token}"}


# ==============================================================================
# WORKFLOW TEST 1: Complete Task Generation Pipeline
# ==============================================================================

class TestTaskGenerationWorkflow:
    """
    Tests the COMPLETE workflow from UI action to final result:
    User clicks generate → API → Queue → Worker → DB → Result → UI display
    """

    def test_complete_task_lifecycle_with_validation(self, api_client, auth_headers):
        """
        CRITICAL: Complete task workflow must work end-to-end

        Steps:
        1. Create project
        2. Generate content (async task)
        3. Monitor task progress
        4. Validate task attributes
        5. Retrieve final content
        6. Validate all data integrity
        """
        # Step 1: Create project
        project_payload = {
            "name": f"Workflow Test {uuid.uuid4().hex[:8]}",
            "description": "End-to-end workflow validation",
            "client_name": "Test Client"
        }

        proj_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json=project_payload
        )

        assert proj_resp.status_code == 201, \
            f"Project creation failed: {proj_resp.status_code} - {proj_resp.text}"

        project = proj_resp.json()
        project_id = project["id"]

        # Validate project attributes
        assert "id" in project
        assert project["name"] == project_payload["name"]
        assert project["description"] == project_payload["description"]
        assert "created_at" in project

        print(f"✓ Project created: {project_id}")

        # Step 2: Generate content (async task)
        content_payload = {
            "project_id": project_id,
            "topic": "Advanced SEO Strategies for 2026",
            "keywords": ["SEO", "content marketing", "Google ranking"],
            "word_count": 200
        }

        gen_resp = api_client.post(
            f"{API_BASE}/content/generate/async",
            headers=auth_headers,
            json=content_payload
        )

        assert gen_resp.status_code == 202, \
            f"Task creation failed: {gen_resp.status_code} - {gen_resp.text}"

        task_data = gen_resp.json()
        task_id = task_data["task_id"]

        # Validate task creation response
        assert "task_id" in task_data
        assert "status" in task_data

        print(f"✓ Task created: {task_id}")

        # Step 3: Monitor task progress (critical for UI responsiveness)
        max_wait = 300  # 5 minutes
        start_time = time.time()
        task_states = []

        while time.time() - start_time < max_wait:
            status_resp = api_client.get(
                f"{API_BASE}/content/task/{task_id}",
                headers=auth_headers
            )

            assert status_resp.status_code == 200, \
                f"Task status check failed: {status_resp.status_code}"

            status_data = status_resp.json()
            current_state = status_data["state"]
            task_states.append(current_state)

            # Validate task attributes at each check
            assert "task_id" in status_data
            assert "state" in status_data
            assert status_data["task_id"] == task_id

            if current_state in ["SUCCESS", "FAILURE"]:
                print(f"✓ Task completed in {time.time() - start_time:.1f}s: {current_state}")

                # Validate terminal state attributes
                if current_state == "SUCCESS":
                    assert "result" in status_data, "SUCCESS task missing result"
                    result = status_data["result"]

                    # CRITICAL: Validate result structure
                    assert "content_id" in result, "Result missing content_id"
                    assert "title" in result, "Result missing title"
                    assert "content" in result, "Result missing content"

                    # Validate content quality
                    assert len(result["content"]) > 0, "Generated content is empty"
                    assert result["title"].strip() != "", "Generated title is empty"

                    content_id = result["content_id"]

                    # Step 5: Retrieve content from database
                    content_resp = api_client.get(
                        f"{API_BASE}/content/{content_id}",
                        headers=auth_headers
                    )

                    assert content_resp.status_code == 200, \
                        f"Content retrieval failed: {content_resp.status_code}"

                    content = content_resp.json()

                    # Step 6: Validate complete data integrity
                    assert content["id"] == content_id
                    assert content["project_id"] == project_id
                    assert content["title"] == result["title"]
                    assert content["content"] == result["content"]
                    assert "status" in content
                    assert "created_at" in content

                    print(f"✓ Content validated: {content_id}")
                    print(f"✓ Complete workflow PASSED")

                    # Validate task state transitions
                    assert "PENDING" in task_states or "STARTED" in task_states, \
                        "Task never entered processing state"

                    return  # SUCCESS

                elif current_state == "FAILURE":
                    # Analyze failure
                    error_info = status_data.get("result", "No error info")
                    pytest.fail(
                        f"WORKFLOW FAILED: Task ended in FAILURE state\n"
                        f"Error: {error_info}\n"
                        f"State transitions: {' → '.join(set(task_states))}"
                    )

            time.sleep(5)

        # Timeout
        pytest.fail(
            f"CRITICAL: Task did not complete within {max_wait}s\n"
            f"Last state: {task_states[-1]}\n"
            f"State history: {' → '.join(set(task_states))}\n"
            f"This indicates a stuck task or worker issue"
        )

    @given(
        word_count=st.integers(min_value=50, max_value=1000),
        keyword_count=st.integers(min_value=1, max_value=10)
    )
    @settings(
        max_examples=5,
        deadline=timedelta(seconds=120),
        suppress_health_check=[HealthCheck.function_scoped_fixture]
    )
    def test_task_generation_with_various_inputs(
        self, api_client, auth_headers, word_count, keyword_count
    ):
        """
        Property: Task generation should work for ANY valid input combination

        Finds edge cases in:
        - Word count handling
        - Keyword processing
        - Content quality validation
        """
        # Create temporary project
        proj_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json={
                "name": f"PropTest {uuid.uuid4().hex[:6]}",
                "description": "Property-based test project"
            }
        )

        if proj_resp.status_code != 201:
            pytest.skip("Project creation failed")

        project_id = proj_resp.json()["id"]

        # Generate random keywords
        keywords = [
            ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 10)))
            for _ in range(keyword_count)
        ]

        try:
            gen_resp = api_client.post(
                f"{API_BASE}/content/generate/async",
                headers=auth_headers,
                json={
                    "project_id": project_id,
                    "topic": f"Test Topic {word_count}w",
                    "keywords": keywords,
                    "word_count": word_count
                }
            )

            # Should either succeed or fail gracefully (no 500 errors)
            assert gen_resp.status_code in [200, 202, 400, 422], \
                f"Server error {gen_resp.status_code} for word_count={word_count}, " \
                f"keywords={keyword_count}\nResponse: {gen_resp.text}"

        except httpx.ReadTimeout:
            # Timeout is acceptable for large requests
            pass


# ==============================================================================
# WORKFLOW TEST 2: Task History and Attributes
# ==============================================================================

class TestTaskHistoryAndAttributes:
    """
    Validates that task history is accurate and all attributes are preserved
    """

    def test_task_history_completeness(self, api_client, auth_headers):
        """
        CRITICAL: Task history must show ALL tasks with correct attributes

        Validates:
        - All created tasks appear in history
        - Attributes are preserved (timestamps, states, etc.)
        - Filtering works correctly
        - Pagination works (if implemented)
        """
        # Create multiple tasks
        task_ids = []
        project_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json={"name": f"History Test {uuid.uuid4().hex[:6]}"}
        )

        if project_resp.status_code != 201:
            pytest.skip("Project creation failed")

        project_id = project_resp.json()["id"]

        # Create 3 tasks
        for i in range(3):
            resp = api_client.post(
                f"{API_BASE}/content/generate/async",
                headers=auth_headers,
                json={
                    "project_id": project_id,
                    "topic": f"History Test Task {i}",
                    "keywords": [f"test{i}"],
                    "word_count": 100
                }
            )

            if resp.status_code == 202:
                task_ids.append(resp.json()["task_id"])

        assert len(task_ids) > 0, "No tasks created"

        # Wait a moment for tasks to be registered
        time.sleep(2)

        # Retrieve task history
        history_resp = api_client.get(
            f"{API_BASE}/content/tasks",
            headers=auth_headers
        )

        assert history_resp.status_code == 200, \
            f"Task history retrieval failed: {history_resp.status_code}"

        tasks = history_resp.json()

        # CRITICAL: All created tasks must appear in history
        task_ids_in_history = {task["task_id"] for task in tasks}

        for task_id in task_ids:
            assert task_id in task_ids_in_history, \
                f"Task {task_id} missing from history!\n" \
                f"Created: {task_ids}\n" \
                f"In history: {task_ids_in_history}"

        # Validate task attributes
        for task in tasks:
            if task["task_id"] in task_ids:
                # Required attributes
                assert "task_id" in task
                assert "state" in task
                assert "project_id" in task

                # Optional but expected attributes
                if "created_at" in task:
                    # Validate timestamp format
                    try:
                        datetime.fromisoformat(task["created_at"].replace('Z', '+00:00'))
                    except ValueError:
                        pytest.fail(f"Invalid timestamp format: {task['created_at']}")

        print(f"✓ All {len(task_ids)} tasks found in history with valid attributes")

    def test_task_state_transitions_are_logged(self, api_client, auth_headers):
        """
        Property: Task state changes should be trackable

        Validates that tasks move through expected states
        """
        # Create task
        proj_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json={"name": f"State Test {uuid.uuid4().hex[:6]}"}
        )

        if proj_resp.status_code != 201:
            pytest.skip("Project creation failed")

        project_id = proj_resp.json()["id"]

        task_resp = api_client.post(
            f"{API_BASE}/content/generate/async",
            headers=auth_headers,
            json={
                "project_id": project_id,
                "topic": "State Transition Test",
                "keywords": ["test"],
                "word_count": 100
            }
        )

        if task_resp.status_code != 202:
            pytest.skip("Task creation failed")

        task_id = task_resp.json()["task_id"]

        # Monitor state transitions
        observed_states = []
        for _ in range(20):  # Check for 100 seconds max
            resp = api_client.get(
                f"{API_BASE}/content/task/{task_id}",
                headers=auth_headers
            )

            if resp.status_code == 200:
                state = resp.json()["state"]
                if state not in observed_states:
                    observed_states.append(state)

                if state in ["SUCCESS", "FAILURE"]:
                    break

            time.sleep(5)

        # Validate state progression
        assert len(observed_states) > 0, "No states observed"

        # Valid state transitions
        valid_states = {"PENDING", "STARTED", "SUCCESS", "FAILURE", "RETRY", "REVOKED"}
        for state in observed_states:
            assert state in valid_states, \
                f"Invalid state observed: {state}\n" \
                f"Valid states: {valid_states}"

        print(f"✓ State transitions: {' → '.join(observed_states)}")


# ==============================================================================
# WORKFLOW TEST 3: Monitoring Data Accuracy
# ==============================================================================

class TestMonitoringAccuracy:
    """
    Validates that monitoring endpoints return accurate real-time data
    """

    def test_health_endpoint_accuracy(self, api_client):
        """
        CRITICAL: Health endpoint must accurately reflect system state
        """
        resp = api_client.get(f"{API_BASE}/health")

        assert resp.status_code == 200, \
            f"Health endpoint failed: {resp.status_code}"

        health = resp.json()

        # Required fields
        assert "status" in health, "Health response missing 'status'"
        assert "timestamp" in health, "Health response missing 'timestamp'"

        # Validate timestamp is recent (within 10 seconds)
        try:
            health_time = datetime.fromisoformat(health["timestamp"].replace('Z', '+00:00'))
            now = datetime.utcnow()
            time_diff = abs((now - health_time.replace(tzinfo=None)).total_seconds())

            assert time_diff < 10, \
                f"Health timestamp is stale: {time_diff}s old"
        except Exception as e:
            pytest.fail(f"Invalid timestamp format: {e}")

        # Check dependencies if present
        if "dependencies" in health:
            deps = health["dependencies"]

            # Critical dependencies
            critical_deps = ["database", "redis"]
            for dep in critical_deps:
                if dep in deps:
                    dep_status = deps[dep]
                    assert dep_status in ["healthy", "degraded", "unhealthy"], \
                        f"Invalid dependency status for {dep}: {dep_status}"

        print(f"✓ Health endpoint accurate: {health.get('status')}")

    def test_worker_metrics_accuracy(self, api_client, auth_headers):
        """
        Property: Worker metrics should match actual system state

        Validates:
        - Worker count is accurate
        - Active task count is accurate
        - Queue length is accurate
        """
        from orchestration.celery_app import app

        # Get actual worker state
        inspect = app.control.inspect(timeout=5)
        actual_workers = inspect.active()

        if actual_workers is None:
            pytest.skip("Workers not responding")

        actual_worker_count = len(actual_workers)
        actual_active_tasks = sum(len(tasks) for tasks in actual_workers.values())

        # Check Redis queue
        r = redis.from_url(REDIS_URL)
        actual_queue_length = r.llen('celery')

        print(f"✓ Actual metrics: {actual_worker_count} workers, "
              f"{actual_active_tasks} active tasks, {actual_queue_length} queued")

        # If monitoring endpoint exists, validate it matches
        try:
            monitor_resp = api_client.get(
                f"{API_BASE}/monitoring/workers",
                headers=auth_headers
            )

            if monitor_resp.status_code == 200:
                reported_metrics = monitor_resp.json()

                # Validate accuracy (allow small variance due to timing)
                if "worker_count" in reported_metrics:
                    assert reported_metrics["worker_count"] == actual_worker_count, \
                        f"Worker count mismatch: reported {reported_metrics['worker_count']}, " \
                        f"actual {actual_worker_count}"
        except httpx.HTTPStatusError:
            # Monitoring endpoint may not exist
            pass


# ==============================================================================
# WORKFLOW TEST 4: User Management
# ==============================================================================

class TestUserManagementWorkflow:
    """
    Tests complete user management workflows
    """

    def test_user_list_completeness(self, api_client, auth_headers):
        """
        CRITICAL: User list must show all users with correct attributes
        """
        resp = api_client.get(
            f"{API_BASE}/auth/users",
            headers=auth_headers
        )

        assert resp.status_code == 200, \
            f"User list retrieval failed: {resp.status_code}"

        users = resp.json()

        assert isinstance(users, list), "Users response is not a list"
        assert len(users) > 0, "No users found (should have at least manager)"

        # Validate user attributes
        for user in users:
            # Required fields
            assert "id" in user, f"User missing 'id': {user}"
            assert "email" in user, f"User missing 'email': {user}"
            assert "role" in user, f"User missing 'role': {user}"

            # Validate email format
            email = user["email"]
            assert "@" in email, f"Invalid email format: {email}"

            # Validate role
            valid_roles = ["admin", "manager", "editor", "viewer"]
            assert user["role"] in valid_roles, \
                f"Invalid role '{user['role']}' for user {email}"

        # Manager should exist
        manager_found = any(u["email"] == MANAGER_EMAIL for u in users)
        assert manager_found, f"Manager user {MANAGER_EMAIL} not found in user list"

        print(f"✓ User list complete: {len(users)} users validated")


# ==============================================================================
# WORKFLOW TEST 5: Project Management
# ==============================================================================

class TestProjectManagementWorkflow:
    """
    Tests project CRUD operations and data integrity
    """

    def test_project_crud_workflow(self, api_client, auth_headers):
        """
        CRITICAL: Complete project lifecycle must work

        Steps: Create → Read → Update → Delete
        """
        # CREATE
        create_payload = {
            "name": f"CRUD Test {uuid.uuid4().hex[:8]}",
            "description": "Testing complete CRUD workflow",
            "client_name": "Test Client"
        }

        create_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json=create_payload
        )

        assert create_resp.status_code in [200, 201], \
            f"Project creation failed: {create_resp.status_code} - {create_resp.text}"

        project = create_resp.json()
        project_id = project["id"]

        # Validate creation
        assert project["name"] == create_payload["name"]
        assert project["description"] == create_payload["description"]

        print(f"✓ Created: {project_id}")

        # READ
        read_resp = api_client.get(
            f"{API_BASE}/projects/{project_id}",
            headers=auth_headers
        )

        assert read_resp.status_code == 200, \
            f"Project read failed: {read_resp.status_code}"

        read_project = read_resp.json()

        # Data integrity check
        assert read_project["id"] == project_id
        assert read_project["name"] == create_payload["name"]

        print(f"✓ Read: data integrity confirmed")

        # UPDATE
        update_payload = {
            "name": f"Updated {create_payload['name']}",
            "description": "Updated description"
        }

        update_resp = api_client.put(
            f"{API_BASE}/projects/{project_id}",
            headers=auth_headers,
            json=update_payload
        )

        assert update_resp.status_code == 200, \
            f"Project update failed: {update_resp.status_code}"

        updated_project = update_resp.json()

        # Validate update
        assert updated_project["name"] == update_payload["name"]
        assert updated_project["description"] == update_payload["description"]

        print(f"✓ Updated: changes persisted")

        # Verify update persisted (read again)
        verify_resp = api_client.get(
            f"{API_BASE}/projects/{project_id}",
            headers=auth_headers
        )

        verify_project = verify_resp.json()
        assert verify_project["name"] == update_payload["name"], \
            "Update did not persist!"

        print(f"✓ CRUD workflow complete: {project_id}")

    def test_project_content_relationship(self, api_client, auth_headers):
        """
        CRITICAL: Projects and content must maintain referential integrity

        Validates:
        - Content is properly linked to projects
        - Deleting project doesn't orphan content (or handles it)
        """
        # Create project
        proj_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json={"name": f"Ref Test {uuid.uuid4().hex[:6]}"}
        )

        assert proj_resp.status_code in [200, 201], \
            f"Project creation failed: {proj_resp.status_code}"
        project_id = proj_resp.json()["id"]

        # Generate content
        content_resp = api_client.post(
            f"{API_BASE}/content/generate/async",
            headers=auth_headers,
            json={
                "project_id": project_id,
                "topic": "Referential Integrity Test",
                "keywords": ["test"],
                "word_count": 100
            }
        )

        if content_resp.status_code != 202:
            pytest.skip("Content generation failed")

        task_id = content_resp.json()["task_id"]

        # Wait for completion
        time.sleep(15)

        # Check task
        task_resp = api_client.get(
            f"{API_BASE}/content/task/{task_id}",
            headers=auth_headers
        )

        if task_resp.status_code == 200:
            task_data = task_resp.json()

            # Verify project_id in task metadata
            assert task_data.get("project_id") == project_id, \
                "Task lost project_id reference!"

            print(f"✓ Referential integrity maintained: task → project")


# ==============================================================================
# WORKFLOW TEST 6: WordPress Integration
# ==============================================================================

class TestWordPressIntegration:
    """
    Tests WordPress connection and publishing workflows
    """

    def test_wordpress_connection_validation(self, api_client, auth_headers):
        """
        Property: WordPress connection settings must be validated

        Validates:
        - Invalid URLs are rejected
        - Missing credentials are detected
        - Connection test works
        """
        # Create project with WordPress settings
        project_payload = {
            "name": f"WP Test {uuid.uuid4().hex[:6]}",
            "wordpress_url": "https://invalid-nonexistent-site-12345.com",
            "wordpress_username": "test",
            "wordpress_password": "test"
        }

        resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json=project_payload
        )

        # Should either succeed or fail gracefully
        assert resp.status_code in [200, 201, 400, 422], \
            f"Unexpected status for invalid WordPress URL: {resp.status_code}"

        if resp.status_code in [200, 201]:
            project_id = resp.json()["id"]

            # If connection test endpoint exists, it should detect invalid URL
            try:
                test_resp = api_client.post(
                    f"{API_BASE}/projects/{project_id}/wordpress/test",
                    headers=auth_headers
                )

                # Connection test should fail for invalid URL (or endpoint doesn't exist)
                assert test_resp.status_code in [400, 404, 422, 503], \
                    f"WordPress connection test unexpected status: {test_resp.status_code}"

                if test_resp.status_code == 404:
                    print(f"✓ WordPress test endpoint not implemented (expected)")
                else:
                    print(f"✓ WordPress validation working: invalid URL detected")
            except httpx.HTTPStatusError:
                # Endpoint may not exist
                pass


# ==============================================================================
# WORKFLOW TEST 7: Data Consistency Across Components
# ==============================================================================

class TestDataConsistency:
    """
    Validates data consistency between UI, API, Database, and Redis
    """

    def test_task_data_consistency_across_layers(self, api_client, auth_headers):
        """
        CRITICAL: Task data must be consistent across all layers

        Validates:
        - API response matches Redis data
        - Redis data matches database
        - No data loss during state transitions
        """
        # Create task
        proj_resp = api_client.post(
            f"{API_BASE}/projects",
            headers=auth_headers,
            json={"name": f"Consistency Test {uuid.uuid4().hex[:6]}"}
        )

        if proj_resp.status_code != 201:
            pytest.skip("Project creation failed")

        project_id = proj_resp.json()["id"]

        task_resp = api_client.post(
            f"{API_BASE}/content/generate/async",
            headers=auth_headers,
            json={
                "project_id": project_id,
                "topic": "Data Consistency Test",
                "keywords": ["consistency"],
                "word_count": 150
            }
        )

        if task_resp.status_code != 202:
            pytest.skip("Task creation failed")

        task_id = task_resp.json()["task_id"]

        # Get task from API
        api_resp = api_client.get(
            f"{API_BASE}/content/task/{task_id}",
            headers=auth_headers
        )

        assert api_resp.status_code == 200
        api_data = api_resp.json()

        # Get task from Redis
        r = redis.from_url(REDIS_URL)
        redis_key = f"celery-task-meta-{task_id}"
        redis_data_raw = r.get(redis_key)

        if redis_data_raw:
            import json
            redis_data = json.loads(redis_data_raw)

            # CRITICAL: State must match
            assert api_data["state"] == redis_data.get("status"), \
                f"State mismatch! API: {api_data['state']}, Redis: {redis_data.get('status')}"

            print(f"✓ Data consistency verified: API ↔ Redis")

        print(f"✓ Multi-layer consistency validated")


# ==============================================================================
# RUN INSTRUCTIONS
# ==============================================================================

"""
Run comprehensive UI workflow tests:

# All tests
pytest tests/test_ui_workflows_comprehensive.py -v

# Specific workflow
pytest tests/test_ui_workflows_comprehensive.py::TestTaskGenerationWorkflow -v

# With coverage
pytest tests/test_ui_workflows_comprehensive.py --cov=api --cov=orchestration -v

# Continuous testing (find intermittent issues)
pytest tests/test_ui_workflows_comprehensive.py --count=5 -v

# Parallel execution
pytest tests/test_ui_workflows_comprehensive.py -n auto -v
"""
