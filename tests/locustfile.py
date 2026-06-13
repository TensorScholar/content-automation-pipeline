"""
Locust Load Testing Configuration for Content Automation API

This file defines load tests for the API endpoints to measure performance
under various load conditions.

Usage:
    # Basic load test (10 users, 2 per second spawn rate)
    locust -f locustfile.py --host=http://localhost:8000

    # Web UI mode
    locust -f locustfile.py --host=http://localhost:8000 --web-port=8089

    # Headless mode (100 users, 10/s spawn, 60s runtime)
    locust -f locustfile.py --host=http://localhost:8000 \
           --users=100 --spawn-rate=10 --run-time=60s --headless

    # Against production
    locust -f locustfile.py --host=https://your-production-url.com
"""

import random

from locust import HttpUser, between, events, task
from locust.contrib.fasthttp import FastHttpUser


class ContentAutomationUser(FastHttpUser):
    """
    Simulates a typical user of the Content Automation API.

    Uses FastHttpUser for better performance (geventhttpclient instead of requests).
    """

    # Wait between 1-3 seconds between tasks
    wait_time = between(1, 3)

    # Test credentials (ensure this user exists in your test database)
    test_email = "test@example.com"
    test_password = "testpassword123"

    def on_start(self):
        """
        Called when a user starts. Performs login to get JWT token.
        """
        # Login to get auth token
        response = self.client.post(
            "/auth/token",
            data={
                "username": self.test_email,
                "password": self.test_password
            },
            catch_response=True
        )

        if response.status_code == 200:
            data = response.json()
            self.token = data.get("access_token")
            response.success()
        else:
            # If login fails, mark as failure and use dummy token
            response.failure(f"Login failed: {response.status_code}")
            self.token = "INVALID_TOKEN"

    def get_headers(self):
        """Get authorization headers for authenticated requests."""
        return {"Authorization": f"Bearer {self.token}"}

    @task(10)  # Weight: 10 (most common operation)
    def health_check(self):
        """Test the health endpoint (no auth required)."""
        with self.client.get(
            "/system/health",
            catch_response=True,
            name="/system/health"
        ) as response:
            if response.status_code == 200:
                response.success()
            else:
                response.failure(f"Health check failed: {response.status_code}")

    @task(8)  # Weight: 8
    def list_projects(self):
        """Test listing projects (requires auth)."""
        with self.client.get(
            "/projects",
            headers=self.get_headers(),
            catch_response=True,
            name="/projects [GET]"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized - token may be invalid")
            else:
                response.failure(f"List projects failed: {response.status_code}")

    @task(5)  # Weight: 5
    def get_current_user(self):
        """Test getting current user info (requires auth)."""
        with self.client.get(
            "/auth/users/me",
            headers=self.get_headers(),
            catch_response=True,
            name="/auth/users/me"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized")
            else:
                response.failure(f"Get user failed: {response.status_code}")

    @task(3)  # Weight: 3
    def get_task_history(self):
        """Test getting task history (requires auth)."""
        with self.client.get(
            "/content/tasks/history?skip=0&limit=10",
            headers=self.get_headers(),
            catch_response=True,
            name="/content/tasks/history"
        ) as response:
            if response.status_code == 200:
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized")
            else:
                response.failure(f"Task history failed: {response.status_code}")

    @task(2)  # Weight: 2 (expensive operation)
    def create_project(self):
        """
        Test creating a project (requires auth).

        Note: This creates actual projects in the database.
        Only use in test environment!
        """
        project_name = f"Load Test Project {random.randint(1000, 9999)}"

        with self.client.post(
            "/projects",
            headers=self.get_headers(),
            json={
                "name": project_name,
                "domain": f"test{random.randint(1000, 9999)}.example.com",
                "vertical": "تکنولوژی و نرم‌افزار",
                "description": "Created by Locust load test"
            },
            catch_response=True,
            name="/projects [POST]"
        ) as response:
            if response.status_code == 201:
                response.success()
            elif response.status_code == 401:
                response.failure("Unauthorized")
            elif response.status_code == 422:
                response.failure("Validation error")
            else:
                response.failure(f"Create project failed: {response.status_code}")

    # Disabled by default - only enable in test environment
    # @task(1)  # Weight: 1 (very expensive operation)
    # def generate_content(self):
    #     """
    #     Test content generation (requires auth).
    #
    #     WARNING: This is expensive and creates real tasks!
    #     Only enable in isolated test environment with mocked LLM.
    #     """
    #     with self.client.post(
    #         "/content/generate/async",
    #         headers=self.get_headers(),
    #         json={
    #             "project_id": "YOUR_TEST_PROJECT_ID",
    #             "topic": "Load Testing Article",
    #             "primary_keyword": "performance",
    #             "priority": "medium",
    #             "language": "en"
    #         },
    #         catch_response=True,
    #         name="/content/generate/async"
    #     ) as response:
    #         if response.status_code == 202:
    #             response.success()
    #         elif response.status_code == 401:
    #             response.failure("Unauthorized")
    #         else:
    #             response.failure(f"Generate content failed: {response.status_code}")


class ReadOnlyUser(FastHttpUser):
    """
    Simulates a read-only user (monitoring, health checks, public endpoints).

    Use this for testing endpoints that don't require authentication.
    """

    wait_time = between(0.5, 2)

    @task(20)
    def health_check(self):
        """Frequent health checks."""
        self.client.get("/system/health", name="/system/health [ReadOnly]")

    @task(1)
    def invalid_endpoint(self):
        """Test 404 handling."""
        with self.client.get(
            "/nonexistent/endpoint",
            catch_response=True,
            name="/invalid [404]"
        ) as response:
            if response.status_code == 404:
                response.success()
            else:
                response.failure(f"Expected 404, got {response.status_code}")


# Event handlers for custom statistics
@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    print("=" * 60)
    print("🚀 Content Automation API Load Test Starting")
    print("=" * 60)
    print(f"Target Host: {environment.host}")
    print(f"Users: {environment.runner.target_user_count if environment.runner else 'N/A'}")
    print("=" * 60)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    print("=" * 60)
    print("🏁 Load Test Completed")
    print("=" * 60)
    if environment.runner and environment.runner.stats:
        stats = environment.runner.stats.total
        print(f"Total Requests: {stats.num_requests}")
        print(f"Failures: {stats.num_failures}")
        print(f"Median Response Time: {stats.median_response_time}ms")
        print(f"95th Percentile: {stats.get_response_time_percentile(0.95)}ms")
        print(f"Requests/sec: {stats.total_rps:.2f}")
    print("=" * 60)


# Quick test scenarios
class QuickTest(FastHttpUser):
    """
    Quick smoke test for basic functionality.

    Run with: locust -f locustfile.py --users=1 --spawn-rate=1 --run-time=10s QuickTest
    """

    wait_time = between(0.5, 1)

    @task
    def smoke_test(self):
        """Basic smoke test of critical endpoints."""
        # Health check
        self.client.get("/system/health")

        # Invalid route (should 404)
        self.client.get("/invalid")
