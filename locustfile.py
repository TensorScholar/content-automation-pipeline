"""
Load Testing Configuration with Locust
======================================

Production workload simulation for content automation pipeline.

Simulates realistic user workflows:
1. Authentication (login with JWT)
2. Project creation
3. Asynchronous content generation
4. Status polling until completion

Usage:
    locust -f locustfile.py --host=http://localhost:8000

    or with Poetry:
    poetry run load-test

Web UI:
    Navigate to http://localhost:8089 to configure:
    - Number of users to simulate
    - Spawn rate (users per second)
    - Host URL

Performance Targets:
    - 95th percentile latency < 2000ms for generation requests
    - 99th percentile latency < 5000ms for generation requests
    - 100 concurrent users with <1% failure rate
    - Authentication: <200ms
    - Status polling: <100ms
"""

import random
import time
from typing import Optional

from locust import HttpUser, between, events, task
from loguru import logger

# ============================================================================
# CONFIGURATION
# ============================================================================


class LoadTestConfig:
    """Configuration for load testing parameters."""

    # Test user credentials (ensure this user exists or will be created)
    TEST_USERNAME = "loadtest@example.com"
    TEST_PASSWORD = "LoadTest123!"  # nosec B105 - Test password for load testing
    TEST_FULL_NAME = "Load Test User"

    # Request timeouts (in seconds)
    AUTH_TIMEOUT = 10
    PROJECT_TIMEOUT = 15
    GENERATION_TIMEOUT = 180  # 3 minutes for async generation
    STATUS_TIMEOUT = 5

    # Polling configuration
    MAX_POLL_ATTEMPTS = 60  # Maximum status polls
    POLL_INTERVAL = 2  # Seconds between polls

    # Test data
    PROJECT_NAMES = [
        "Load Test Project Alpha",
        "Load Test Project Beta",
        "Load Test Project Gamma",
        "Load Test Project Delta",
        "Load Test Project Epsilon",
    ]

    TOPICS = [
        "Advanced Machine Learning Techniques",
        "Cloud Infrastructure Best Practices",
        "Modern Web Development Frameworks",
        "Data Science and Analytics Strategies",
        "Cybersecurity in the Digital Age",
        "Artificial Intelligence Applications",
        "DevOps and Continuous Integration",
        "Scalable Microservices Architecture",
        "Blockchain Technology Explained",
        "Internet of Things Innovation",
    ]

    PRIORITIES = ["low", "medium", "high"]


# ============================================================================
# USER BEHAVIOR SIMULATION
# ============================================================================


class ContentAutomationUser(HttpUser):
    """
    Simulates a user of the content automation platform.

    Workflow:
    1. Authenticate on start (get JWT token)
    2. Create projects (20% of requests)
    3. Generate content asynchronously (70% of requests)
    4. Poll for completion status (10% of requests)

    Wait time between tasks: 1-5 seconds (simulates real user behavior)
    """

    # Wait between 1-5 seconds between tasks (realistic user behavior)
    wait_time = between(1, 5)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.access_token: Optional[str] = None
        self.project_ids: list = []
        self.generation_task_ids: list = []

    def on_start(self):
        """
        Called when a simulated user starts.

        Authenticates the user and retrieves JWT token for subsequent requests.
        Creates a test user if it doesn't exist.
        """
        logger.info(f"Starting user session for {LoadTestConfig.TEST_USERNAME}")

        # Try to register (might fail if user already exists, which is fine)
        try:
            self.client.post(
                "/auth/register",
                json={
                    "email": LoadTestConfig.TEST_USERNAME,
                    "password": LoadTestConfig.TEST_PASSWORD,
                    "full_name": LoadTestConfig.TEST_FULL_NAME,
                },
                timeout=LoadTestConfig.AUTH_TIMEOUT,
                name="/auth/register (setup)",
            )
        except Exception:  # nosec B110 - User might already exist, continue to login
            # User might already exist, continue to login
            pass

        # Login to get access token
        self.authenticate()

    def authenticate(self):
        """
        Authenticate user and retrieve JWT access token.

        This is a critical operation - if it fails, the user cannot proceed.
        """
        try:
            response = self.client.post(
                "/auth/token",
                json={
                    "username": LoadTestConfig.TEST_USERNAME,
                    "password": LoadTestConfig.TEST_PASSWORD,
                },
                timeout=LoadTestConfig.AUTH_TIMEOUT,
                name="/auth/token",
            )

            if response.status_code == 200:
                token_data = response.json()
                self.access_token = token_data["access_token"]
                logger.info("Authentication successful")
            else:
                logger.error(f"Authentication failed: {response.status_code} - {response.text}")
                self.access_token = None

        except Exception as e:
            logger.error(f"Authentication error: {e}")
            self.access_token = None

    def get_auth_headers(self) -> dict:
        """Get authorization headers with JWT token."""
        if not self.access_token:
            self.authenticate()

        return {"Authorization": f"Bearer {self.access_token}"}

    @task(2)
    def create_project(self):
        """
        Create a new project (weight: 2).

        Projects are required for content generation.
        Weight of 2 means this task is less frequent than content generation.
        """
        if not self.access_token:
            logger.warning("No access token, skipping project creation")
            return

        project_name = random.choice(
            LoadTestConfig.PROJECT_NAMES
        )  # nosec B311 - Load testing randomization
        project_data = {
            "name": f"{project_name} {random.randint(1000, 9999)}",  # nosec B311 - Load testing randomization
            "domain": f"https://loadtest-{random.randint(1000, 9999)}.com",  # nosec B311 - Load testing randomization
            "telegram_channel": f"@loadtest_{random.randint(1000, 9999)}",  # nosec B311 - Load testing randomization
        }

        try:
            response = self.client.post(
                "/projects/",
                json=project_data,
                headers=self.get_auth_headers(),
                timeout=LoadTestConfig.PROJECT_TIMEOUT,
                name="/projects/ (create)",
            )

            if response.status_code == 201:
                project = response.json()
                project_id = project.get("id")
                if project_id:
                    self.project_ids.append(project_id)
                    logger.info(f"Created project: {project_id}")
            else:
                logger.warning(f"Project creation failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Project creation error: {e}")

    @task(7)
    def generate_content_async(self):
        """
        Generate content asynchronously (weight: 7).

        This is the primary workflow of the system.
        Higher weight means this task runs more frequently.
        """
        if not self.access_token:
            logger.warning("No access token, skipping content generation")
            return

        # Need at least one project
        if not self.project_ids:
            self.create_project()
            if not self.project_ids:
                return

        project_id = random.choice(self.project_ids)  # nosec B311 - Load testing randomization
        topic = random.choice(LoadTestConfig.TOPICS)  # nosec B311 - Load testing randomization
        priority = random.choice(
            LoadTestConfig.PRIORITIES
        )  # nosec B311 - Load testing randomization

        generation_request = {
            "project_id": project_id,
            "topic": topic,
            "priority": priority,
        }

        try:
            response = self.client.post(
                "/content/generate/async",
                json=generation_request,
                headers=self.get_auth_headers(),
                timeout=LoadTestConfig.GENERATION_TIMEOUT,
                name="/content/generate/async",
            )

            if response.status_code == 202:  # Accepted
                result = response.json()
                task_id = result.get("task_id")
                if task_id:
                    self.generation_task_ids.append(task_id)
                    logger.info(f"Started content generation: {task_id}")

                    # Poll for completion (simulates real user checking status)
                    self.poll_generation_status(task_id)
            else:
                logger.warning(f"Content generation failed: {response.status_code}")

        except Exception as e:
            logger.error(f"Content generation error: {e}")

    def poll_generation_status(self, task_id: str):
        """
        Poll for content generation status until completion.

        Simulates a user checking the status of their content generation task.
        Polls at regular intervals until completion or max attempts reached.
        """
        if not self.access_token:
            return

        for attempt in range(LoadTestConfig.MAX_POLL_ATTEMPTS):
            try:
                response = self.client.get(
                    f"/content/status/{task_id}",
                    headers=self.get_auth_headers(),
                    timeout=LoadTestConfig.STATUS_TIMEOUT,
                    name="/content/status/{task_id}",
                )

                if response.status_code == 200:
                    status_data = response.json()
                    state = status_data.get("state")

                    if state in ["completed", "failed", "cancelled"]:
                        logger.info(f"Task {task_id} finished with state: {state}")
                        break

                    # Still processing, wait and poll again
                    time.sleep(LoadTestConfig.POLL_INTERVAL)
                else:
                    logger.warning(f"Status check failed: {response.status_code}")
                    break

            except Exception as e:
                logger.error(f"Status polling error: {e}")
                break
        else:
            logger.warning(
                f"Task {task_id} did not complete within {LoadTestConfig.MAX_POLL_ATTEMPTS} polls"
            )

    @task(1)
    def check_random_task_status(self):
        """
        Check status of a random previously created task (weight: 1).

        Simulates a user checking the status of older tasks.
        """
        if not self.access_token or not self.generation_task_ids:
            return

        task_id = random.choice(self.generation_task_ids)  # nosec B311 - Load testing randomization

        try:
            response = self.client.get(
                f"/content/status/{task_id}",
                headers=self.get_auth_headers(),
                timeout=LoadTestConfig.STATUS_TIMEOUT,
                name="/content/status/{task_id} (random)",
            )

            if response.status_code == 200:
                status_data = response.json()
                logger.debug(f"Task {task_id} status: {status_data.get('state')}")

        except Exception as e:
            logger.error(f"Random status check error: {e}")


# ============================================================================
# EVENT LISTENERS (for custom reporting)
# ============================================================================


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    """Called when load test starts."""
    logger.info("=" * 70)
    logger.info("Load Test Starting")
    logger.info("=" * 70)
    logger.info(f"Target host: {environment.host}")
    logger.info(f"Test user: {LoadTestConfig.TEST_USERNAME}")
    logger.info("=" * 70)


@events.test_stop.add_listener
def on_test_stop(environment, **kwargs):
    """Called when load test stops."""
    logger.info("=" * 70)
    logger.info("Load Test Completed")
    logger.info("=" * 70)

    # Print summary statistics
    stats = environment.stats
    logger.info(f"Total requests: {stats.total.num_requests}")
    logger.info(f"Total failures: {stats.total.num_failures}")
    logger.info(f"Failure rate: {stats.total.fail_ratio * 100:.2f}%")
    logger.info(f"Average response time: {stats.total.avg_response_time:.2f}ms")
    logger.info(f"95th percentile: {stats.total.get_response_time_percentile(0.95):.2f}ms")
    logger.info(f"99th percentile: {stats.total.get_response_time_percentile(0.99):.2f}ms")
    logger.info("=" * 70)


# ============================================================================
# CUSTOM LOAD SHAPES (Optional)
# ============================================================================


from locust import LoadTestShape


class StepLoadShape(LoadTestShape):
    """
    A load test shape that steps up users gradually.

    Stages:
    1. Warm-up: 10 users for 60 seconds
    2. Ramp-up: 50 users for 120 seconds
    3. Peak load: 100 users for 180 seconds
    4. Cool-down: 25 users for 60 seconds

    To use this, run: locust -f locustfile.py --host=http://localhost:8000 --headless
    """

    stages = [
        {"duration": 60, "users": 10, "spawn_rate": 2},
        {"duration": 120, "users": 50, "spawn_rate": 5},
        {"duration": 180, "users": 100, "spawn_rate": 10},
        {"duration": 60, "users": 25, "spawn_rate": 5},
    ]

    def tick(self):
        """
        Returns the number of users and spawn rate for the current time.

        Returns:
            Tuple of (user_count, spawn_rate) or None to stop the test
        """
        run_time = self.get_run_time()

        for stage in self.stages:
            if run_time < stage["duration"]:
                return (stage["users"], stage["spawn_rate"])
            run_time -= stage["duration"]

        return None  # Test complete
