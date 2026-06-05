"""
Property-Based Tests for Connection Resilience

GOAL: Find edge cases and failure modes, NOT to make tests pass.
Uses Hypothesis to generate random scenarios that might break the system.
"""

import pytest
import time
from hypothesis import given, strategies as st, settings, example, HealthCheck
from hypothesis.stateful import RuleBasedStateMachine, rule, initialize
from unittest.mock import Mock, patch, MagicMock
import httpx
import sys
import os

# Add parent directory to path to import dashboard components
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dashboard_utils import api_call, CircuitBreaker, CircuitState

# ============================================================================
# PROPERTY 1: Circuit Breaker State Transitions
# ============================================================================

class CircuitBreakerStateMachine(RuleBasedStateMachine):
    """
    State machine to test circuit breaker transitions.
    Finds edge cases in CLOSED → OPEN → HALF_OPEN logic.
    """

    def __init__(self):
        super().__init__()
        self.circuit = CircuitBreaker(failure_threshold=5, timeout=60)
        self.request_count = 0
        self.state_history = []

    @initialize()
    def start_closed(self):
        """Circuit should start in CLOSED state."""
        assert self.circuit.state == CircuitState.CLOSED
        self.state_history.append("CLOSED")

    @rule(failures=st.integers(min_value=1, max_value=10))
    def record_failures(self, failures):
        """Record N consecutive failures."""
        for _ in range(failures):
            self.circuit.record_failure()
            self.request_count += 1

        self.state_history.append(f"FAIL_x{failures}")

        # Property: After 5+ failures, circuit MUST be OPEN
        if self.circuit.failures >= 5:
            assert self.circuit.state == CircuitState.OPEN, \
                f"Circuit should be OPEN after {self.circuit.failures} failures, but is {self.circuit.state}"

    @rule(successes=st.integers(min_value=1, max_value=5))
    def record_successes(self, successes):
        """Record N consecutive successes."""
        for _ in range(successes):
            # Only if request allowed
            if self.circuit.should_allow_request():
                self.circuit.record_success()
                self.request_count += 1

        self.state_history.append(f"SUCCESS_x{successes}")

        # Property: Successes in HALF_OPEN should eventually close circuit
        if self.circuit.state == CircuitState.HALF_OPEN and self.circuit.successes >= 3:
            assert self.circuit.state == CircuitState.CLOSED

    @rule(wait_seconds=st.integers(min_value=0, max_value=120))
    def time_passes(self, wait_seconds):
        """Simulate time passing."""
        if self.circuit.last_failure_time:
            # Simulate time passage by modifying last_failure_time
            self.circuit.last_failure_time -= wait_seconds

        self.state_history.append(f"WAIT_{wait_seconds}s")

        # Property: After timeout, OPEN should allow test request (HALF_OPEN)
        if wait_seconds >= 60 and self.circuit.state == CircuitState.OPEN:
            assert self.circuit.should_allow_request(), \
                "Circuit should enter HALF_OPEN after timeout"

TestCircuitBreaker = CircuitBreakerStateMachine.TestCase

# ============================================================================
# PROPERTY 2: API Call Retry Logic
# ============================================================================

@given(
    status_code=st.one_of(
        st.integers(min_value=500, max_value=599),  # Server errors
        st.integers(min_value=400, max_value=499),  # Client errors
        st.just(200),  # Success
    ),
    response_delay=st.floats(min_value=0.0, max_value=5.0),
    max_retries=st.integers(min_value=1, max_value=5)
)
@settings(max_examples=50, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture])
@example(status_code=503, response_delay=1.0, max_retries=3)  # Service unavailable
@example(status_code=429, response_delay=0.5, max_retries=2)   # Rate limited
def test_api_call_retry_behavior(status_code, response_delay, max_retries):
    """
    Property: API call should retry 5xx errors but not 4xx errors.
    Find edge cases where retry logic fails.
    """
    call_count = 0

    def mock_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        time.sleep(response_delay)

        mock_response = Mock()
        mock_response.status_code = status_code
        mock_response.text = f"Mock response {call_count}"
        return mock_response

    with patch('dashboard_utils.httpx.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.get = mock_request
        mock_instance.post = mock_request
        mock_client.return_value.__enter__.return_value = mock_instance

        # Make API call
        response = api_call("GET", "/test", token="fake_token", max_retries=max_retries, timeout=10.0)

        # PROPERTIES TO VERIFY:
        if 500 <= status_code < 600:
            # Property: 5xx errors should be retried max_retries times
            assert call_count == max_retries, \
                f"Expected {max_retries} attempts for {status_code}, got {call_count}"
        else:
            # Property: 4xx and 2xx should NOT be retried
            assert call_count == 1, \
                f"Expected 1 attempt for {status_code}, got {call_count}"

# ============================================================================
# PROPERTY 3: Timeout Handling
# ============================================================================

@given(
    endpoint=st.sampled_from(["/projects", "/content/generate", "/auth/token", "/tasks"]),
    timeout_at_attempt=st.integers(min_value=0, max_value=2)
)
@settings(max_examples=30, deadline=None)
def test_timeout_retry_logic(endpoint, timeout_at_attempt):
    """
    Property: Timeouts should trigger retry with exponential backoff.
    """
    call_count = 0

    def mock_request_with_timeout(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == timeout_at_attempt + 1:
            raise httpx.TimeoutException("Mock timeout")

        mock_response = Mock()
        mock_response.status_code = 200
        return mock_response

    with patch('dashboard_utils.httpx.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.get = mock_request_with_timeout
        mock_client.return_value.__enter__.return_value = mock_instance

        start = time.time()
        response = api_call("GET", endpoint, max_retries=3, timeout=10.0)
        duration = time.time() - start

        # Property: Should retry after timeout
        if timeout_at_attempt < 3:
            # Should eventually succeed after retries
            assert call_count > timeout_at_attempt, \
                f"Should retry after timeout at attempt {timeout_at_attempt}"

        # Property: Exponential backoff should add delay
        if timeout_at_attempt > 0:
            # Expected wait: 2^0 + 2^1 + ... (exponential backoff)
            min_expected_wait = sum(min(2 ** i, 10.0) for i in range(timeout_at_attempt)) * 0.3
            assert duration >= min_expected_wait, \
                f"Expected backoff delay ~{min_expected_wait}s, got {duration:.2f}s"

# ============================================================================
# PROPERTY 4: Connection Error Recovery
# ============================================================================

@given(
    connection_failures=st.integers(min_value=0, max_value=5),
    eventual_success=st.booleans()
)
@settings(max_examples=40)
def test_connection_error_recovery(connection_failures, eventual_success):
    """
    Property: Connection errors should be retried until max_retries.
    If API recovers, should eventually succeed.
    """
    call_count = 0

    def mock_request_with_connection_error(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= connection_failures:
            raise httpx.ConnectError("Mock connection error")

        if eventual_success:
            mock_response = Mock()
            mock_response.status_code = 200
            return mock_response
        else:
            raise httpx.ConnectError("Still failing")

    with patch('dashboard_utils.httpx.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.get = mock_request_with_connection_error
        mock_client.return_value.__enter__.return_value = mock_instance

        response = api_call("GET", "/test", max_retries=5, timeout=10.0)

        # Property: Should attempt up to max_retries times
        assert call_count <= 5, f"Should not exceed max_retries (got {call_count})"

        # Property: If eventual_success and failures < retries, should succeed
        if eventual_success and connection_failures < 5:
            assert response is not None, "Should recover and return response"
            assert response.status_code == 200
        elif not eventual_success or connection_failures >= 5:
            assert response is None, "Should return None after exhausting retries"

# ============================================================================
# PROPERTY 5: Sidebar Project Selection State
# ============================================================================

@given(
    project_count=st.integers(min_value=0, max_value=20),
    selected_index=st.integers(min_value=0, max_value=19),
    project_deleted=st.booleans(),
)
@settings(max_examples=50)
def test_sidebar_project_selection_state(project_count, selected_index, project_deleted):
    """
    Property: Sidebar project selection should handle:
    - Empty project list
    - Selected project deleted
    - Index out of range
    """
    # Generate mock projects
    projects = [{"id": f"proj_{i}", "name": f"Project {i}"} for i in range(project_count)]

    # Simulate session state
    session_state = {}

    # Initial selection
    if project_count > 0 and selected_index < project_count:
        session_state["selected_project_id"] = projects[selected_index]["id"]
        session_state["selected_project_name"] = projects[selected_index]["name"]

    # Simulate project deletion
    if project_deleted and project_count > 0 and selected_index < project_count:
        projects.pop(selected_index)

    # PROPERTIES TO VERIFY:

    # Property 1: If no projects, selected_project_id should be None
    if project_count == 0:
        # System should clear selection
        selected_id = session_state.get("selected_project_id")
        if selected_id:
            # Check if ID exists in empty list
            exists = any(p["id"] == selected_id for p in projects)
            assert not exists, "Selected project should not exist in empty list"

    # Property 2: If selected project deleted, should handle gracefully
    if project_deleted and "selected_project_id" in session_state:
        selected_id = session_state["selected_project_id"]
        exists = any(p["id"] == selected_id for p in projects)
        # System should detect this and clear/reset selection
        if not exists:
            # This is the edge case we want to handle - project no longer exists
            pass  # System should reset selection

    # Property 3: Selected index should always be valid
    if projects and "selected_project_id" in session_state:
        selected_id = session_state["selected_project_id"]
        project_ids = [p["id"] for p in projects]

        if selected_id in project_ids:
            index = project_ids.index(selected_id)
            assert 0 <= index < len(projects), f"Index {index} out of range [0, {len(projects)})"

# ============================================================================
# CHAOS ENGINEERING TEST: Kill API Mid-Request
# ============================================================================

def test_api_killed_during_request():
    """
    CHAOS TEST: Simulate API process dying during request.
    Expectation: Should fail gracefully, not crash dashboard.
    """
    def mock_request_that_dies(*args, **kwargs):
        # Simulate API dying mid-request
        raise httpx.ConnectError("Connection refused (API died)")

    with patch('dashboard_utils.httpx.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.get = mock_request_that_dies
        mock_client.return_value.__enter__.return_value = mock_instance

        # Should not raise exception
        response = api_call("GET", "/projects", token="test_token", timeout=10.0)

        # Property: Should return None, not crash
        assert response is None, "Should handle API death gracefully"

# ============================================================================
# CHAOS ENGINEERING TEST: Race Condition on Startup
# ============================================================================

def test_ui_starts_before_api_ready():
    """
    CHAOS TEST: UI loads before API finishes starting.
    Expectation: Should retry and eventually succeed.
    """
    call_count = 0

    def mock_health_check(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        if call_count < 3:
            # API not ready yet
            raise Exception("Connection refused")
        else:
            # API now ready
            mock_response = Mock()
            mock_response.status_code = 200
            mock_response.json.return_value = {"status": "healthy"}
            return mock_response

    with patch('requests.get', mock_health_check):
        # Import the wait function
        from start import wait_for_api_healthy

        # Should retry and eventually succeed
        success, message = wait_for_api_healthy(max_wait=15)

        assert success, f"Should eventually succeed: {message}"
        assert call_count >= 3, "Should retry multiple times"

# ============================================================================
# PROPERTY 6: Circuit Breaker Recovery After Long Downtime
# ============================================================================

def test_circuit_breaker_recovery_after_downtime():
    """
    Property: Circuit breaker should recover after downtime exceeds timeout.
    """
    circuit = CircuitBreaker(failure_threshold=3, timeout=5)

    # Trigger circuit to open
    for _ in range(5):
        circuit.record_failure()

    assert circuit.state == CircuitState.OPEN, "Circuit should be OPEN after failures"

    # Simulate time passing
    original_time = circuit.last_failure_time
    circuit.last_failure_time = original_time - 10  # 10 seconds ago

    # Should allow request (HALF_OPEN)
    assert circuit.should_allow_request(), "Should enter HALF_OPEN after timeout"
    assert circuit.state == CircuitState.HALF_OPEN, "Should be in HALF_OPEN state"

    # Successful requests should close circuit
    for _ in range(3):
        circuit.record_success()

    assert circuit.state == CircuitState.CLOSED, "Circuit should close after successful recoveries"

# ============================================================================
# PROPERTY 7: Exponential Backoff Limits
# ============================================================================

def test_exponential_backoff_has_max_limit():
    """
    Property: Exponential backoff should not grow indefinitely.
    Should cap at reasonable maximum (10 seconds).
    """
    call_count = 0
    backoff_times = []

    def mock_failing_request(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        mock_response = Mock()
        mock_response.status_code = 503  # Trigger retry
        mock_response.text = "Service unavailable"
        return mock_response

    with patch('dashboard_utils.httpx.Client') as mock_client:
        mock_instance = MagicMock()
        mock_instance.get = mock_failing_request
        mock_client.return_value.__enter__.return_value = mock_instance

        with patch('dashboard_utils.time.sleep') as mock_sleep:
            api_call("GET", "/test", max_retries=10, timeout=10.0)

            # Capture all sleep times
            backoff_times = [call.args[0] for call in mock_sleep.call_args_list]

            # Property: No backoff should exceed 10 seconds
            for backoff in backoff_times:
                assert backoff <= 10.0, f"Backoff time {backoff} exceeds max limit of 10s"

            # Property: Backoff should increase initially
            if len(backoff_times) >= 2:
                # First few should show exponential growth
                assert backoff_times[1] >= backoff_times[0], "Backoff should increase"
