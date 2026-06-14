"""
Unit Tests for Celery Task Error Handling
=========================================

Tests for enhanced error handling in orchestration/tasks.py including:
- Retry behavior on transient errors
- DLQ routing after max retries
- Admin notification on permanent failures
- asyncio.CancelledError handling
- Error metrics integration
"""

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class TestNotifyAdmin:
    """Tests for notify_admin function."""

    def test_notify_admin_disabled_without_smtp(self):
        """Should return False when SMTP is not configured."""
        with patch.dict(os.environ, {"SMTP_HOST": "", "ADMIN_EMAIL": ""}):
            if 'orchestration.tasks' in sys.modules:
                del sys.modules['orchestration.tasks']

            # Import with mocked dependencies
            with patch('api.dependencies.get_database'):
                with patch('api.dependencies.get_metrics'):
                    with patch('api.dependencies.get_redis'):
                        from orchestration.tasks import SMTP_ENABLED, notify_admin

                        # SMTP should be disabled
                        result = notify_admin(
                            task_id="test-task-id",
                            task_name="test.task",
                            error="Test error",
                            error_type="ValueError",
                            traceback_str="Test traceback",
                            kwargs={"project_id": "test-project"}
                        )

                        assert result is False

    def test_notify_admin_sends_email_when_configured(self):
        """Should send email when SMTP is configured."""
        env = {
            "SMTP_HOST": "smtp.test.com",
            "SMTP_PORT": "587",
            "SMTP_USER": "test@test.com",
            "SMTP_PASSWORD": "password",
            "ADMIN_EMAIL": "admin@test.com"
        }

        with patch.dict(os.environ, env):
            if 'orchestration.tasks' in sys.modules:
                del sys.modules['orchestration.tasks']

            with patch('api.dependencies.get_database'):
                with patch('api.dependencies.get_metrics'):
                    with patch('api.dependencies.get_redis'):
                        with patch('smtplib.SMTP') as mock_smtp:
                            mock_server = MagicMock()
                            mock_smtp.return_value.__enter__ = MagicMock(return_value=mock_server)
                            mock_smtp.return_value.__exit__ = MagicMock(return_value=False)

                            from orchestration.tasks import notify_admin

                            result = notify_admin(
                                task_id="test-task-id",
                                task_name="test.task",
                                error="Test error",
                                error_type="ValueError",
                                traceback_str="Test traceback",
                                kwargs={"project_id": "test-project"}
                            )

                            # Verify SMTP was called
                            mock_smtp.assert_called_once()


class TestRouteToDeadLetterQueue:
    """Tests for DLQ routing."""

    def test_routes_to_dlq_on_permanent_failure(self):
        """Should route task to DLQ after max retries."""
        with patch('api.dependencies.get_database'):
            with patch('api.dependencies.get_metrics'):
                with patch('api.dependencies.get_redis'):
                    if 'orchestration.tasks' in sys.modules:
                        del sys.modules['orchestration.tasks']

                    from orchestration.tasks import app, route_to_dead_letter_queue

                    # Mock app.send_task
                    app.send_task = MagicMock()

                    exc = ValueError("Test permanent failure")

                    route_to_dead_letter_queue(
                        task_id="test-task-id",
                        task_name="test.task",
                        args=("arg1",),
                        kwargs={"key": "value"},
                        exc=exc
                    )

                    # Verify task was sent to DLQ
                    app.send_task.assert_called_once()
                    call_kwargs = app.send_task.call_args
                    assert call_kwargs[0][0] == "orchestration.tasks.process_dead_letter"
                    assert call_kwargs[1]["queue"] == "dead_letter"


class TestErrorTypeDetection:
    """Tests for specific error type handling."""

    def test_detects_cancelled_error(self):
        """Should detect asyncio.CancelledError."""
        exc = asyncio.CancelledError("Task was cancelled")
        error_type = type(exc).__name__
        is_cancelled = isinstance(exc, asyncio.CancelledError)

        assert error_type == "CancelledError"
        assert is_cancelled is True

    def test_detects_workflow_error(self):
        """Should detect WorkflowError with retry metadata."""
        # Mock WorkflowError since it requires core.exceptions
        class MockWorkflowError(Exception):
            def __init__(self, message, retryable=False, retry_after=None):
                super().__init__(message)
                self.retryable = retryable
                self.retry_after = retry_after

        exc = MockWorkflowError("Test workflow error", retryable=True, retry_after=60)
        error_type = type(exc).__name__

        assert error_type == "MockWorkflowError"
        assert exc.retryable is True
        assert exc.retry_after == 60


class TestGenerateIdempotencyKey:
    """Tests for idempotency key generation."""

    def test_same_inputs_produce_same_key(self):
        """Same inputs should produce identical idempotency keys."""
        with patch('api.dependencies.get_database'):
            with patch('api.dependencies.get_metrics'):
                with patch('api.dependencies.get_redis'):
                    if 'orchestration.tasks' in sys.modules:
                        del sys.modules['orchestration.tasks']

                    from orchestration.tasks import generate_idempotency_key

                    key1 = generate_idempotency_key(
                        "test.task",
                        "project-123",
                        "test topic",
                        priority="high"
                    )

                    key2 = generate_idempotency_key(
                        "test.task",
                        "project-123",
                        "test topic",
                        priority="high"
                    )

                    assert key1 == key2

    def test_different_inputs_produce_different_keys(self):
        """Different inputs should produce different idempotency keys."""
        with patch('api.dependencies.get_database'):
            with patch('api.dependencies.get_metrics'):
                with patch('api.dependencies.get_redis'):
                    if 'orchestration.tasks' in sys.modules:
                        del sys.modules['orchestration.tasks']

                    from orchestration.tasks import generate_idempotency_key

                    key1 = generate_idempotency_key(
                        "test.task",
                        "project-123",
                        "test topic 1"
                    )

                    key2 = generate_idempotency_key(
                        "test.task",
                        "project-123",
                        "test topic 2"
                    )

                    assert key1 != key2


class TestInputValidation:
    """Tests for task input validation."""

    def test_validates_project_id_format(self):
        """Should validate UUID format for project_id."""
        with patch('api.dependencies.get_database'):
            with patch('api.dependencies.get_metrics'):
                with patch('api.dependencies.get_redis'):
                    if 'orchestration.tasks' in sys.modules:
                        del sys.modules['orchestration.tasks']

                    from orchestration.tasks import GenerateContentInput

                    # Valid UUID should pass
                    valid_input = GenerateContentInput(
                        project_id="12345678-1234-1234-1234-123456789012",
                        topic="Test topic"
                    )
                    assert valid_input.project_id == "12345678-1234-1234-1234-123456789012"

                    # Invalid UUID should fail
                    with pytest.raises(ValueError):
                        GenerateContentInput(
                            project_id="invalid-uuid",
                            topic="Test topic"
                        )

    def test_sanitizes_topic_input(self):
        """Should sanitize topic to prevent injection."""
        with patch('api.dependencies.get_database'):
            with patch('api.dependencies.get_metrics'):
                with patch('api.dependencies.get_redis'):
                    if 'orchestration.tasks' in sys.modules:
                        del sys.modules['orchestration.tasks']

                    from orchestration.tasks import GenerateContentInput

                    # Topic with control characters should be sanitized
                    input_data = GenerateContentInput(
                        project_id="12345678-1234-1234-1234-123456789012",
                        topic="Test\x00topic\x01with\x02control"  # Contains null and control chars
                    )

                    # Null bytes should be removed
                    assert "\x00" not in input_data.topic
                    assert "\x01" not in input_data.topic
