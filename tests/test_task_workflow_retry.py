from types import SimpleNamespace
from unittest.mock import ANY, Mock

import pytest

from core.exceptions import (
    CacheWriteError,
    DatabaseConnectionError,
    LLMTimeoutError,
    SchemaValidationError,
    WorkflowError,
)
from orchestration.content_agent import _wrap_workflow_error
from orchestration.tasks import (
    CONTENT_GENERATION_MAX_RETRIES,
    ContentGenerationBaseTask,
    _cache_finalized_result,
    _retry_wrapped_workflow_error,
)


class RetrySignal(Exception):
    pass


def _task(retries: int = 0):
    task = SimpleNamespace(request=SimpleNamespace(retries=retries), retry=Mock())
    task.retry.return_value = RetrySignal("celery retry requested")
    return task


@pytest.mark.parametrize(
    "source_error",
    [
        LLMTimeoutError(timeout_seconds=30),
        CacheWriteError(cache_key="task:test"),
        DatabaseConnectionError(database="content"),
    ],
)
def test_wrapped_transient_failures_enter_celery_retry(source_error):
    wrapped = _wrap_workflow_error(source_error, workflow_step="generation")
    task = _task(retries=1)

    with pytest.raises(RetrySignal, match="celery retry requested"):
        _retry_wrapped_workflow_error(task, wrapped)

    task.retry.assert_called_once_with(
        exc=wrapped,
        countdown=ANY,
        max_retries=CONTENT_GENERATION_MAX_RETRIES,
    )
    assert wrapped.retryable is True
    assert wrapped.context["source_error_code"] == source_error.error_code


@pytest.mark.parametrize(
    "source_error",
    [
        SchemaValidationError(validation_errors={"topic": "required"}),
        ValueError("Invalid task parameters"),
    ],
)
def test_wrapped_validation_failure_remains_terminal(source_error):
    wrapped = _wrap_workflow_error(source_error, workflow_step="planning")
    task = _task()

    assert _retry_wrapped_workflow_error(task, wrapped) is False
    task.retry.assert_not_called()
    assert wrapped.retryable is False


def test_release_quality_failure_remains_terminal_with_diagnostics():
    error = WorkflowError("Article failed release gate", workflow_step="quality_validation")
    error.retryable = False
    error.context.update(
        {
            "terminal_reason": "release_quality",
            "quality_diagnostics": {"actual_word_count": 100},
        }
    )
    wrapped = _wrap_workflow_error(error, workflow_step="quality_validation")
    task = _task()

    assert _retry_wrapped_workflow_error(task, wrapped) is False
    task.retry.assert_not_called()
    assert wrapped.context["terminal_reason"] == "release_quality"
    assert wrapped.context["quality_diagnostics"] == {"actual_word_count": 100}


def test_success_hook_does_not_rewrite_atomic_generation_result(monkeypatch):
    task = object.__new__(ContentGenerationBaseTask)
    task.name = "orchestration.tasks.generate_content_task"
    repository_constructed = False

    def unexpected_repository_constructor():
        nonlocal repository_constructed
        repository_constructed = True
        raise AssertionError("on_success must not rewrite a finalized task result")

    class Metrics:
        def record_workflow_completion(self, **kwargs):
            return None

    monkeypatch.setattr("orchestration.tasks.get_metrics", lambda: Metrics())
    monkeypatch.setattr(
        "orchestration.tasks.SyncTaskResultRepository", unexpected_repository_constructor
    )

    task.on_success(
        {"status": "success", "article_id": "article-1"},
        "task-atomic-result",
        (),
        {"project_id": "project-1"},
    )

    assert repository_constructed is False


@pytest.mark.asyncio
async def test_cache_failure_does_not_raise_after_durable_finalization(monkeypatch):
    async def failed_cache_write(*args, **kwargs):
        raise ConnectionError("redis unavailable")

    monkeypatch.setattr("orchestration.tasks.mark_task_complete", failed_cache_write)

    await _cache_finalized_result(
        object(),
        idempotency_key="idempotency-key",
        result={"status": "success", "article_id": "article-1"},
        task_id="task-finalized",
    )
