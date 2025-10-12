"""
Integration Tests: Prometheus Metrics Collection

Validates comprehensive metrics collection across the content automation pipeline.
"""

import pytest
from unittest.mock import AsyncMock, MagicMock
from infrastructure.monitoring import MetricsCollector
from orchestration.content_agent import ContentAgent
from infrastructure.llm_client import LLMClient
from optimization.cache_manager import CacheManager
from core.models import GeneratedArticle, Project
from datetime import datetime
from uuid import uuid4


@pytest.mark.asyncio
async def test_metrics_collector_records_workflow_metrics():
    """Verify that MetricsCollector records workflow completion metrics."""
    metrics_collector = MetricsCollector()
    
    # Record workflow completion
    project_id = str(uuid4())
    metrics_collector.record_workflow_completion(
        project_id=project_id,
        workflow_type="content_generation",
        duration_seconds=45.2,
        cost=0.125,
        success=True
    )
    
    # Record workflow failure
    metrics_collector.record_workflow_completion(
        project_id=project_id,
        workflow_type="content_generation",
        duration_seconds=12.8,
        cost=0.0,
        success=False,
        error_type="LLMTimeoutError"
    )
    
    # Verify metrics were recorded (checking internal state)
    assert metrics_collector.workflow_success_total._value.get() == 1
    assert metrics_collector.workflow_failure_total._value.get() == 1
    
    print("✓ Workflow metrics recording test passed.")


@pytest.mark.asyncio
async def test_metrics_collector_records_llm_metrics():
    """Verify that MetricsCollector records LLM API call metrics."""
    metrics_collector = MetricsCollector()
    
    # Record successful LLM API call
    metrics_collector.record_llm_api_call(
        model="gpt-4",
        provider="openai",
        status="success",
        tokens_used=150,
        cost=0.045,
        latency_seconds=2.3,
        token_type="total"
    )
    
    # Record failed LLM API call
    metrics_collector.record_llm_api_call(
        model="gpt-4",
        provider="openai",
        status="failure",
        tokens_used=0,
        cost=0.0,
        latency_seconds=0.0,
        token_type="total"
    )
    
    # Verify metrics were recorded
    assert metrics_collector.llm_api_requests_total._value.get() == 2
    assert metrics_collector.llm_api_tokens_total._value.get() == 150
    assert metrics_collector.llm_api_cost_total._value.get() == 0.045
    
    print("✓ LLM API metrics recording test passed.")


@pytest.mark.asyncio
async def test_metrics_collector_records_cache_metrics():
    """Verify that MetricsCollector records cache performance metrics."""
    metrics_collector = MetricsCollector()
    
    # Record cache hits
    metrics_collector.record_cache_hit("memory", "general")
    metrics_collector.record_cache_hit("redis", "general")
    metrics_collector.record_cache_hit("memory", "llm_responses")
    
    # Record cache misses
    metrics_collector.record_cache_miss("memory", "general")
    metrics_collector.record_cache_miss("redis", "general")
    
    # Verify metrics were recorded
    assert metrics_collector.cache_hits_total._value.get() == 3
    assert metrics_collector.cache_misses_total._value.get() == 2
    
    print("✓ Cache metrics recording test passed.")


@pytest.mark.asyncio
async def test_metrics_collector_exports_prometheus_format():
    """Verify that MetricsCollector exports metrics in Prometheus format."""
    metrics_collector = MetricsCollector()
    
    # Record some sample metrics
    metrics_collector.record_workflow_completion(
        project_id="test-project",
        workflow_type="content_generation",
        duration_seconds=30.0,
        cost=0.1,
        success=True
    )
    
    metrics_collector.record_llm_api_call(
        model="gpt-4",
        provider="openai",
        status="success",
        tokens_used=100,
        cost=0.03,
        latency_seconds=1.5,
        token_type="total"
    )
    
    # Export metrics
    metrics_content = metrics_collector.export_metrics()
    content_type = metrics_collector.get_content_type()
    
    # Verify format
    assert content_type == "text/plain; version=0.0.4; charset=utf-8"
    assert isinstance(metrics_content, str)
    assert "workflow_success_total" in metrics_content
    assert "llm_api_requests_total" in metrics_content
    assert "test-project" in metrics_content
    assert "gpt-4" in metrics_content
    
    print("✓ Prometheus format export test passed.")


@pytest.mark.asyncio
async def test_metrics_collector_health_summary():
    """Verify that MetricsCollector provides health summary."""
    metrics_collector = MetricsCollector()
    
    summary = metrics_collector.get_metrics_summary()
    
    # Verify summary structure
    assert isinstance(summary, dict)
    assert "metrics_initialized" in summary
    assert "total_metrics" in summary
    assert "workflow_metrics" in summary
    assert "llm_metrics" in summary
    assert "cache_metrics" in summary
    assert "system_metrics" in summary
    
    assert summary["metrics_initialized"] is True
    assert summary["workflow_metrics"] is True
    assert summary["llm_metrics"] is True
    assert summary["cache_metrics"] is True
    assert summary["system_metrics"] is True
    
    print("✓ Metrics health summary test passed.")


@pytest.mark.asyncio
async def test_content_agent_integrates_with_metrics():
    """Verify that ContentAgent integrates with MetricsCollector."""
    # Mock dependencies
    mock_project_repo = AsyncMock()
    mock_rulebook_manager = AsyncMock()
    mock_website_analyzer = AsyncMock()
    mock_decision_engine = AsyncMock()
    mock_context_synthesizer = AsyncMock()
    mock_keyword_researcher = AsyncMock()
    mock_content_planner = AsyncMock()
    mock_content_generator = AsyncMock()
    mock_distributor = AsyncMock()
    mock_budget_manager = AsyncMock()
    mock_metrics_collector = MetricsCollector()
    
    # Create ContentAgent with metrics collector
    content_agent = ContentAgent(
        project_repository=mock_project_repo,
        rulebook_manager=mock_rulebook_manager,
        website_analyzer=mock_website_analyzer,
        decision_engine=mock_decision_engine,
        context_synthesizer=mock_context_synthesizer,
        keyword_researcher=mock_keyword_researcher,
        content_planner=mock_content_planner,
        content_generator=mock_content_generator,
        distributor=mock_distributor,
        budget_manager=mock_budget_manager,
        metrics_collector=mock_metrics_collector,
    )
    
    # Verify metrics collector is properly integrated
    assert content_agent.metrics_collector is not None
    assert content_agent.metrics_collector == mock_metrics_collector
    
    print("✓ ContentAgent metrics integration test passed.")


@pytest.mark.asyncio
async def test_llm_client_integrates_with_metrics():
    """Verify that LLMClient integrates with MetricsCollector."""
    mock_redis_client = AsyncMock()
    mock_cache_manager = AsyncMock()
    mock_metrics_collector = MetricsCollector()
    
    # Create LLMClient with metrics collector
    llm_client = LLMClient(
        redis_client=mock_redis_client,
        cache_manager=mock_cache_manager,
        metrics_collector=mock_metrics_collector,
    )
    
    # Verify metrics collector is properly integrated
    assert llm_client.metrics_collector is not None
    assert llm_client.metrics_collector == mock_metrics_collector
    
    print("✓ LLMClient metrics integration test passed.")


@pytest.mark.asyncio
async def test_cache_manager_integrates_with_metrics():
    """Verify that CacheManager integrates with MetricsCollector."""
    mock_metrics_collector = MetricsCollector()
    
    # Create CacheManager with metrics collector
    cache_manager = CacheManager(
        max_memory_entries=100,
        metrics_collector=mock_metrics_collector
    )
    
    # Verify metrics collector is properly integrated
    assert cache_manager.metrics_collector is not None
    assert cache_manager.metrics_collector == mock_metrics_collector
    
    print("✓ CacheManager metrics integration test passed.")
