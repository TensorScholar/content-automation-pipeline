"""
End-to-End Integration Tests

Validates complete system integration across all architectural layers:
- Project context resolution (3-layer hierarchy)
- Keyword research with semantic clustering
- Content planning and outline synthesis
- Article generation with quality validation
- Multi-channel distribution
- Observability and metrics collection

Testing Philosophy: Production scenario simulation with chaos engineering
Theoretical Foundation: Contract testing + temporal logic verification
"""

import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List
from unittest.mock import AsyncMock, Mock, patch
from uuid import UUID, uuid4

import pytest
import pytest_asyncio

from core.exceptions import TokenBudgetExceededError, WorkflowError
from execution.content_generator import ContentGenerator
from execution.content_planner import ContentPlanner
from execution.distributer import Distributor
from execution.keyword_researcher import KeywordResearcher
from infrastructure.llm_client import LLMClient
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager
from knowledge.website_analyzer import WebsiteAnalyzer
from optimization.cache_manager import CacheManager
from optimization.model_router import ModelRouter
from optimization.prompt_compressor import PromptCompressor
from optimization.token_budget_manager import TokenBudgetManager
from orchestration.content_agent import ContentAgent, ContentAgentConfig, WorkflowState
from orchestration.task_queue import TaskManager

# ============================================================================
# INTEGRATION TEST FIXTURES
# ============================================================================


@pytest_asyncio.fixture
async def integrated_system(clean_db, redis):
    """
    Fully integrated system with real components (mocked LLM only).

    Constructs complete dependency graph for end-to-end testing.
    Uses actual implementations except for expensive external calls.
    """
    # Infrastructure
    db = clean_db
    metrics = MetricsCollector()

    # Mock LLM client to avoid API costs
    llm = AsyncMock(spec=LLMClient)

    # Configure realistic LLM responses
    def mock_complete(**kwargs):
        prompt = kwargs.get("prompt", "")
        max_tokens = kwargs.get("max_tokens", 1000)

        # Generate realistic content based on prompt
        if "keyword" in prompt.lower():
            content = json.dumps(
                {
                    "keywords": [
                        {"phrase": "test keyword 1", "volume": 5000, "difficulty": 0.6},
                        {"phrase": "test keyword 2", "volume": 3000, "difficulty": 0.4},
                        {"phrase": "test keyword 3", "volume": 1500, "difficulty": 0.3},
                    ]
                }
            )
        elif "outline" in prompt.lower() or "plan" in prompt.lower():
            content = json.dumps(
                {
                    "title": "Advanced Guide to Test-Driven Content Generation",
                    "meta_description": "Comprehensive guide exploring modern approaches to automated content creation with quality assurance.",
                    "sections": [
                        {"heading": "Introduction to Automated Content", "words": 300},
                        {"heading": "Understanding NLP-Driven Generation", "words": 400},
                        {"heading": "Quality Validation Strategies", "words": 350},
                        {"heading": "Economic Optimization Techniques", "words": 300},
                        {"heading": "Conclusion and Future Directions", "words": 250},
                    ],
                }
            )
        else:
            # Article section content
            word_count = min(max_tokens // 2, 400)
            content = " ".join(
                [
                    f"This is generated content for testing purposes. "
                    f"The system demonstrates advanced NLP capabilities and intelligent content synthesis. "
                    f"Integration testing validates end-to-end workflow execution with realistic scenarios."
                ]
                * (word_count // 30)
            )

        return Mock(
            content=content,
            usage=Mock(
                prompt_tokens=len(prompt.split()) * 1.3,
                completion_tokens=len(content.split()) * 1.3,
            ),
            cost=0.015,
        )

    llm.complete = AsyncMock(side_effect=mock_complete)

    # Knowledge layer
    projects = ProjectRepository(db)
    rulebook_mgr = RulebookManager(db)
    website_analyzer = WebsiteAnalyzer()

    # Intelligence layer
    semantic_analyzer = SemanticAnalyzer()
    decision_engine = DecisionEngine(db, semantic_analyzer)
    cache = CacheManager(redis)
    context_synthesizer = ContextSynthesizer(projects, rulebook_mgr, decision_engine, cache)

    # Optimization layer
    model_router = ModelRouter()
    budget_manager = TokenBudgetManager(redis, metrics)
    prompt_compressor = PromptCompressor(semantic_analyzer)

    # Execution layer
    keyword_researcher = KeywordResearcher(llm, semantic_analyzer, cache)
    content_planner = ContentPlanner(llm, decision_engine, context_synthesizer, model_router)
    content_generator = ContentGenerator(
        llm,
        context_synthesizer,
        semantic_analyzer,
        model_router,
        budget_manager,
        prompt_compressor,
        metrics,
    )

    # Mock distributor to avoid external API calls
    distributor = AsyncMock(spec=Distributor)
    distributor.distribute_to_telegram.return_value = {
        "channel": "telegram",
        "message_id": 12345,
        "timestamp": datetime.utcnow(),
    }

    # Orchestration
    agent = ContentAgent(
        project_repository=projects,
        rulebook_manager=rulebook_mgr,
        website_analyzer=website_analyzer,
        decision_engine=decision_engine,
        context_synthesizer=context_synthesizer,
        keyword_researcher=keyword_researcher,
        content_planner=content_planner,
        content_generator=content_generator,
        distributor=distributor,
        budget_manager=budget_manager,
        metrics_collector=metrics,
        config=ContentAgentConfig(enable_auto_distribution=False, require_manual_approval=False),
    )

    return {
        "agent": agent,
        "projects": projects,
        "rulebook_mgr": rulebook_mgr,
        "db": db,
        "redis": redis,
        "metrics": metrics,
        "llm": llm,
        "distributor": distributor,
    }


@pytest_asyncio.fixture
async def sample_project_with_rulebook(integrated_system):
    """Create fully configured project for testing."""
    db = integrated_system["db"]
    projects = integrated_system["projects"]
    rulebook_mgr = integrated_system["rulebook_mgr"]

    # Create project
    project = await projects.create(
        name="Integration Test Project",
        domain="https://integration-test.com",
        telegram_channel="@integration_test",
    )

    # Add rulebook
    rulebook_content = """
    # Content Guidelines
    
    ## Tone
    - Use professional yet accessible language
    - Target audience: technical professionals
    - Maintain authoritative voice without condescension
    
    ## Structure
    - Begin with compelling hook
    - Use clear hierarchical headings
    - Include practical examples
    - Conclude with actionable takeaways
    
    ## Quality Standards
    - Target Flesch-Kincaid grade 10-12
    - Keyword density: 0.5-2%
    - Article length: 1500-2500 words
    - Include internal links where relevant
    """

    await rulebook_mgr.create_rulebook(project_id=project.id, content=rulebook_content)

    return project


# ============================================================================
# HAPPY PATH INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_complete_content_generation_workflow(
    integrated_system, sample_project_with_rulebook
):
    """
    Test complete workflow: topic → published article.

    Validates all stages execute correctly with proper data flow.
    This is the golden path scenario.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    # Execute complete workflow
    article = await agent.create_content(
        project_id=project.id,
        topic="Advanced NLP Techniques for Content Automation",
        priority="high",
    )

    # Validate article creation
    assert article is not None
    assert article.id is not None
    assert article.project_id == project.id
    assert article.title is not None
    assert len(article.title) > 10

    # Validate content generation
    assert article.content is not None
    assert article.word_count > 800
    assert article.word_count < 3500

    # Validate quality metrics
    assert article.readability_score > 0
    assert article.readability_score <= 100
    assert len(article.keyword_density) > 0

    # Validate cost tracking
    assert article.total_cost > 0
    assert article.total_cost < 1.0  # Should be well under $1
    assert article.total_tokens_used > 0

    # Validate timing
    assert article.generation_time > 0
    assert article.generation_time < 300  # Should complete in < 5 minutes

    # Validate workflow events
    workflow_status = await agent.get_workflow_status()
    assert workflow_status["state"] == WorkflowState.COMPLETED
    assert len(workflow_status["events"]) >= 5  # Multiple workflow stages


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_with_new_project_no_context(integrated_system, clean_db):
    """
    Test workflow with brand new project (no rulebook, no patterns).

    Should fall back to best practices and still generate quality content.
    """
    agent = integrated_system["agent"]
    projects = integrated_system["projects"]

    # Create minimal project
    project = await projects.create(name="New Project", domain=None, telegram_channel=None)

    # Generate content with no context
    article = await agent.create_content(
        project_id=project.id, topic="Getting Started with Content Automation", priority="medium"
    )

    # Should still succeed
    assert article is not None
    assert article.word_count > 500
    assert article.readability_score > 0

    # Workflow should use best practices
    workflow_status = await agent.get_workflow_status()
    events = workflow_status["events"]

    # Should have context loading event indicating Layer 3
    context_events = [e for e in events if e["state"] == WorkflowState.CONTEXT_LOADING]
    assert len(context_events) > 0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_with_custom_instructions(integrated_system, sample_project_with_rulebook):
    """
    Test workflow with custom generation instructions.

    Custom instructions should override defaults while respecting rulebook.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    custom_instructions = """
    Focus heavily on practical implementation examples.
    Include code snippets where appropriate.
    Target advanced audience with deep technical knowledge.
    """

    article = await agent.create_content(
        project_id=project.id,
        topic="Building Production-Grade NLP Pipelines",
        priority="high",
        custom_instructions=custom_instructions,
    )

    assert article is not None
    assert article.word_count > 1000

    # Custom instructions should be reflected in metadata
    workflow_status = await agent.get_workflow_status()
    planning_events = [
        e for e in workflow_status["events"] if e["state"] == WorkflowState.CONTENT_PLANNING
    ]
    assert len(planning_events) > 0


# ============================================================================
# DECISION HIERARCHY INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_layer1_explicit_rules_precedence(integrated_system, sample_project_with_rulebook):
    """
    Test that explicit rulebook rules take precedence.

    Validates Layer 1 (explicit) > Layer 2 (inferred) > Layer 3 (best practices).
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    article = await agent.create_content(
        project_id=project.id, topic="Technical Documentation Best Practices", priority="high"
    )

    # Workflow should have used explicit rules
    workflow_status = await agent.get_workflow_status()
    context_event = next(
        e for e in workflow_status["events"] if e["state"] == WorkflowState.CONTEXT_LOADING
    )

    # Should indicate explicit rules strategy
    assert context_event["metadata"]["strategy"] == "explicit_rules"

    # Article should reflect rulebook guidance
    # (Professional tone, grade 10-12 readability)
    assert 50 <= article.readability_score <= 80  # Grade 10-12 range


@pytest.mark.integration
@pytest.mark.asyncio
async def test_layer2_inferred_patterns_fallback(integrated_system, clean_db):
    """
    Test Layer 2 fallback when no explicit rules exist.

    Creates project with inferred patterns but no rulebook.
    """
    agent = integrated_system["agent"]
    projects = integrated_system["projects"]
    db = clean_db

    # Create project
    project = await projects.create(name="Pattern Project", domain="https://pattern.com")

    # Add inferred patterns
    import numpy as np

    pattern_id = uuid4()
    tone_embedding = np.random.rand(384).astype(np.float32)

    await db.execute(
        """
        INSERT INTO inferred_patterns 
        (id, project_id, avg_sentence_length, sentence_length_std,
         lexical_diversity, readability_score, tone_embedding, confidence, sample_size)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        pattern_id,
        project.id,
        18.5,
        2.3,
        0.72,
        65.0,
        tone_embedding.tolist(),
        0.85,
        15,
    )

    # Generate content
    article = await agent.create_content(
        project_id=project.id, topic="Content Strategy Analysis", priority="medium"
    )

    assert article is not None

    # Should have used inferred patterns
    workflow_status = await agent.get_workflow_status()
    context_event = next(
        e for e in workflow_status["events"] if e["state"] == WorkflowState.CONTEXT_LOADING
    )

    assert context_event["metadata"]["strategy"] == "inferred_patterns"


# ============================================================================
# QUALITY VALIDATION INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_quality_validation_rejects_poor_content(
    integrated_system, sample_project_with_rulebook
):
    """
    Test quality validation with intentionally poor content.

    Simulates generation failure and validates retry mechanism.
    """
    agent = integrated_system["agent"]
    llm = integrated_system["llm"]
    project = sample_project_with_rulebook

    # Mock LLM to return poor quality content first time
    call_count = 0

    def poor_then_good_content(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count <= 3:
            # First few calls: very short, low quality
            content = "Short bad content." * 10
        else:
            # Subsequent calls: good quality
            content = (
                "This is high quality generated content with proper length and structure. " * 50
            )

        return Mock(
            content=content,
            usage=Mock(prompt_tokens=100, completion_tokens=len(content.split())),
            cost=0.015,
        )

    llm.complete = AsyncMock(side_effect=poor_then_good_content)

    # Should retry and eventually succeed
    article = await agent.create_content(
        project_id=project.id, topic="Quality Content Generation", priority="high"
    )

    # Should have good quality despite initial failures
    assert article.word_count > 500
    assert call_count > 3  # Proves retry happened


@pytest.mark.integration
@pytest.mark.asyncio
async def test_keyword_density_validation(integrated_system, sample_project_with_rulebook):
    """
    Test keyword density validation.

    Ensures generated content has appropriate keyword integration.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    article = await agent.create_content(
        project_id=project.id, topic="SEO Optimization Strategies", priority="high"
    )

    # Validate keyword density is within acceptable range
    for keyword, density in article.keyword_density.items():
        assert 0.005 <= density <= 0.025, f"Keyword '{keyword}' density {density} out of range"


# ============================================================================
# ECONOMIC OPTIMIZATION INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cost_stays_under_budget(integrated_system, sample_project_with_rulebook):
    """
    Test that generation cost stays within economic targets.

    Target: $0.08-0.30 per article
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    # Generate multiple articles to test consistency
    articles = []
    for i in range(3):
        article = await agent.create_content(
            project_id=project.id, topic=f"Test Topic {i}: Content Automation", priority="high"
        )
        articles.append(article)

    # Validate costs
    for article in articles:
        assert article.total_cost < 0.50, f"Cost ${article.total_cost:.2f} exceeds target"
        assert article.total_cost > 0.01, "Cost suspiciously low"

        # Cost per word should be reasonable
        cost_per_word = article.total_cost / article.word_count
        assert cost_per_word < 0.001, f"Cost per word ${cost_per_word:.4f} too high"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_cache_hit_reduces_cost(integrated_system, sample_project_with_rulebook):
    """
    Test that cache hits reduce generation cost.

    Generate same topic twice; second should be cheaper due to caching.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook
    topic = "Cache Test: Identical Topic Generation"

    # First generation
    article1 = await agent.create_content(project_id=project.id, topic=topic, priority="high")

    # Second generation (should hit cache)
    article2 = await agent.create_content(project_id=project.id, topic=topic, priority="high")

    # Second should be faster and/or cheaper
    # (In practice, full cache hit would skip generation entirely,
    #  but partial hits still provide benefit)
    assert article2.generation_time <= article1.generation_time * 1.2  # Within 20%


@pytest.mark.integration
@pytest.mark.asyncio
async def test_token_budget_enforcement(integrated_system, sample_project_with_rulebook):
    """
    Test token budget hard limits are enforced.

    Attempts generation with very low budget to trigger limit.
    """
    agent = integrated_system["agent"]
    budget_manager = integrated_system["agent"].budget_manager
    project = sample_project_with_rulebook

    # Set very low daily budget
    await budget_manager.set_daily_budget(1000)  # Only 1k tokens

    # Attempt generation (should fail or be heavily constrained)
    with pytest.raises((TokenBudgetExceededError, WorkflowError)):
        await agent.create_content(
            project_id=project.id,
            topic="Very Long Comprehensive Guide Requiring Many Tokens",
            priority="low",  # Low priority gets less budget
        )


# ============================================================================
# DISTRIBUTION INTEGRATION TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_successful_distribution(integrated_system, sample_project_with_rulebook):
    """
    Test article distribution after generation.

    Validates distribution workflow integration.
    """
    agent = integrated_system["agent"]
    distributor = integrated_system["distributor"]
    project = sample_project_with_rulebook

    # Enable auto-distribution
    agent.config.enable_auto_distribution = True

    article = await agent.create_content(
        project_id=project.id, topic="Distribution Test Article", priority="high"
    )

    # Should have distributed
    assert article.distributed_at is not None
    assert len(article.distribution_channels) > 0

    # Distributor should have been called
    distributor.distribute_to_telegram.assert_called_once()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_distribution_failure_doesnt_fail_workflow(
    integrated_system, sample_project_with_rulebook
):
    """
    Test that distribution failures don't cause workflow failure.

    Generation should succeed even if distribution fails.
    """
    agent = integrated_system["agent"]
    distributor = integrated_system["distributor"]
    project = sample_project_with_rulebook

    # Mock distribution failure
    from core.exceptions import DistributionError

    distributor.distribute_to_telegram.side_effect = DistributionError("Network error")

    agent.config.enable_auto_distribution = True

    # Should still succeed
    article = await agent.create_content(
        project_id=project.id, topic="Resilient Distribution Test", priority="high"
    )

    # Article generated despite distribution failure
    assert article is not None
    assert article.distributed_at is None  # Distribution failed


# ============================================================================
# CONCURRENCY & PERFORMANCE TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_article_generation(integrated_system, sample_project_with_rulebook):
    """
    Test concurrent article generation.

    Validates system handles parallel requests without interference.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    topics = [
        "Concurrent Generation Test 1",
        "Concurrent Generation Test 2",
        "Concurrent Generation Test 3",
        "Concurrent Generation Test 4",
        "Concurrent Generation Test 5",
    ]

    # Generate concurrently
    tasks = [
        agent.create_content(project_id=project.id, topic=topic, priority="medium")
        for topic in topics
    ]

    articles = await asyncio.gather(*tasks, return_exceptions=True)

    # All should succeed
    successful = [a for a in articles if not isinstance(a, Exception)]
    assert len(successful) == len(topics)

    # Each should be unique
    titles = [a.title for a in successful]
    assert len(set(titles)) == len(titles)  # All unique


@pytest.mark.integration
@pytest.mark.slow
@pytest.mark.asyncio
async def test_end_to_end_latency_under_3_minutes(
    integrated_system, sample_project_with_rulebook, benchmark_timer
):
    """
    Test end-to-end workflow completes within performance target.

    Target: < 3 minutes for standard article generation.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    with benchmark_timer() as timer:
        article = await agent.create_content(
            project_id=project.id, topic="Performance Test Article", priority="high"
        )

    # Should complete in reasonable time
    assert timer.elapsed < 180  # 3 minutes
    assert article is not None


# ============================================================================
# OBSERVABILITY & MONITORING TESTS
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_metrics_collection_throughout_workflow(
    integrated_system, sample_project_with_rulebook
):
    """
    Test that metrics are collected at all workflow stages.

    Validates observability infrastructure integration.
    """
    agent = integrated_system["agent"]
    metrics = integrated_system["metrics"]
    project = sample_project_with_rulebook

    # Clear any existing metrics
    await metrics.reset()

    article = await agent.create_content(
        project_id=project.id, topic="Metrics Collection Test", priority="high"
    )

    # Validate metrics were collected
    # TODO: Implement actual metrics retrieval and validation
    # For now, just ensure workflow completed
    assert article is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_event_sourcing(integrated_system, sample_project_with_rulebook):
    """
    Test complete workflow event sourcing for audit trail.

    Validates all major workflow stages are recorded.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    article = await agent.create_content(
        project_id=project.id, topic="Event Sourcing Test", priority="high"
    )

    workflow_status = await agent.get_workflow_status()
    events = workflow_status["events"]

    # Validate required workflow stages present
    expected_states = [
        WorkflowState.CONTEXT_LOADING,
        WorkflowState.KEYWORD_RESEARCH,
        WorkflowState.CONTENT_PLANNING,
        WorkflowState.CONTENT_GENERATION,
        WorkflowState.QUALITY_VALIDATION,
        WorkflowState.COMPLETED,
    ]

    event_states = [e["state"] for e in events]

    for expected_state in expected_states:
        assert expected_state in event_states, f"Missing workflow state: {expected_state}"


# ============================================================================
# FAILURE & RECOVERY TESTS (Chaos Engineering)
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_database_connection_failure_recovery(
    integrated_system, sample_project_with_rulebook
):
    """
    Test recovery from transient database failures.

    Simulates database connection loss and validates retry logic.
    """
    agent = integrated_system["agent"]
    db = integrated_system["db"]
    project = sample_project_with_rulebook

    # Simulate intermittent database failure
    original_execute = db.execute
    call_count = 0

    async def failing_execute(*args, **kwargs):
        nonlocal call_count
        call_count += 1

        # Fail first 2 calls, then succeed
        if call_count <= 2:
            raise Exception("Database connection lost")

        return await original_execute(*args, **kwargs)

    db.execute = failing_execute

    # Should retry and succeed
    article = await agent.create_content(
        project_id=project.id, topic="Database Failure Recovery Test", priority="high"
    )

    assert article is not None
    assert call_count > 2  # Proves retry happened


@pytest.mark.integration
@pytest.mark.asyncio
async def test_llm_api_timeout_handling(integrated_system, sample_project_with_rulebook):
    """
    Test handling of LLM API timeouts.

    Validates timeout detection and retry logic.
    """
    agent = integrated_system["agent"]
    llm = integrated_system["llm"]
    project = sample_project_with_rulebook

    # Simulate timeout then success
    call_count = 0

    async def timeout_then_success(**kwargs):
        nonlocal call_count
        call_count += 1

        if call_count == 1:
            await asyncio.sleep(0.1)
            raise asyncio.TimeoutError("API timeout")

        # Success on retry
        return Mock(
            content="Successful content after timeout",
            usage=Mock(prompt_tokens=100, completion_tokens=200),
            cost=0.015,
        )

    llm.complete = AsyncMock(side_effect=timeout_then_success)

    # Should handle timeout and retry
    article = await agent.create_content(
        project_id=project.id, topic="Timeout Handling Test", priority="high"
    )

    assert article is not None
    assert call_count > 1  # Retry occurred


@pytest.mark.integration
@pytest.mark.asyncio
async def test_partial_workflow_failure_cleanup(integrated_system, sample_project_with_rulebook):
    """
    Test cleanup on mid-workflow failures.

    Validates state management and resource cleanup.
    """
    agent = integrated_system["agent"]
    llm = integrated_system["llm"]
    project = sample_project_with_rulebook

    # Cause failure during content generation
    async def always_fail(**kwargs):
        raise Exception("Intentional generation failure")

    llm.complete = AsyncMock(side_effect=always_fail)

    # Should fail gracefully
    with pytest.raises(WorkflowError):
        await agent.create_content(
            project_id=project.id, topic="Failure Cleanup Test", priority="high"
        )

    # Workflow should have failed state
    workflow_status = await agent.get_workflow_status()
    assert workflow_status["state"] == WorkflowState.FAILED


# ============================================================================
# CONTRACT TESTING
# ============================================================================


@pytest.mark.integration
@pytest.mark.asyncio
async def test_article_schema_contract(integrated_system, sample_project_with_rulebook):
    """
    Test generated article adheres to schema contract.

    Validates all required fields present with correct types.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    article = await agent.create_content(
        project_id=project.id, topic="Schema Contract Test", priority="high"
    )

    # Validate schema contract
    assert isinstance(article.id, UUID)
    assert isinstance(article.project_id, UUID)
    assert isinstance(article.title, str) and len(article.title) > 0
    assert isinstance(article.content, str) and len(article.content) > 0
    assert isinstance(article.meta_description, str)
    assert isinstance(article.word_count, int) and article.word_count > 0
    assert isinstance(article.readability_score, (int, float))
    assert isinstance(article.keyword_density, dict)
    assert isinstance(article.total_tokens_used, int) and article.total_tokens_used > 0
    assert isinstance(article.total_cost, (int, float)) and article.total_cost > 0
    assert isinstance(article.generation_time, (int, float)) and article.generation_time > 0
    assert isinstance(article.created_at, datetime)
    # Validate optional fields

    if article.distributed_at is not None:
        assert isinstance(article.distributed_at, datetime)

    if article.distribution_channels is not None:
        assert isinstance(article.distribution_channels, list)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_workflow_state_machine_contract(integrated_system, sample_project_with_rulebook):
    """
    Test workflow state transitions follow valid state machine.
    Validates temporal logic: states must progress in valid order.
    """
    agent = integrated_system["agent"]
    project = sample_project_with_rulebook

    article = await agent.create_content(
        project_id=project.id, topic="State Machine Contract Test", priority="high"
    )

    workflow_status = await agent.get_workflow_status()
    events = workflow_status["events"]

    # Define valid state transitions
    valid_transitions = {
        WorkflowState.INITIALIZED: [WorkflowState.CONTEXT_LOADING],
        WorkflowState.CONTEXT_LOADING: [WorkflowState.KEYWORD_RESEARCH],
        WorkflowState.KEYWORD_RESEARCH: [WorkflowState.CONTENT_PLANNING],
        WorkflowState.CONTENT_PLANNING: [WorkflowState.CONTENT_GENERATION],
        WorkflowState.CONTENT_GENERATION: [WorkflowState.QUALITY_VALIDATION],
        WorkflowState.QUALITY_VALIDATION: [
            WorkflowState.DISTRIBUTION,
            WorkflowState.COMPLETED,
            WorkflowState.CONTENT_GENERATION,  # Retry allowed
        ],
        WorkflowState.DISTRIBUTION: [WorkflowState.COMPLETED],
    }

    # Validate state transitions
    states = [e["state"] for e in events]

    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]

        if current_state in valid_transitions:
            assert (
                next_state in valid_transitions[current_state]
            ), f"Invalid transition: {current_state} → {next_state}"
