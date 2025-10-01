"""
Decision Engine Unit Tests

Validates the 3-layer adaptive decision hierarchy:
- Layer 1: Explicit rulebook queries
- Layer 2: Inferred pattern analysis
- Layer 3: Best practices fallback

Tests semantic understanding, confidence weighting, and graceful degradation.

Testing Philosophy: Behavior-driven with property-based validation
"""

from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

import numpy as np
import pytest
import pytest_asyncio

from core.models import InferredPatterns, Rule, Rulebook
from infrastructure.database import DatabaseManager
from intelligence.decision_engine import Decision, DecisionContext, DecisionEngine
from intelligence.semantic_analyzer import SemanticAnalyzer

# ============================================================================
# TEST FIXTURES
# ============================================================================


@pytest.fixture
def mock_semantic_analyzer():
    """Mock semantic analyzer with controllable similarity scores."""
    mock = Mock(spec=SemanticAnalyzer)

    # Default: return high similarity for explicit rules
    def compute_similarity(embedding1, embedding2):
        # Deterministic based on embedding values
        similarity = np.dot(embedding1, embedding2) / (
            np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        )
        return float(similarity)

    mock.compute_similarity = Mock(side_effect=compute_similarity)

    # Mock embedding generation
    def generate_embedding(text: str):
        # Hash-based deterministic embedding
        seed = hash(text) % (2**32)
        np.random.seed(seed)
        return np.random.rand(384).astype(np.float32)

    mock.generate_embedding = Mock(side_effect=generate_embedding)

    return mock


@pytest_asyncio.fixture
async def decision_engine(db, mock_semantic_analyzer):
    """Create decision engine with mocked dependencies."""
    engine = DecisionEngine(db, mock_semantic_analyzer)
    await engine.initialize()
    return engine


@pytest_asyncio.fixture
async def project_with_explicit_rules(db):
    """Create project with explicit rulebook."""
    project_id = uuid4()

    # Create project
    await db.execute(
        "INSERT INTO projects (id, name, domain) VALUES ($1, $2, $3)",
        project_id,
        "Test Project",
        "https://test.com",
    )

    # Create rulebook with explicit rules
    rulebook_id = uuid4()
    await db.execute(
        "INSERT INTO rulebooks (id, project_id, raw_content, version) VALUES ($1, $2, $3, $4)",
        rulebook_id,
        project_id,
        "Test rulebook content",
        1,
    )

    # Add specific tone rule
    rule_id = uuid4()
    tone_embedding = np.random.rand(384).astype(np.float32)

    await db.execute(
        """
        INSERT INTO rules (id, rulebook_id, rule_type, content, embedding, priority, context)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        rule_id,
        rulebook_id,
        "tone",
        "Use conversational, friendly tone for all content",
        tone_embedding.tolist(),
        9,
        "General content tone guidance",
    )

    return project_id


@pytest_asyncio.fixture
async def project_with_inferred_patterns(db):
    """Create project with inferred patterns but no rulebook."""
    project_id = uuid4()

    await db.execute(
        "INSERT INTO projects (id, name, domain) VALUES ($1, $2, $3)",
        project_id,
        "Pattern Project",
        "https://pattern.com",
    )

    # Add inferred patterns
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
        project_id,
        18.5,
        2.3,
        0.72,
        65.0,
        tone_embedding.tolist(),
        0.85,
        15,
    )

    return project_id


@pytest_asyncio.fixture
async def project_with_no_context(db):
    """Create bare project with no rulebook or patterns."""
    project_id = uuid4()

    await db.execute("INSERT INTO projects (id, name) VALUES ($1, $2)", project_id, "Bare Project")

    return project_id


# ============================================================================
# LAYER 1: EXPLICIT RULES TESTS
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer1_explicit_rule_high_similarity(decision_engine, project_with_explicit_rules):
    """
    Test Layer 1: High similarity query matches explicit rule.

    Validates semantic matching with confidence scoring.
    """
    context = DecisionContext(
        project_id=project_with_explicit_rules,
        decision_type="tone",
        query="What tone should I use for this article?",
    )

    decision = await decision_engine.make_decision(context)

    # Should match explicit rule
    assert decision.layer == 1
    assert decision.source == "explicit_rule"
    assert decision.confidence > 0.80
    assert "conversational" in decision.choice.lower() or "friendly" in decision.choice.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer1_no_match_falls_to_layer2(decision_engine, project_with_explicit_rules):
    """
    Test Layer 1 → Layer 2 fallback.

    When explicit rules don't match query, should fall to inferred patterns.
    """
    context = DecisionContext(
        project_id=project_with_explicit_rules,
        decision_type="structure",  # No explicit structure rules
        query="How should I structure the introduction?",
    )

    decision = await decision_engine.make_decision(context)

    # Should skip to Layer 2 or 3 (no Layer 1 match)
    assert decision.layer in [2, 3]
    assert decision.source in ["inferred_pattern", "best_practice"]


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer1_similarity_threshold_enforcement(
    decision_engine, project_with_explicit_rules, mock_semantic_analyzer
):
    """
    Test similarity threshold (0.85) enforcement.

    Queries with similarity < 0.85 should not match Layer 1.
    """
    # Mock low similarity
    mock_semantic_analyzer.compute_similarity = Mock(return_value=0.70)

    context = DecisionContext(
        project_id=project_with_explicit_rules,
        decision_type="tone",
        query="Completely unrelated query about cooking recipes",
    )

    decision = await decision_engine.make_decision(context)

    # Should not use Layer 1 with low similarity
    assert decision.layer != 1
    assert decision.confidence < 0.80


# ============================================================================
# LAYER 2: INFERRED PATTERNS TESTS
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer2_uses_inferred_patterns(decision_engine, project_with_inferred_patterns):
    """
    Test Layer 2: Uses inferred patterns when no explicit rules.

    Validates pattern-based decision making with statistical confidence.
    """
    context = DecisionContext(
        project_id=project_with_inferred_patterns,
        decision_type="tone",
        query="What writing style should I use?",
    )

    decision = await decision_engine.make_decision(context)

    # Should use inferred patterns
    assert decision.layer == 2
    assert decision.source == "inferred_pattern"
    assert decision.confidence >= 0.70
    assert decision.confidence <= decision.metadata.get("pattern_confidence", 1.0)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer2_confidence_propagation(decision_engine, project_with_inferred_patterns):
    """
    Test confidence score propagation from patterns.

    Decision confidence should not exceed pattern confidence.
    """
    context = DecisionContext(
        project_id=project_with_inferred_patterns,
        decision_type="readability",
        query="What readability level should I target?",
    )

    decision = await decision_engine.make_decision(context)

    # Confidence should be bounded by pattern confidence (0.85)
    assert decision.confidence <= 0.85
    assert "pattern_confidence" in decision.metadata
    assert decision.metadata["pattern_confidence"] == 0.85


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer2_requires_minimum_sample_size(decision_engine, db):
    """
    Test minimum sample size validation (5 examples).

    Patterns with insufficient samples should be ignored.
    """
    project_id = uuid4()

    await db.execute(
        "INSERT INTO projects (id, name) VALUES ($1, $2)", project_id, "Insufficient Sample Project"
    )

    # Add pattern with only 3 samples (below threshold)
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
        project_id,
        18.5,
        2.3,
        0.72,
        65.0,
        tone_embedding.tolist(),
        0.75,
        3,  # Only 3 samples
    )

    context = DecisionContext(
        project_id=project_id, decision_type="tone", query="What tone should I use?"
    )

    decision = await decision_engine.make_decision(context)

    # Should skip Layer 2 due to insufficient samples
    assert decision.layer == 3
    assert decision.source == "best_practice"


# ============================================================================
# LAYER 3: BEST PRACTICES TESTS
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer3_fallback_for_new_project(decision_engine, project_with_no_context):
    """
    Test Layer 3: Fallback to best practices for new projects.

    Projects without rulebook or patterns should use best practices.
    """
    context = DecisionContext(
        project_id=project_with_no_context,
        decision_type="tone",
        query="What tone should I use for B2B content?",
    )

    decision = await decision_engine.make_decision(context)

    # Should use best practices
    assert decision.layer == 3
    assert decision.source == "best_practice"
    assert decision.confidence > 0.0
    assert decision.choice is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer3_semantic_best_practice_matching(decision_engine, project_with_no_context):
    """
    Test semantic matching in best practices knowledge base.

    Should find relevant best practices via embedding similarity.
    """
    context = DecisionContext(
        project_id=project_with_no_context,
        decision_type="seo",
        query="How should I optimize keywords?",
    )

    decision = await decision_engine.make_decision(context)

    assert decision.layer == 3
    assert decision.source == "best_practice"
    # Should contain SEO-related guidance
    assert any(
        keyword in decision.choice.lower() for keyword in ["keyword", "seo", "optimize", "density"]
    )


@pytest.mark.unit
@pytest.mark.asyncio
async def test_layer3_priority_weighted_selection(decision_engine, project_with_no_context, db):
    """
    Test priority-weighted best practice selection.

    Higher priority practices should be preferred when multiple match.
    """
    # Seed best practices with different priorities
    practice1_id = uuid4()
    practice2_id = uuid4()

    embedding = np.random.rand(384).astype(np.float32)

    await db.execute(
        """
        INSERT INTO best_practices (id, category, subcategory, guideline, embedding, priority, context)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        practice1_id,
        "tone",
        "general",
        "Low priority guidance",
        embedding.tolist(),
        3,
        "General",
    )

    await db.execute(
        """
        INSERT INTO best_practices (id, category, subcategory, guideline, embedding, priority, context)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        practice2_id,
        "tone",
        "b2b",
        "High priority B2B guidance",
        embedding.tolist(),
        10,
        "B2B content",
    )

    context = DecisionContext(
        project_id=project_with_no_context, decision_type="tone", query="What tone for B2B?"
    )

    decision = await decision_engine.make_decision(context)

    # Should prefer high priority practice
    assert "high priority" in decision.choice.lower() or decision.metadata.get("priority", 0) >= 9


# ============================================================================
# DECISION HIERARCHY INTEGRATION TESTS
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decision_hierarchy_layer_priority(decision_engine, db):
    """
    Test that Layer 1 > Layer 2 > Layer 3 priority is enforced.

    Creates project with all three layers and validates Layer 1 wins.
    """
    project_id = uuid4()

    # Create project
    await db.execute(
        "INSERT INTO projects (id, name, domain) VALUES ($1, $2, $3)",
        project_id,
        "Priority Test",
        "https://priority.com",
    )

    # Add explicit rule (Layer 1)
    rulebook_id = uuid4()
    await db.execute(
        "INSERT INTO rulebooks (id, project_id, raw_content, version) VALUES ($1, $2, $3, $4)",
        rulebook_id,
        project_id,
        "Test content",
        1,
    )

    rule_id = uuid4()
    rule_embedding = np.random.rand(384).astype(np.float32)

    await db.execute(
        """
        INSERT INTO rules (id, rulebook_id, rule_type, content, embedding, priority, context)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        rule_id,
        rulebook_id,
        "tone",
        "EXPLICIT: Use formal, authoritative tone",
        rule_embedding.tolist(),
        10,
        "Explicit rule",
    )

    # Add inferred pattern (Layer 2)
    pattern_id = uuid4()
    pattern_embedding = np.random.rand(384).astype(np.float32)

    await db.execute(
        """
        INSERT INTO inferred_patterns 
        (id, project_id, avg_sentence_length, sentence_length_std,
         lexical_diversity, readability_score, tone_embedding, confidence, sample_size)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        pattern_id,
        project_id,
        18.5,
        2.3,
        0.72,
        65.0,
        pattern_embedding.tolist(),
        0.85,
        15,
    )

    # Query that would match all layers
    context = DecisionContext(
        project_id=project_id,
        decision_type="tone",
        query="What tone should I use for formal business content?",
    )

    decision = await decision_engine.make_decision(context)

    # Layer 1 should win
    assert decision.layer == 1
    assert decision.source == "explicit_rule"
    assert "explicit" in decision.choice.lower() or "formal" in decision.choice.lower()


@pytest.mark.unit
@pytest.mark.asyncio
async def test_confidence_scoring_consistency(decision_engine, project_with_explicit_rules):
    """
    Test confidence scores are consistent and bounded [0, 1].

    Validates confidence calculation across multiple decisions.
    """
    queries = [
        "What tone should I use?",
        "How should I structure content?",
        "What readability level to target?",
        "How to integrate keywords?",
        "What length should articles be?",
    ]

    for query in queries:
        context = DecisionContext(
            project_id=project_with_explicit_rules, decision_type="general", query=query
        )

        decision = await decision_engine.make_decision(context)

        # Confidence should be valid
        assert 0.0 <= decision.confidence <= 1.0

        # Layer and source should be consistent
        if decision.layer == 1:
            assert decision.source == "explicit_rule"
        elif decision.layer == 2:
            assert decision.source == "inferred_pattern"
        elif decision.layer == 3:
            assert decision.source == "best_practice"


# ============================================================================
# EDGE CASES & ERROR HANDLING
# ============================================================================


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handles_empty_query_gracefully(decision_engine, project_with_explicit_rules):
    """Test handling of empty/invalid queries."""
    context = DecisionContext(
        project_id=project_with_explicit_rules, decision_type="tone", query=""
    )

    decision = await decision_engine.make_decision(context)

    # Should still return a decision (fallback to Layer 3)
    assert decision.layer == 3
    assert decision.source == "best_practice"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_handles_nonexistent_project(decision_engine):
    """Test handling of invalid project IDs."""
    context = DecisionContext(
        project_id=uuid4(), decision_type="tone", query="What tone?"  # Non-existent
    )

    decision = await decision_engine.make_decision(context)

    # Should fallback to Layer 3
    assert decision.layer == 3
    assert decision.source == "best_practice"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_decision_provenance_tracking(decision_engine, project_with_explicit_rules):
    """
    Test decision provenance for audit trail.

    Validates metadata includes source information for reproducibility.
    """
    context = DecisionContext(
        project_id=project_with_explicit_rules,
        decision_type="tone",
        query="What tone for technical content?",
    )

    decision = await decision_engine.make_decision(context)

    # Should include provenance metadata
    assert "query" in decision.metadata
    assert decision.metadata["query"] == context.query
    assert "decision_type" in decision.metadata
    assert "timestamp" in decision.metadata


# ============================================================================
# PERFORMANCE TESTS
# ============================================================================


@pytest.mark.slow
@pytest.mark.asyncio
async def test_decision_latency_under_100ms(
    decision_engine, project_with_explicit_rules, benchmark_timer
):
    """
    Test decision latency meets performance requirements.

    Decisions should complete in < 100ms for real-time use.
    """
    context = DecisionContext(
        project_id=project_with_explicit_rules, decision_type="tone", query="What tone?"
    )

    with benchmark_timer() as timer:
        decision = await decision_engine.make_decision(context)

    # Should be fast (< 100ms)
    assert timer.elapsed < 0.100
    assert decision is not None


@pytest.mark.slow
@pytest.mark.asyncio
async def test_concurrent_decision_handling(decision_engine, project_with_explicit_rules):
    """
    Test concurrent decision requests.

    Validates thread-safety and performance under load.
    """
    import asyncio

    contexts = [
        DecisionContext(
            project_id=project_with_explicit_rules, decision_type="tone", query=f"Query {i}"
        )
        for i in range(50)
    ]

    # Execute concurrently
    decisions = await asyncio.gather(*[decision_engine.make_decision(ctx) for ctx in contexts])

    # All should succeed
    assert len(decisions) == 50
    assert all(d.confidence > 0 for d in decisions)
