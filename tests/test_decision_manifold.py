"""Focused regression tests for DecisionManifold analyzer injection.

Uses a deterministic stub analyzer only — no models, network, Redis,
PostgreSQL, or LLM providers.
"""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

# Importing decision_engine pulls settings via transitive dependencies.
# Provide hermetic placeholders so collection does not require a real stack.
os.environ.setdefault(
    "DATABASE_URL", "postgresql+asyncpg://test:test@127.0.0.1:5432/test_decision_manifold"
)
os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/15")
os.environ.setdefault("CELERY_BROKER_URL", "redis://127.0.0.1:6379/14")
os.environ.setdefault("CELERY_RESULT_BACKEND", "redis://127.0.0.1:6379/13")
os.environ.setdefault("SECRET_KEY", "test-secret-key-for-decision-manifold-unit-tests")

from core.exceptions import DecisionError
from intelligence.decision_engine import (
    DecisionEngine,
    DecisionLayer,
    DecisionManifold,
    Evidence,
)
from intelligence.semantic_analyzer import SimilarityMetric


class RecordingAnalyzer:
    """Minimal stand-in for SemanticAnalyzer with call recording."""

    def __init__(self) -> None:
        self.similarity_calls: list[tuple[Any, ...]] = []
        self.normalize_calls: list[np.ndarray] = []

    def compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> float:
        self.similarity_calls.append((vec1, vec2, metric))
        norm1 = float(np.linalg.norm(vec1))
        norm2 = float(np.linalg.norm(vec2))
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        return float(np.dot(vec1, vec2) / (norm1 * norm2))

    def normalize_vector(self, vector: np.ndarray) -> np.ndarray:
        self.normalize_calls.append(vector)
        norm = float(np.linalg.norm(vector))
        if norm == 0.0:
            return vector
        return vector / norm


def _unit(vec: list[float]) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(arr))
    return arr if norm == 0.0 else arr / norm


def _evidence(embedding: np.ndarray, *, confidence: float = 1.0, authority: float = 1.0) -> Evidence:
    return Evidence(
        source_layer=DecisionLayer.BEST_PRACTICE,
        content="fixture",
        embedding=embedding,
        confidence=confidence,
        authority=authority,
    )


def test_resolve_uses_injected_analyzer_not_bare_name():
    analyzer = RecordingAnalyzer()
    manifold = DecisionManifold(analyzer)
    query = _unit([1.0, 0.0, 0.0])
    evidence = [_evidence(_unit([1.0, 0.0, 0.0])), _evidence(_unit([0.0, 1.0, 0.0]))]

    decision_vector, confidence = manifold.resolve(query, evidence)

    assert analyzer.similarity_calls, "resolve must call injected analyzer.compute_similarity"
    assert analyzer.normalize_calls, "resolve must call injected analyzer.normalize_vector"
    assert all(call[2] == SimilarityMetric.COSINE for call in analyzer.similarity_calls)
    assert decision_vector.shape == query.shape
    assert 0.0 <= confidence <= 1.0
    # Former bare-name path would raise NameError before any result.
    assert not isinstance(decision_vector, BaseException)


def test_find_best_match_uses_injected_analyzer():
    analyzer = RecordingAnalyzer()
    manifold = DecisionManifold(analyzer)
    decision_vector = _unit([1.0, 0.0, 0.0])
    candidates = {
        "near": _unit([0.9, 0.1, 0.0]),
        "far": _unit([0.0, 1.0, 0.0]),
    }

    choice, score = manifold.find_best_match(decision_vector, candidates)

    assert analyzer.similarity_calls, "find_best_match must call injected analyzer"
    assert len(analyzer.similarity_calls) == 2
    assert choice == "near"
    assert score > 0.0


def test_resolve_empty_evidence_raises_decision_error():
    manifold = DecisionManifold(RecordingAnalyzer())
    with pytest.raises(DecisionError, match="Cannot resolve decision without evidence"):
        manifold.resolve(_unit([1.0, 0.0]), [])


def test_decision_engine_passes_same_analyzer_instance_into_manifold():
    analyzer = RecordingAnalyzer()
    engine = DecisionEngine(
        database_manager=MagicMock(),
        best_practices=MagicMock(),
        semantic_analyzer=analyzer,  # type: ignore[arg-type]
    )

    assert engine.manifold.semantic_analyzer is analyzer
    assert engine.semantic_analyzer is analyzer
    # Smoke the wired manifold so the old NameError path cannot regress.
    vector, _ = engine.manifold.resolve(
        _unit([1.0, 0.0]),
        [_evidence(_unit([1.0, 0.0]))],
    )
    assert vector.shape == (2,)
    assert analyzer.similarity_calls
    assert analyzer.normalize_calls
