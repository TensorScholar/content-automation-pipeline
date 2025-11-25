"""
Adaptive model selection router. Uses strategies (e.g., Thompson Sampling) to choose the optimal LLM based on cost, latency, and capability constraints.
"""

from __future__ import annotations

import asyncio
import hashlib
from abc import ABC, abstractmethod
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, Generic, List, Optional, Protocol, TypeVar

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field

from core.exceptions import ModelRoutingError, ValidationError
from optimization.token_budget_manager import TokenBudgetManager

T = TypeVar("T")


# =========================================================================
# TYPE SYSTEM & PROTOCOLS
# =========================================================================


class ModelCapability(str, Enum):
    """Semantic task capability taxonomy."""

    CREATIVE_GENERATION = "creative_generation"  # Long-form content
    STRUCTURED_EXTRACTION = "structured_extraction"  # JSON, data extraction
    CLASSIFICATION = "classification"  # Categorization, labeling
    SUMMARIZATION = "summarization"  # Condensing information
    REASONING = "reasoning"  # Complex logical inference
    CODE_GENERATION = "code_generation"  # Programming tasks
    TRANSLATION = "translation"  # Language translation
    GENERAL = "general"  # Catch-all capability


class TaskComplexity(str, Enum):
    """Computational complexity taxonomy."""

    TRIVIAL = "trivial"  # <100 tokens, simple pattern
    SIMPLE = "simple"  # 100-500 tokens, straightforward
    MODERATE = "moderate"  # 500-1500 tokens, some reasoning
    COMPLEX = "complex"  # 1500-4000 tokens, deep reasoning
    EXTREME = "extreme"  # >4000 tokens, multi-step reasoning


class ModelClass(str, Enum):
    """Model tier classification."""

    FLAGSHIP = "flagship"  # GPT-4, Claude Opus
    STANDARD = "standard"  # GPT-3.5-turbo, Claude Sonnet
    EFFICIENT = "efficient"  # Fine-tuned, specialized models
    LOCAL = "local"  # On-device, zero-cost models


@dataclass
class ModelSpec:
    """
    Complete model specification with capabilities and cost structure.

    Represents the interface contract for any LLM in the routing system.
    """

    identifier: str
    display_name: str
    model_class: ModelClass

    # Cost structure (USD per 1K tokens)
    cost_per_1k_input: float
    cost_per_1k_output: float

    # Capabilities (weighted 0-1)
    capabilities: Dict[ModelCapability, float]

    # Performance characteristics
    max_tokens: int
    typical_latency_ms: int
    context_window: int

    # Availability
    enabled: bool = True
    rate_limit_rpm: Optional[int] = None

    def __post_init__(self):
        """Validate model specification."""
        if not 0 <= min(self.capabilities.values()) <= max(self.capabilities.values()) <= 1:
            raise ValidationError(f"Capability scores must be in [0,1] for {self.identifier}")

        if self.cost_per_1k_input < 0 or self.cost_per_1k_output < 0:
            raise ValidationError(f"Costs must be non-negative for {self.identifier}")


@dataclass
class RoutingTask:
    """
    Task specification for model routing decisions.

    Encapsulates all information needed for optimal model selection.
    """

    task_id: str
    capability_required: ModelCapability
    complexity: TaskComplexity

    estimated_input_tokens: int
    estimated_output_tokens: int

    # Constraints
    max_cost_usd: Optional[float] = None
    max_latency_ms: Optional[int] = None

    # Context
    priority: int = 5  # 1-10
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RoutingDecision:
    """
    Model routing decision with full provenance.

    Represents the outcome of the routing algorithm.
    """

    task_id: str
    selected_model: str
    rationale: str

    expected_cost: float
    expected_latency: int
    confidence: float  # 0-1

    alternatives_considered: List[Tuple[str, float]]  # (model_id, score)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for logging."""
        return {
            "task_id": self.task_id,
            "model": self.selected_model,
            "cost": self.expected_cost,
            "latency": self.expected_latency,
            "confidence": self.confidence,
            "alternatives": len(self.alternatives_considered),
        }


@dataclass
class PerformanceMetrics:
    """
    Real-time performance tracking for online learning.

    Maintains sufficient statistics for Bayesian updates.
    """

    model_id: str
    capability: ModelCapability

    # Success tracking
    attempts: int = 0
    successes: int = 0
    failures: int = 0

    # Cost tracking
    total_cost: float = 0.0
    total_tokens_input: int = 0
    total_tokens_output: int = 0

    # Latency tracking (exponential moving average)
    avg_latency_ms: float = 0.0

    # Thompson Sampling parameters (Beta distribution)
    alpha: float = 1.0  # Prior successes
    beta: float = 1.0  # Prior failures

    # Quality tracking (user feedback)
    quality_scores: List[float] = field(default_factory=list)

    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def success_rate(self) -> float:
        """Empirical success rate."""
        return self.successes / self.attempts if self.attempts > 0 else 0.5

    @property
    def avg_cost_per_call(self) -> float:
        """Average cost per invocation."""
        return self.total_cost / self.attempts if self.attempts > 0 else 0.0

    def sample_success_probability(self) -> float:
        """
        Sample from posterior success probability (Thompson Sampling).

        Returns:
            Sample from Beta(alpha, beta) distribution
        """
        return np.random.beta(self.alpha, self.beta)

    def update(
        self,
        success: bool,
        cost: float,
        tokens_input: int,
        tokens_output: int,
        latency_ms: int,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Bayesian update with new observation.

        Args:
            success: Whether task succeeded
            cost: Actual cost incurred
            tokens_input: Input tokens used
            tokens_output: Output tokens generated
            latency_ms: Actual latency
            quality_score: Optional quality rating (0-1)
        """
        self.attempts += 1

        if success:
            self.successes += 1
            self.alpha += 1
        else:
            self.failures += 1
            self.beta += 1

        # Update cost tracking
        self.total_cost += cost
        self.total_tokens_input += tokens_input
        self.total_tokens_output += tokens_output

        # Update latency (EMA with decay 0.9)
        if self.avg_latency_ms == 0:
            self.avg_latency_ms = latency_ms
        else:
            self.avg_latency_ms = 0.9 * self.avg_latency_ms + 0.1 * latency_ms

        # Track quality
        if quality_score is not None:
            self.quality_scores.append(quality_score)
            if len(self.quality_scores) > 100:
                self.quality_scores.pop(0)

        self.last_updated = datetime.now(timezone.utc)


# =========================================================================
# ROUTING STRATEGIES (STRATEGY PATTERN)
# =========================================================================


class RoutingStrategy(ABC):
    """Abstract base for routing strategies."""

    @abstractmethod
    def select_model(
        self,
        task: RoutingTask,
        models: List[ModelSpec],
        metrics: Dict[str, Dict[ModelCapability, PerformanceMetrics]],
    ) -> RoutingDecision:
        """
        Select optimal model for task.

        Args:
            task: Task specification
            models: Available models
            metrics: Performance history

        Returns:
            Routing decision
        """
        pass


class GreedyStrategy(RoutingStrategy):
    """
    Greedy strategy: Select cheapest model meeting constraints.

    Time Complexity: O(n) where n = number of models
    """

    def select_model(
        self,
        task: RoutingTask,
        models: List[ModelSpec],
        metrics: Dict[str, Dict[ModelCapability, PerformanceMetrics]],
    ) -> RoutingDecision:
        """Greedy selection by cost."""

        # Filter by capability
        candidates = [
            m
            for m in models
            if m.enabled and m.capabilities.get(task.capability_required, 0) >= 0.6
        ]

        if not candidates:
            raise ModelRoutingError(f"No models available for {task.capability_required}")

        # Calculate expected costs
        scored = []
        for model in candidates:
            cost = self._estimate_cost(model, task)
            latency = model.typical_latency_ms

            # Check constraints
            if task.max_cost_usd and cost > task.max_cost_usd:
                continue
            if task.max_latency_ms and latency > task.max_latency_ms:
                continue

            scored.append((model, cost))

        if not scored:
            raise ModelRoutingError("No models satisfy constraints")

        # Select cheapest
        scored.sort(key=lambda x: x[1])
        selected, cost = scored[0]

        return RoutingDecision(
            task_id=task.task_id,
            selected_model=selected.identifier,
            rationale=f"Greedy: Cheapest model meeting constraints",
            expected_cost=cost,
            expected_latency=selected.typical_latency_ms,
            confidence=0.8,
            alternatives_considered=[(m.identifier, c) for m, c in scored[:3]],
        )

    @staticmethod
    def _estimate_cost(model: ModelSpec, task: RoutingTask) -> float:
        """Estimate task cost for model."""
        input_cost = (task.estimated_input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (task.estimated_output_tokens / 1000) * model.cost_per_1k_output
        return input_cost + output_cost


class ThompsonSamplingStrategy(RoutingStrategy):
    """
    Thompson Sampling: Bayesian bandit algorithm with exploration.

    Theoretical Property: O(√(KT log T)) expected regret
    Exploration Rate: Automatically calibrated via posterior sampling
    """

    def __init__(self, exploration_bonus: float = 0.1):
        """
        Initialize Thompson Sampling strategy.

        Args:
            exploration_bonus: Bonus for exploration (higher = more exploration)
        """
        self.exploration_bonus = exploration_bonus

    def select_model(
        self,
        task: RoutingTask,
        models: List[ModelSpec],
        metrics: Dict[str, Dict[ModelCapability, PerformanceMetrics]],
    ) -> RoutingDecision:
        """Thompson Sampling with cost-awareness."""

        # Filter candidates
        candidates = [
            m
            for m in models
            if m.enabled and m.capabilities.get(task.capability_required, 0) >= 0.5
        ]

        if not candidates:
            raise ModelRoutingError(f"No models for {task.capability_required}")

        # Score each candidate
        scored = []
        for model in candidates:
            # Get performance metrics
            perf = metrics.get(model.identifier, {}).get(task.capability_required)

            if perf is None:
                # Uninitiated model: Use optimistic prior
                success_prob = 0.9
                cost = self._estimate_cost(model, task)
            else:
                # Sample from posterior
                success_prob = perf.sample_success_probability()

                # Use empirical cost
                if perf.attempts > 0:
                    cost = perf.avg_cost_per_call
                else:
                    cost = self._estimate_cost(model, task)

            # Expected utility: success_prob / cost (higher is better)
            # Add exploration bonus for underexplored models
            attempts = perf.attempts if perf else 0
            exploration = self.exploration_bonus / (1 + attempts)

            utility = (success_prob + exploration) / (cost + 1e-9)

            # Check constraints
            if task.max_cost_usd and cost > task.max_cost_usd:
                utility *= 0.1  # Penalize but don't eliminate

            scored.append((model, utility, success_prob, cost))

        # Select highest utility
        scored.sort(key=lambda x: x[1], reverse=True)
        selected, utility, success_prob, cost = scored[0]

        return RoutingDecision(
            task_id=task.task_id,
            selected_model=selected.identifier,
            rationale=f"Thompson Sampling: success_prob={success_prob:.2f}, cost=${cost:.4f}",
            expected_cost=cost,
            expected_latency=selected.typical_latency_ms,
            confidence=success_prob,
            alternatives_considered=[(m.identifier, u) for m, u, _, _ in scored[:3]],
        )

    @staticmethod
    def _estimate_cost(model: ModelSpec, task: RoutingTask) -> float:
        """Estimate task cost."""
        input_cost = (task.estimated_input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (task.estimated_output_tokens / 1000) * model.cost_per_1k_output
        return input_cost + output_cost


class CapabilityMatchingStrategy(RoutingStrategy):
    """
    Capability-first matching: Select best capability match within budget.

    Prioritizes quality over cost, suitable for critical tasks.
    """

    def select_model(
        self,
        task: RoutingTask,
        models: List[ModelSpec],
        metrics: Dict[str, Dict[ModelCapability, PerformanceMetrics]],
    ) -> RoutingDecision:
        """Select by capability strength."""

        candidates = [m for m in models if m.enabled]

        if not candidates:
            raise ModelRoutingError("No enabled models")

        # Score by capability
        scored = []
        for model in candidates:
            capability_score = model.capabilities.get(task.capability_required, 0)

            # Adjust by empirical performance
            perf = metrics.get(model.identifier, {}).get(task.capability_required)
            if perf and perf.attempts > 10:
                # Blend with empirical success rate
                empirical_success = perf.success_rate
                capability_score = 0.6 * capability_score + 0.4 * empirical_success

            cost = self._estimate_cost(model, task)

            # Check constraints
            if task.max_cost_usd and cost > task.max_cost_usd:
                continue

            scored.append((model, capability_score, cost))

        if not scored:
            raise ModelRoutingError("No models within budget")

        # Select best capability
        scored.sort(key=lambda x: x[1], reverse=True)
        selected, cap_score, cost = scored[0]

        return RoutingDecision(
            task_id=task.task_id,
            selected_model=selected.identifier,
            rationale=f"Capability-first: score={cap_score:.2f}",
            expected_cost=cost,
            expected_latency=selected.typical_latency_ms,
            confidence=cap_score,
            alternatives_considered=[(m.identifier, s) for m, s, _ in scored[:3]],
        )

    @staticmethod
    def _estimate_cost(model: ModelSpec, task: RoutingTask) -> float:
        """Estimate cost."""
        input_cost = (task.estimated_input_tokens / 1000) * model.cost_per_1k_input
        output_cost = (task.estimated_output_tokens / 1000) * model.cost_per_1k_output
        return input_cost + output_cost


# =========================================================================
# MODEL ROUTER (FACADE PATTERN)
# =========================================================================


class ModelRouter:
    """
    Adaptive model routing orchestrator with online learning.

    Architecture:
    - Strategy Pattern: Pluggable routing algorithms
    - Observer Pattern: Performance tracking and updates
    - Facade Pattern: Unified interface for model selection

    Learning Mechanism:
    - Thompson Sampling for exploration-exploitation
    - Bayesian inference for capability estimation
    - Exponential moving average for latency tracking
    """

    def __init__(
        self,
        budget_manager: TokenBudgetManager,
        default_strategy: str = "thompson_sampling",
    ):
        """
        Initialize model router.

        Args:
            budget_manager: Token budget manager
            default_strategy: Default routing strategy
        """
        self.budget_manager = budget_manager

        # Model registry
        self.models: Dict[str, ModelSpec] = {}

        # Performance tracking (model_id -> capability -> metrics)
        self.metrics: Dict[str, Dict[ModelCapability, PerformanceMetrics]] = defaultdict(dict)

        # Routing strategies
        self.strategies: Dict[str, RoutingStrategy] = {
            "greedy": GreedyStrategy(),
            "thompson_sampling": ThompsonSamplingStrategy(exploration_bonus=0.1),
            "capability_matching": CapabilityMatchingStrategy(),
        }

        self.default_strategy = default_strategy

        # Decision history
        self.decision_history: deque = deque(maxlen=1000)

        # Initialize with standard models
        self._register_standard_models()

        logger.info(f"Model router initialized with strategy: {default_strategy}")

    # =========================================================================
    # MODEL REGISTRATION
    # =========================================================================

    def _register_standard_models(self) -> None:
        """Register standard LLM models."""

        # GPT-4 (Flagship)
        self.register_model(
            ModelSpec(
                identifier="gpt-4",
                display_name="GPT-4",
                model_class=ModelClass.FLAGSHIP,
                cost_per_1k_input=0.03,
                cost_per_1k_output=0.06,
                capabilities={
                    ModelCapability.CREATIVE_GENERATION: 0.95,
                    ModelCapability.REASONING: 0.95,
                    ModelCapability.STRUCTURED_EXTRACTION: 0.90,
                    ModelCapability.SUMMARIZATION: 0.90,
                    ModelCapability.CODE_GENERATION: 0.95,
                    ModelCapability.CLASSIFICATION: 0.92,
                    ModelCapability.TRANSLATION: 0.90,
                    ModelCapability.GENERAL: 0.95,
                },
                max_tokens=8192,
                typical_latency_ms=3000,
                context_window=8192,
                rate_limit_rpm=3000,
            )
        )

        # GPT-3.5-turbo (Standard)
        self.register_model(
            ModelSpec(
                identifier="gpt-3.5-turbo",
                display_name="GPT-3.5 Turbo",
                model_class=ModelClass.STANDARD,
                cost_per_1k_input=0.0005,
                cost_per_1k_output=0.0015,
                capabilities={
                    ModelCapability.CREATIVE_GENERATION: 0.80,
                    ModelCapability.REASONING: 0.75,
                    ModelCapability.STRUCTURED_EXTRACTION: 0.85,
                    ModelCapability.SUMMARIZATION: 0.85,
                    ModelCapability.CODE_GENERATION: 0.80,
                    ModelCapability.CLASSIFICATION: 0.88,
                    ModelCapability.TRANSLATION: 0.82,
                    ModelCapability.GENERAL: 0.82,
                },
                max_tokens=4096,
                typical_latency_ms=1500,
                context_window=16385,
                rate_limit_rpm=10000,
            )
        )

        # Claude Sonnet (Standard)
        self.register_model(
            ModelSpec(
                identifier="claude-sonnet-4",
                display_name="Claude Sonnet 4",
                model_class=ModelClass.STANDARD,
                cost_per_1k_input=0.003,
                cost_per_1k_output=0.015,
                capabilities={
                    ModelCapability.CREATIVE_GENERATION: 0.92,
                    ModelCapability.REASONING: 0.90,
                    ModelCapability.STRUCTURED_EXTRACTION: 0.88,
                    ModelCapability.SUMMARIZATION: 0.90,
                    ModelCapability.CODE_GENERATION: 0.88,
                    ModelCapability.CLASSIFICATION: 0.85,
                    ModelCapability.TRANSLATION: 0.85,
                    ModelCapability.GENERAL: 0.90,
                },
                max_tokens=4096,
                typical_latency_ms=2000,
                context_window=200000,
                rate_limit_rpm=4000,
            )
        )

    def register_model(self, model: ModelSpec) -> None:
        """
        Register model in routing system.

        Args:
            model: Model specification
        """
        self.models[model.identifier] = model

        # Initialize metrics for all capabilities
        if model.identifier not in self.metrics:
            self.metrics[model.identifier] = {}

            for capability in ModelCapability:
                self.metrics[model.identifier][capability] = PerformanceMetrics(
                    model_id=model.identifier,
                    capability=capability,
                )

        logger.info(f"Registered model: {model.display_name} ({model.identifier})")

    # =========================================================================
    # ROUTING INTERFACE
    # =========================================================================

    async def route(
        self,
        task: RoutingTask,
        strategy: Optional[str] = None,
    ) -> RoutingDecision:
        """
        Route task to optimal model.

        Args:
            task: Task specification
            strategy: Override default strategy

        Returns:
            Routing decision
        """
        strategy_name = strategy or self.default_strategy

        if strategy_name not in self.strategies:
            raise ValidationError(f"Unknown strategy: {strategy_name}")

        routing_strategy = self.strategies[strategy_name]

        # Check budget
        if not await self.budget_manager.can_afford(
            task.estimated_input_tokens + task.estimated_output_tokens
        ):
            raise ModelRoutingError("Insufficient token budget")

        # Execute routing strategy
        decision = routing_strategy.select_model(
            task=task,
            models=list(self.models.values()),
            metrics=self.metrics,
        )

        # Record decision
        self.decision_history.append(decision)

        logger.info(
            f"Routed task {task.task_id} to {decision.selected_model} "
            f"(cost: ${decision.expected_cost:.4f})"
        )

        return decision

    async def report_outcome(
        self,
        task_id: str,
        success: bool,
        actual_cost: float,
        actual_tokens_input: int,
        actual_tokens_output: int,
        actual_latency_ms: int,
        quality_score: Optional[float] = None,
    ) -> None:
        """
        Report task outcome for online learning.

        Updates performance metrics via Bayesian inference.

        Args:
            task_id: Task identifier
            success: Whether task succeeded
            actual_cost: Actual cost incurred
            actual_tokens_input: Actual input tokens
            actual_tokens_output: Actual output tokens
            actual_latency_ms: Actual latency
            quality_score: Optional quality rating
        """
        # Find decision
        decision = None
        for d in reversed(self.decision_history):
            if d.task_id == task_id:
                decision = d
                break

        if not decision:
            logger.warning(f"No decision found for task {task_id}")
            return

        # Find task capability (infer from decision metadata if needed)
        # For now, update all capabilities
        model_id = decision.selected_model

        for capability in ModelCapability:
            if model_id in self.metrics and capability in self.metrics[model_id]:
                self.metrics[model_id][capability].update(
                    success=success,
                    cost=actual_cost,
                    tokens_input=actual_tokens_input,
                    tokens_output=actual_tokens_output,
                    latency_ms=actual_latency_ms,
                    quality_score=quality_score,
                )

        logger.debug(f"Updated metrics for {model_id} (task {task_id}, success={success})")

    # =========================================================================
    # ANALYTICS & MONITORING
    # =========================================================================

    def get_model_statistics(self, model_id: str) -> Dict[str, Any]:
        """Get performance statistics for model."""
        if model_id not in self.models:
            raise ValidationError(f"Unknown model: {model_id}")

        model = self.models[model_id]
        model_metrics = self.metrics.get(model_id, {})

        # Aggregate across capabilities
        total_attempts = sum(m.attempts for m in model_metrics.values())
        total_successes = sum(m.successes for m in model_metrics.values())
        total_cost = sum(m.total_cost for m in model_metrics.values())

        avg_latency = np.mean(
            [m.avg_latency_ms for m in model_metrics.values() if m.avg_latency_ms > 0]
        )

        return {
            "model_id": model_id,
            "display_name": model.display_name,
            "model_class": model.model_class.value,
            "total_attempts": total_attempts,
            "total_successes": total_successes,
            "success_rate": total_successes / total_attempts if total_attempts > 0 else 0,
            "total_cost": total_cost,
            "avg_cost_per_call": total_cost / total_attempts if total_attempts > 0 else 0,
            "avg_latency_ms": avg_latency if not np.isnan(avg_latency) else 0,
            "by_capability": {
                cap.value: {
                    "attempts": metrics.attempts,
                    "success_rate": metrics.success_rate,
                    "avg_cost": metrics.avg_cost_per_call,
                }
                for cap, metrics in model_metrics.items()
                if metrics.attempts > 0
            },
        }

    def get_routing_statistics(self) -> Dict[str, Any]:
        """Get overall routing statistics."""
        total_decisions = len(self.decision_history)

        # Count by model
        by_model = defaultdict(int)
        for decision in self.decision_history:
            by_model[decision.selected_model] += 1

        # Calculate total cost
        total_cost = sum(d.expected_cost for d in self.decision_history)

        return {
            "total_decisions": total_decisions,
            "total_expected_cost": total_cost,
            "decisions_by_model": dict(by_model),
            "registered_models": len(self.models),
            "enabled_models": sum(1 for m in self.models.values() if m.enabled),
        }

    def recommend_strategy(self) -> str:
        """
        Recommend routing strategy based on current state.

        Returns:
            Recommended strategy name
        """
        # Simple heuristic: use Thompson Sampling after warmup
        total_attempts = sum(
            sum(m.attempts for m in model_metrics.values())
            for model_metrics in self.metrics.values()
        )

        if total_attempts < 50:
            return "greedy"  # Warmup phase
        else:
            return "thompson_sampling"  # Exploration phase


# =========================================================================
# GLOBAL INSTANCE
# =========================================================================

# Note: Requires TokenBudgetManager, will be initialized in orchestration layer
# model_router = ModelRouter(budget_manager)
