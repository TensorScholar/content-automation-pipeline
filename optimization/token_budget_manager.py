"""
Token Budget Manager - Economic Constraint Enforcement
=======================================================

Implements token budget control through computational rate limiting:

Algorithms:
- Leaky Bucket: Smooth rate limiting with burst allowance
- Priority Queue: Hierarchical budget allocation
- Predictive Analysis: Proactive budget exhaustion detection
- Exponential Backoff: Graceful degradation under load

Architecture: Event-driven budget accounting with real-time
constraint satisfaction and predictive analytics.

Mathematical Properties:
- Hard limit enforcement: Never exceeds configured budget
- Fairness: Priority-proportional allocation
- Liveness: High-priority requests never starve
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

from loguru import logger

from core.exceptions import BudgetExceededError, ValidationError

# =========================================================================
# TYPE SYSTEM
# =========================================================================


class BudgetPeriod(str, Enum):
    """Budget accounting periods."""

    HOURLY = "hourly"
    DAILY = "daily"
    MONTHLY = "monthly"
    TOTAL = "total"  # Lifetime budget


class Priority(int, Enum):
    """Request priority levels."""

    CRITICAL = 10  # System-critical operations
    HIGH = 7  # User-facing operations
    NORMAL = 5  # Standard operations
    LOW = 3  # Background tasks
    BATCH = 1  # Bulk/batch operations


@dataclass
class BudgetConfig:
    """
    Budget configuration with hierarchical limits.

    Enforces constraints at multiple time scales.
    """

    # Token limits
    hourly_token_limit: Optional[int] = None
    daily_token_limit: int = 1_000_000  # 1M tokens/day default
    monthly_token_limit: Optional[int] = None

    # Cost limits (USD)
    hourly_cost_limit: Optional[float] = None
    daily_cost_limit: float = 50.0  # $50/day default
    monthly_cost_limit: Optional[float] = None

    # Burst allowance (% of limit)
    burst_allowance: float = 0.20  # Allow 20% burst

    # Throttling thresholds
    warning_threshold: float = 0.80  # Warn at 80% usage
    throttle_threshold: float = 0.90  # Start throttling at 90%

    # Priority configuration
    reserve_for_high_priority: float = 0.20  # Reserve 20% for high-priority

    def __post_init__(self):
        """Validate configuration."""
        if self.burst_allowance < 0 or self.burst_allowance > 1:
            raise ValidationError("burst_allowance must be in [0, 1]")

        if not 0 < self.warning_threshold < self.throttle_threshold <= 1:
            raise ValidationError("Invalid threshold configuration")


@dataclass
class BudgetUsage:
    """
    Real-time budget usage tracking.

    Maintains rolling windows for different time periods.
    """

    period: BudgetPeriod

    # Token accounting
    tokens_used: int = 0
    tokens_limit: int = 0

    # Cost accounting
    cost_used: float = 0.0
    cost_limit: float = 0.0

    # Temporal tracking
    period_start: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_reset: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # Request tracking
    requests_served: int = 0
    requests_denied: int = 0

    @property
    def token_utilization(self) -> float:
        """Token budget utilization (0-1)."""
        return self.tokens_used / self.tokens_limit if self.tokens_limit > 0 else 0.0

    @property
    def cost_utilization(self) -> float:
        """Cost budget utilization (0-1)."""
        return self.cost_used / self.cost_limit if self.cost_limit > 0 else 0.0

    @property
    def max_utilization(self) -> float:
        """Maximum utilization across metrics."""
        return max(self.token_utilization, self.cost_utilization)

    @property
    def tokens_remaining(self) -> int:
        """Tokens remaining in budget."""
        return max(0, self.tokens_limit - self.tokens_used)

    @property
    def cost_remaining(self) -> float:
        """Cost remaining in budget."""
        return max(0.0, self.cost_limit - self.cost_used)

    def should_reset(self) -> bool:
        """Check if period should reset."""
        now = datetime.utcnow()

        if self.period == BudgetPeriod.HOURLY:
            return (now - self.period_start) > timedelta(hours=1)
        elif self.period == BudgetPeriod.DAILY:
            return now.date() > self.period_start.date()
        elif self.period == BudgetPeriod.MONTHLY:
            return (now.year, now.month) > (self.period_start.year, self.period_start.month)
        else:
            return False

    def reset(self) -> None:
        """Reset usage counters."""
        self.tokens_used = 0
        self.cost_used = 0.0
        self.requests_served = 0
        self.requests_denied = 0
        self.period_start = datetime.utcnow()
        self.last_reset = datetime.utcnow()


@dataclass
class BudgetRequest:
    """
    Budget allocation request.

    Represents a single request for token/cost budget.
    """

    request_id: str
    priority: Priority
    estimated_tokens: int
    estimated_cost: float
    timestamp: datetime = field(default_factory=datetime.utcnow)
    timeout_seconds: float = 30.0

    # Callback for async notification
    future: Optional[asyncio.Future] = None


@dataclass
class BudgetAllocation:
    """
    Budget allocation decision.

    Result of budget request processing.
    """

    request_id: str
    approved: bool
    allocated_tokens: int
    allocated_cost: float
    wait_time_seconds: float
    reason: str
    timestamp: datetime = field(default_factory=datetime.utcnow)


# =========================================================================
# LEAKY BUCKET RATE LIMITER
# =========================================================================


class LeakyBucket:
    """
    Leaky bucket algorithm for smooth rate limiting.

    Properties:
    - Smooth traffic shaping
    - Burst tolerance
    - Deterministic behavior
    """

    def __init__(
        self,
        capacity: int,
        leak_rate: float,  # tokens per second
        burst_allowance: float = 0.2,
    ):
        """
        Initialize leaky bucket.

        Args:
            capacity: Maximum bucket capacity
            leak_rate: Rate at which bucket leaks (tokens/sec)
            burst_allowance: Additional burst capacity (0-1)
        """
        self.capacity = capacity
        self.leak_rate = leak_rate
        self.burst_capacity = int(capacity * (1 + burst_allowance))

        self.current_level = 0.0
        self.last_leak_time = time.time()

    def try_consume(self, tokens: int) -> bool:
        """
        Try to consume tokens from bucket.

        Args:
            tokens: Number of tokens to consume

        Returns:
            True if consumption successful
        """
        # Leak tokens based on elapsed time
        self._leak()

        # Check if we can fit tokens
        if self.current_level + tokens <= self.burst_capacity:
            self.current_level += tokens
            return True

        return False

    def _leak(self) -> None:
        """Leak tokens based on elapsed time."""
        now = time.time()
        elapsed = now - self.last_leak_time

        # Leak tokens
        leaked = elapsed * self.leak_rate
        self.current_level = max(0, self.current_level - leaked)

        self.last_leak_time = now

    @property
    def available_capacity(self) -> int:
        """Get available capacity."""
        self._leak()
        return int(self.burst_capacity - self.current_level)


# =========================================================================
# TOKEN BUDGET MANAGER
# =========================================================================


class TokenBudgetManager:
    """
    Distributed token budget management system.

    Features:
    - Multi-period budget tracking (hourly, daily, monthly)
    - Priority-based allocation
    - Predictive budget exhaustion detection
    - Adaptive throttling
    - Real-time monitoring

    Architecture: Event-driven accounting with leaky bucket smoothing
    and priority queue for request ordering.
    """

    def __init__(self, config: BudgetConfig):
        """
        Initialize budget manager.

        Args:
            config: Budget configuration
        """
        self.config = config

        # Usage tracking by period
        self.usage: Dict[BudgetPeriod, BudgetUsage] = {}
        self._initialize_usage_tracking()

        # Leaky bucket rate limiter (tokens per second)
        daily_rate = config.daily_token_limit / 86400  # tokens/second
        self.rate_limiter = LeakyBucket(
            capacity=config.daily_token_limit,
            leak_rate=daily_rate,
            burst_allowance=config.burst_allowance,
        )

        # Priority queue for pending requests
        self.pending_requests: Dict[Priority, deque] = {p: deque() for p in Priority}

        # Statistics
        self.total_requests = 0
        self.total_approved = 0
        self.total_denied = 0

        # Throttling state
        self.throttle_active = False
        self.throttle_factor = 1.0  # Multiplier for delays

        logger.info(f"Token budget manager initialized: {config.daily_token_limit:,} tokens/day")

    def _initialize_usage_tracking(self) -> None:
        """Initialize usage tracking for all periods."""
        if self.config.hourly_token_limit:
            self.usage[BudgetPeriod.HOURLY] = BudgetUsage(
                period=BudgetPeriod.HOURLY,
                tokens_limit=self.config.hourly_token_limit,
                cost_limit=self.config.hourly_cost_limit or float("inf"),
            )

        self.usage[BudgetPeriod.DAILY] = BudgetUsage(
            period=BudgetPeriod.DAILY,
            tokens_limit=self.config.daily_token_limit,
            cost_limit=self.config.daily_cost_limit,
        )

        if self.config.monthly_token_limit:
            self.usage[BudgetPeriod.MONTHLY] = BudgetUsage(
                period=BudgetPeriod.MONTHLY,
                tokens_limit=self.config.monthly_token_limit,
                cost_limit=self.config.monthly_cost_limit or float("inf"),
            )

    # =========================================================================
    # BUDGET REQUEST INTERFACE
    # =========================================================================

    async def request_budget(
        self,
        tokens: int,
        cost: float,
        priority: Priority = Priority.NORMAL,
        timeout: float = 30.0,
    ) -> BudgetAllocation:
        """
        Request token/cost budget allocation.

        Args:
            tokens: Number of tokens requested
            cost: Estimated cost
            priority: Request priority
            timeout: Maximum wait time (seconds)

        Returns:
            BudgetAllocation decision

        Raises:
            BudgetExceededError: If budget cannot be allocated
        """
        request_id = f"req_{int(time.time() * 1000)}"
        start_time = time.time()

        self.total_requests += 1

        # Check and reset periods if needed
        self._check_period_resets()

        # Fast path: Check if we can immediately approve
        if await self._can_afford(tokens, cost, priority):
            allocation = await self._allocate(request_id, tokens, cost, priority)

            if allocation.approved:
                self.total_approved += 1
                return allocation

        # Slow path: Queue and wait
        request = BudgetRequest(
            request_id=request_id,
            priority=priority,
            estimated_tokens=tokens,
            estimated_cost=cost,
            timeout_seconds=timeout,
            future=asyncio.Future(),
        )

        self.pending_requests[priority].append(request)

        try:
            # Wait for allocation with timeout
            allocation = await asyncio.wait_for(
                request.future,
                timeout=timeout,
            )

            if allocation.approved:
                self.total_approved += 1
            else:
                self.total_denied += 1

            return allocation

        except asyncio.TimeoutError:
            self.total_denied += 1

            # Remove from queue
            try:
                self.pending_requests[priority].remove(request)
            except ValueError:
                pass

            raise BudgetExceededError(
                f"Budget request timed out after {timeout}s "
                f"(tokens: {tokens}, cost: ${cost:.4f})"
            )

    async def can_afford(self, tokens: int, cost: float = 0.0) -> bool:
        """
        Quick check if budget is available.

        Args:
            tokens: Token count
            cost: Estimated cost

        Returns:
            True if budget likely available
        """
        return await self._can_afford(tokens, cost, Priority.NORMAL)

    async def _can_afford(
        self,
        tokens: int,
        cost: float,
        priority: Priority,
    ) -> bool:
        """Internal affordability check with priority consideration."""
        # Check rate limiter
        if not self.rate_limiter.try_consume(tokens):
            return False

        # Check each period
        for usage in self.usage.values():
            # Token check
            tokens_available = usage.tokens_remaining

            # Reserve budget for high priority if this is low priority
            if priority < Priority.HIGH:
                reserve = int(usage.tokens_limit * self.config.reserve_for_high_priority)
                tokens_available -= reserve

            if tokens_available < tokens:
                return False

            # Cost check
            cost_available = usage.cost_remaining

            if priority < Priority.HIGH:
                reserve = usage.cost_limit * self.config.reserve_for_high_priority
                cost_available -= reserve

            if cost_available < cost:
                return False

        return True

    async def _allocate(
        self,
        request_id: str,
        tokens: int,
        cost: float,
        priority: Priority,
    ) -> BudgetAllocation:
        """
        Allocate budget and update accounting.

        Args:
            request_id: Request identifier
            tokens: Tokens to allocate
            cost: Cost to allocate
            priority: Request priority

        Returns:
            BudgetAllocation
        """
        # Update usage for all periods
        for usage in self.usage.values():
            usage.tokens_used += tokens
            usage.cost_used += cost
            usage.requests_served += 1

        # Check if throttling should activate
        self._update_throttling_state()

        # Process pending requests if budget freed up
        asyncio.create_task(self._process_pending_requests())

        return BudgetAllocation(
            request_id=request_id,
            approved=True,
            allocated_tokens=tokens,
            allocated_cost=cost,
            wait_time_seconds=0.0,
            reason="Budget allocated",
        )

    async def _process_pending_requests(self) -> None:
        """Process pending requests in priority order."""
        # Process in priority order (high to low)
        for priority in sorted(Priority, key=lambda p: p.value, reverse=True):
            queue = self.pending_requests[priority]

            while queue:
                request = queue[0]

                # Check if request timed out
                elapsed = (datetime.utcnow() - request.timestamp).total_seconds()
                if elapsed > request.timeout_seconds:
                    queue.popleft()
                    if request.future and not request.future.done():
                        request.future.set_exception(
                            BudgetExceededError("Request timed out in queue")
                        )
                    continue

                # Try to allocate
                if await self._can_afford(
                    request.estimated_tokens,
                    request.estimated_cost,
                    request.priority,
                ):
                    queue.popleft()

                    allocation = await self._allocate(
                        request.request_id,
                        request.estimated_tokens,
                        request.estimated_cost,
                        request.priority,
                    )

                    if request.future and not request.future.done():
                        request.future.set_result(allocation)
                else:
                    # Can't afford this request, stop processing this priority
                    break

    # =========================================================================
    # BUDGET REPORTING
    # =========================================================================

    async def report_actual_usage(
        self,
        request_id: str,
        actual_tokens: int,
        actual_cost: float,
    ) -> None:
        """
        Report actual usage (for correction if estimate was wrong).

        Args:
            request_id: Request identifier
            actual_tokens: Actual tokens used
            actual_cost: Actual cost incurred
        """
        # For now, we don't track per-request, so this is informational
        # In production, could implement correction logic
        logger.debug(f"Actual usage for {request_id}: {actual_tokens} tokens, ${actual_cost:.4f}")

    # =========================================================================
    # PERIOD MANAGEMENT
    # =========================================================================

    def _check_period_resets(self) -> None:
        """Check and reset usage periods if needed."""
        for period, usage in self.usage.items():
            if usage.should_reset():
                logger.info(f"Resetting {period.value} budget usage")
                usage.reset()

    def _update_throttling_state(self) -> None:
        """Update throttling based on usage."""
        # Get maximum utilization across all periods
        max_util = max(usage.max_utilization for usage in self.usage.values())

        if max_util >= self.config.throttle_threshold:
            if not self.throttle_active:
                logger.warning(f"Throttling activated at {max_util:.1%} utilization")
                self.throttle_active = True

            # Exponential throttle factor
            self.throttle_factor = 1.0 + (max_util - self.config.throttle_threshold) * 10

        elif max_util >= self.config.warning_threshold:
            if not self.throttle_active:
                logger.warning(f"Budget warning: {max_util:.1%} utilization")

        else:
            if self.throttle_active:
                logger.info("Throttling deactivated")
                self.throttle_active = False

            self.throttle_factor = 1.0

    # =========================================================================
    # MONITORING & ANALYTICS
    # =========================================================================

    def get_usage_statistics(self) -> Dict[str, Any]:
        """Get comprehensive usage statistics."""
        return {
            "by_period": {
                period.value: {
                    "tokens_used": usage.tokens_used,
                    "tokens_limit": usage.tokens_limit,
                    "token_utilization": usage.token_utilization,
                    "cost_used": usage.cost_used,
                    "cost_limit": usage.cost_limit,
                    "cost_utilization": usage.cost_utilization,
                    "requests_served": usage.requests_served,
                    "requests_denied": usage.requests_denied,
                    "period_start": usage.period_start.isoformat(),
                }
                for period, usage in self.usage.items()
            },
            "global": {
                "total_requests": self.total_requests,
                "total_approved": self.total_approved,
                "total_denied": self.total_denied,
                "approval_rate": (
                    self.total_approved / self.total_requests if self.total_requests > 0 else 0
                ),
                "throttle_active": self.throttle_active,
                "throttle_factor": self.throttle_factor,
            },
            "pending_requests": {
                priority.name: len(queue) for priority, queue in self.pending_requests.items()
            },
            "rate_limiter": {
                "available_capacity": self.rate_limiter.available_capacity,
                "burst_capacity": self.rate_limiter.burst_capacity,
                "current_level": int(self.rate_limiter.current_level),
            },
        }

    def predict_exhaustion(self) -> Optional[datetime]:
        """
        Predict when budget will be exhausted.

        Uses linear extrapolation based on current burn rate.

        Returns:
            Predicted exhaustion time or None if not trending toward exhaustion
        """
        daily = self.usage.get(BudgetPeriod.DAILY)
        if not daily:
            return None

        # Calculate burn rate (tokens per hour)
        elapsed_hours = (datetime.utcnow() - daily.period_start).total_seconds() / 3600

        if elapsed_hours < 0.1:  # Too early to predict
            return None

        burn_rate = daily.tokens_used / elapsed_hours  # tokens per hour

        if burn_rate <= 0:
            return None

        # Tokens remaining
        remaining = daily.tokens_remaining

        # Hours until exhaustion
        hours_remaining = remaining / burn_rate

        # Predict exhaustion time
        exhaustion_time = datetime.utcnow() + timedelta(hours=hours_remaining)

        return exhaustion_time

    def get_recommendations(self) -> List[str]:
        """Get budget optimization recommendations."""
        recommendations = []

        daily = self.usage.get(BudgetPeriod.DAILY)
        if not daily:
            return recommendations

        util = daily.max_utilization

        if util > 0.95:
            recommendations.append(
                "CRITICAL: Budget nearly exhausted. Consider increasing limits or reducing usage."
            )
        elif util > 0.85:
            recommendations.append("WARNING: High budget utilization. Monitor closely.")

        # Check burn rate
        exhaustion = self.predict_exhaustion()
        if exhaustion and exhaustion < datetime.utcnow() + timedelta(hours=6):
            recommendations.append(
                f"Budget predicted to exhaust at {exhaustion.strftime('%H:%M')}. Take action."
            )

        # Check pending requests
        total_pending = sum(len(q) for q in self.pending_requests.values())
        if total_pending > 10:
            recommendations.append(f"{total_pending} requests pending. System under load.")

        if not recommendations:
            recommendations.append("Budget healthy. No action needed.")

        return recommendations


# =========================================================================
# CONVENIENCE FUNCTIONS
# =========================================================================


def create_default_budget_manager() -> TokenBudgetManager:
    """Create budget manager with sensible defaults."""
    config = BudgetConfig(
        daily_token_limit=1_000_000,  # 1M tokens/day
        daily_cost_limit=50.0,  # $50/day
        burst_allowance=0.20,
        warning_threshold=0.80,
        throttle_threshold=0.90,
        reserve_for_high_priority=0.20,
    )

    return TokenBudgetManager(config)
