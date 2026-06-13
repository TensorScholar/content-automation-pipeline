"""Persistent LLM usage accounting and distributed budget enforcement."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Iterator, Optional
from uuid import UUID, uuid4

from loguru import logger
from sqlalchemy import insert, text, update

from config.settings import get_settings
from core.exceptions import TokenBudgetExceededError
from infrastructure.database import DatabaseManager
from infrastructure.schema import llm_usage_records_table


@dataclass(frozen=True)
class LLMUsageContext:
    project_id: Optional[UUID] = None
    user_id: Optional[UUID] = None
    task_id: Optional[str] = None
    operation_type: str = "generation"


_usage_context: ContextVar[LLMUsageContext] = ContextVar(
    "llm_usage_context",
    default=LLMUsageContext(),
)


def _optional_uuid(value: UUID | str | None) -> Optional[UUID]:
    if value is None or value == "":
        return None
    return value if isinstance(value, UUID) else UUID(str(value))


@contextmanager
def llm_usage_context(
    *,
    project_id: UUID | str | None = None,
    user_id: UUID | str | None = None,
    task_id: Optional[str] = None,
    operation_type: str = "generation",
) -> Iterator[None]:
    """Attach project/user/task attribution to nested LLM calls."""
    token = _usage_context.set(
        LLMUsageContext(
            project_id=_optional_uuid(project_id),
            user_id=_optional_uuid(user_id),
            task_id=task_id,
            operation_type=operation_type,
        )
    )
    try:
        yield
    finally:
        _usage_context.reset(token)


def get_llm_usage_context() -> LLMUsageContext:
    return _usage_context.get()


@dataclass(frozen=True)
class BudgetLimit:
    name: str
    amount: Decimal
    period_start: datetime
    project_id: Optional[UUID] = None
    user_id: Optional[UUID] = None


class LLMUsageService:
    """Reserve budget and persist actual provider usage in PostgreSQL."""

    RESERVATION_TTL = timedelta(minutes=20)

    def __init__(self, database_manager: DatabaseManager):
        self.database_manager = database_manager
        self.settings = get_settings()

    def _limits(self, context: LLMUsageContext, now: datetime) -> list[BudgetLimit]:
        llm = self.settings.llm
        day_start = now.replace(hour=0, minute=0, second=0, microsecond=0)
        month_start = day_start.replace(day=1)
        limits = [
            BudgetLimit(
                "global_daily",
                Decimal(str(llm.daily_cost_limit_usd)),
                day_start,
            ),
            BudgetLimit(
                "global_monthly",
                Decimal(str(llm.monthly_cost_limit_usd)),
                month_start,
            ),
        ]
        if context.project_id:
            limits.extend(
                [
                    BudgetLimit(
                        "project_daily",
                        Decimal(str(llm.project_daily_cost_limit_usd)),
                        day_start,
                        project_id=context.project_id,
                    ),
                    BudgetLimit(
                        "project_monthly",
                        Decimal(str(llm.project_monthly_cost_limit_usd)),
                        month_start,
                        project_id=context.project_id,
                    ),
                ]
            )
        if context.user_id:
            limits.extend(
                [
                    BudgetLimit(
                        "user_daily",
                        Decimal(str(llm.user_daily_cost_limit_usd)),
                        day_start,
                        user_id=context.user_id,
                    ),
                    BudgetLimit(
                        "user_monthly",
                        Decimal(str(llm.user_monthly_cost_limit_usd)),
                        month_start,
                        user_id=context.user_id,
                    ),
                ]
            )
        return [limit for limit in limits if limit.amount > 0]

    @staticmethod
    async def _committed_and_reserved_cost(session, limit: BudgetLimit, now: datetime) -> Decimal:
        filters = ["created_at >= :period_start"]
        params = {
            "now": now,
            "period_start": limit.period_start,
        }
        if limit.project_id is not None:
            filters.append("project_id = :project_id")
            params["project_id"] = limit.project_id
        if limit.user_id is not None:
            filters.append("user_id = :user_id")
            params["user_id"] = limit.user_id

        result = await session.execute(
            text(
                f"""
                SELECT COALESCE(SUM(
                    CASE
                        WHEN status = 'reserved'
                             AND reservation_expires_at > :now
                            THEN estimated_cost_usd
                        WHEN status = 'failure'
                            THEN estimated_cost_usd
                        ELSE actual_cost_usd
                    END
                ), 0)
                FROM llm_usage_records
                WHERE {" AND ".join(filters)}
                """
            ),
            params,
        )
        return Decimal(str(result.scalar_one()))

    @staticmethod
    def _log_threshold(limit: BudgetLimit, projected: Decimal) -> None:
        utilization = projected / limit.amount
        threshold = 100 if utilization >= 1 else 80 if utilization >= Decimal("0.8") else 50
        if utilization >= Decimal("0.5"):
            logger.warning(
                "LLM budget threshold reached | scope={} | threshold={} | "
                "projected_usd={} | limit_usd={}",
                limit.name,
                threshold,
                projected,
                limit.amount,
            )

    async def reserve(
        self,
        *,
        provider: str,
        model: str,
        estimated_cost: float,
        context: Optional[LLMUsageContext] = None,
    ) -> UUID:
        """Atomically enforce limits and create a short-lived cost reservation."""
        now = datetime.now(timezone.utc)
        context = context or get_llm_usage_context()
        estimated = Decimal(str(max(0.0, estimated_cost))).quantize(Decimal("0.000001"))
        record_id = uuid4()

        async with self.database_manager.session() as session:
            # One short global transaction lock makes all scope checks atomic
            # across API and Celery processes without adding a new coordinator.
            await session.execute(
                text("SELECT pg_advisory_xact_lock(hashtext('smarlux_llm_budget'))")
            )
            for limit in self._limits(context, now):
                used = await self._committed_and_reserved_cost(session, limit, now)
                projected = used + estimated
                self._log_threshold(limit, projected)
                if projected > limit.amount:
                    raise TokenBudgetExceededError(
                        f"LLM {limit.name.replace('_', ' ')} budget is exhausted",
                        current_cost=float(used),
                        budget_limit=float(limit.amount),
                    )

            await session.execute(
                insert(llm_usage_records_table).values(
                    id=record_id,
                    provider=provider,
                    model=model,
                    operation_type=context.operation_type,
                    project_id=context.project_id,
                    user_id=context.user_id,
                    task_id=context.task_id,
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                    estimated_cost_usd=estimated,
                    actual_cost_usd=Decimal("0"),
                    status="reserved",
                    created_at=now,
                    reservation_expires_at=now + self.RESERVATION_TTL,
                )
            )

        return record_id

    async def record_success(
        self,
        record_id: UUID,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        total_tokens: int,
        actual_cost: float,
    ) -> None:
        now = datetime.now(timezone.utc)
        async with self.database_manager.session() as session:
            await session.execute(
                update(llm_usage_records_table)
                .where(llm_usage_records_table.c.id == record_id)
                .values(
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    actual_cost_usd=Decimal(str(max(0.0, actual_cost))).quantize(
                        Decimal("0.000001")
                    ),
                    status="success",
                    completed_at=now,
                    reservation_expires_at=None,
                )
            )

    async def record_failure(self, record_id: UUID, error_category: str) -> None:
        """Persist failure without storing provider prompts or exception text."""
        async with self.database_manager.session() as session:
            await session.execute(
                update(llm_usage_records_table)
                .where(llm_usage_records_table.c.id == record_id)
                .values(
                    status="failure",
                    error_category=error_category[:100],
                    completed_at=datetime.now(timezone.utc),
                    reservation_expires_at=None,
                )
            )
