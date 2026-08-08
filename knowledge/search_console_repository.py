"""Persistence for Google Search Console OAuth connections and durable sync runs."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from typing import Any
from uuid import UUID, uuid4

from sqlalchemy import delete, desc, func, insert, or_, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert

from infrastructure.database import DatabaseManager
from infrastructure.redaction import redact_text
from infrastructure.schema import (
    search_console_connections_table,
    search_console_oauth_states_table,
    search_console_properties_table,
    search_console_sync_runs_table,
)


def _utc_now_naive() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


@dataclass(frozen=True)
class SearchConsoleSyncClaim:
    claimed: bool
    run: dict[str, Any]
    reason: str | None = None


class SearchConsoleRepository:
    """Repository with one-time OAuth state and idempotent sync-window claims."""

    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager

    async def create_oauth_state(
        self,
        *,
        state_hash: str,
        project_id: UUID,
        user_id: UUID,
        expires_at: datetime,
    ) -> UUID:
        state_id = uuid4()
        await self.db.execute(
            insert(search_console_oauth_states_table).values(
                id=state_id,
                state_hash=state_hash,
                project_id=project_id,
                user_id=user_id,
                expires_at=expires_at.replace(tzinfo=None),
            )
        )
        return state_id

    async def consume_oauth_state(self, *, state_hash: str) -> dict[str, Any] | None:
        """Atomically consume an unexpired OAuth state exactly once."""
        now = _utc_now_naive()
        async with self.db.transaction() as session:
            result = await session.execute(
                select(search_console_oauth_states_table)
                .where(search_console_oauth_states_table.c.state_hash == state_hash)
                .with_for_update()
            )
            row = result.fetchone()
            if not row:
                return None
            state = dict(row._mapping)
            if state.get("consumed_at") is not None or state["expires_at"] <= now:
                return None
            await session.execute(
                update(search_console_oauth_states_table)
                .where(search_console_oauth_states_table.c.id == state["id"])
                .values(consumed_at=now)
            )
            return state

    async def delete_expired_oauth_states(self) -> int:
        now = _utc_now_naive()
        result = await self.db.execute(
            delete(search_console_oauth_states_table).where(
                (search_console_oauth_states_table.c.expires_at < now)
                | (search_console_oauth_states_table.c.consumed_at.is_not(None))
            )
        )
        return int(getattr(result, "rowcount", 0) or 0)

    async def upsert_connection(
        self,
        *,
        project_id: UUID,
        encrypted_refresh_token: str,
        scope: str,
        connected_by: UUID | None,
    ) -> dict[str, Any]:
        now = _utc_now_naive()
        statement = pg_insert(search_console_connections_table).values(
            id=uuid4(),
            project_id=project_id,
            encrypted_refresh_token=encrypted_refresh_token,
            scope=scope,
            status="connected",
            connected_by=connected_by,
            created_at=now,
            updated_at=now,
        )
        statement = statement.on_conflict_do_update(
            index_elements=[search_console_connections_table.c.project_id],
            set_={
                "encrypted_refresh_token": statement.excluded.encrypted_refresh_token,
                "scope": statement.excluded.scope,
                "status": "connected",
                "connected_by": connected_by,
                "last_error_category": None,
                "last_error_message": None,
                "updated_at": now,
            },
        ).returning(search_console_connections_table)
        async with self.db.session() as session:
            result = await session.execute(statement)
            return dict(result.mappings().one())

    async def get_connection(self, project_id: UUID) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(search_console_connections_table).where(
                search_console_connections_table.c.project_id == project_id
            )
        )
        return dict(row) if row else None

    async def list_active_connections(self) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            select(search_console_connections_table).where(
                search_console_connections_table.c.status == "connected",
                search_console_connections_table.c.selected_site_url.is_not(None),
            )
        )
        return [dict(row) for row in rows]

    async def set_connection_error(
        self,
        *,
        project_id: UUID,
        category: str,
        message: str,
        status_value: str | None = None,
    ) -> None:
        values: dict[str, Any] = {
            "last_error_category": category,
            "last_error_message": redact_text(message),
            "updated_at": _utc_now_naive(),
        }
        if status_value:
            values["status"] = status_value
        await self.db.execute(
            update(search_console_connections_table)
            .where(search_console_connections_table.c.project_id == project_id)
            .values(**values)
        )

    async def clear_connection_error(self, project_id: UUID, *, synced: bool = False) -> None:
        now = _utc_now_naive()
        values: dict[str, Any] = {
            "status": "connected",
            "last_error_category": None,
            "last_error_message": None,
            "updated_at": now,
        }
        if synced:
            values["last_sync_at"] = now
        await self.db.execute(
            update(search_console_connections_table)
            .where(search_console_connections_table.c.project_id == project_id)
            .values(**values)
        )

    async def replace_properties(
        self,
        *,
        connection_id: UUID,
        project_id: UUID,
        properties: list[dict[str, str]],
    ) -> list[dict[str, Any]]:
        now = _utc_now_naive()
        async with self.db.transaction() as session:
            for prop in properties:
                statement = pg_insert(search_console_properties_table).values(
                    id=uuid4(),
                    connection_id=connection_id,
                    project_id=project_id,
                    site_url=prop["site_url"],
                    permission_level=prop["permission_level"],
                    last_seen_at=now,
                    created_at=now,
                )
                statement = statement.on_conflict_do_update(
                    constraint="uq_search_console_connection_site",
                    set_={
                        "permission_level": statement.excluded.permission_level,
                        "last_seen_at": now,
                    },
                )
                await session.execute(statement)

            if properties:
                site_urls = [prop["site_url"] for prop in properties]
                await session.execute(
                    delete(search_console_properties_table).where(
                        search_console_properties_table.c.connection_id == connection_id,
                        search_console_properties_table.c.site_url.not_in(site_urls),
                    )
                )
                # A previously selected property can be removed or access can be
                # revoked independently in Search Console. Never keep syncing a
                # property that is no longer present in the authenticated list.
                await session.execute(
                    update(search_console_connections_table)
                    .where(
                        search_console_connections_table.c.id == connection_id,
                        search_console_connections_table.c.selected_site_url.is_not(None),
                        search_console_connections_table.c.selected_site_url.not_in(site_urls),
                    )
                    .values(
                        selected_site_url=None,
                        permission_level=None,
                        last_error_category="property_access_removed",
                        last_error_message=(
                            "The selected Search Console property is no longer accessible; "
                            "select an available property before syncing."
                        ),
                        updated_at=now,
                    )
                )
            else:
                await session.execute(
                    delete(search_console_properties_table).where(
                        search_console_properties_table.c.connection_id == connection_id
                    )
                )
                await session.execute(
                    update(search_console_connections_table)
                    .where(search_console_connections_table.c.id == connection_id)
                    .values(
                        selected_site_url=None,
                        permission_level=None,
                        last_error_category="no_accessible_properties",
                        last_error_message="No Search Console properties are accessible.",
                        updated_at=now,
                    )
                )

        return await self.list_properties(project_id)

    async def list_properties(self, project_id: UUID) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            select(search_console_properties_table)
            .where(search_console_properties_table.c.project_id == project_id)
            .order_by(search_console_properties_table.c.site_url.asc())
        )
        return [dict(row) for row in rows]

    async def select_property(
        self,
        *,
        project_id: UUID,
        site_url: str,
    ) -> dict[str, Any] | None:
        prop = await self.db.fetch_one(
            select(search_console_properties_table).where(
                search_console_properties_table.c.project_id == project_id,
                search_console_properties_table.c.site_url == site_url,
            )
        )
        if not prop:
            return None
        await self.db.execute(
            update(search_console_connections_table)
            .where(search_console_connections_table.c.project_id == project_id)
            .values(
                selected_site_url=site_url,
                permission_level=prop["permission_level"],
                status="connected",
                last_error_category=None,
                last_error_message=None,
                updated_at=_utc_now_naive(),
            )
        )
        return dict(prop)

    async def disconnect(self, project_id: UUID) -> bool:
        """Revoke local use while retaining immutable synchronization audit rows."""
        now = _utc_now_naive()
        async with self.db.transaction() as session:
            result = await session.execute(
                update(search_console_connections_table)
                .where(search_console_connections_table.c.project_id == project_id)
                .values(
                    encrypted_refresh_token=None,
                    status="disconnected",
                    selected_site_url=None,
                    permission_level=None,
                    last_error_category=None,
                    last_error_message=None,
                    updated_at=now,
                )
            )
            await session.execute(
                delete(search_console_properties_table).where(
                    search_console_properties_table.c.project_id == project_id
                )
            )
        return bool(getattr(result, "rowcount", 0))

    async def claim_sync(
        self,
        *,
        project_id: UUID,
        connection_id: UUID,
        site_url: str,
        date_from: date,
        date_to: date,
        task_id: str,
    ) -> SearchConsoleSyncClaim:
        """Claim one project/property/date window; successful windows are immutable."""
        now = _utc_now_naive()
        async with self.db.transaction() as session:
            result = await session.execute(
                select(search_console_sync_runs_table)
                .where(
                    search_console_sync_runs_table.c.project_id == project_id,
                    search_console_sync_runs_table.c.site_url == site_url,
                    search_console_sync_runs_table.c.date_from == date_from,
                    search_console_sync_runs_table.c.date_to == date_to,
                )
                .with_for_update()
            )
            row = result.fetchone()
            if row:
                existing = dict(row._mapping)
                if existing["status"] == "succeeded":
                    return SearchConsoleSyncClaim(False, existing, "already_succeeded")
                if existing["status"] in {"queued", "running", "retrying"}:
                    stale_before = now - timedelta(minutes=30)
                    if existing.get("updated_at") and existing["updated_at"] > stale_before:
                        return SearchConsoleSyncClaim(False, existing, "already_running")
                await session.execute(
                    update(search_console_sync_runs_table)
                    .where(search_console_sync_runs_table.c.id == existing["id"])
                    .values(
                        status="queued",
                        task_id=task_id,
                        retry_count=search_console_sync_runs_table.c.retry_count + 1,
                        error_category=None,
                        error_message=None,
                        started_at=None,
                        finished_at=None,
                        updated_at=now,
                    )
                )
                existing.update({"status": "queued", "task_id": task_id})
                return SearchConsoleSyncClaim(True, existing)

            run_id = uuid4()
            await session.execute(
                insert(search_console_sync_runs_table).values(
                    id=run_id,
                    connection_id=connection_id,
                    project_id=project_id,
                    site_url=site_url,
                    date_from=date_from,
                    date_to=date_to,
                    status="queued",
                    task_id=task_id,
                    created_at=now,
                    updated_at=now,
                )
            )
            return SearchConsoleSyncClaim(
                True,
                {
                    "id": run_id,
                    "connection_id": connection_id,
                    "project_id": project_id,
                    "site_url": site_url,
                    "date_from": date_from,
                    "date_to": date_to,
                    "status": "queued",
                    "task_id": task_id,
                    "retry_count": 0,
                },
            )

    async def mark_sync_running(self, run_id: UUID, *, task_id: str) -> bool:
        """Acquire one execution slot; duplicate deliveries become safe no-ops."""
        now = _utc_now_naive()
        result = await self.db.execute(
            update(search_console_sync_runs_table)
            .where(
                search_console_sync_runs_table.c.id == run_id,
                search_console_sync_runs_table.c.task_id == task_id,
                search_console_sync_runs_table.c.status.in_({"queued", "retrying"}),
            )
            .values(status="running", started_at=now, updated_at=now)
        )
        return bool(getattr(result, "rowcount", 0))

    async def mark_sync_retry(
        self,
        *,
        run_id: UUID,
        task_id: str,
        category: str,
        message: str,
        retry_count: int,
    ) -> bool:
        result = await self.db.execute(
            update(search_console_sync_runs_table)
            .where(
                search_console_sync_runs_table.c.id == run_id,
                search_console_sync_runs_table.c.task_id == task_id,
                search_console_sync_runs_table.c.status.in_({"queued", "running", "retrying"}),
            )
            .values(
                status="retrying",
                retry_count=retry_count,
                error_category=category,
                error_message=redact_text(message),
                updated_at=_utc_now_naive(),
            )
        )
        return bool(getattr(result, "rowcount", 0))

    async def mark_sync_success(
        self,
        *,
        run_id: UUID,
        task_id: str,
        row_count: int,
        pages_fetched: int,
        truncated: bool = False,
    ) -> bool:
        now = _utc_now_naive()
        result = await self.db.execute(
            update(search_console_sync_runs_table)
            .where(
                search_console_sync_runs_table.c.id == run_id,
                search_console_sync_runs_table.c.task_id == task_id,
                search_console_sync_runs_table.c.status == "running",
            )
            .values(
                status="succeeded",
                row_count=row_count,
                pages_fetched=pages_fetched,
                truncated=truncated,
                error_category=None,
                error_message=None,
                finished_at=now,
                updated_at=now,
            )
        )
        return bool(getattr(result, "rowcount", 0))

    async def mark_sync_failure(
        self,
        *,
        run_id: UUID,
        task_id: str,
        category: str,
        message: str,
        retry_count: int,
    ) -> bool:
        now = _utc_now_naive()
        result = await self.db.execute(
            update(search_console_sync_runs_table)
            .where(
                search_console_sync_runs_table.c.id == run_id,
                search_console_sync_runs_table.c.task_id == task_id,
                search_console_sync_runs_table.c.status.in_({"queued", "running", "retrying"}),
            )
            .values(
                status="failed",
                retry_count=retry_count,
                error_category=category,
                error_message=redact_text(message),
                finished_at=now,
                updated_at=now,
            )
        )
        return bool(getattr(result, "rowcount", 0))


    async def list_stale_sync_runs(
        self,
        *,
        stale_after_minutes: int = 30,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """Return active sync runs whose worker heartbeat/transition expired."""
        cutoff = _utc_now_naive() - timedelta(minutes=max(5, stale_after_minutes))
        rows = await self.db.fetch_all(
            select(search_console_sync_runs_table)
            .where(
                search_console_sync_runs_table.c.status.in_({"queued", "running", "retrying"}),
                search_console_sync_runs_table.c.updated_at < cutoff,
            )
            .order_by(search_console_sync_runs_table.c.updated_at.asc())
            .limit(max(1, min(limit, 500)))
        )
        return [dict(row) for row in rows]

    async def requeue_stale_sync_run(
        self,
        *,
        run_id: UUID,
        task_id: str,
        stale_after_minutes: int = 30,
    ) -> dict[str, Any] | None:
        """Atomically recover one stale active run; fresh or terminal runs are untouched."""
        now = _utc_now_naive()
        cutoff = now - timedelta(minutes=max(5, stale_after_minutes))
        async with self.db.transaction() as session:
            result = await session.execute(
                select(search_console_sync_runs_table)
                .where(search_console_sync_runs_table.c.id == run_id)
                .with_for_update()
            )
            row = result.fetchone()
            if not row:
                return None
            existing = dict(row._mapping)
            if existing["status"] not in {"queued", "running", "retrying"}:
                return None
            if existing.get("updated_at") and existing["updated_at"] >= cutoff:
                return None
            await session.execute(
                update(search_console_sync_runs_table)
                .where(search_console_sync_runs_table.c.id == run_id)
                .values(
                    status="queued",
                    task_id=task_id,
                    retry_count=search_console_sync_runs_table.c.retry_count + 1,
                    error_category="stale_worker_recovered",
                    error_message="Recovered after an interrupted integration worker",
                    started_at=None,
                    finished_at=None,
                    updated_at=now,
                )
            )
            existing.update(
                {
                    "status": "queued",
                    "task_id": task_id,
                    "retry_count": int(existing.get("retry_count") or 0) + 1,
                    "updated_at": now,
                }
            )
            return existing

    async def get_operational_summary(
        self,
        *,
        project_id: UUID | None = None,
        lookback_hours: int = 24,
        recent_limit: int = 10,
    ) -> dict[str, Any]:
        """Return bounded Search Console connection and sync health signals."""
        now = _utc_now_naive()
        cutoff = now - timedelta(hours=max(1, min(lookback_hours, 168)))
        run_filters = []
        connection_filters = []
        if project_id is not None:
            run_filters.append(search_console_sync_runs_table.c.project_id == project_id)
            connection_filters.append(search_console_connections_table.c.project_id == project_id)

        status_rows = await self.db.fetch_all(
            select(
                search_console_sync_runs_table.c.status,
                func.count().label("count"),
            )
            .where(
                or_(
                    search_console_sync_runs_table.c.status.in_(
                        {"queued", "running", "retrying"}
                    ),
                    search_console_sync_runs_table.c.created_at >= cutoff,
                ),
                *run_filters,
            )
            .group_by(search_console_sync_runs_table.c.status)
        )
        status_counts = {str(row["status"]): int(row["count"] or 0) for row in status_rows}

        duration_seconds = func.extract(
            "epoch",
            search_console_sync_runs_table.c.finished_at
            - search_console_sync_runs_table.c.started_at,
        )
        recent = await self.db.fetch_one(
            select(
                func.count().label("total"),
                func.count().filter(search_console_sync_runs_table.c.status == "succeeded").label("succeeded"),
                func.count().filter(search_console_sync_runs_table.c.status == "failed").label("failed"),
                func.count().filter(search_console_sync_runs_table.c.truncated.is_(True)).label("truncated"),
                func.max(search_console_sync_runs_table.c.finished_at)
                .filter(search_console_sync_runs_table.c.status == "succeeded")
                .label("latest_success_at"),
                func.percentile_cont(0.95)
                .within_group(duration_seconds)
                .label("p95_duration_seconds"),
            ).where(
                search_console_sync_runs_table.c.created_at >= cutoff,
                search_console_sync_runs_table.c.status.in_({"succeeded", "failed"}),
                *run_filters,
            )
        )
        recent_values = dict(recent) if recent else {}

        stale_cutoff = now - timedelta(minutes=30)
        stale = await self.db.fetch_one(
            select(func.count().label("count")).where(
                search_console_sync_runs_table.c.status.in_({"queued", "running", "retrying"}),
                search_console_sync_runs_table.c.updated_at < stale_cutoff,
                *run_filters,
            )
        )
        stale_count = int(stale["count"] or 0) if stale else 0

        connection_rows = await self.db.fetch_all(
            select(
                search_console_connections_table.c.status,
                func.count().label("count"),
            )
            .where(*connection_filters)
            .group_by(search_console_connections_table.c.status)
        )
        connection_counts = {str(row["status"]): int(row["count"] or 0) for row in connection_rows}

        failure_rows = await self.db.fetch_all(
            select(
                search_console_sync_runs_table.c.id,
                search_console_sync_runs_table.c.project_id,
                search_console_sync_runs_table.c.site_url,
                search_console_sync_runs_table.c.error_category,
                search_console_sync_runs_table.c.error_message,
                search_console_sync_runs_table.c.retry_count,
                search_console_sync_runs_table.c.updated_at,
            )
            .where(
                search_console_sync_runs_table.c.status == "failed",
                search_console_sync_runs_table.c.updated_at >= cutoff,
                *run_filters,
            )
            .order_by(search_console_sync_runs_table.c.updated_at.desc())
            .limit(max(1, min(recent_limit, 50)))
        )
        return {
            "connection_counts": connection_counts,
            "status_counts": status_counts,
            "active_count": sum(status_counts.get(state, 0) for state in {"queued", "running", "retrying"}),
            "stale_count": stale_count,
            "recent_total": int(recent_values.get("total") or 0),
            "recent_succeeded": int(recent_values.get("succeeded") or 0),
            "recent_failed": int(recent_values.get("failed") or 0),
            "recent_truncated": int(recent_values.get("truncated") or 0),
            "latest_success_at": recent_values.get("latest_success_at"),
            "p95_duration_seconds": float(recent_values.get("p95_duration_seconds") or 0),
            "recent_failures": [
                {**dict(row), "error_message": redact_text(str(row.get("error_message") or ""))}
                for row in failure_rows
            ],
        }

    async def get_sync_run(self, run_id: UUID) -> dict[str, Any] | None:
        row = await self.db.fetch_one(
            select(search_console_sync_runs_table).where(
                search_console_sync_runs_table.c.id == run_id
            )
        )
        return dict(row) if row else None

    async def list_sync_runs(self, project_id: UUID, limit: int = 20) -> list[dict[str, Any]]:
        rows = await self.db.fetch_all(
            select(search_console_sync_runs_table)
            .where(search_console_sync_runs_table.c.project_id == project_id)
            .order_by(desc(search_console_sync_runs_table.c.created_at))
            .limit(limit)
        )
        return [dict(row) for row in rows]
