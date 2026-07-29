"""PostgreSQL concurrency coverage for durable generation finalization.

This test is opt-in because it needs an isolated PostgreSQL database supplied
through TEST_DATABASE_URL. It creates and drops a random schema only; it never
uses application tables or configured production credentials.
"""

import asyncio
import os
from contextlib import asynccontextmanager
from datetime import datetime
from decimal import Decimal
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import NullPool

from orchestration.generation_finalization import GenerationFinalizationRepository

TEST_DATABASE_URL = os.getenv("TEST_DATABASE_URL")
pytestmark = pytest.mark.integration


def _async_database_url(url: str) -> str:
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


class _SchemaDatabase:
    def __init__(self, engine, schema: str):
        self._session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
        self._schema = schema

    @asynccontextmanager
    async def transaction(self):
        session = self._session_factory()
        try:
            async with session.begin():
                await session.execute(text(f'SET LOCAL search_path TO "{self._schema}"'))
                yield session
        finally:
            await session.close()

    @asynccontextmanager
    async def read_session(self):
        async with self.transaction() as session:
            yield session


def _article(project_id, article_id):
    now = datetime(2026, 7, 22, 12, 0, 0)
    return SimpleNamespace(
        id=article_id,
        project_id=project_id,
        content_plan_id=uuid4(),
        title="Concurrent finalization",
        content="<h2>Section</h2><p>Durable content.</p>",
        meta_description="Durable finalization integration test.",
        keywords=["durability"],
        quality_metrics=SimpleNamespace(
            word_count=900,
            readability_score=8.0,
            keyword_density={"durability": 0.02},
        ),
        total_tokens_used=1200,
        total_cost_usd=Decimal("0.04"),
        generation_time_seconds=4.2,
        created_at=now,
        updated_at=now,
    )


@pytest.mark.asyncio
@pytest.mark.skipif(not TEST_DATABASE_URL, reason="requires an isolated TEST_DATABASE_URL")
async def test_postgres_concurrent_finalization_commits_once():
    engine = create_async_engine(_async_database_url(TEST_DATABASE_URL), poolclass=NullPool)
    schema = f"generation_finalization_{uuid4().hex}"
    project_id = uuid4()
    task_id = f"generation-finalization-{uuid4()}"
    article_id = uuid4()

    try:
        async with engine.begin() as connection:
            await connection.execute(text(f'CREATE SCHEMA "{schema}"'))
            await connection.execute(
                text(
                    f'''CREATE TABLE "{schema}".projects (
                        id UUID PRIMARY KEY,
                        total_articles_generated INTEGER NOT NULL DEFAULT 0,
                        total_tokens_consumed INTEGER NOT NULL DEFAULT 0,
                        total_cost_usd NUMERIC(14, 6) NOT NULL DEFAULT 0,
                        last_active TIMESTAMP,
                        updated_at TIMESTAMP
                    )'''
                )
            )
            await connection.execute(
                text(
                    f'''CREATE TABLE "{schema}".generated_articles (
                        id UUID PRIMARY KEY,
                        generation_task_id VARCHAR(255),
                        project_id UUID NOT NULL,
                        content_plan_id UUID NOT NULL,
                        title VARCHAR(500) NOT NULL,
                        content TEXT NOT NULL,
                        meta_description TEXT,
                        keywords JSONB,
                        word_count INTEGER,
                        readability_score FLOAT,
                        keyword_density JSONB,
                        total_tokens_used INTEGER,
                        total_cost NUMERIC(14, 6),
                        generation_time FLOAT,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL
                    )'''
                )
            )
            await connection.execute(
                text(
                    f'''CREATE UNIQUE INDEX "uq_articles_{schema}" ON "{schema}".generated_articles
                    (generation_task_id) WHERE generation_task_id IS NOT NULL'''
                )
            )
            await connection.execute(
                text(
                    f'''CREATE TABLE "{schema}".task_results (
                        task_id VARCHAR(255) PRIMARY KEY,
                        status VARCHAR(50) NOT NULL,
                        result JSONB,
                        error TEXT,
                        traceback TEXT,
                        start_time TIMESTAMP,
                        end_time TIMESTAMP,
                        duration_seconds FLOAT,
                        updated_at TIMESTAMP
                    )'''
                )
            )
            await connection.execute(
                text(
                    f'''CREATE TABLE "{schema}".generation_outbox_events (
                        id UUID PRIMARY KEY,
                        task_id VARCHAR(255) NOT NULL,
                        article_id UUID NOT NULL,
                        event_type VARCHAR(80) NOT NULL,
                        payload JSONB NOT NULL,
                        status VARCHAR(32) NOT NULL,
                        attempt_count INTEGER NOT NULL DEFAULT 0,
                        last_error TEXT,
                        available_at TIMESTAMP,
                        completed_at TIMESTAMP,
                        created_at TIMESTAMP NOT NULL,
                        updated_at TIMESTAMP NOT NULL,
                        UNIQUE (task_id, event_type)
                    )'''
                )
            )
            await connection.execute(
                text(f'INSERT INTO "{schema}".projects (id) VALUES (:project_id)'),
                {"project_id": project_id},
            )
            await connection.execute(
                text(
                    f'''INSERT INTO "{schema}".task_results (task_id, status, start_time)
                    VALUES (:task_id, 'started', :start_time)'''
                ),
                {"task_id": task_id, "start_time": datetime(2026, 7, 22, 11, 59, 0)},
            )

        database = _SchemaDatabase(engine, schema)
        repository = GenerationFinalizationRepository(database)
        result = {"status": "success", "cost": 0.04}
        first, second = await asyncio.gather(
            repository.finalize(
                task_id=task_id,
                article=_article(project_id, article_id),
                task_result=result,
                language="fa",
            ),
            repository.finalize(
                task_id=task_id,
                article=_article(project_id, uuid4()),
                task_result=result,
                language="fa",
            ),
        )

        assert sorted((first.newly_finalized, second.newly_finalized)) == [False, True]

        first_export, second_export = await asyncio.gather(
            repository.claim_pending_export(task_id),
            repository.claim_pending_export(task_id),
        )
        claimed_exports = [item for item in (first_export, second_export) if item is not None]
        assert len(claimed_exports) == 1
        assert claimed_exports[0].attempt_number == 1

        async with engine.connect() as connection:
            await connection.execute(text(f'SET search_path TO "{schema}"'))
            article_count = await connection.execute(text("SELECT count(*) FROM generated_articles"))
            project = await connection.execute(
                text(
                    "SELECT total_articles_generated, total_tokens_consumed, total_cost_usd FROM projects"
                )
            )
            outbox = await connection.execute(
                text("SELECT count(*), status, attempt_count FROM generation_outbox_events GROUP BY status, attempt_count")
            )

        assert article_count.scalar_one() == 1
        assert project.one() == (1, 1200, Decimal("0.040000"))
        assert outbox.one() == (1, "processing", 1)
    finally:
        async with engine.begin() as connection:
            await connection.execute(text(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE'))
        await engine.dispose()
