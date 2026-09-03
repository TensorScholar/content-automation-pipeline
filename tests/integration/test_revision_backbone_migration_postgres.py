"""End-to-end PostgreSQL evidence for the Phase 4A revision migration.

The test creates a disposable database only in CI (or when TEST_DATABASE_URL is
explicitly provided), migrates to the pre-Phase-4A head, seeds legacy data, then
runs the real Alembic upgrade and exercises mixed-version writes.
"""

from __future__ import annotations

import os
import subprocess
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text

BASE_DATABASE_URL = os.getenv("TEST_DATABASE_URL") or (
    os.getenv("DATABASE_URL") if os.getenv("CI") == "true" else None
)
pytestmark = pytest.mark.integration


def _sync_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)


def _database_url(base_url: str, database_name: str) -> str:
    parsed = urlsplit(_sync_url(base_url))
    return urlunsplit(parsed._replace(path=f"/{database_name}"))


def _run_alembic(database_url: str, target: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    result = subprocess.run(
        ["alembic", "upgrade", target],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.skipif(not BASE_DATABASE_URL, reason="requires CI DATABASE_URL or TEST_DATABASE_URL")
def test_revision_backbone_backfills_and_captures_mixed_version_writes():
    database_name = f"revision_backbone_{uuid4().hex}"
    admin_engine = create_engine(_sync_url(BASE_DATABASE_URL), isolation_level="AUTOCOMMIT")
    database_url = _database_url(BASE_DATABASE_URL, database_name)
    target_engine = None

    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))

        _run_alembic(database_url, "20260801_001")
        target_engine = create_engine(database_url)

        project_id = uuid4()
        article_id = uuid4()
        historical_revision_id = uuid4()
        created_at = datetime(2026, 9, 1, 10, 0, 0)

        with target_engine.begin() as connection:
            connection.execute(
                text("INSERT INTO projects (id, name) VALUES (:id, 'Revision Test')"),
                {"id": project_id},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO generated_articles (
                        id, generation_task_id, project_id, title, content,
                        meta_description, keywords, word_count, created_at, updated_at
                    ) VALUES (
                        :id, :task_id, :project_id, :title, :content,
                        :meta_description, CAST(:keywords AS jsonb), :word_count,
                        :created_at, :updated_at
                    )
                    """
                ),
                {
                    "id": article_id,
                    "task_id": "generation-task-1",
                    "project_id": project_id,
                    "title": "Current article",
                    "content": "Current immutable payload",
                    "meta_description": "Current metadata",
                    "keywords": '["revision", "integrity"]',
                    "word_count": 3,
                    "created_at": created_at,
                    "updated_at": created_at + timedelta(hours=1),
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO article_revisions (
                        id, article_id, title, content, revision_note, word_count, created_at
                    ) VALUES (
                        :id, :article_id, :title, :content, :note, :word_count, :created_at
                    )
                    """
                ),
                {
                    "id": historical_revision_id,
                    "article_id": article_id,
                    "title": "Current article",
                    "content": "Historical payload",
                    "note": "Legacy manual edit snapshot",
                    "word_count": 2,
                    "created_at": created_at,
                },
            )

        _run_alembic(database_url, "head")

        with target_engine.connect() as connection:
            revisions = (
                connection.execute(
                    text(
                        """
                    SELECT id, revision_number, content, meta_description,
                           revision_source, snapshot_completeness, generation_task_id
                    FROM article_revisions
                    WHERE article_id = :article_id
                    ORDER BY revision_number
                    """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .all()
            )
            current_revision_id = connection.execute(
                text("SELECT current_revision_id FROM generated_articles WHERE id = :id"),
                {"id": article_id},
            ).scalar_one()

        assert len(revisions) == 2
        assert revisions[0]["id"] == historical_revision_id
        assert revisions[0]["revision_number"] == 1
        assert revisions[0]["content"] == "Historical payload"
        assert revisions[0]["meta_description"] is None
        assert revisions[0]["revision_source"] == "legacy_snapshot"
        assert revisions[0]["snapshot_completeness"] == "legacy_partial"

        assert revisions[1]["revision_number"] == 2
        assert revisions[1]["content"] == "Current immutable payload"
        assert revisions[1]["meta_description"] == "Current metadata"
        assert revisions[1]["revision_source"] == "migration_current_backfill"
        assert revisions[1]["snapshot_completeness"] == "complete"
        assert revisions[1]["generation_task_id"] == "generation-task-1"
        assert current_revision_id == revisions[1]["id"]

        # New-binary write: only the article projection is updated. The DB trigger
        # appends the immutable complete revision and advances the pointer.
        with target_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET content = 'New binary payload', word_count = 3, updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": article_id},
            )

        with target_engine.connect() as connection:
            current = (
                connection.execute(
                    text(
                        """
                    SELECT article.current_revision_id, revision.revision_number,
                           revision.content, revision.revision_source,
                           revision.snapshot_completeness
                    FROM generated_articles AS article
                    JOIN article_revisions AS revision
                      ON revision.id = article.current_revision_id
                    WHERE article.id = :id
                    """
                    ),
                    {"id": article_id},
                )
                .mappings()
                .one()
            )
        assert current["revision_number"] == 3
        assert current["content"] == "New binary payload"
        assert current["revision_source"] == "article_payload_update"
        assert current["snapshot_completeness"] == "complete"

        # Rolling-deploy compatibility: an old binary may still write a partial
        # pre-edit snapshot without revision_number, then mutate the article.
        # The BEFORE trigger numbers that legacy row; the article UPDATE trigger
        # then appends the new complete current revision.
        legacy_runtime_revision_id = uuid4()
        with target_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO article_revisions (
                        id, article_id, title, content, revision_note, word_count, created_at
                    ) VALUES (
                        :id, :article_id, 'Current article', 'New binary payload',
                        'Old binary compatibility snapshot', 3, NOW()
                    )
                    """
                ),
                {"id": legacy_runtime_revision_id, "article_id": article_id},
            )
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET content = 'Old binary edited payload', word_count = 4, updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": article_id},
            )

        with target_engine.connect() as connection:
            rows = (
                connection.execute(
                    text(
                        """
                    SELECT id, revision_number, content, revision_source, snapshot_completeness
                    FROM article_revisions
                    WHERE article_id = :article_id
                    ORDER BY revision_number
                    """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .all()
            )
            pointer = connection.execute(
                text("SELECT current_revision_id FROM generated_articles WHERE id = :id"),
                {"id": article_id},
            ).scalar_one()

        assert [row["revision_number"] for row in rows] == [1, 2, 3, 4, 5]
        assert rows[3]["id"] == legacy_runtime_revision_id
        assert rows[3]["revision_source"] == "legacy_application_snapshot"
        assert rows[3]["snapshot_completeness"] == "legacy_partial"
        assert rows[4]["content"] == "Old binary edited payload"
        assert rows[4]["snapshot_completeness"] == "complete"
        assert pointer == rows[4]["id"]

        # Independent article creation is also covered without application dual-write.
        second_article_id = uuid4()
        with target_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    INSERT INTO generated_articles (
                        id, project_id, title, content, word_count, created_at, updated_at
                    ) VALUES (
                        :id, :project_id, 'Second article', 'Second payload', 2, NOW(), NOW()
                    )
                    """
                ),
                {"id": second_article_id, "project_id": project_id},
            )
        with target_engine.connect() as connection:
            second = connection.execute(
                text(
                    """
                    SELECT revision.revision_number, revision.revision_source,
                           revision.snapshot_completeness
                    FROM generated_articles AS article
                    JOIN article_revisions AS revision
                      ON revision.id = article.current_revision_id
                    WHERE article.id = :id
                    """
                ),
                {"id": second_article_id},
            ).one()
        assert second == (1, "article_initial", "complete")

        # Concurrent payload updates serialize on the article row and must never
        # produce duplicate revision numbers.
        def update_payload(payload: str) -> None:
            with target_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE generated_articles
                        SET content = :payload, updated_at = NOW()
                        WHERE id = :id
                        """
                    ),
                    {"payload": payload, "id": article_id},
                )

        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [
                executor.submit(update_payload, "Concurrent payload A"),
                executor.submit(update_payload, "Concurrent payload B"),
            ]
            for future in futures:
                future.result(timeout=30)

        with target_engine.connect() as connection:
            revision_numbers = (
                connection.execute(
                    text(
                        """
                    SELECT revision_number
                    FROM article_revisions
                    WHERE article_id = :article_id
                    ORDER BY revision_number
                    """
                    ),
                    {"article_id": article_id},
                )
                .scalars()
                .all()
            )
        assert revision_numbers == list(range(1, len(revision_numbers) + 1))
        assert len(revision_numbers) == 7

        # The cyclic ownership relation must remain deletion-safe.
        with target_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM generated_articles WHERE id = :id"),
                {"id": second_article_id},
            )
        with target_engine.connect() as connection:
            orphan_count = connection.execute(
                text("SELECT count(*) FROM article_revisions WHERE article_id = :id"),
                {"id": second_article_id},
            ).scalar_one()
        assert orphan_count == 0

    finally:
        if target_engine is not None:
            target_engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(
                text(
                    """
                    SELECT pg_terminate_backend(pid)
                    FROM pg_stat_activity
                    WHERE datname = :database_name AND pid <> pg_backend_pid()
                    """
                ),
                {"database_name": database_name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database_name}"'))
        admin_engine.dispose()
