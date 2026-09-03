from __future__ import annotations

import re
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def write(path: str, content: str) -> None:
    target = ROOT / path
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(content, encoding="utf-8")


def replace_once(path: str, old: str, new: str) -> None:
    content = read(path)
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one literal match, found {count}")
    write(path, content.replace(old, new, 1))


def regex_replace_once(path: str, pattern: str, replacement: str) -> None:
    content = read(path)
    updated, count = re.subn(pattern, replacement, content, count=1, flags=re.DOTALL)
    if count != 1:
        raise RuntimeError(f"{path}: expected exactly one regex match, found {count}: {pattern}")
    write(path, updated)


# ---------------------------------------------------------------------------
# SQLAlchemy metadata: expose the new revision identity contract to runtime.
# ---------------------------------------------------------------------------
replace_once(
    "infrastructure/schema.py",
    "    Boolean,\n    Column,",
    "    Boolean,\n    CheckConstraint,\n    Column,",
)
replace_once(
    "infrastructure/schema.py",
    '    Column("generation_task_id", String(255)),\n',
    '''    Column("generation_task_id", String(255)),
    Column(
        "current_revision_id",
        PG_UUID,
        ForeignKey(
            "article_revisions.id",
            name="fk_generated_articles_current_revision",
            ondelete="SET NULL",
            use_alter=True,
        ),
    ),
''',
)
regex_replace_once(
    "infrastructure/schema.py",
    r"# Article Revisions Table\narticle_revisions_table = Table\(.*?\n\)\n\n# Projects Table",
    '''# Article Revisions Table
article_revisions_table = Table(
    "article_revisions",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "article_id",
        PG_UUID,
        ForeignKey("generated_articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("revision_number", Integer, nullable=False),
    Column("title", String(500), nullable=False),
    Column("content", Text),
    Column("meta_description", String(500)),
    Column("keywords", JSONB),
    Column("revision_note", Text),
    Column("word_count", Integer),
    Column(
        "revision_source",
        String(64),
        nullable=False,
        server_default="legacy_application_snapshot",
    ),
    Column(
        "snapshot_completeness",
        String(32),
        nullable=False,
        server_default="legacy_partial",
    ),
    Column("generation_task_id", String(255)),
    Column("created_at", DateTime, default=func.now(), index=True),
    CheckConstraint(
        "revision_number > 0",
        name="ck_article_revisions_revision_number_positive",
    ),
    CheckConstraint(
        "snapshot_completeness IN ('legacy_partial', 'complete')",
        name="ck_article_revisions_snapshot_completeness",
    ),
    Index("idx_revisions_article_created", "article_id", "created_at"),
    Index(
        "uq_article_revisions_article_number",
        "article_id",
        "revision_number",
        unique=True,
    ),
)

# Projects Table''',
)


# ---------------------------------------------------------------------------
# Repository semantics: the database trigger now owns snapshot capture.
# Keep the application write path atomic and expose current revision identity.
# ---------------------------------------------------------------------------
regex_replace_once(
    "knowledge/article_repository.py",
    r"    async def update_content_with_revision\(.*?\n    async def get_review_state\(",
    '''    async def update_content_with_revision(
        self,
        *,
        article_id: UUID,
        content: str,
        word_count: int,
        revision_note: Optional[str],
    ) -> Optional[Dict[str, Any]]:
        """Atomically apply a manual edit; DB triggers append the new immutable revision."""
        del revision_note  # Revision identity is payload-derived; edit events are tracked separately.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            current_result = await session.execute(
                select(generated_articles_table.c.id)
                .where(generated_articles_table.c.id == article_id)
                .with_for_update()
            )
            current = current_result.scalar_one_or_none()
            if current is None:
                return None

            updated_result = await session.execute(
                update(generated_articles_table)
                .where(generated_articles_table.c.id == article_id)
                .values(
                    content=content,
                    word_count=word_count,
                    readability_score=None,
                    keyword_density={},
                    review_status="pending_review",
                    review_note=None,
                    reviewed_by=None,
                    reviewed_at=None,
                    review_updated_at=now,
                    updated_at=now,
                )
                .returning(generated_articles_table)
            )
            updated = updated_result.mappings().one()

        if self.redis_client:
            await self.redis_client.delete_pattern("article:get_by_id:*")
        return dict(updated)

    async def get_review_state(''',
)
regex_replace_once(
    "knowledge/article_repository.py",
    r"    async def get_article_history\(.*?\n    async def export_articles\(",
    '''    async def get_article_history(self, article_id: UUID) -> Optional[Dict[str, Any]]:
        """Return the mutable article projection plus immutable prior revisions."""
        current_revision = article_revisions_table.alias("current_revision")
        current_query = (
            select(
                generated_articles_table.c.id,
                generated_articles_table.c.current_revision_id,
                generated_articles_table.c.title,
                generated_articles_table.c.content,
                generated_articles_table.c.meta_description,
                generated_articles_table.c.keywords,
                generated_articles_table.c.created_at,
                generated_articles_table.c.word_count,
                current_revision.c.revision_number.label("revision_number"),
                current_revision.c.revision_source.label("revision_source"),
                current_revision.c.snapshot_completeness.label("snapshot_completeness"),
                current_revision.c.generation_task_id.label("revision_generation_task_id"),
            )
            .select_from(
                generated_articles_table.outerjoin(
                    current_revision,
                    generated_articles_table.c.current_revision_id == current_revision.c.id,
                )
            )
            .where(generated_articles_table.c.id == article_id)
        )
        current = await self.db.fetch_one(current_query)
        if not current:
            return None

        revisions_query = select(
            article_revisions_table.c.id,
            article_revisions_table.c.revision_number,
            article_revisions_table.c.title,
            article_revisions_table.c.content,
            article_revisions_table.c.meta_description,
            article_revisions_table.c.keywords,
            article_revisions_table.c.created_at,
            article_revisions_table.c.revision_note,
            article_revisions_table.c.word_count,
            article_revisions_table.c.revision_source,
            article_revisions_table.c.snapshot_completeness,
            article_revisions_table.c.generation_task_id,
        ).where(article_revisions_table.c.article_id == article_id)
        if current["current_revision_id"] is not None:
            revisions_query = revisions_query.where(
                article_revisions_table.c.id != current["current_revision_id"]
            )
        revisions_query = revisions_query.order_by(
            article_revisions_table.c.revision_number.desc(),
            article_revisions_table.c.created_at.desc(),
        )
        revisions = await self.db.fetch_all(revisions_query)

        return {
            "current_version": {
                "id": str(current["id"]),
                "revision_id": (
                    str(current["current_revision_id"])
                    if current["current_revision_id"]
                    else None
                ),
                "revision_number": current["revision_number"],
                "title": current["title"],
                "content": current["content"],
                "meta_description": current["meta_description"],
                "keywords": current["keywords"],
                "created_at": current["created_at"],
                "word_count": current["word_count"],
                "revision_source": current["revision_source"],
                "snapshot_completeness": current["snapshot_completeness"],
                "generation_task_id": current["revision_generation_task_id"],
            },
            "revisions": [
                {
                    "id": str(rev["id"]),
                    "revision_number": rev["revision_number"],
                    "title": rev["title"],
                    "content": rev["content"],
                    "meta_description": rev["meta_description"],
                    "keywords": rev["keywords"],
                    "revision_note": rev["revision_note"],
                    "created_at": rev["created_at"],
                    "word_count": rev["word_count"],
                    "revision_source": rev["revision_source"],
                    "snapshot_completeness": rev["snapshot_completeness"],
                    "generation_task_id": rev["generation_task_id"],
                }
                for rev in revisions
            ],
            "total_revisions": len(revisions),
        }

    async def export_articles(''',
)
regex_replace_once(
    "knowledge/article_repository.py",
    r"\n    async def create_revision\(.*?\n    async def save_content_plan\(",
    "\n    async def save_content_plan(",
)


# ---------------------------------------------------------------------------
# Revision requests no longer write a redundant pre-regeneration snapshot.
# current_revision_id already names the immutable base payload.
# ---------------------------------------------------------------------------
regex_replace_once(
    "services/content_service.py",
    r"\n        # Save current version as a revision snapshot before regeneration\n.*?\n        # Dispatch regeneration task via Celery with feedback as custom instructions",
    '''
        # The current article payload is already represented by current_revision_id.
        # Regeneration lineage will bind the generation task to that revision in Phase 4D.

        # Dispatch regeneration task via Celery with feedback as custom instructions''',
)


migration = textwrap.dedent(r'''\
"""establish immutable article revision identity backbone

Revision ID: 20260903_001
Revises: 20260801_001
Create Date: 2026-09-03

This is an expand-safe migration. Existing article rows remain the mutable
projection used by the current application, while article_revisions becomes
the immutable identity ledger. PostgreSQL triggers guarantee that old binaries
participating in a rolling deployment cannot bypass revision capture.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260903_001"
down_revision = "20260801_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generated_articles",
        sa.Column("current_revision_id", PG_UUID(), nullable=True),
    )
    op.add_column(
        "article_revisions",
        sa.Column("revision_number", sa.Integer(), nullable=True),
    )
    op.add_column(
        "article_revisions",
        sa.Column("meta_description", sa.String(500), nullable=True),
    )
    op.add_column(
        "article_revisions",
        sa.Column("keywords", JSONB, nullable=True),
    )
    op.add_column(
        "article_revisions",
        sa.Column(
            "revision_source",
            sa.String(64),
            nullable=False,
            server_default="legacy_application_snapshot",
        ),
    )
    op.add_column(
        "article_revisions",
        sa.Column(
            "snapshot_completeness",
            sa.String(32),
            nullable=False,
            server_default="legacy_partial",
        ),
    )
    op.add_column(
        "article_revisions",
        sa.Column("generation_task_id", sa.String(255), nullable=True),
    )

    # Historical rows are preserved exactly as captured. Their missing SEO
    # metadata is not reconstructed or invented.
    op.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT
                    id,
                    ROW_NUMBER() OVER (
                        PARTITION BY article_id
                        ORDER BY created_at ASC NULLS FIRST, id ASC
                    ) AS revision_number
                FROM article_revisions
            )
            UPDATE article_revisions AS revision
            SET revision_number = ranked.revision_number,
                revision_source = 'legacy_snapshot',
                snapshot_completeness = 'legacy_partial'
            FROM ranked
            WHERE revision.id = ranked.id
            """
        )
    )

    # Every article receives one complete immutable snapshot of its current
    # payload. Historical snapshots remain earlier numbered revisions.
    op.execute(
        sa.text(
            """
            WITH next_numbers AS (
                SELECT
                    article.id AS article_id,
                    COALESCE(MAX(revision.revision_number), 0) + 1 AS revision_number
                FROM generated_articles AS article
                LEFT JOIN article_revisions AS revision
                    ON revision.article_id = article.id
                GROUP BY article.id
            ),
            inserted AS (
                INSERT INTO article_revisions (
                    id,
                    article_id,
                    revision_number,
                    title,
                    content,
                    meta_description,
                    keywords,
                    revision_note,
                    word_count,
                    revision_source,
                    snapshot_completeness,
                    generation_task_id,
                    created_at
                )
                SELECT
                    uuid_generate_v4(),
                    article.id,
                    next_numbers.revision_number,
                    article.title,
                    article.content,
                    article.meta_description,
                    article.keywords,
                    'Current payload captured during immutable revision migration',
                    article.word_count,
                    'migration_current_backfill',
                    'complete',
                    article.generation_task_id,
                    COALESCE(article.updated_at, article.created_at, NOW())
                FROM generated_articles AS article
                JOIN next_numbers ON next_numbers.article_id = article.id
                RETURNING id, article_id
            )
            UPDATE generated_articles AS article
            SET current_revision_id = inserted.id
            FROM inserted
            WHERE article.id = inserted.article_id
            """
        )
    )

    op.alter_column(
        "article_revisions",
        "revision_number",
        existing_type=sa.Integer(),
        nullable=False,
    )
    op.create_check_constraint(
        "ck_article_revisions_revision_number_positive",
        "article_revisions",
        "revision_number > 0",
    )
    op.create_check_constraint(
        "ck_article_revisions_snapshot_completeness",
        "article_revisions",
        "snapshot_completeness IN ('legacy_partial', 'complete')",
    )
    op.create_index(
        "uq_article_revisions_article_number",
        "article_revisions",
        ["article_id", "revision_number"],
        unique=True,
    )
    op.create_foreign_key(
        "fk_generated_articles_current_revision",
        "generated_articles",
        "article_revisions",
        ["current_revision_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Old binaries may still insert article_revisions without revision_number.
    # Locking the parent row serializes numbering per article and closes the
    # mixed-version race without a global sequence or application convention.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION assign_article_revision_identity()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                next_revision_number integer;
            BEGIN
                PERFORM 1
                FROM generated_articles
                WHERE id = NEW.article_id
                FOR UPDATE;

                IF NOT FOUND THEN
                    RAISE EXCEPTION 'Cannot create revision for missing article %', NEW.article_id
                        USING ERRCODE = '23503';
                END IF;

                IF NEW.revision_number IS NULL THEN
                    SELECT COALESCE(MAX(revision_number), 0) + 1
                    INTO next_revision_number
                    FROM article_revisions
                    WHERE article_id = NEW.article_id;
                    NEW.revision_number := next_revision_number;
                END IF;

                IF NEW.revision_source IS NULL THEN
                    NEW.revision_source := 'legacy_application_snapshot';
                END IF;
                IF NEW.snapshot_completeness IS NULL THEN
                    NEW.snapshot_completeness := 'legacy_partial';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_article_revisions_assign_identity
            BEFORE INSERT ON article_revisions
            FOR EACH ROW
            EXECUTE FUNCTION assign_article_revision_identity()
            """
        )
    )

    # The article row remains the mutable projection for compatibility, but
    # every persisted payload change appends a complete revision in the same
    # transaction and advances current_revision_id.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION capture_generated_article_revision()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                captured_revision_id uuid;
                captured_source text;
                captured_at timestamp;
            BEGIN
                IF TG_OP = 'UPDATE' AND NOT (
                    OLD.title IS DISTINCT FROM NEW.title OR
                    OLD.content IS DISTINCT FROM NEW.content OR
                    OLD.meta_description IS DISTINCT FROM NEW.meta_description OR
                    OLD.keywords IS DISTINCT FROM NEW.keywords OR
                    OLD.word_count IS DISTINCT FROM NEW.word_count
                ) THEN
                    RETURN NEW;
                END IF;

                IF TG_OP = 'INSERT' THEN
                    captured_source := CASE
                        WHEN NEW.generation_task_id IS NOT NULL THEN 'generation_initial'
                        ELSE 'article_initial'
                    END;
                    captured_at := COALESCE(NEW.created_at, NOW());
                ELSE
                    captured_source := 'article_payload_update';
                    captured_at := COALESCE(NEW.updated_at, NOW());
                END IF;

                INSERT INTO article_revisions (
                    id,
                    article_id,
                    revision_number,
                    title,
                    content,
                    meta_description,
                    keywords,
                    revision_note,
                    word_count,
                    revision_source,
                    snapshot_completeness,
                    generation_task_id,
                    created_at
                ) VALUES (
                    uuid_generate_v4(),
                    NEW.id,
                    NULL,
                    NEW.title,
                    NEW.content,
                    NEW.meta_description,
                    NEW.keywords,
                    NULL,
                    NEW.word_count,
                    captured_source,
                    'complete',
                    NEW.generation_task_id,
                    captured_at
                )
                RETURNING id INTO captured_revision_id;

                UPDATE generated_articles
                SET current_revision_id = captured_revision_id
                WHERE id = NEW.id;

                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_generated_articles_capture_revision
            AFTER INSERT OR UPDATE OF title, content, meta_description, keywords, word_count
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION capture_generated_article_revision()
            """
        )
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_generated_articles_capture_revision ON generated_articles")
    op.execute("DROP FUNCTION IF EXISTS capture_generated_article_revision()")
    op.execute("DROP TRIGGER IF EXISTS trg_article_revisions_assign_identity ON article_revisions")
    op.execute("DROP FUNCTION IF EXISTS assign_article_revision_identity()")

    op.drop_constraint(
        "fk_generated_articles_current_revision",
        "generated_articles",
        type_="foreignkey",
    )
    op.drop_index(
        "uq_article_revisions_article_number",
        table_name="article_revisions",
    )
    op.drop_constraint(
        "ck_article_revisions_snapshot_completeness",
        "article_revisions",
        type_="check",
    )
    op.drop_constraint(
        "ck_article_revisions_revision_number_positive",
        "article_revisions",
        type_="check",
    )
    op.drop_column("generated_articles", "current_revision_id")
    op.drop_column("article_revisions", "generation_task_id")
    op.drop_column("article_revisions", "snapshot_completeness")
    op.drop_column("article_revisions", "revision_source")
    op.drop_column("article_revisions", "keywords")
    op.drop_column("article_revisions", "meta_description")
    op.drop_column("article_revisions", "revision_number")
''')
write("alembic/versions/20260903_revision_backbone.py", migration)


truth_test = textwrap.dedent(r'''\
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_revision_backbone_is_explicit_in_schema_and_migration():
    schema = _read("infrastructure/schema.py")
    migration = _read("alembic/versions/20260903_revision_backbone.py")

    assert 'Column("current_revision_id"' in schema
    assert 'Column("revision_number", Integer, nullable=False)' in schema
    assert 'Column("meta_description", String(500))' in schema
    assert 'Column("keywords", JSONB)' in schema
    assert '"uq_article_revisions_article_number"' in schema

    assert 'revision = "20260903_001"' in migration
    assert 'down_revision = "20260801_001"' in migration
    assert "migration_current_backfill" in migration
    assert "assign_article_revision_identity" in migration
    assert "capture_generated_article_revision" in migration
    assert "trg_generated_articles_capture_revision" in migration
    assert "current_revision_id = captured_revision_id" in migration
    assert "legacy_partial" in migration
    assert "snapshot_completeness = 'complete'" not in migration  # never fabricates historical completeness


def test_application_no_longer_writes_redundant_pre_edit_snapshots():
    repository = _read("knowledge/article_repository.py")
    service = _read("services/content_service.py")

    update_method = repository.split("async def update_content_with_revision", 1)[1].split(
        "async def get_review_state", 1
    )[0]
    assert "insert(article_revisions_table)" not in update_method
    assert "DB triggers append the new immutable revision" in update_method
    assert "async def create_revision" not in repository
    assert "create_revision(" not in service


def test_history_exposes_current_revision_identity_without_duplicating_it():
    repository = _read("knowledge/article_repository.py")

    assert '"revision_id"' in repository
    assert 'generated_articles_table.c.current_revision_id == current_revision.c.id' in repository
    assert 'article_revisions_table.c.id != current["current_revision_id"]' in repository
    assert 'article_revisions_table.c.revision_number.desc()' in repository
''')
write("tests/test_revision_backbone_truth.py", truth_test)


integration_test = textwrap.dedent(r'''\
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
            revisions = connection.execute(
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
            ).mappings().all()
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
            current = connection.execute(
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
            ).mappings().one()
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
            rows = connection.execute(
                text(
                    """
                    SELECT id, revision_number, content, revision_source, snapshot_completeness
                    FROM article_revisions
                    WHERE article_id = :article_id
                    ORDER BY revision_number
                    """
                ),
                {"article_id": article_id},
            ).mappings().all()
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
            revision_numbers = connection.execute(
                text(
                    """
                    SELECT revision_number
                    FROM article_revisions
                    WHERE article_id = :article_id
                    ORDER BY revision_number
                    """
                ),
                {"article_id": article_id},
            ).scalars().all()
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
''')
write("tests/integration/test_revision_backbone_migration_postgres.py", integration_test)

print("Phase 4A implementation patch staged successfully")
