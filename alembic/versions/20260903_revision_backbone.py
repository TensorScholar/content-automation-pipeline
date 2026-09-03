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
                captured_note text;
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
                    captured_note := NULLIF(current_setting('app.revision_note', true), '');
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
                    captured_note,
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

    # The simple FK proves that current_revision_id exists. This trigger also
    # proves ownership: the pointed revision must belong to the same article.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION validate_generated_article_current_revision()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.current_revision_id IS NULL THEN
                    RETURN NEW;
                END IF;

                PERFORM 1
                FROM article_revisions
                WHERE id = NEW.current_revision_id
                  AND article_id = NEW.id;

                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'Current revision % does not belong to article %',
                        NEW.current_revision_id,
                        NEW.id
                        USING ERRCODE = '23514';
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
            CREATE TRIGGER trg_generated_articles_validate_current_revision
            BEFORE INSERT OR UPDATE OF current_revision_id
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION validate_generated_article_current_revision()
            """
        )
    )

    # Revisions are append-only. Deletion remains available only because an
    # article delete must cascade its historical revisions safely.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION prevent_article_revision_update()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'Article revisions are immutable once inserted'
                    USING ERRCODE = '55000';
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_article_revisions_prevent_update
            BEFORE UPDATE ON article_revisions
            FOR EACH ROW
            EXECUTE FUNCTION prevent_article_revision_update()
            """
        )
    )


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_article_revisions_prevent_update ON article_revisions")
    op.execute("DROP FUNCTION IF EXISTS prevent_article_revision_update()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_validate_current_revision ON generated_articles"
    )
    op.execute("DROP FUNCTION IF EXISTS validate_generated_article_current_revision()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_capture_revision ON generated_articles"
    )
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
