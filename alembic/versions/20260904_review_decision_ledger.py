"""add revision-bound immutable article review decision ledger

Revision ID: 20260904_001
Revises: 20260903_001
Create Date: 2026-09-04

Phase 4B1 is an expand-safe migration. Existing review columns on
``generated_articles`` remain a compatibility projection while immutable
``article_review_decisions`` becomes the durable audit ledger. Review decisions
are bound to the exact article revision current when the decision is persisted.

Native callers may set the transaction-local ``app.expected_review_revision_id``
setting. When present, stale review writes are rejected at the PostgreSQL
boundary. Missing expected revision remains temporarily compatible for old
binaries; Phase 4B2 contracts that compatibility window after all callers send
revision identity explicitly.
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260904_001"
down_revision = "20260903_001"
branch_labels = None
depends_on = None

TERMINAL_REVIEW_STATES = "'approved', 'rejected', 'changes_requested'"
ALL_REVIEW_STATES = f"'pending_review', {TERMINAL_REVIEW_STATES}"


def upgrade() -> None:
    # Fail before creating a partial ledger if legacy data already violates the
    # Phase 4A invariant required to bind an existing decision to a revision.
    op.execute(
        sa.text(
            f"""
            DO $$
            BEGIN
                IF EXISTS (
                    SELECT 1
                    FROM generated_articles
                    WHERE review_status IN ({TERMINAL_REVIEW_STATES})
                      AND current_revision_id IS NULL
                ) THEN
                    RAISE EXCEPTION
                        'Cannot migrate terminal article review state without current revision'
                        USING ERRCODE = '23514';
                END IF;
            END;
            $$
            """
        )
    )

    op.create_check_constraint(
        "ck_generated_articles_review_status",
        "generated_articles",
        f"review_status IN ({ALL_REVIEW_STATES})",
    )

    op.create_table(
        "article_review_decisions",
        sa.Column("id", PG_UUID(), primary_key=True),
        sa.Column(
            "article_id",
            PG_UUID(),
            sa.ForeignKey("generated_articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "article_revision_id",
            PG_UUID(),
            sa.ForeignKey("article_revisions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("decision_number", sa.Integer(), nullable=False),
        sa.Column("decision", sa.String(40), nullable=False),
        sa.Column("note", sa.Text(), nullable=True),
        # Reviewer identity is an audit snapshot, not a live relational owner.
        # Keeping this UUID free of an ON DELETE action means deleting a user
        # cannot mutate an immutable historical decision.
        sa.Column("reviewer_id_snapshot", PG_UUID(), nullable=True),
        sa.Column("reviewer_name_snapshot", sa.String(255), nullable=True),
        sa.Column("reviewer_email_snapshot", sa.String(255), nullable=True),
        sa.Column(
            "decision_source",
            sa.String(64),
            nullable=False,
            server_default="legacy_projection_write",
        ),
        sa.Column("decided_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("recorded_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.CheckConstraint(
            f"decision IN ({TERMINAL_REVIEW_STATES})",
            name="ck_article_review_decisions_decision",
        ),
        sa.CheckConstraint(
            "decision_number > 0",
            name="ck_article_review_decisions_number_positive",
        ),
    )
    op.create_index(
        "uq_article_review_decisions_article_number",
        "article_review_decisions",
        ["article_id", "decision_number"],
        unique=True,
    )
    op.create_index(
        "idx_article_review_decisions_revision",
        "article_review_decisions",
        ["article_revision_id", "recorded_at"],
    )
    op.create_index(
        "idx_article_review_decisions_article_recorded",
        "article_review_decisions",
        ["article_id", "recorded_at"],
    )

    op.add_column(
        "generated_articles",
        sa.Column("current_review_decision_id", PG_UUID(), nullable=True),
    )
    op.create_foreign_key(
        "fk_generated_articles_current_review_decision",
        "generated_articles",
        "article_review_decisions",
        ["current_review_decision_id"],
        ["id"],
        ondelete="SET NULL",
    )

    # Numbering and revision ownership are enforced inside PostgreSQL so all
    # writers participate in one decision identity contract.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION assign_article_review_decision_identity()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                next_decision_number integer;
                reviewer_name text;
                reviewer_email text;
            BEGIN
                PERFORM 1
                FROM generated_articles
                WHERE id = NEW.article_id
                FOR UPDATE;

                IF NOT FOUND THEN
                    RAISE EXCEPTION 'Cannot create review decision for missing article %', NEW.article_id
                        USING ERRCODE = '23503';
                END IF;

                PERFORM 1
                FROM article_revisions
                WHERE id = NEW.article_revision_id
                  AND article_id = NEW.article_id;

                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'Review decision revision % does not belong to article %',
                        NEW.article_revision_id,
                        NEW.article_id
                        USING ERRCODE = '23514';
                END IF;

                IF NEW.decision_number IS NULL THEN
                    SELECT COALESCE(MAX(decision_number), 0) + 1
                    INTO next_decision_number
                    FROM article_review_decisions
                    WHERE article_id = NEW.article_id;
                    NEW.decision_number := next_decision_number;
                END IF;

                IF NEW.reviewer_id_snapshot IS NOT NULL
                   AND (NEW.reviewer_name_snapshot IS NULL OR NEW.reviewer_email_snapshot IS NULL) THEN
                    SELECT full_name, email
                    INTO reviewer_name, reviewer_email
                    FROM users
                    WHERE id = NEW.reviewer_id_snapshot;

                    NEW.reviewer_name_snapshot := COALESCE(
                        NEW.reviewer_name_snapshot,
                        reviewer_name
                    );
                    NEW.reviewer_email_snapshot := COALESCE(
                        NEW.reviewer_email_snapshot,
                        reviewer_email
                    );
                END IF;

                IF NEW.decision_source IS NULL OR btrim(NEW.decision_source) = '' THEN
                    NEW.decision_source := 'legacy_projection_write';
                END IF;
                IF NEW.decided_at IS NULL THEN
                    NEW.decided_at := NOW();
                END IF;
                IF NEW.recorded_at IS NULL THEN
                    NEW.recorded_at := NOW();
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
            CREATE TRIGGER trg_article_review_decisions_assign_identity
            BEFORE INSERT ON article_review_decisions
            FOR EACH ROW
            EXECUTE FUNCTION assign_article_review_decision_identity()
            """
        )
    )

    # Preserve pre-migration terminal review state as one immutable decision.
    # Pending review is a projection state, not a historical decision event.
    op.execute(
        sa.text(
            f"""
            WITH inserted AS (
                INSERT INTO article_review_decisions (
                    id,
                    article_id,
                    article_revision_id,
                    decision_number,
                    decision,
                    note,
                    reviewer_id_snapshot,
                    reviewer_name_snapshot,
                    reviewer_email_snapshot,
                    decision_source,
                    decided_at,
                    recorded_at
                )
                SELECT
                    uuid_generate_v4(),
                    article.id,
                    article.current_revision_id,
                    1,
                    article.review_status,
                    article.review_note,
                    article.reviewed_by,
                    reviewer.full_name,
                    reviewer.email,
                    'legacy_review_backfill',
                    COALESCE(
                        article.reviewed_at,
                        article.review_updated_at,
                        article.updated_at,
                        article.created_at,
                        NOW()
                    ),
                    NOW()
                FROM generated_articles AS article
                LEFT JOIN users AS reviewer
                  ON reviewer.id = article.reviewed_by
                WHERE article.review_status IN ({TERMINAL_REVIEW_STATES})
                RETURNING id, article_id
            )
            UPDATE generated_articles AS article
            SET current_review_decision_id = inserted.id
            FROM inserted
            WHERE article.id = inserted.article_id
            """
        )
    )

    # A current-review pointer is valid only when the decision belongs to the
    # same article, exact current revision, and same projected decision state.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION validate_generated_article_current_review_decision()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.current_review_decision_id IS NULL THEN
                    RETURN NEW;
                END IF;

                PERFORM 1
                FROM article_review_decisions
                WHERE id = NEW.current_review_decision_id
                  AND article_id = NEW.id
                  AND article_revision_id = NEW.current_revision_id
                  AND decision = NEW.review_status;

                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'Current review decision % does not match article % current revision/state',
                        NEW.current_review_decision_id,
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
            CREATE TRIGGER trg_generated_articles_validate_current_review_decision
            BEFORE INSERT OR UPDATE OF current_review_decision_id
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION validate_generated_article_current_review_decision()
            """
        )
    )

    # A new revision always supersedes the previous review projection. This
    # also covers a direct current_revision_id switch.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION invalidate_article_review_on_revision_change()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF OLD.current_revision_id IS NOT DISTINCT FROM NEW.current_revision_id THEN
                    RETURN NEW;
                END IF;

                UPDATE generated_articles
                SET current_review_decision_id = NULL,
                    review_status = 'pending_review',
                    review_note = NULL,
                    reviewed_by = NULL,
                    reviewed_at = NULL,
                    review_updated_at = NOW()
                WHERE id = NEW.id
                  AND (
                    current_review_decision_id IS NOT NULL OR
                    review_status IS DISTINCT FROM 'pending_review' OR
                    review_note IS NOT NULL OR
                    reviewed_by IS NOT NULL OR
                    reviewed_at IS NOT NULL
                  );

                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_generated_articles_invalidate_review_on_revision
            AFTER UPDATE OF current_revision_id
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION invalidate_article_review_on_revision_change()
            """
        )
    )

    # Existing review columns stay writable during the expand window. Terminal
    # projection writes append immutable decisions. New callers provide the
    # revision they actually reviewed; stale writes fail at the DB boundary.
    op.execute(
        sa.text(
            f"""
            CREATE OR REPLACE FUNCTION capture_article_review_decision_from_projection()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                expected_revision_text text;
                expected_revision uuid;
                captured_decision_id uuid;
                captured_source text;
            BEGIN
                IF NOT (
                    OLD.review_status IS DISTINCT FROM NEW.review_status OR
                    OLD.review_note IS DISTINCT FROM NEW.review_note OR
                    OLD.reviewed_by IS DISTINCT FROM NEW.reviewed_by OR
                    OLD.reviewed_at IS DISTINCT FROM NEW.reviewed_at
                ) THEN
                    RETURN NEW;
                END IF;

                IF NEW.review_status = 'pending_review' THEN
                    UPDATE generated_articles
                    SET current_review_decision_id = NULL
                    WHERE id = NEW.id
                      AND current_review_decision_id IS NOT NULL;
                    RETURN NEW;
                END IF;

                IF NEW.review_status NOT IN ({TERMINAL_REVIEW_STATES}) THEN
                    RAISE EXCEPTION 'Unsupported article review status %', NEW.review_status
                        USING ERRCODE = '23514';
                END IF;

                IF NEW.current_revision_id IS NULL THEN
                    RAISE EXCEPTION 'Cannot review article % without a current revision', NEW.id
                        USING ERRCODE = '23514';
                END IF;

                expected_revision_text := NULLIF(
                    current_setting('app.expected_review_revision_id', true),
                    ''
                );
                IF expected_revision_text IS NOT NULL THEN
                    BEGIN
                        expected_revision := expected_revision_text::uuid;
                    EXCEPTION
                        WHEN invalid_text_representation THEN
                            RAISE EXCEPTION 'Invalid expected review revision id %', expected_revision_text
                                USING ERRCODE = '22023';
                    END;

                    IF expected_revision IS DISTINCT FROM NEW.current_revision_id THEN
                        RAISE EXCEPTION
                            'Stale review revision: expected %, current %',
                            expected_revision,
                            NEW.current_revision_id
                            USING ERRCODE = '40001';
                    END IF;
                END IF;

                captured_source := COALESCE(
                    NULLIF(current_setting('app.review_decision_source', true), ''),
                    'legacy_projection_write'
                );

                INSERT INTO article_review_decisions (
                    id,
                    article_id,
                    article_revision_id,
                    decision_number,
                    decision,
                    note,
                    reviewer_id_snapshot,
                    decision_source,
                    decided_at,
                    recorded_at
                ) VALUES (
                    uuid_generate_v4(),
                    NEW.id,
                    NEW.current_revision_id,
                    NULL,
                    NEW.review_status,
                    NEW.review_note,
                    NEW.reviewed_by,
                    captured_source,
                    COALESCE(NEW.reviewed_at, NOW()),
                    NOW()
                )
                RETURNING id INTO captured_decision_id;

                UPDATE generated_articles
                SET current_review_decision_id = captured_decision_id
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
            CREATE TRIGGER trg_generated_articles_capture_review_decision
            AFTER UPDATE OF review_status, review_note, reviewed_by, reviewed_at, review_updated_at
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION capture_article_review_decision_from_projection()
            """
        )
    )

    # Decisions are append-only audit events.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION prevent_article_review_decision_update()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'Article review decisions are immutable once inserted'
                    USING ERRCODE = '55000';
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_article_review_decisions_prevent_update
            BEFORE UPDATE ON article_review_decisions
            FOR EACH ROW
            EXECUTE FUNCTION prevent_article_review_decision_update()
            """
        )
    )

    # Terminal review state must always have one current decision bound to the
    # exact current revision. A DEFERRABLE constraint trigger allows the legacy
    # projection UPDATE and its AFTER-trigger decision insert to complete within
    # the same transaction while still failing inconsistent commits.
    op.execute(
        sa.text(
            f"""
            CREATE OR REPLACE FUNCTION validate_article_review_projection_consistency()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            DECLARE
                current_row record;
            BEGIN
                SELECT review_status, current_revision_id, current_review_decision_id
                INTO current_row
                FROM generated_articles
                WHERE id = NEW.id;

                IF NOT FOUND THEN
                    RETURN NULL;
                END IF;

                IF current_row.review_status = 'pending_review' THEN
                    IF current_row.current_review_decision_id IS NOT NULL THEN
                        RAISE EXCEPTION
                            'Pending article % cannot retain a current review decision',
                            NEW.id
                            USING ERRCODE = '23514';
                    END IF;
                    RETURN NULL;
                END IF;

                IF current_row.review_status NOT IN ({TERMINAL_REVIEW_STATES}) THEN
                    RAISE EXCEPTION 'Unsupported article review status %', current_row.review_status
                        USING ERRCODE = '23514';
                END IF;

                IF current_row.current_review_decision_id IS NULL THEN
                    RAISE EXCEPTION
                        'Terminal article review state requires a current review decision for article %',
                        NEW.id
                        USING ERRCODE = '23514';
                END IF;

                PERFORM 1
                FROM article_review_decisions
                WHERE id = current_row.current_review_decision_id
                  AND article_id = NEW.id
                  AND article_revision_id = current_row.current_revision_id
                  AND decision = current_row.review_status;

                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'Article % review projection does not match its immutable current decision',
                        NEW.id
                        USING ERRCODE = '23514';
                END IF;
                RETURN NULL;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE CONSTRAINT TRIGGER trg_generated_articles_review_projection_consistency
            AFTER INSERT OR UPDATE OF review_status, current_revision_id, current_review_decision_id
            ON generated_articles
            DEFERRABLE INITIALLY DEFERRED
            FOR EACH ROW
            EXECUTE FUNCTION validate_article_review_projection_consistency()
            """
        )
    )


def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_review_projection_consistency "
        "ON generated_articles"
    )
    op.execute("DROP FUNCTION IF EXISTS validate_article_review_projection_consistency()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_article_review_decisions_prevent_update "
        "ON article_review_decisions"
    )
    op.execute("DROP FUNCTION IF EXISTS prevent_article_review_decision_update()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_capture_review_decision "
        "ON generated_articles"
    )
    op.execute("DROP FUNCTION IF EXISTS capture_article_review_decision_from_projection()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_invalidate_review_on_revision "
        "ON generated_articles"
    )
    op.execute("DROP FUNCTION IF EXISTS invalidate_article_review_on_revision_change()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_validate_current_review_decision "
        "ON generated_articles"
    )
    op.execute("DROP FUNCTION IF EXISTS validate_generated_article_current_review_decision()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_article_review_decisions_assign_identity "
        "ON article_review_decisions"
    )
    op.execute("DROP FUNCTION IF EXISTS assign_article_review_decision_identity()")

    op.drop_constraint(
        "fk_generated_articles_current_review_decision",
        "generated_articles",
        type_="foreignkey",
    )
    op.drop_column("generated_articles", "current_review_decision_id")
    op.drop_index(
        "idx_article_review_decisions_article_recorded",
        table_name="article_review_decisions",
    )
    op.drop_index(
        "idx_article_review_decisions_revision",
        table_name="article_review_decisions",
    )
    op.drop_index(
        "uq_article_review_decisions_article_number",
        table_name="article_review_decisions",
    )
    op.drop_table("article_review_decisions")
    op.drop_constraint(
        "ck_generated_articles_review_status",
        "generated_articles",
        type_="check",
    )
