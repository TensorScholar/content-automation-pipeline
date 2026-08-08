"""add reliable publishing and Search Console integration

Revision ID: 20260801_001
Revises: 20260721_001
Create Date: 2026-08-01
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260801_001"
down_revision = "20260721_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("generated_articles", sa.Column("publish_task_id", sa.String(255)))
    op.add_column("generated_articles", sa.Column("publish_requested_status", sa.String(40)))
    op.add_column("generated_articles", sa.Column("publish_scheduled_at", sa.DateTime()))
    op.add_column("generated_articles", sa.Column("publish_lease_expires_at", sa.DateTime()))

    op.add_column(
        "publishing_attempts",
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
    )
    op.add_column("publishing_attempts", sa.Column("lease_expires_at", sa.DateTime()))
    op.add_column(
        "publishing_attempts",
        sa.Column("warnings", JSONB, nullable=False, server_default=sa.text("'[]'::jsonb")),
    )
    op.add_column("publishing_attempts", sa.Column("remote_verified_at", sa.DateTime()))
    op.add_column(
        "publishing_attempts",
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    # Backfill historical attempts before enforcing active-state semantics and
    # the successful-idempotency uniqueness invariant.
    op.execute(
        sa.text(
            """
            UPDATE publishing_attempts
            SET status = CASE
                WHEN success = true THEN 'succeeded'
                WHEN finished_at IS NOT NULL THEN 'failed'
                ELSE 'failed'
            END,
            updated_at = COALESCE(finished_at, started_at, NOW())
            """
        )
    )
    op.execute(
        sa.text(
            """
            WITH ranked AS (
                SELECT id,
                       ROW_NUMBER() OVER (
                           PARTITION BY article_id, idempotency_key
                           ORDER BY finished_at DESC NULLS LAST, started_at DESC, id DESC
                       ) AS rn
                FROM publishing_attempts
                WHERE success = true
            )
            UPDATE publishing_attempts AS attempt
            SET success = false,
                status = 'superseded',
                error_category = COALESCE(error_category, 'historical_duplicate'),
                error_message = COALESCE(
                    error_message,
                    'Superseded during P0 idempotency migration'
                )
            FROM ranked
            WHERE attempt.id = ranked.id AND ranked.rn > 1
            """
        )
    )
    op.execute(
        sa.text(
            """
            UPDATE generated_articles
            SET publish_status = 'publish_failed',
                publish_error_category = COALESCE(publish_error_category, 'stale_legacy_attempt'),
                publish_error_message = COALESCE(
                    publish_error_message,
                    'Legacy publishing state was incomplete during reliability migration'
                ),
                publish_updated_at = NOW()
            WHERE publish_status = 'publishing'
            """
        )
    )
    op.create_index(
        "idx_publishing_attempts_status_lease",
        "publishing_attempts",
        ["status", "lease_expires_at"],
    )
    op.create_index(
        "uq_publishing_success_idempotency",
        "publishing_attempts",
        ["article_id", "idempotency_key"],
        unique=True,
        postgresql_where=sa.text("success = true"),
    )

    op.create_table(
        "search_console_oauth_states",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column("state_hash", sa.String(64), nullable=False, unique=True),
        sa.Column("project_id", PG_UUID, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("user_id", PG_UUID, sa.ForeignKey("users.id", ondelete="CASCADE"), nullable=False),
        sa.Column("expires_at", sa.DateTime(), nullable=False),
        sa.Column("consumed_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_search_console_oauth_expiry",
        "search_console_oauth_states",
        ["expires_at", "consumed_at"],
    )

    op.create_table(
        "search_console_connections",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column("project_id", PG_UUID, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False, unique=True),
        sa.Column("encrypted_refresh_token", sa.Text()),
        sa.Column("scope", sa.Text(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="connected"),
        sa.Column("selected_site_url", sa.String(1000)),
        sa.Column("permission_level", sa.String(80)),
        sa.Column("last_sync_at", sa.DateTime()),
        sa.Column("last_error_category", sa.String(100)),
        sa.Column("last_error_message", sa.Text()),
        sa.Column("connected_by", PG_UUID, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "idx_search_console_connections_status",
        "search_console_connections",
        ["status"],
    )

    op.create_table(
        "search_console_properties",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column("connection_id", PG_UUID, sa.ForeignKey("search_console_connections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", PG_UUID, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("site_url", sa.String(1000), nullable=False),
        sa.Column("permission_level", sa.String(80), nullable=False),
        sa.Column("last_seen_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("connection_id", "site_url", name="uq_search_console_connection_site"),
    )
    op.create_index(
        "idx_search_console_properties_project",
        "search_console_properties",
        ["project_id", "last_seen_at"],
    )

    op.create_table(
        "search_console_sync_runs",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column("connection_id", PG_UUID, sa.ForeignKey("search_console_connections.id", ondelete="CASCADE"), nullable=False),
        sa.Column("project_id", PG_UUID, sa.ForeignKey("projects.id", ondelete="CASCADE"), nullable=False),
        sa.Column("site_url", sa.String(1000), nullable=False),
        sa.Column("date_from", sa.Date(), nullable=False),
        sa.Column("date_to", sa.Date(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False, server_default="queued"),
        sa.Column("task_id", sa.String(255)),
        sa.Column("row_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("pages_fetched", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("truncated", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("retry_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("error_category", sa.String(100)),
        sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime()),
        sa.Column("finished_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint(
            "project_id", "site_url", "date_from", "date_to",
            name="uq_search_console_sync_window",
        ),
    )
    op.create_index(
        "idx_search_console_sync_project_created",
        "search_console_sync_runs",
        ["project_id", "created_at"],
    )
    op.create_index(
        "idx_search_console_sync_status",
        "search_console_sync_runs",
        ["status", "created_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_search_console_sync_status", table_name="search_console_sync_runs")
    op.drop_index("idx_search_console_sync_project_created", table_name="search_console_sync_runs")
    op.drop_table("search_console_sync_runs")
    op.drop_index("idx_search_console_properties_project", table_name="search_console_properties")
    op.drop_table("search_console_properties")
    op.drop_index("idx_search_console_connections_status", table_name="search_console_connections")
    op.drop_table("search_console_connections")
    op.drop_index("idx_search_console_oauth_expiry", table_name="search_console_oauth_states")
    op.drop_table("search_console_oauth_states")

    op.drop_index("uq_publishing_success_idempotency", table_name="publishing_attempts")
    op.drop_index("idx_publishing_attempts_status_lease", table_name="publishing_attempts")
    op.drop_column("publishing_attempts", "updated_at")
    op.drop_column("publishing_attempts", "remote_verified_at")
    op.drop_column("publishing_attempts", "warnings")
    op.drop_column("publishing_attempts", "lease_expires_at")
    op.drop_column("publishing_attempts", "status")

    op.drop_column("generated_articles", "publish_lease_expires_at")
    op.drop_column("generated_articles", "publish_scheduled_at")
    op.drop_column("generated_articles", "publish_requested_status")
    op.drop_column("generated_articles", "publish_task_id")
