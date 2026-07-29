"""add durable generation finalization

Revision ID: 20260721_001
Revises: 20260614_003
Create Date: 2026-07-21
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260721_001"
down_revision = "20260614_003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generated_articles",
        sa.Column("generation_task_id", sa.String(255), nullable=True),
    )
    op.create_index(
        "uq_generated_articles_generation_task_id",
        "generated_articles",
        ["generation_task_id"],
        unique=True,
        postgresql_where=sa.text("generation_task_id IS NOT NULL"),
    )
    op.create_table(
        "generation_outbox_events",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column("task_id", sa.String(255), nullable=False),
        sa.Column(
            "article_id",
            PG_UUID,
            sa.ForeignKey("generated_articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("event_type", sa.String(80), nullable=False),
        sa.Column("payload", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("attempt_count", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("last_error", sa.Text()),
        sa.Column("available_at", sa.DateTime()),
        sa.Column("completed_at", sa.DateTime()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.UniqueConstraint("task_id", "event_type", name="uq_generation_outbox_task_event"),
    )
    op.create_index(
        "ix_generation_outbox_events_task_id",
        "generation_outbox_events",
        ["task_id"],
    )
    op.create_index(
        "ix_generation_outbox_events_article_id",
        "generation_outbox_events",
        ["article_id"],
    )
    op.create_index(
        "idx_generation_outbox_pending",
        "generation_outbox_events",
        ["status", "available_at"],
    )


def downgrade() -> None:
    op.drop_index("idx_generation_outbox_pending", table_name="generation_outbox_events")
    op.drop_index(
        "ix_generation_outbox_events_article_id",
        table_name="generation_outbox_events",
    )
    op.drop_index(
        "ix_generation_outbox_events_task_id",
        table_name="generation_outbox_events",
    )
    op.drop_table("generation_outbox_events")
    op.drop_index("uq_generated_articles_generation_task_id", table_name="generated_articles")
    op.drop_column("generated_articles", "generation_task_id")
