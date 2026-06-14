"""add performance feedback tables

Revision ID: 20260614_003
Revises: 20260614_002
Create Date: 2026-06-14
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260614_003"
down_revision = "20260614_002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "content_performance_snapshots",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column(
            "project_id",
            PG_UUID,
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "article_id",
            PG_UUID,
            sa.ForeignKey("generated_articles.id", ondelete="SET NULL"),
        ),
        sa.Column("url", sa.String(1000), nullable=False),
        sa.Column("date_from", sa.Date(), nullable=False),
        sa.Column("date_to", sa.Date(), nullable=False),
        sa.Column("clicks", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("impressions", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("ctr", sa.Float(), nullable=False, server_default="0"),
        sa.Column("average_position", sa.Float(), nullable=False, server_default="0"),
        sa.Column("source", sa.String(40), nullable=False, server_default="manual_csv"),
        sa.Column("imported_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "uq_performance_snapshot_project_url_period_source",
        "content_performance_snapshots",
        ["project_id", "url", "date_from", "date_to", "source"],
        unique=True,
    )
    op.create_index(
        "idx_performance_snapshots_project_period",
        "content_performance_snapshots",
        ["project_id", "date_to"],
    )
    op.create_index(
        "idx_performance_snapshots_article_period",
        "content_performance_snapshots",
        ["article_id", "date_to"],
    )

    op.create_table(
        "content_improvement_opportunities",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column(
            "project_id",
            PG_UUID,
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "article_id",
            PG_UUID,
            sa.ForeignKey("generated_articles.id", ondelete="SET NULL"),
        ),
        sa.Column(
            "snapshot_id",
            PG_UUID,
            sa.ForeignKey("content_performance_snapshots.id", ondelete="CASCADE"),
        ),
        sa.Column("url", sa.String(1000), nullable=False),
        sa.Column("type", sa.String(80), nullable=False),
        sa.Column("severity", sa.String(20), nullable=False),
        sa.Column("reason", sa.Text(), nullable=False),
        sa.Column("suggested_action", sa.Text(), nullable=False),
        sa.Column("supporting_metrics", JSONB, nullable=False, server_default=sa.text("'{}'::jsonb")),
        sa.Column("status", sa.String(40), nullable=False, server_default="open"),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "uq_improvement_opportunity_project_url_type",
        "content_improvement_opportunities",
        ["project_id", "url", "type"],
        unique=True,
    )
    op.create_index(
        "idx_improvement_opportunities_project_status",
        "content_improvement_opportunities",
        ["project_id", "status"],
    )
    op.create_index(
        "idx_improvement_opportunities_article_status",
        "content_improvement_opportunities",
        ["article_id", "status"],
    )


def downgrade() -> None:
    op.drop_index(
        "idx_improvement_opportunities_article_status",
        table_name="content_improvement_opportunities",
    )
    op.drop_index(
        "idx_improvement_opportunities_project_status",
        table_name="content_improvement_opportunities",
    )
    op.drop_index(
        "uq_improvement_opportunity_project_url_type",
        table_name="content_improvement_opportunities",
    )
    op.drop_table("content_improvement_opportunities")
    op.drop_index(
        "idx_performance_snapshots_article_period",
        table_name="content_performance_snapshots",
    )
    op.drop_index(
        "idx_performance_snapshots_project_period",
        table_name="content_performance_snapshots",
    )
    op.drop_index(
        "uq_performance_snapshot_project_url_period_source",
        table_name="content_performance_snapshots",
    )
    op.drop_table("content_performance_snapshots")
