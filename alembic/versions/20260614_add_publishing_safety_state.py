"""add WordPress publishing safety state

Revision ID: 20260614_001
Revises: 20260613_001
Create Date: 2026-06-14
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260614_001"
down_revision = "20260613_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generated_articles",
        sa.Column(
            "publish_status",
            sa.String(40),
            nullable=False,
            server_default="not_published",
        ),
    )
    op.add_column("generated_articles", sa.Column("wordpress_post_id", sa.String(64)))
    op.add_column("generated_articles", sa.Column("wordpress_post_url", sa.String(1000)))
    op.add_column("generated_articles", sa.Column("wordpress_post_status", sa.String(40)))
    op.add_column("generated_articles", sa.Column("wordpress_published_at", sa.DateTime()))
    op.add_column("generated_articles", sa.Column("publish_idempotency_key", sa.String(160)))
    op.add_column("generated_articles", sa.Column("publish_error_category", sa.String(100)))
    op.add_column("generated_articles", sa.Column("publish_error_message", sa.Text()))
    op.add_column(
        "generated_articles",
        sa.Column(
            "publish_attempt_count",
            sa.Integer,
            nullable=False,
            server_default="0",
        ),
    )
    op.add_column("generated_articles", sa.Column("publish_updated_at", sa.DateTime()))
    op.create_index(
        "idx_articles_publish_status",
        "generated_articles",
        ["project_id", "publish_status"],
    )
    op.create_index(
        "idx_articles_wordpress_post",
        "generated_articles",
        ["project_id", "wordpress_post_id"],
    )

    op.create_table(
        "publishing_attempts",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column(
            "article_id",
            PG_UUID,
            sa.ForeignKey("generated_articles.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "project_id",
            PG_UUID,
            sa.ForeignKey("projects.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("user_id", PG_UUID, sa.ForeignKey("users.id", ondelete="SET NULL")),
        sa.Column("target_site_url", sa.String(500)),
        sa.Column("requested_publish_mode", sa.String(40), nullable=False),
        sa.Column("final_wordpress_status", sa.String(40)),
        sa.Column("wordpress_post_id", sa.String(64)),
        sa.Column("wordpress_post_url", sa.String(1000)),
        sa.Column("idempotency_key", sa.String(160), nullable=False),
        sa.Column("started_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
        sa.Column("finished_at", sa.DateTime()),
        sa.Column("success", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("error_category", sa.String(100)),
        sa.Column("error_message", sa.Text()),
        sa.Column("retry_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("task_id", sa.String(255)),
        sa.Column("correlation_id", sa.String(255)),
        sa.Column("created_at", sa.DateTime(), nullable=False, server_default=sa.func.now()),
    )
    op.create_index(
        "ix_publishing_attempts_article_id",
        "publishing_attempts",
        ["article_id"],
    )
    op.create_index(
        "ix_publishing_attempts_project_id",
        "publishing_attempts",
        ["project_id"],
    )
    op.create_index(
        "ix_publishing_attempts_idempotency_key",
        "publishing_attempts",
        ["idempotency_key"],
    )
    op.create_index(
        "idx_publishing_attempts_article_started",
        "publishing_attempts",
        ["article_id", "started_at"],
    )
    op.create_index(
        "idx_publishing_attempts_project_started",
        "publishing_attempts",
        ["project_id", "started_at"],
    )
    op.create_index(
        "idx_publishing_attempts_idempotency",
        "publishing_attempts",
        ["article_id", "idempotency_key"],
    )


def downgrade() -> None:
    op.drop_table("publishing_attempts")
    op.drop_index("idx_articles_wordpress_post", table_name="generated_articles")
    op.drop_index("idx_articles_publish_status", table_name="generated_articles")
    for column in (
        "publish_updated_at",
        "publish_attempt_count",
        "publish_error_message",
        "publish_error_category",
        "publish_idempotency_key",
        "wordpress_published_at",
        "wordpress_post_status",
        "wordpress_post_url",
        "wordpress_post_id",
        "publish_status",
    ):
        op.drop_column("generated_articles", column)
