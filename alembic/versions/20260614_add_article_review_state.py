"""add article review state

Revision ID: 20260614_002
Revises: 20260614_001
Create Date: 2026-06-14
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260614_002"
down_revision = "20260614_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "generated_articles",
        sa.Column(
            "review_status",
            sa.String(40),
            nullable=False,
            server_default="pending_review",
        ),
    )
    op.add_column("generated_articles", sa.Column("review_note", sa.Text()))
    op.add_column(
        "generated_articles",
        sa.Column(
            "reviewed_by",
            PG_UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
        ),
    )
    op.add_column("generated_articles", sa.Column("reviewed_at", sa.DateTime()))
    op.add_column("generated_articles", sa.Column("review_updated_at", sa.DateTime()))
    op.create_index(
        "idx_articles_review_status",
        "generated_articles",
        ["project_id", "review_status"],
    )


def downgrade() -> None:
    op.drop_index("idx_articles_review_status", table_name="generated_articles")
    op.drop_column("generated_articles", "review_updated_at")
    op.drop_column("generated_articles", "reviewed_at")
    op.drop_column("generated_articles", "reviewed_by")
    op.drop_column("generated_articles", "review_note")
    op.drop_column("generated_articles", "review_status")
