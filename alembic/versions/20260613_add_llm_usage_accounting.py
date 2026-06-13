"""add persistent LLM usage accounting

Revision ID: 20260613_001
Revises: 20260607_001
Create Date: 2026-06-13
"""

import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID

from alembic import op

revision = "20260613_001"
down_revision = "20260607_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "llm_usage_records",
        sa.Column("id", PG_UUID, primary_key=True),
        sa.Column("provider", sa.String(50), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("operation_type", sa.String(100), nullable=False),
        sa.Column(
            "project_id",
            PG_UUID,
            sa.ForeignKey("projects.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column(
            "user_id",
            PG_UUID,
            sa.ForeignKey("users.id", ondelete="SET NULL"),
            nullable=True,
        ),
        sa.Column("task_id", sa.String(255), nullable=True),
        sa.Column("prompt_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("completion_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column("total_tokens", sa.Integer, nullable=False, server_default="0"),
        sa.Column(
            "estimated_cost_usd",
            sa.Numeric(14, 6),
            nullable=False,
            server_default="0",
        ),
        sa.Column(
            "actual_cost_usd",
            sa.Numeric(14, 6),
            nullable=False,
            server_default="0",
        ),
        sa.Column("status", sa.String(20), nullable=False),
        sa.Column("error_category", sa.String(100), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("reservation_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    for column in (
        "provider",
        "model",
        "operation_type",
        "project_id",
        "user_id",
        "task_id",
        "status",
        "created_at",
        "reservation_expires_at",
    ):
        op.create_index(f"ix_llm_usage_records_{column}", "llm_usage_records", [column])
    op.create_index(
        "idx_llm_usage_project_created",
        "llm_usage_records",
        ["project_id", "created_at"],
    )
    op.create_index(
        "idx_llm_usage_user_created",
        "llm_usage_records",
        ["user_id", "created_at"],
    )
    op.create_index(
        "idx_llm_usage_status_expiry",
        "llm_usage_records",
        ["status", "reservation_expires_at"],
    )


def downgrade() -> None:
    op.drop_table("llm_usage_records")
