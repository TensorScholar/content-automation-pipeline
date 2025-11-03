"""add_task_results_table_for_audit_trail

Revision ID: 001
Revises: 
Create Date: 2024-01-15 10:00:00.000000

Phase 5: Idempotent & Atomic Task Execution
- Creates task_results table for comprehensive audit trail
- Indexes on task_id (unique), task_name, idempotency_key, status for fast queries
- Tracks full lifecycle: args, kwargs, result, errors, traceback, timing, retries
"""

from typing import Sequence, Union

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "001"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Create task_results table with comprehensive tracking fields."""
    op.create_table(
        "task_results",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("task_id", sa.String(255), nullable=False, unique=True, index=True),
        sa.Column("task_name", sa.String(255), nullable=False, index=True),
        sa.Column("idempotency_key", sa.String(64), nullable=True, index=True),
        sa.Column(
            "status",
            sa.Enum(
                "PENDING",
                "STARTED",
                "RETRY",
                "SUCCESS",
                "FAILURE",
                "REVOKED",
                name="taskstatus",
            ),
            nullable=False,
            index=True,
        ),
        sa.Column("args", postgresql.JSONB, nullable=True),
        sa.Column("kwargs", postgresql.JSONB, nullable=True),
        sa.Column("result", postgresql.JSONB, nullable=True),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("traceback", sa.Text, nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("retry_count", sa.Integer, nullable=False, default=0),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.func.now(),
            onupdate=sa.func.now(),
        ),
    )

    # Create composite indexes for common queries
    op.create_index(
        "idx_task_results_status_created",
        "task_results",
        ["status", "created_at"],
    )
    op.create_index(
        "idx_task_results_task_name_status",
        "task_results",
        ["task_name", "status"],
    )


def downgrade() -> None:
    """Drop task_results table and enum type."""
    op.drop_index("idx_task_results_task_name_status", table_name="task_results")
    op.drop_index("idx_task_results_status_created", table_name="task_results")
    op.drop_table("task_results")
    op.execute("DROP TYPE taskstatus")
