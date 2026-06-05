"""create task_results table

Revision ID: 20250225_001
Revises: 20251228_001
Create Date: 2025-02-25

Adds the task_results table used by TaskResultRepository for persisting
Celery task execution state. Without this table, task reconciliation
falls back to Celery-only state, losing durability guarantees.
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSON


# revision identifiers, used by Alembic.
revision = '20250225_001'
down_revision = '20251228_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Create task_results table for task persistence."""
    op.create_table(
        'task_results',
        sa.Column('id', PG_UUID, primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('task_id', sa.String(255), nullable=False, unique=True, index=True),
        sa.Column('task_name', sa.String(255), nullable=False, index=True),
        sa.Column('idempotency_key', sa.String(255), index=True),
        sa.Column('status', sa.String(50), nullable=False, index=True),
        sa.Column('args', JSON),
        sa.Column('kwargs', JSON),
        sa.Column('result', JSON),
        sa.Column('error', sa.Text),
        sa.Column('traceback', sa.Text),
        sa.Column('start_time', sa.DateTime),
        sa.Column('end_time', sa.DateTime),
        sa.Column('duration_seconds', sa.Float),
        sa.Column('retry_count', sa.Integer, server_default='0'),
        sa.Column('worker_name', sa.String(255)),
        sa.Column('created_at', sa.DateTime, server_default=sa.func.now()),
        sa.Column('updated_at', sa.DateTime, server_default=sa.func.now()),
    )
    print("Created task_results table")


def downgrade() -> None:
    """Drop task_results table."""
    op.drop_table('task_results')
    print("Dropped task_results table")
