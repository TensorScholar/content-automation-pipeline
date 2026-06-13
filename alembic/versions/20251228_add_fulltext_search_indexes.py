"""add fulltext search indexes

Revision ID: 20251228_001
Revises: 20250101_001
Create Date: 2025-12-28

CRITICAL FIX: Adds GIN index for full-text search on articles table.
This fixes the performance issue where article searches perform full table scans.

Performance Impact:
- Before: O(n) full table scan on ILIKE queries
- After: O(log n) indexed search using PostgreSQL ts_vector

"""
import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision = '20251228_001'
down_revision = '20250101_001'
branch_labels = None
depends_on = None


def upgrade() -> None:
    """Add full-text search GIN index to generated_articles table."""
    # Create GIN index for full-text search on title and content
    # This enables fast searches using PostgreSQL's ts_vector functionality
    op.execute("""
        CREATE INDEX IF NOT EXISTS idx_articles_fulltext
        ON generated_articles
        USING gin(to_tsvector('english', COALESCE(title, '') || ' ' || COALESCE(content, '')))
    """)

    print("✓ Created full-text search index idx_articles_fulltext")


def downgrade() -> None:
    """Remove full-text search index."""
    op.execute("DROP INDEX IF EXISTS idx_articles_fulltext")

    print("✓ Removed full-text search index idx_articles_fulltext")
