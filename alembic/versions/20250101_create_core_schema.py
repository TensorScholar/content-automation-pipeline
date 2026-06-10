"""create core application schema

Revision ID: 20250101_001
Revises:
Create Date: 2025-01-01

Creates the core Smarlux application tables that were previously created by
metadata.create_all setup scripts. Production deployments must be able to build
a fresh database using Alembic alone, so this revision is the explicit base
schema for article/project/rule/auth tables.
"""

from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql


revision = "20250101_001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"')
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "projects",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("domain", sa.String(500)),
        sa.Column("description", sa.Text()),
        sa.Column("vertical", sa.String(100)),
        sa.Column("telegram_channel", sa.String(255)),
        sa.Column("wordpress_url", sa.String(500)),
        sa.Column("wordpress_username", sa.String(255)),
        sa.Column("wordpress_app_password", sa.String(500)),
        sa.Column("total_articles_generated", sa.Integer()),
        sa.Column("total_tokens_consumed", sa.Integer()),
        sa.Column("total_cost_usd", sa.Numeric(10, 2)),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
        sa.Column("last_active", sa.DateTime()),
        sa.Column("deleted_at", sa.DateTime()),
    )
    op.create_index("ix_projects_domain", "projects", ["domain"])
    op.create_index("ix_projects_created_at", "projects", ["created_at"])
    op.create_index("ix_projects_last_active", "projects", ["last_active"])
    op.create_index("ix_projects_deleted_at", "projects", ["deleted_at"])
    op.create_index("idx_projects_active", "projects", ["deleted_at", "last_active"])

    op.create_table(
        "users",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("email", sa.String(255), nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255)),
        sa.Column("is_active", sa.Boolean()),
        sa.Column("is_superuser", sa.Boolean()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_users_email", "users", ["email"], unique=True)

    op.create_table(
        "rulebooks",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("project_id", postgresql.UUID(), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("version", sa.Integer()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_rulebooks_project_id", "rulebooks", ["project_id"])
    op.create_index("idx_rulebooks_project_version", "rulebooks", ["project_id", "version"])

    op.create_table(
        "content_plans",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("project_id", postgresql.UUID(), nullable=False),
        sa.Column("topic", sa.String(500), nullable=False),
        sa.Column("outline_json", postgresql.JSONB(), nullable=False),
        sa.Column("primary_keywords", postgresql.JSONB()),
        sa.Column("secondary_keywords", postgresql.JSONB()),
        sa.Column("target_word_count", sa.Integer()),
        sa.Column("readability_target", sa.String(50)),
        sa.Column("estimated_cost", sa.Numeric(6, 4)),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_content_plans_project_id", "content_plans", ["project_id"])
    op.create_index("ix_content_plans_created_at", "content_plans", ["created_at"])
    op.create_index("idx_plans_project_created", "content_plans", ["project_id", "created_at"])

    op.create_table(
        "generated_articles",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("project_id", postgresql.UUID(), nullable=False),
        sa.Column("content_plan_id", postgresql.UUID()),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text()),
        sa.Column("meta_description", sa.String(500)),
        sa.Column("keywords", postgresql.JSONB()),
        sa.Column("word_count", sa.Integer()),
        sa.Column("readability_score", sa.Float()),
        sa.Column("keyword_density", sa.JSON()),
        sa.Column("total_tokens_used", sa.Integer()),
        sa.Column("total_cost", sa.Float()),
        sa.Column("generation_time", sa.Float()),
        sa.Column("distributed_at", sa.DateTime()),
        sa.Column("distribution_channels", sa.JSON()),
        sa.Column("created_at", sa.DateTime()),
        sa.Column("updated_at", sa.DateTime()),
    )
    op.create_index("ix_generated_articles_project_id", "generated_articles", ["project_id"])
    op.create_index(
        "ix_generated_articles_content_plan_id",
        "generated_articles",
        ["content_plan_id"],
    )
    op.create_index("ix_generated_articles_created_at", "generated_articles", ["created_at"])
    op.create_index(
        "idx_articles_project_created",
        "generated_articles",
        ["project_id", "created_at"],
    )
    op.create_index(
        "idx_articles_project_distributed",
        "generated_articles",
        ["project_id", "distributed_at"],
    )
    op.create_index(
        "idx_articles_keywords_gin",
        "generated_articles",
        ["keywords"],
        postgresql_using="gin",
    )

    op.create_table(
        "inferred_patterns",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("project_id", postgresql.UUID(), nullable=False),
        sa.Column("avg_sentence_length", sa.JSON()),
        sa.Column("lexical_diversity", sa.Float()),
        sa.Column("readability_score", sa.Float()),
        sa.Column("confidence", sa.Float()),
        sa.Column("sample_size", sa.Integer()),
        sa.Column("analyzed_at", sa.DateTime()),
    )
    op.create_index("ix_inferred_patterns_project_id", "inferred_patterns", ["project_id"])
    op.create_index("ix_inferred_patterns_analyzed_at", "inferred_patterns", ["analyzed_at"])
    op.create_index(
        "idx_patterns_project_analyzed",
        "inferred_patterns",
        ["project_id", "analyzed_at"],
    )

    op.create_table(
        "rules",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("rulebook_id", postgresql.UUID(), nullable=False),
        sa.Column("rule_type", sa.String(50), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(384)),
        sa.Column("priority", sa.Integer()),
        sa.Column("context", sa.Text()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_rules_rulebook_id", "rules", ["rulebook_id"])
    op.create_index("ix_rules_rule_type", "rules", ["rule_type"])
    op.create_index("ix_rules_priority", "rules", ["priority"])
    op.create_index("idx_rules_rulebook_type", "rules", ["rulebook_id", "rule_type"])
    op.create_index("idx_rules_rulebook_priority", "rules", ["rulebook_id", "priority"])
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rules_embedding_hnsw
        ON rules
        USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )

    op.create_table(
        "article_revisions",
        sa.Column("id", postgresql.UUID(), primary_key=True),
        sa.Column("article_id", postgresql.UUID(), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("content", sa.Text()),
        sa.Column("revision_note", sa.Text()),
        sa.Column("word_count", sa.Integer()),
        sa.Column("created_at", sa.DateTime()),
    )
    op.create_index("ix_article_revisions_article_id", "article_revisions", ["article_id"])
    op.create_index("ix_article_revisions_created_at", "article_revisions", ["created_at"])
    op.create_index(
        "idx_revisions_article_created",
        "article_revisions",
        ["article_id", "created_at"],
    )


def downgrade() -> None:
    op.drop_table("article_revisions")
    op.drop_table("rules")
    op.drop_table("inferred_patterns")
    op.drop_table("generated_articles")
    op.drop_table("content_plans")
    op.drop_table("rulebooks")
    op.drop_table("users")
    op.drop_table("projects")
