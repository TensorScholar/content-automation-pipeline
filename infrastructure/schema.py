"""
Database Schema: SQLAlchemy Core Table Definitions

Defines database tables using SQLAlchemy Core for type-safe query building.
Replaces raw SQL strings with structured table definitions.
"""

from datetime import datetime
from typing import Optional

from pgvector.sqlalchemy import Vector
from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from sqlalchemy.orm import declarative_base

# Metadata instance for all tables
metadata = MetaData()

# Declarative base for Alembic autogenerate
Base = declarative_base(metadata=metadata)

# Generated Articles Table
generated_articles_table = Table(
    "generated_articles",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "project_id",
        PG_UUID,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column(
        "content_plan_id",
        PG_UUID,
        ForeignKey("content_plans.id", ondelete="SET NULL"),
        index=True,
    ),
    Column("title", String(500), nullable=False),
    Column("content", Text),
    Column("meta_description", String(500)),
    Column("keywords", JSONB, default=[]),
    Column("word_count", Integer),
    Column("readability_score", Float),
    Column("keyword_density", JSON),
    Column("total_tokens_used", Integer),
    Column("total_cost", Float),
    Column("generation_time", Float),
    Column("distributed_at", DateTime),
    Column("distribution_channels", JSON),
    Column("publish_status", String(40), nullable=False, server_default="not_published"),
    Column("wordpress_post_id", String(64)),
    Column("wordpress_post_url", String(1000)),
    Column("wordpress_post_status", String(40)),
    Column("wordpress_published_at", DateTime),
    Column("publish_idempotency_key", String(160)),
    Column("publish_error_category", String(100)),
    Column("publish_error_message", Text),
    Column("publish_attempt_count", Integer, nullable=False, server_default="0"),
    Column("publish_updated_at", DateTime),
    Column("created_at", DateTime, default=func.now(), index=True),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
    # Composite indexes for common query patterns
    Index("idx_articles_project_created", "project_id", "created_at"),
    Index("idx_articles_project_distributed", "project_id", "distributed_at"),
    Index("idx_articles_publish_status", "project_id", "publish_status"),
    Index("idx_articles_wordpress_post", "project_id", "wordpress_post_id"),
    # Full-text search index for title and content (PostgreSQL GIN index)
    # This enables fast ILIKE and full-text searches on articles
    Index(
        "idx_articles_fulltext",
        text(
            "to_tsvector('english'::regconfig, "
            "(COALESCE(title, ''::character varying)::text || ' '::text) "
            "|| COALESCE(content, ''::text))"
        ),
        postgresql_using="gin",
    ),
    # GIN index for JSONB keywords column for fast keyword queries
    Index(
        "idx_articles_keywords_gin",
        "keywords",
        postgresql_using="gin",
    ),
)


publishing_attempts_table = Table(
    "publishing_attempts",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "article_id",
        PG_UUID,
        ForeignKey("generated_articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column(
        "project_id",
        PG_UUID,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("user_id", PG_UUID, ForeignKey("users.id", ondelete="SET NULL")),
    Column("target_site_url", String(500)),
    Column("requested_publish_mode", String(40), nullable=False),
    Column("final_wordpress_status", String(40)),
    Column("wordpress_post_id", String(64)),
    Column("wordpress_post_url", String(1000)),
    Column("idempotency_key", String(160), nullable=False, index=True),
    Column("started_at", DateTime, nullable=False, server_default=func.now()),
    Column("finished_at", DateTime),
    Column("success", Boolean, nullable=False, server_default="false"),
    Column("error_category", String(100)),
    Column("error_message", Text),
    Column("retry_count", Integer, nullable=False, server_default="0"),
    Column("task_id", String(255)),
    Column("correlation_id", String(255)),
    Column("created_at", DateTime, nullable=False, server_default=func.now()),
    Index("idx_publishing_attempts_article_started", "article_id", "started_at"),
    Index("idx_publishing_attempts_project_started", "project_id", "started_at"),
    Index("idx_publishing_attempts_idempotency", "article_id", "idempotency_key"),
)

# Article Revisions Table
article_revisions_table = Table(
    "article_revisions",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "article_id",
        PG_UUID,
        ForeignKey("generated_articles.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("title", String(500), nullable=False),
    Column("content", Text),
    Column("revision_note", Text),
    Column("word_count", Integer),
    Column("created_at", DateTime, default=func.now(), index=True),
    # Composite index for revision history queries
    Index("idx_revisions_article_created", "article_id", "created_at"),
)

# Projects Table
projects_table = Table(
    "projects",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("domain", String(500), index=True),
    Column("description", Text),  # Project description
    Column("vertical", String(100)),  # Business vertical/category
    Column("telegram_channel", String(255)),
    Column("wordpress_url", String(500)),
    Column("wordpress_username", String(255)),
    Column("wordpress_app_password", String(500)),  # Stored as encrypted string
    Column("total_articles_generated", Integer, default=0),
    Column("total_tokens_consumed", Integer, default=0),
    Column("total_cost_usd", Numeric(14, 6), default=0),
    Column("created_at", DateTime, default=func.now(), index=True),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
    Column("last_active", DateTime, index=True),
    Column("deleted_at", DateTime, index=True),
    # Composite index for active project queries
    Index("idx_projects_active", "deleted_at", "last_active"),
)

# Rulebooks Table
rulebooks_table = Table(
    "rulebooks",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "project_id",
        PG_UUID,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("content", Text, nullable=False),
    Column("version", Integer, default=1),
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
    # Composite index for latest version queries
    Index("idx_rulebooks_project_version", "project_id", "version"),
)

# Inferred Patterns Table
inferred_patterns_table = Table(
    "inferred_patterns",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "project_id",
        PG_UUID,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("avg_sentence_length", JSON),
    Column("lexical_diversity", Float),
    Column("readability_score", Float),
    Column("confidence", Float),
    Column("sample_size", Integer),
    Column("analyzed_at", DateTime, default=func.now(), index=True),
    # Composite index for latest pattern queries
    Index("idx_patterns_project_analyzed", "project_id", "analyzed_at"),
)

# Content Plans Table
content_plans_table = Table(
    "content_plans",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "project_id",
        PG_UUID,
        ForeignKey("projects.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("topic", String(500), nullable=False),
    Column("outline_json", JSONB, nullable=False),
    Column("primary_keywords", JSONB),
    Column("secondary_keywords", JSONB),
    Column("target_word_count", Integer, default=1500),
    Column("readability_target", String(50)),
    Column("estimated_cost", Numeric(6, 4)),
    Column("created_at", DateTime, default=func.now(), index=True),
    # Composite index for recent plans queries
    Index("idx_plans_project_created", "project_id", "created_at"),
)

# Rules Table
rules_table = Table(
    "rules",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column(
        "rulebook_id",
        PG_UUID,
        ForeignKey("rulebooks.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    ),
    Column("rule_type", String(50), nullable=False, index=True),
    Column("content", Text, nullable=False),
    Column("embedding", Vector(384)),  # pgvector type for 384-dim embeddings
    Column("priority", Integer, default=5, index=True),
    Column("context", Text),
    Column("created_at", DateTime, default=func.now()),
    Index("idx_rules_rulebook_type", "rulebook_id", "rule_type"),
    Index("idx_rules_rulebook_priority", "rulebook_id", "priority"),
    # HNSW Index for high-performance vector similarity search
    Index(
        "idx_rules_embedding_hnsw",
        "embedding",
        postgresql_using="hnsw",
        postgresql_with={"m": 16, "ef_construction": 64},
        postgresql_ops={"embedding": "vector_cosine_ops"},
    ),
)

# Users Table
users_table = Table(
    "users",
    metadata,
    Column("id", PG_UUID, primary_key=True, default=func.uuid_generate_v4()),
    Column("email", String(255), nullable=False, unique=True, index=True),
    Column("hashed_password", String(255), nullable=False),
    Column("full_name", String(255), nullable=True),
    Column("is_active", Boolean, default=True),
    Column("is_superuser", Boolean, default=False),
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
)

# Persistent provider usage and distributed cost reservations. Prompt bodies
# are deliberately excluded to avoid retaining customer content.
llm_usage_records_table = Table(
    "llm_usage_records",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column("provider", String(50), nullable=False, index=True),
    Column("model", String(255), nullable=False, index=True),
    Column("operation_type", String(100), nullable=False, index=True),
    Column(
        "project_id",
        PG_UUID,
        ForeignKey("projects.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    ),
    Column(
        "user_id",
        PG_UUID,
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    ),
    Column("task_id", String(255), nullable=True, index=True),
    Column("prompt_tokens", Integer, nullable=False, default=0),
    Column("completion_tokens", Integer, nullable=False, default=0),
    Column("total_tokens", Integer, nullable=False, default=0),
    Column("estimated_cost_usd", Numeric(14, 6), nullable=False, default=0),
    Column("actual_cost_usd", Numeric(14, 6), nullable=False, default=0),
    Column("status", String(20), nullable=False, index=True),
    Column("error_category", String(100), nullable=True),
    Column("created_at", DateTime(timezone=True), nullable=False, default=func.now(), index=True),
    Column("completed_at", DateTime(timezone=True), nullable=True),
    Column("reservation_expires_at", DateTime(timezone=True), nullable=True, index=True),
    Index("idx_llm_usage_project_created", "project_id", "created_at"),
    Index("idx_llm_usage_user_created", "user_id", "created_at"),
    Index("idx_llm_usage_status_expiry", "status", "reservation_expires_at"),
)
