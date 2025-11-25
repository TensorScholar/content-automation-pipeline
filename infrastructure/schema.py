"""
Database Schema: SQLAlchemy Core Table Definitions

Defines database tables using SQLAlchemy Core for type-safe query building.
Replaces raw SQL strings with structured table definitions.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Index,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    func,
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
    Column("project_id", PG_UUID, nullable=False, index=True),
    Column("content_plan_id", PG_UUID, index=True),
    Column("title", String(500), nullable=False),
    Column("content", Text),
    Column("meta_description", String(500)),
    Column("word_count", Integer),
    Column("readability_score", Float),
    Column("keyword_density", JSON),
    Column("total_tokens_used", Integer),
    Column("total_cost", Float),
    Column("generation_time", Float),
    Column("distributed_at", DateTime),
    Column("distribution_channels", JSON),
    Column("created_at", DateTime, default=func.now(), index=True),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
    # Composite indexes for common query patterns
    Index("idx_articles_project_created", "project_id", "created_at"),
    Index("idx_articles_project_distributed", "project_id", "distributed_at"),
)

# Article Revisions Table
article_revisions_table = Table(
    "article_revisions",
    metadata,
    Column("id", PG_UUID, primary_key=True),
    Column("article_id", PG_UUID, nullable=False, index=True),
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
    Column("telegram_channel", String(255)),
    Column("wordpress_url", String(500)),
    Column("wordpress_username", String(255)),
    Column("wordpress_app_password", String(500)),  # Stored as encrypted string
    Column("total_articles_generated", Integer, default=0),
    Column("total_tokens_consumed", Integer, default=0),
    Column("total_cost_usd", Numeric(10, 2), default=0),
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
    Column("project_id", PG_UUID, nullable=False, index=True),
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
    Column("project_id", PG_UUID, nullable=False, index=True),
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
    Column("project_id", PG_UUID, nullable=False, index=True),
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
    Column("rulebook_id", PG_UUID, nullable=False, index=True),
    Column("rule_type", String(50), nullable=False, index=True),
    Column("content", Text, nullable=False),
    Column("embedding", Text),  # Stored as text representation of vector
    Column("priority", Integer, default=5, index=True),
    Column("context", Text),
    Column("created_at", DateTime, default=func.now()),
    # Composite indexes for rule queries
    Index("idx_rules_rulebook_type", "rulebook_id", "rule_type"),
    Index("idx_rules_rulebook_priority", "rulebook_id", "priority"),
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
