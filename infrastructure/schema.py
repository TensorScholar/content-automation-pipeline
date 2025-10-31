"""
Database Schema: SQLAlchemy Core Table Definitions

Defines database tables using SQLAlchemy Core for type-safe query building.
Replaces raw SQL strings with structured table definitions.
"""

from datetime import datetime
from typing import Optional

from sqlalchemy import (
    JSON,
    JSONB,
    UUID,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    MetaData,
    Numeric,
    String,
    Table,
    Text,
    func,
)

# Metadata instance for all tables
metadata = MetaData()

# Generated Articles Table
generated_articles_table = Table(
    "generated_articles",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("project_id", UUID, nullable=False),
    Column("content_plan_id", UUID),
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
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
)

# Article Revisions Table
article_revisions_table = Table(
    "article_revisions",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("article_id", UUID, nullable=False),
    Column("title", String(500), nullable=False),
    Column("content", Text),
    Column("revision_note", Text),
    Column("word_count", Integer),
    Column("created_at", DateTime, default=func.now()),
)

# Projects Table
projects_table = Table(
    "projects",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("name", String(255), nullable=False),
    Column("domain", String(500)),
    Column("telegram_channel", String(255)),
    Column("wordpress_url", String(500)),
    Column("wordpress_username", String(255)),
    Column("wordpress_app_password", String(500)),  # Stored as encrypted string
    Column("total_articles_generated", Integer, default=0),
    Column("total_tokens_consumed", Integer, default=0),
    Column("total_cost_usd", Numeric(10, 2), default=0),
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
    Column("last_active", DateTime),
    Column("deleted_at", DateTime),
)

# Rulebooks Table
rulebooks_table = Table(
    "rulebooks",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("project_id", UUID, nullable=False),
    Column("content", Text, nullable=False),
    Column("version", Integer, default=1),
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
)

# Inferred Patterns Table
inferred_patterns_table = Table(
    "inferred_patterns",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("project_id", UUID, nullable=False),
    Column("avg_sentence_length", JSON),
    Column("lexical_diversity", Float),
    Column("readability_score", Float),
    Column("confidence", Float),
    Column("sample_size", Integer),
    Column("analyzed_at", DateTime, default=func.now()),
)

# Content Plans Table
content_plans_table = Table(
    "content_plans",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("project_id", UUID, nullable=False),
    Column("topic", String(500), nullable=False),
    Column("outline_json", JSONB, nullable=False),
    Column("primary_keywords", JSONB),
    Column("secondary_keywords", JSONB),
    Column("target_word_count", Integer, default=1500),
    Column("readability_target", String(50)),
    Column("estimated_cost", Numeric(6, 4)),
    Column("created_at", DateTime, default=func.now()),
)

# Rules Table
rules_table = Table(
    "rules",
    metadata,
    Column("id", UUID, primary_key=True),
    Column("rulebook_id", UUID, nullable=False),
    Column("rule_type", String(50), nullable=False),
    Column("content", Text, nullable=False),
    Column("embedding", Text),  # Stored as text representation of vector
    Column("priority", Integer, default=5),
    Column("context", Text),
    Column("created_at", DateTime, default=func.now()),
)

# Users Table
users_table = Table(
    "users",
    metadata,
    Column("id", UUID, primary_key=True, default=func.uuid_generate_v4()),
    Column("email", String(255), nullable=False, unique=True, index=True),
    Column("hashed_password", String(255), nullable=False),
    Column("full_name", String(255), nullable=True),
    Column("is_active", Boolean, default=True),
    Column("is_superuser", Boolean, default=False),
    Column("created_at", DateTime, default=func.now()),
    Column("updated_at", DateTime, default=func.now(), onupdate=func.now()),
)
