"""
Database Schema Initialization & Migration Script
==================================================

Idempotent schema creation with:
- pgvector extension activation
- Table creation with foreign key constraints
- Vector similarity indices (IVFFlat)
- Trigger functions for audit trails
- Initial system data seeding

Usage:
    python -m scripts.setup_database
    python -m scripts.setup_database --drop-existing  # Dangerous!
"""

import argparse
import asyncio
import sys
from datetime import datetime
from pathlib import Path
from typing import List

from loguru import logger
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import settings
from infrastructure.database import database_manager


class DatabaseSetup:
    """
    Orchestrates database schema creation with transaction safety.
    """

    def __init__(self, session: AsyncSession):
        self.session = session
        self.schema_version = "1.0.0"

    async def setup_complete_schema(self, drop_existing: bool = False) -> None:
        """
        Execute complete schema setup with rollback safety.

        Args:
            drop_existing: If True, drops all tables before creation (DANGEROUS)
        """
        try:
            logger.info("=" * 70)
            logger.info("DATABASE SCHEMA INITIALIZATION")
            logger.info("=" * 70)

            if drop_existing:
                logger.warning("⚠️  DROP EXISTING flag is SET - all data will be lost!")
                await asyncio.sleep(2)  # Give time to cancel
                await self._drop_all_tables()

            # Execute schema creation steps
            await self._enable_extensions()
            await self._create_tables()
            await self._create_indices()
            await self._create_triggers()
            await self._seed_initial_data()
            await self._record_schema_version()

            logger.info("✓ Database schema initialized successfully")
            logger.info(f"✓ Schema version: {self.schema_version}")

        except Exception as e:
            logger.error(f"✗ Schema initialization failed: {e}")
            raise

    # =========================================================================
    # SCHEMA CREATION STEPS
    # =========================================================================

    async def _enable_extensions(self) -> None:
        """Enable required PostgreSQL extensions."""
        logger.info("Enabling PostgreSQL extensions...")

        extensions = [
            "CREATE EXTENSION IF NOT EXISTS vector;",
            'CREATE EXTENSION IF NOT EXISTS "uuid-ossp";',
            "CREATE EXTENSION IF NOT EXISTS pg_trgm;",  # For text search
        ]

        for ext_sql in extensions:
            await self.session.execute(text(ext_sql))

        await self.session.commit()
        logger.info("  ✓ Extensions enabled")

    async def _create_tables(self) -> None:
        """Create all application tables."""
        logger.info("Creating database tables...")

        # Projects table (root entity)
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS projects (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                name VARCHAR(255) NOT NULL,
                domain VARCHAR(255),
                telegram_channel VARCHAR(255),
                
                -- Metadata
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_active TIMESTAMP,
                deleted_at TIMESTAMP,
                
                -- Statistics
                total_articles_generated INTEGER NOT NULL DEFAULT 0,
                total_tokens_consumed BIGINT NOT NULL DEFAULT 0,
                total_cost_usd DECIMAL(10, 2) NOT NULL DEFAULT 0.00,
                
                -- Constraints
                CONSTRAINT projects_name_unique UNIQUE (name),
                CONSTRAINT projects_domain_valid CHECK (domain ~ '^[a-zA-Z0-9][a-zA-Z0-9-]{1,61}[a-zA-Z0-9]\\.[a-zA-Z]{2,}$' OR domain IS NULL)
            );
        """
            )
        )

        # Rulebooks table
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS rulebooks (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                
                raw_content TEXT NOT NULL,
                version INTEGER NOT NULL DEFAULT 1,
                
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                updated_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                CONSTRAINT rulebooks_project_version_unique UNIQUE (project_id, version)
            );
        """
            )
        )

        # Rules table with vector embeddings
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS rules (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                rulebook_id UUID NOT NULL REFERENCES rulebooks(id) ON DELETE CASCADE,
                
                rule_type VARCHAR(50) NOT NULL,
                content TEXT NOT NULL,
                embedding vector(384),  -- Sentence-BERT embedding dimension
                priority INTEGER NOT NULL DEFAULT 5,
                context TEXT,
                
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                CONSTRAINT rules_priority_range CHECK (priority BETWEEN 1 AND 10)
            );
        """
            )
        )

        # Inferred patterns table
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS inferred_patterns (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                
                -- Linguistic features
                avg_sentence_length FLOAT,
                sentence_length_std FLOAT,
                lexical_diversity FLOAT,
                readability_score FLOAT,
                
                -- Semantic representation
                tone_embedding vector(384),
                
                -- Structural patterns (stored as JSONB for flexibility)
                structure_patterns JSONB,
                
                -- Statistical metadata
                confidence FLOAT NOT NULL,
                sample_size INTEGER NOT NULL,
                analyzed_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                CONSTRAINT inferred_patterns_confidence_range CHECK (confidence BETWEEN 0 AND 1),
                CONSTRAINT inferred_patterns_sample_size_min CHECK (sample_size >= 5)
            );
        """
            )
        )

        # Content plans table
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS content_plans (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                
                topic VARCHAR(500) NOT NULL,
                outline_json JSONB NOT NULL,
                
                -- Keywords (stored as JSONB array)
                primary_keywords JSONB,
                secondary_keywords JSONB,
                
                -- Targets
                target_word_count INTEGER NOT NULL DEFAULT 1500,
                readability_target VARCHAR(50),
                
                -- Cost estimation
                estimated_cost DECIMAL(6, 4),
                
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                CONSTRAINT content_plans_word_count_positive CHECK (target_word_count > 0)
            );
        """
            )
        )

        # Generated articles table
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS generated_articles (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                project_id UUID NOT NULL REFERENCES projects(id) ON DELETE CASCADE,
                content_plan_id UUID REFERENCES content_plans(id) ON DELETE SET NULL,
                
                -- Content
                title VARCHAR(500) NOT NULL,
                content TEXT NOT NULL,
                meta_description TEXT,
                
                -- Quality metrics
                word_count INTEGER NOT NULL,
                readability_score FLOAT,
                keyword_density JSONB,
                
                -- Cost tracking
                total_tokens_used INTEGER NOT NULL,
                total_cost DECIMAL(6, 4) NOT NULL,
                generation_time FLOAT NOT NULL,  -- seconds
                
                -- Distribution
                distributed_at TIMESTAMP,
                distribution_channels JSONB,
                
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                
                CONSTRAINT articles_word_count_positive CHECK (word_count > 0),
                CONSTRAINT articles_generation_time_positive CHECK (generation_time > 0)
            );
        """
            )
        )

        # LLM response cache table
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS llm_response_cache (
                id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
                prompt_hash VARCHAR(64) NOT NULL UNIQUE,  -- SHA-256
                
                response TEXT NOT NULL,
                model VARCHAR(50) NOT NULL,
                tokens_used INTEGER NOT NULL,
                
                created_at TIMESTAMP NOT NULL DEFAULT NOW(),
                last_accessed TIMESTAMP NOT NULL DEFAULT NOW(),
                access_count INTEGER NOT NULL DEFAULT 1,
                
                CONSTRAINT cache_tokens_positive CHECK (tokens_used > 0),
                CONSTRAINT cache_access_count_positive CHECK (access_count > 0)
            );
        """
            )
        )

        # System metadata table
        await self.session.execute(
            text(
                """
            CREATE TABLE IF NOT EXISTS system_metadata (
                key VARCHAR(100) PRIMARY KEY,
                value JSONB NOT NULL,
                updated_at TIMESTAMP NOT NULL DEFAULT NOW()
            );
        """
            )
        )

        await self.session.commit()
        logger.info("  ✓ Tables created")

    async def _create_indices(self) -> None:
        """Create performance and vector similarity indices."""
        logger.info("Creating database indices...")

        indices = [
            # Vector similarity indices (IVFFlat algorithm)
            """
            CREATE INDEX IF NOT EXISTS rules_embedding_idx 
            ON rules USING ivfflat (embedding vector_cosine_ops) 
            WITH (lists = 100);
            """,
            """
            CREATE INDEX IF NOT EXISTS inferred_patterns_tone_embedding_idx 
            ON inferred_patterns USING ivfflat (tone_embedding vector_cosineops)WITH (lists = 50);
	""",
            # Standard B-tree indices for foreign keys
            "CREATE INDEX IF NOT EXISTS idx_rules_rulebook_id ON rules(rulebook_id);",
            "CREATE INDEX IF NOT EXISTS idx_rulebooks_project_id ON rulebooks(project_id);",
            "CREATE INDEX IF NOT EXISTS idx_inferred_patterns_project_id ON inferred_patterns(project_id);",
            "CREATE INDEX IF NOT EXISTS idx_content_plans_project_id ON content_plans(project_id);",
            "CREATE INDEX IF NOT EXISTS idx_articles_project_id ON generated_articles(project_id);",
            "CREATE INDEX IF NOT EXISTS idx_articles_content_plan_id ON generated_articles(content_plan_id);",
            # Cache lookup optimization
            "CREATE INDEX IF NOT EXISTS idx_cache_prompt_hash ON llm_response_cache(prompt_hash);",
            "CREATE INDEX IF NOT EXISTS idx_cache_last_accessed ON llm_response_cache(last_accessed DESC);",
            # Temporal indices for analytics
            "CREATE INDEX IF NOT EXISTS idx_articles_created_at ON generated_articles(created_at DESC);",
            "CREATE INDEX IF NOT EXISTS idx_articles_distributed_at ON generated_articles(distributed_at DESC) WHERE distributed_at IS NOT NULL;",
            # Full-text search on content
            "CREATE INDEX IF NOT EXISTS idx_articles_title_trgm ON generated_articles USING gin (title gin_trgm_ops);",
            # Soft delete support
            "CREATE INDEX IF NOT EXISTS idx_projects_deleted_at ON projects(deleted_at) WHERE deleted_at IS NULL;",
        ]

        for idx_sql in indices:
            await self.session.execute(text(idx_sql))

        await self.session.commit()
        logger.info("  ✓ Indices created")


async def _create_triggers(self) -> None:
    """Create database triggers for automated behaviors."""
    logger.info("Creating database triggers...")

    # Trigger: Update projects.updated_at on modification
    await self.session.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION update_updated_at_column()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.updated_at = NOW();
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
        )
    )

    await self.session.execute(
        text(
            """
        DROP TRIGGER IF EXISTS projects_updated_at ON projects;
        CREATE TRIGGER projects_updated_at
        BEFORE UPDATE ON projects
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """
        )
    )

    await self.session.execute(
        text(
            """
        DROP TRIGGER IF EXISTS rulebooks_updated_at ON rulebooks;
        CREATE TRIGGER rulebooks_updated_at
        BEFORE UPDATE ON rulebooks
        FOR EACH ROW
        EXECUTE FUNCTION update_updated_at_column();
    """
        )
    )

    # Trigger: Update cache access statistics
    await self.session.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION update_cache_access()
        RETURNS TRIGGER AS $$
        BEGIN
            NEW.last_accessed = NOW();
            NEW.access_count = OLD.access_count + 1;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
        )
    )

    await self.session.execute(
        text(
            """
        DROP TRIGGER IF EXISTS cache_access_update ON llm_response_cache;
        CREATE TRIGGER cache_access_update
        BEFORE UPDATE ON llm_response_cache
        FOR EACH ROW
        EXECUTE FUNCTION update_cache_access();
    """
        )
    )

    # Trigger: Increment project statistics on article generation
    await self.session.execute(
        text(
            """
        CREATE OR REPLACE FUNCTION increment_project_stats()
        RETURNS TRIGGER AS $$
        BEGIN
            UPDATE projects
            SET 
                total_articles_generated = total_articles_generated + 1,
                total_tokens_consumed = total_tokens_consumed + NEW.total_tokens_used,
                total_cost_usd = total_cost_usd + NEW.total_cost,
                last_active = NOW()
            WHERE id = NEW.project_id;
            RETURN NEW;
        END;
        $$ LANGUAGE plpgsql;
    """
        )
    )

    await self.session.execute(
        text(
            """
        DROP TRIGGER IF EXISTS article_generation_stats ON generated_articles;
        CREATE TRIGGER article_generation_stats
        AFTER INSERT ON generated_articles
        FOR EACH ROW
        EXECUTE FUNCTION increment_project_stats();
    """
        )
    )

    await self.session.commit()
    logger.info("  ✓ Triggers created")


async def _seed_initial_data(self) -> None:
    """Seed initial system data and best practices."""
    logger.info("Seeding initial system data...")

    # Insert schema version
    await self.session.execute(
        text(
            """
        INSERT INTO system_metadata (key, value, updated_at)
        VALUES ('schema_version', :version, NOW())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = NOW();
    """
        ),
        {"version": f'"{self.schema_version}"'},
    )

    # Insert system configuration
    await self.session.execute(
        text(
            """
        INSERT INTO system_metadata (key, value, updated_at)
        VALUES ('system_config', :config, NOW())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = NOW();
    """
        ),
        {
            "config": """{
            "default_token_budget": 1000000,
            "cache_ttl_days": 30,
            "max_concurrent_generations": 5,
            "default_target_word_count": 1500
        }"""
        },
    )

    await self.session.commit()
    logger.info("  ✓ Initial data seeded")


async def _record_schema_version(self) -> None:
    """Record schema version and migration timestamp."""
    await self.session.execute(
        text(
            """
        INSERT INTO system_metadata (key, value, updated_at)
        VALUES ('last_migration', :timestamp, NOW())
        ON CONFLICT (key) DO UPDATE
        SET value = EXCLUDED.value, updated_at = NOW();
    """
        ),
        {"timestamp": f'"{datetime.utcnow().isoformat()}"'},
    )

    await self.session.commit()


async def _drop_all_tables(self) -> None:
    """Drop all tables (DANGEROUS - only for development)."""
    logger.warning("Dropping all existing tables...")

    await self.session.execute(
        text(
            """
        DROP TABLE IF EXISTS generated_articles CASCADE;
        DROP TABLE IF EXISTS content_plans CASCADE;
        DROP TABLE IF EXISTS inferred_patterns CASCADE;
        DROP TABLE IF EXISTS rules CASCADE;
        DROP TABLE IF EXISTS rulebooks CASCADE;
        DROP TABLE IF EXISTS projects CASCADE;
        DROP TABLE IF EXISTS llm_response_cache CASCADE;
        DROP TABLE IF EXISTS system_metadata CASCADE;
    """
        )
    )

    await self.session.commit()
    logger.warning("  ✓ All tables dropped")


# =========================================================================
# VALIDATION & VERIFICATION
# =========================================================================


async def verify_schema(self) -> bool:
    """Verify schema integrity after setup."""
    logger.info("Verifying schema integrity...")

    try:
        # Check all tables exist
        result = await self.session.execute(
            text(
                """
            SELECT table_name 
            FROM information_schema.tables 
            WHERE table_schema = 'public'
            AND table_type = 'BASE TABLE';
        """
            )
        )

        tables = {row[0] for row in result.fetchall()}
        expected_tables = {
            "projects",
            "rulebooks",
            "rules",
            "inferred_patterns",
            "content_plans",
            "generated_articles",
            "llm_response_cache",
            "system_metadata",
        }

        missing_tables = expected_tables - tables
        if missing_tables:
            logger.error(f"  ✗ Missing tables: {missing_tables}")
            return False

        # Check pgvector extension
        result = await self.session.execute(
            text(
                """
            SELECT * FROM pg_extension WHERE extname = 'vector';
        """
            )
        )

        if not result.fetchone():
            logger.error("  ✗ pgvector extension not installed")
            return False

        # Check vector indices exist
        result = await self.session.execute(
            text(
                """
            SELECT indexname 
            FROM pg_indexes 
            WHERE schemaname = 'public' 
            AND indexname LIKE '%embedding%';
        """
            )
        )

        vector_indices = {row[0] for row in result.fetchall()}
        if len(vector_indices) < 2:
            logger.error(f"  ✗ Missing vector indices: {vector_indices}")
            return False

        logger.info("  ✓ Schema verification passed")
        return True

    except Exception as e:
        logger.error(f"  ✗ Schema verification failed: {e}")
        return False


async def main():
    """Main entry point for database setup script."""
    parser = argparse.ArgumentParser(
        description="Initialize Content Automation Engine database schema"
    )
    parser.add_argument(
        "--drop-existing",
        action="store_true",
        help="Drop all existing tables before creation (DANGEROUS - destroys all data)",
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify schema without making changes"
    )
    args = parser.parse_args()

    try:
        # Initialize database connection
        await database_manager.initialize()

        async with database_manager.session() as session:
            setup = DatabaseSetup(session)

            if args.verify_only:
                logger.info("Running schema verification only...")
                success = await setup.verify_schema()
                sys.exit(0 if success else 1)

            # Run full setup
            await setup.setup_complete_schema(drop_existing=args.drop_existing)

            # Verify after setup
            success = await setup.verify_schema()

            if success:
                logger.info("=" * 70)
                logger.info("✓ DATABASE SETUP COMPLETED SUCCESSFULLY")
                logger.info("=" * 70)
                sys.exit(0)
            else:
                logger.error("✗ Schema verification failed after setup")
                sys.exit(1)

    except Exception as e:
        logger.error(f"✗ Database setup failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    finally:
        await database_manager.close()


if __name__ == "__main__":
    asyncio.run(main())
