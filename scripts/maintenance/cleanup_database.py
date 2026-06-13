"""
Database Cleanup Script
Removes all test/development data from the production database
"""

import asyncio
import sys

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from config.settings import get_settings


async def cleanup_database():
    """Clean all test data from database"""

    settings = get_settings()
    engine = create_async_engine(str(settings.database.url), echo=False)

    print("🧹 Starting database cleanup...")
    print("=" * 60)

    async with engine.begin() as conn:
        # Count before cleanup
        print("\n📊 Data BEFORE cleanup:")
        projects_count = await conn.execute(text("SELECT COUNT(*) FROM projects WHERE deleted_at IS NULL"))
        tasks_count = await conn.execute(text("SELECT COUNT(*) FROM tasks"))
        articles_count = await conn.execute(text("SELECT COUNT(*) FROM generated_articles"))

        print(f"  Projects: {projects_count.scalar()}")
        print(f"  Tasks: {tasks_count.scalar()}")
        print(f"  Articles: {articles_count.scalar()}")

        # Delete all generated articles
        print("\n🗑️  Deleting all generated articles...")
        result = await conn.execute(text("DELETE FROM generated_articles"))
        print(f"  ✓ Deleted {result.rowcount} articles")

        # Delete all tasks
        print("\n🗑️  Deleting all tasks...")
        result = await conn.execute(text("DELETE FROM tasks"))
        print(f"  ✓ Deleted {result.rowcount} tasks")

        # Delete all rulebooks
        print("\n🗑️  Deleting all rulebooks...")
        result = await conn.execute(text("DELETE FROM rulebooks"))
        print(f"  ✓ Deleted {result.rowcount} rulebooks")

        # Delete all content plans
        print("\n🗑️  Deleting all content plans...")
        result = await conn.execute(text("DELETE FROM content_plans"))
        print(f"  ✓ Deleted {result.rowcount} content plans")

        # Soft delete all projects (keep structure)
        print("\n🗑️  Soft deleting all projects...")
        result = await conn.execute(text("""
            UPDATE projects
            SET deleted_at = NOW()
            WHERE deleted_at IS NULL
        """))
        print(f"  ✓ Soft deleted {result.rowcount} projects")

        # Count after cleanup
        print("\n📊 Data AFTER cleanup:")
        projects_count = await conn.execute(text("SELECT COUNT(*) FROM projects WHERE deleted_at IS NULL"))
        tasks_count = await conn.execute(text("SELECT COUNT(*) FROM tasks"))
        articles_count = await conn.execute(text("SELECT COUNT(*) FROM generated_articles"))

        print(f"  Active Projects: {projects_count.scalar()}")
        print(f"  Tasks: {tasks_count.scalar()}")
        print(f"  Articles: {articles_count.scalar()}")

    await engine.dispose()

    print("\n" + "=" * 60)
    print("✅ Database cleanup completed successfully!")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(cleanup_database())
