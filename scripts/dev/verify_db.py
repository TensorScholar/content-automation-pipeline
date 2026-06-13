
import asyncio
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sqlalchemy import text

from infrastructure.database import DatabaseManager


async def main():
    print("🔌 Connecting to database...")
    db = DatabaseManager()
    await db.initialize()

    print("🔍 Querying database...")
    async with db.session() as session:
        # Get total count
        count_result = await session.execute(text("SELECT COUNT(*) FROM generated_articles"))
        total_count = count_result.scalar()
        print(f"📊 Total Articles: {total_count}")

        # Get recent articles
        result = await session.execute(text(
            "SELECT id, title, created_at FROM generated_articles ORDER BY created_at DESC LIMIT 5"
        ))
        articles = result.fetchall()

        if not articles:
            print("⚠️ No articles found in the database.")
        else:
            print(f"✅ Found {len(articles)} recent articles:")
            print("-" * 80)
            print(f"{'ID':<38} | {'Created At':<20} | {'Title'}")
            print("-" * 80)
            for row in articles:
                print(f"{str(row.id):<38} | {row.created_at.strftime('%Y-%m-%d %H:%M') if row.created_at else 'N/A':<20} | {row.title}")
            print("-" * 80)

    await db.close()

if __name__ == "__main__":
    asyncio.run(main())
