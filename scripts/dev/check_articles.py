
import asyncio
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.database import DatabaseManager
from sqlalchemy import text

async def check_articles():
    db = DatabaseManager()
    await db.initialize()

    try:
        async with db.session() as session:
            # Query generated_articles table
            query = text("""
                SELECT id, title, created_at, word_count, total_cost
                FROM generated_articles
                ORDER BY created_at DESC
                LIMIT 10
            """)
            result = await session.execute(query)
            rows = result.fetchall()

            print(f"📊 Checking 'generated_articles' table...")

            if rows:
                print(f"✅ Found {len(rows)} articles:")
                for row in rows:
                    print("-" * 50)
                    print(f"   🆔 ID: {row.id}")
                    print(f"   📝 Title: {row.title}")
                    print(f"   📅 Created: {row.created_at}")
                    print(f"   📏 Words: {row.word_count}")
                    print(f"   💰 Cost: ${row.total_cost}")
            else:
                print("❌ No articles found in the database yet.")

            # Also check task_results for failures
            print("\n🔍 Checking recent failed tasks:")
            fail_query = text("""
                SELECT task_id, status, error, created_at
                FROM task_results
                WHERE status = 'FAILURE'
                ORDER BY created_at DESC
                LIMIT 5
            """)
            fail_result = await session.execute(fail_query)
            fail_rows = fail_result.fetchall()

            if fail_rows:
                for row in fail_rows:
                     print(f"   ⚠️ Task {row.task_id} FAILED at {row.created_at}")
                     print(f"      Error: {row.error}")
            else:
                print("   ✅ No recent failed tasks found.")

    except Exception as e:
        print(f"Error querying DB: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    asyncio.run(check_articles())
