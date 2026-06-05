
import asyncio
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from infrastructure.database import DatabaseManager
from sqlalchemy import text

async def check_task(task_id):
    db = DatabaseManager()
    await db.initialize()

    try:
        async with db.session() as session:
            # Query task_results table
            query = text("SELECT * FROM task_results WHERE task_id = :tid")
            result = await session.execute(query, {"tid": task_id})
            row = result.fetchone()

            if row:
                print(f"✅ Found Task in DB:")
                # Convert row to dict for printing
                print(f"   Status: {row.status}")
                print(f"   Result: {row.result}")
                print(f"   Error:  {row.error}")
            else:
                print(f"❌ Task {task_id} NOT found in 'task_results' table.")

            # Also check generated_articles just in case
            # We don't have task_id in articles directly usually, but maybe in metadata?
            # Let's stick to task_results first.

    except Exception as e:
        print(f"Error querying DB: {e}")
    finally:
        await db.close()

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_task.py <task_id>")
        sys.exit(1)

    asyncio.run(check_task(sys.argv[1]))
