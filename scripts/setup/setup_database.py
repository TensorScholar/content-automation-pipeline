"""
Safe Database Setup — Non-Destructive
======================================
Creates tables ONLY if they don't exist (metadata.create_all uses IF NOT EXISTS).
Seeds admin user ONLY if no users exist.
Safe to run multiple times — idempotent.
"""

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.schema import metadata
from security import get_password_hash

load_dotenv()


async def setup_database():
    """Create missing tables and seed admin if needed."""

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        return 1

    # Convert to async URL if needed
    if db_url.startswith("postgresql://"):
        async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    elif db_url.startswith("postgresql+asyncpg://"):
        async_url = db_url
    else:
        async_url = db_url

    print("Connecting to database...")
    engine = create_async_engine(async_url, echo=False)

    try:
        async with engine.begin() as conn:
            # Ensure required extensions
            print("Ensuring extensions...")
            try:
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                print("  uuid-ossp: OK")
            except Exception as e:
                print(f"  uuid-ossp: {e}")

            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("  vector: OK")
            except Exception as e:
                print(f"  vector: skipped ({e})")

            # Create all tables IF NOT EXISTS (non-destructive)
            print("Creating tables (IF NOT EXISTS)...")
            await conn.run_sync(metadata.create_all)
            print("  Tables ready")

            # Check if admin user exists
            result = await conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = result.scalar()

            if user_count == 0:
                print("Seeding admin user...")
                user_id = str(uuid.uuid4())
                password = "secure123"
                hashed = get_password_hash(password)

                await conn.execute(
                    text("""
                        INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
                        VALUES (:id, :email, :password, :name, true, true, NOW(), NOW())
                    """),
                    {
                        "id": user_id,
                        "email": "manager@smarlux.com",
                        "password": hashed,
                        "name": "Manager",
                    },
                )
                print("  Admin created: manager@smarlux.com / secure123")
            else:
                print(f"  {user_count} user(s) already exist — skipping seed")

        await engine.dispose()
        print("\nDatabase setup complete.")
        return 0

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        await engine.dispose()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(setup_database()))
