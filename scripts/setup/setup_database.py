"""
Safe Database Setup — Non-Destructive
======================================
Creates tables only when they do not exist. An initial administrator is seeded
only when the database has no users and ADMIN_PASSWORD is provided securely.
"""

import asyncio
import os
import sys
import uuid

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.schema import metadata
from security import get_password_hash, validate_password_strength

load_dotenv()


async def setup_database() -> int:
    """Create missing tables and optionally seed a securely configured admin."""
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        return 1

    if db_url.startswith("postgresql://"):
        async_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    else:
        async_url = db_url

    print("Connecting to database...")
    engine = create_async_engine(async_url, echo=False)

    try:
        async with engine.begin() as conn:
            print("Ensuring extensions...")
            try:
                await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
                print("  uuid-ossp: OK")
            except Exception as exc:
                print(f"  uuid-ossp: {exc}")

            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("  vector: OK")
            except Exception as exc:
                print(f"  vector: skipped ({exc})")

            print("Creating tables (IF NOT EXISTS)...")
            await conn.run_sync(metadata.create_all)
            print("  Tables ready")

            result = await conn.execute(text("SELECT COUNT(*) FROM users"))
            user_count = int(result.scalar() or 0)

            if user_count == 0:
                password = os.getenv("ADMIN_PASSWORD", "")
                if not password:
                    print(
                        "  No users exist; admin seed skipped because ADMIN_PASSWORD is not set.\n"
                        "  Create an administrator with scripts/setup/create_user.py or rerun with a secret-managed ADMIN_PASSWORD."
                    )
                else:
                    valid, issues = validate_password_strength(password)
                    if not valid:
                        raise RuntimeError("ADMIN_PASSWORD is not strong enough: " + "; ".join(issues))

                    email = os.getenv("ADMIN_EMAIL", "admin@smarlux.com").strip().lower()
                    full_name = os.getenv("ADMIN_FULL_NAME", "Admin").strip() or "Admin"
                    if not email or "@" not in email:
                        raise RuntimeError("ADMIN_EMAIL must be a valid email address")

                    await conn.execute(
                        text(
                            """
                            INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
                            VALUES (:id, :email, :password, :name, true, true, NOW(), NOW())
                            """
                        ),
                        {
                            "id": str(uuid.uuid4()),
                            "email": email,
                            "password": get_password_hash(password),
                            "name": full_name,
                        },
                    )
                    print(f"  Administrator created: {email} (password not printed)")
            else:
                print(f"  {user_count} user(s) already exist — skipping seed")

        print("\nDatabase setup complete.")
        return 0
    except Exception as exc:
        print(f"\nERROR: {exc}")
        return 1
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(setup_database()))
