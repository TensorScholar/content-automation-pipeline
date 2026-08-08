"""Destructively recreate the database after explicit operator confirmation."""

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

CONFIRMATION_VALUE = "YES-I-UNDERSTAND-THIS-DELETES-DATA"


async def init_database() -> int:
    """Drop and recreate the public schema, then create one configured admin."""
    if os.getenv("ALLOW_DESTRUCTIVE_DATABASE_RESET") != CONFIRMATION_VALUE:
        print(
            "REFUSED: this command deletes the public schema. Set "
            f"ALLOW_DESTRUCTIVE_DATABASE_RESET={CONFIRMATION_VALUE} only for an approved disposable database."
        )
        return 2

    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("ERROR: DATABASE_URL not set")
        return 1
    if db_url.startswith("postgresql://"):
        db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)

    password = os.getenv("ADMIN_PASSWORD", "")
    if not password:
        print("ERROR: ADMIN_PASSWORD must be supplied through a secret manager or environment variable")
        return 1
    valid, issues = validate_password_strength(password)
    if not valid:
        print("ERROR: ADMIN_PASSWORD is not strong enough: " + "; ".join(issues))
        return 1

    email = os.getenv("ADMIN_EMAIL", "admin@smarlux.com").strip().lower()
    full_name = os.getenv("ADMIN_FULL_NAME", "Admin").strip() or "Admin"
    if not email or "@" not in email:
        print("ERROR: ADMIN_EMAIL must be a valid email address")
        return 1

    print("Connecting to database...")
    engine = create_async_engine(db_url, echo=False)

    try:
        async with engine.begin() as conn:
            print("Dropping and recreating the public schema...")
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            await conn.execute(text('CREATE EXTENSION IF NOT EXISTS "uuid-ossp"'))
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception as exc:
                print(f"  vector extension skipped: {exc}")

            await conn.run_sync(metadata.create_all)
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

        print(f"Database initialized; administrator created: {email} (password not printed)")
        return 0
    except Exception as exc:
        print(f"ERROR: database initialization failed: {type(exc).__name__}: {exc}")
        return 1
    finally:
        await engine.dispose()


if __name__ == "__main__":
    sys.exit(asyncio.run(init_database()))
