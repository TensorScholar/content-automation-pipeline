#!/usr/bin/env python3
"""Create or update an administrator without embedding credentials in source."""

import asyncio
import getpass
import os
import sys
import uuid

import asyncpg
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

load_dotenv()

from security import get_password_hash, validate_password_strength  # noqa: E402

DEFAULT_EMAIL = "admin@smarlux.com"
DEFAULT_FULL_NAME = "Admin"


def _admin_identity() -> tuple[str, str, str]:
    """Resolve administrator identity from environment or an interactive prompt."""
    email = os.getenv("ADMIN_EMAIL", DEFAULT_EMAIL).strip().lower()
    full_name = os.getenv("ADMIN_FULL_NAME", DEFAULT_FULL_NAME).strip() or DEFAULT_FULL_NAME
    password = os.getenv("ADMIN_PASSWORD")

    if not password:
        if not sys.stdin.isatty():
            raise RuntimeError(
                "ADMIN_PASSWORD is required in non-interactive environments. "
                "Provide it through a secret manager or environment variable."
            )
        password = getpass.getpass("Admin password: ")
        confirmation = getpass.getpass("Confirm admin password: ")
        if password != confirmation:
            raise RuntimeError("Password confirmation does not match.")

    valid, issues = validate_password_strength(password)
    if not valid:
        raise RuntimeError("Admin password is not strong enough: " + "; ".join(issues))

    if not email or "@" not in email:
        raise RuntimeError("ADMIN_EMAIL must be a valid email address.")

    return email, full_name, password


async def main() -> None:
    raw_url = os.environ["DATABASE_URL"]
    url = raw_url.replace("postgresql+asyncpg://", "postgresql://")
    email, full_name, password = _admin_identity()

    print("Connecting to PostgreSQL (timeout=15s)...")
    conn: asyncpg.Connection | None = None
    for attempt in range(1, 4):
        try:
            conn = await asyncio.wait_for(asyncpg.connect(url, timeout=15), timeout=20)
            print("  Connected.")
            break
        except Exception as exc:
            print(f"  Attempt {attempt} failed: {type(exc).__name__}: {exc}")
            if attempt == 3:
                raise RuntimeError(
                    "Could not connect after 3 attempts. Check the database URL and network access."
                ) from exc
            await asyncio.sleep(3)

    if conn is None:  # Defensive; the retry loop either connects or raises.
        raise RuntimeError("Database connection was not established.")

    try:
        hashed = get_password_hash(password)
        user_id = str(uuid.uuid4())
        existing = await conn.fetchval("SELECT id FROM users WHERE email = $1", email)

        if existing:
            await conn.execute(
                "UPDATE users SET hashed_password=$1, full_name=$2, is_active=true, is_superuser=true WHERE email=$3",
                hashed,
                full_name,
                email,
            )
            print(f"Administrator {email} updated and activated.")
        else:
            await conn.execute(
                """INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
                   VALUES ($1,$2,$3,$4,true,true,NOW(),NOW())""",
                user_id,
                email,
                hashed,
                full_name,
            )
            print(f"Administrator created: {email}")
    finally:
        await conn.close()

    print("Password was accepted securely and was not printed.")


if __name__ == "__main__":
    asyncio.run(main())
