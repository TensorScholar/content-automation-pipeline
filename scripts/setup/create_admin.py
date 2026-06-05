#!/usr/bin/env python3
"""Quick admin user creation with explicit connect timeout for Neon cloud DB."""
import asyncio, os, sys, uuid
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv()

import asyncpg
from security import get_password_hash

EMAIL = "admin@smarlux.com"
PASSWORD = "Admin@123456"
FULL_NAME = "Admin"

async def main():
    raw_url = os.environ["DATABASE_URL"]
    url = raw_url.replace("postgresql+asyncpg://", "postgresql://")

    print(f"Connecting to Neon DB (timeout=15s)...")
    conn = None
    for attempt in range(1, 4):
        try:
            conn = await asyncio.wait_for(asyncpg.connect(url, timeout=15), timeout=20)
            print("  Connected.")
            break
        except Exception as e:
            print(f"  Attempt {attempt} failed: {type(e).__name__}: {e}")
            if attempt == 3:
                print("Could not connect after 3 attempts. Check your network / Neon project status.")
                return
            await asyncio.sleep(3)

    hashed = get_password_hash(PASSWORD)
    user_id = str(uuid.uuid4())

    existing = await conn.fetchval("SELECT id FROM users WHERE email = $1", EMAIL)
    if existing:
        await conn.execute(
            "UPDATE users SET hashed_password=$1, is_active=true, is_superuser=true WHERE email=$2",
            hashed, EMAIL
        )
        print(f"User {EMAIL} already existed — password reset and superuser confirmed.")
    else:
        await conn.execute(
            """INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
               VALUES ($1,$2,$3,$4,true,true,NOW(),NOW())""",
            user_id, EMAIL, hashed, FULL_NAME
        )
        print(f"Admin user created: {EMAIL}")

    await conn.close()
    print(f"\nLogin credentials:\n  Email:    {EMAIL}\n  Password: {PASSWORD}")

asyncio.run(main())
