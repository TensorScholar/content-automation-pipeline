"""
Initialize Fresh Database
Creates all tables and adds initial superuser
"""

import asyncio
import os
import sys
import uuid

import bcrypt
from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

from infrastructure.schema import metadata

# Load environment variables
load_dotenv()

async def init_database():
    """Initialize database with tables and first user"""

    # Get database URL from environment
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        print("❌ DATABASE_URL not set in environment")
        return 1

    print(f"🔗 Connecting to database...")
    engine = create_async_engine(db_url, echo=False)

    try:
        async with engine.begin() as conn:
            print("\n📦 Creating database extensions...")
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
                print("  ✓ uuid-ossp extension ready")
            except Exception as e:
                print(f"  ⚠️  uuid-ossp: {e}")

            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
                print("  ✓ vector extension ready")
            except Exception as e:
                print(f"  ⚠️  vector extension not available (skipping): {e}")

            print("\n🏗️  Dropping existing tables...")
            # Drop all tables with CASCADE
            await conn.execute(text("DROP SCHEMA public CASCADE"))
            await conn.execute(text("CREATE SCHEMA public"))
            print("  ✓ Old tables dropped")

            # Recreate extensions after schema drop
            await conn.execute(text("CREATE EXTENSION IF NOT EXISTS \"uuid-ossp\""))
            try:
                await conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector"))
            except Exception:
                pass

            print("\n🏗️  Creating fresh database tables...")
            await conn.run_sync(metadata.create_all)
            print("  ✓ All tables created")

            print("\n👤 Creating initial superuser...")
            user_id = str(uuid.uuid4())
            password = "secure123"
            hashed_password = bcrypt.hashpw(password.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')

            await conn.execute(text("""
                INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
                VALUES (:id, :email, :password, :name, true, true, NOW(), NOW())
            """), {
                "id": user_id,
                "email": "manager@smarlux.com",
                "password": hashed_password,
                "name": "Manager"
            })
            print("  ✓ Superuser created: manager@smarlux.com / secure123")

        await engine.dispose()

        print("\n" + "=" * 60)
        print("✅ Database initialized successfully!")
        print("=" * 60)
        print("\n📝 Login credentials:")
        print("   Email: manager@smarlux.com")
        print("   Password: secure123")
        print("=" * 60)

        return 0

    except Exception as e:
        print(f"\n❌ Error initializing database: {e}")
        import traceback
        traceback.print_exc()
        await engine.dispose()
        return 1

if __name__ == "__main__":
    sys.exit(asyncio.run(init_database()))
