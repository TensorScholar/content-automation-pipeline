#!/usr/bin/env python3
"""
Secure User Creation Script
============================

Creates a new user in the database with proper security practices:
- CLI arguments for credentials (no hardcoding)
- Hidden password input via getpass
- Environment variable fallbacks for CI/CD
- Email format validation
- Password strength validation
- Graceful error handling

Usage:
    python scripts/create_user.py --email admin@example.com --full-name "Admin User"

    # With password as environment variable (for CI/CD):
    ADMIN_PASSWORD=SecurePass123! python scripts/create_user.py --email admin@example.com
"""

import argparse
import asyncio
import getpass
import os
import re
import sys
import uuid
from datetime import datetime, timezone

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from loguru import logger
from security import get_password_hash, validate_password_strength
from sqlalchemy.ext.asyncio import create_async_engine
from sqlalchemy.sql import text

# Email regex pattern for validation
EMAIL_REGEX = re.compile(
    r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
)


def validate_email(email: str) -> tuple[bool, str]:
    """
    Validate email format using regex.

    Args:
        email: Email address to validate

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    if not email:
        return False, "Email is required"
    if not EMAIL_REGEX.match(email):
        return False, f"Invalid email format: {email}"
    return True, ""


def get_credentials_from_args_or_env(args: argparse.Namespace) -> tuple[str, str, str]:
    """
    Get credentials from CLI args, environment variables, or interactive prompt.

    Priority: CLI args > Environment variables > Interactive prompt (password only)

    Args:
        args: Parsed argparse namespace

    Returns:
        tuple[str, str, str]: (email, password, full_name)
    """
    # Email: CLI arg > env var
    email = args.email or os.getenv("ADMIN_EMAIL", "")

    # Full name: CLI arg > env var > default
    full_name = args.full_name or os.getenv("ADMIN_FULL_NAME", "Admin User")

    # Password: CLI arg > env var > interactive prompt
    password = args.password or os.getenv("ADMIN_PASSWORD", "")

    if not password:
        # Interactive password prompt with hidden input
        password = getpass.getpass("Enter password: ")

    return email, password, full_name


async def create_user(email: str, password: str, full_name: str, is_superuser: bool = True) -> bool:
    """
    Create a new user in the database.

    Args:
        email: User's email address
        password: Plain text password (will be hashed)
        full_name: User's full name
        is_superuser: Whether user has admin privileges

    Returns:
        bool: True if user created successfully, False otherwise
    """
    db_url = os.getenv("DATABASE_URL")
    if not db_url:
        logger.error("DATABASE_URL environment variable not set")
        print("Error: DATABASE_URL environment variable not set")
        return False

    # Hash the password
    hashed_pw = get_password_hash(password)
    user_id = str(uuid.uuid4())

    try:
        # Create async engine
        engine = create_async_engine(db_url)

        async with engine.begin() as conn:
            # Check if user exists
            result = await conn.execute(
                text("SELECT 1 FROM users WHERE email = :email"),
                {"email": email}
            )
            if result.scalar():
                print(f"User with this email already exists.")
                logger.info(f"user_creation_skipped | User already exists")
                await engine.dispose()
                return False

            # Insert user
            await conn.execute(
                text("""
                    INSERT INTO users (id, email, hashed_password, full_name, is_active, is_superuser, created_at, updated_at)
                    VALUES (:id, :email, :hashed_password, :full_name, :is_active, :is_superuser, NOW(), NOW())
                """),
                {
                    "id": user_id,
                    "email": email,
                    "hashed_password": hashed_pw,
                    "full_name": full_name,
                    "is_active": True,
                    "is_superuser": is_superuser
                }
            )

        await engine.dispose()

        # Success message without echoing credentials
        print("User created successfully!")
        logger.info(f"user_created | User created successfully")
        return True

    except Exception as e:
        # Log error without exposing sensitive data
        logger.error(f"user_creation_failed | Database error occurred: {type(e).__name__}")
        print(f"Error: Failed to create user. Check logs for details.")
        return False


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a new user in the database securely.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Environment Variables:
  ADMIN_EMAIL       User's email address (fallback if --email not provided)
  ADMIN_PASSWORD    User's password (fallback if --password not provided)
  ADMIN_FULL_NAME   User's full name (fallback if --full-name not provided)
  DATABASE_URL      PostgreSQL connection string (required)

Examples:
  # Interactive password input:
  python scripts/create_user.py --email admin@example.com --full-name "Admin User"

  # With environment variables (for CI/CD):
  ADMIN_EMAIL=admin@example.com ADMIN_PASSWORD=SecurePass123! python scripts/create_user.py

  # All via CLI (not recommended for production):
  python scripts/create_user.py -e admin@example.com -p "SecurePass123!" -n "Admin User"
"""
    )

    parser.add_argument(
        "-e", "--email",
        type=str,
        help="User's email address"
    )
    parser.add_argument(
        "-p", "--password",
        type=str,
        help="User's password (use env var or interactive prompt for security)"
    )
    parser.add_argument(
        "-n", "--full-name",
        type=str,
        default="",
        help="User's full name (default: 'Admin User')"
    )
    parser.add_argument(
        "--no-superuser",
        action="store_true",
        help="Create as regular user instead of superuser"
    )

    return parser.parse_args()


def main() -> int:
    """
    Main entry point for user creation script.

    Returns:
        int: Exit code (0 for success, 1 for failure)
    """
    args = parse_args()

    # Get credentials from args, env vars, or interactive prompt
    email, password, full_name = get_credentials_from_args_or_env(args)

    # Validate email format
    email_valid, email_error = validate_email(email)
    if not email_valid:
        print(f"Validation Error: {email_error}")
        return 1

    # Validate password strength
    password_valid, password_issues = validate_password_strength(password)
    if not password_issues:
        # The existing function checks for min 8 chars, we need 12
        if len(password) < 12:
            password_issues.append("Password must be at least 12 characters long")
            password_valid = False
    elif len(password) < 12:
        password_issues = ["Password must be at least 12 characters long"] + password_issues
        password_valid = False

    if not password_valid:
        print("Validation Error: Password does not meet strength requirements:")
        for issue in password_issues:
            print(f"  - {issue}")
        return 1

    # Create user
    is_superuser = not args.no_superuser
    success = asyncio.run(create_user(email, password, full_name, is_superuser))

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
