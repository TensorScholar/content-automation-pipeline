import os
from logging.config import fileConfig
from urllib.parse import urlparse, urlunparse

from dotenv import load_dotenv
from sqlalchemy import create_engine, engine_from_config, pool

from alembic import context
from infrastructure.schema import Base
from orchestration import task_persistence as _task_persistence  # noqa: F401

# Load environment variables from .env file
load_dotenv()

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config


def _set_sqlalchemy_url(url: str) -> None:
    """Set URL through ConfigParser, escaping percent-encoded DSN values."""
    config.set_main_option("sqlalchemy.url", url.replace("%", "%%"))


# Override sqlalchemy.url with environment variable if available
database_url = os.getenv("DATABASE_URL")
if database_url:
    # Convert asyncpg URL to psycopg2 for Alembic migrations (sync)
    # Use proper URL parsing to handle special characters in passwords
    try:
        from urllib.parse import parse_qs, urlencode
        parsed = urlparse(database_url)
        # Replace asyncpg driver with psycopg2
        scheme = parsed.scheme.replace("postgresql+asyncpg", "postgresql+psycopg2")
        if scheme == "postgresql":
            scheme = "postgresql+psycopg2"
        # Convert asyncpg-specific 'ssl' param to psycopg2 'sslmode'
        query_params = parse_qs(parsed.query)
        if "ssl" in query_params:
            ssl_val = query_params.pop("ssl")[0]
            if "sslmode" not in query_params:
                sslmode = "require" if ssl_val in ("require", "true", "1") else "disable"
                query_params["sslmode"] = [sslmode]
        new_query = urlencode(query_params, doseq=True)
        # Reconstruct URL with new scheme and cleaned query
        database_url = urlunparse(parsed._replace(scheme=scheme, query=new_query))
        _set_sqlalchemy_url(database_url)
    except Exception as e:
        print(f"Warning: Failed to parse DATABASE_URL: {e}")
        # Fallback to simple replacement
        database_url = database_url.replace("+asyncpg", "").replace("postgresql://", "postgresql+psycopg2://")
        _set_sqlalchemy_url(database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# add your model's MetaData object here
# for 'autogenerate' support
target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    In this scenario we need to create an Engine
    and associate a connection with the context.

    """
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection, target_metadata=target_metadata
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
