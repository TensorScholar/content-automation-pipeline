#!/bin/bash
# =============================================================================
# Docker Entrypoint Script
# Handles database migrations and service startup
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Wait for database to be ready
wait_for_db() {
    echo -e "${YELLOW}Waiting for PostgreSQL to be ready...${NC}"

    local max_attempts=30
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if python -c "
import asyncio
from infrastructure.database import DatabaseManager

async def check():
    db = DatabaseManager()
    try:
        await db.initialize()
        await db.health_check()
        await db.close()
        return True
    except Exception:
        return False

exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null; then
            echo -e "${GREEN}PostgreSQL is ready!${NC}"
            return 0
        fi

        echo "Attempt $attempt/$max_attempts - Database not ready, waiting..."
        sleep 2
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Database connection failed after $max_attempts attempts${NC}"
    exit 1
}

# Wait for Redis to be ready
wait_for_redis() {
    echo -e "${YELLOW}Waiting for Redis to be ready...${NC}"

    local max_attempts=20
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if python -c "
import asyncio
from infrastructure.redis_client import RedisClient

async def check():
    redis = RedisClient()
    try:
        return await redis.health_check()
    except Exception:
        return False

exit(0 if asyncio.run(check()) else 1)
" 2>/dev/null; then
            echo -e "${GREEN}Redis is ready!${NC}"
            return 0
        fi

        echo "Attempt $attempt/$max_attempts - Redis not ready, waiting..."
        sleep 1
        attempt=$((attempt + 1))
    done

    echo -e "${RED}Redis connection failed after $max_attempts attempts${NC}"
    exit 1
}

# Run database migrations
run_migrations() {
    echo -e "${YELLOW}Running database migrations...${NC}"

    if [ -f "alembic.ini" ]; then
        alembic upgrade head
        if [ $? -eq 0 ]; then
            echo -e "${GREEN}Migrations completed successfully!${NC}"
        else
            echo -e "${RED}Migration failed!${NC}"
            exit 1
        fi
    else
        echo -e "${YELLOW}No alembic.ini found, skipping migrations${NC}"
    fi
}

should_run_migrations_for_api() {
    local default_run="true"
    if [ "${ENVIRONMENT:-development}" = "production" ]; then
        default_run="false"
    fi

    [ "${RUN_MIGRATIONS:-$default_run}" = "true" ]
}

# Main entrypoint logic
main() {
    local command="${1:-api}"

    echo "========================================"
    echo "Content Automation Pipeline"
    echo "Starting: $command"
    echo "========================================"

    case "$command" in
        api)
            wait_for_db
            wait_for_redis

            # Production deployments must run migrations explicitly via
            # `entrypoint.sh migrate` or the compose migration service. Running
            # migrations from replicated API startup risks concurrent Alembic
            # upgrades during rolling deploys.
            if should_run_migrations_for_api; then
                run_migrations
            else
                echo -e "${YELLOW}Skipping API startup migrations (RUN_MIGRATIONS=${RUN_MIGRATIONS:-unset}, ENVIRONMENT=${ENVIRONMENT:-development})${NC}"
            fi

            echo -e "${GREEN}Starting API server...${NC}"
            exec gunicorn api.main:app \
                --workers ${GUNICORN_WORKERS:-4} \
                --worker-class uvicorn.workers.UvicornWorker \
                --bind 0.0.0.0:${API_PORT:-8000} \
                --timeout ${GUNICORN_TIMEOUT:-300} \
                --graceful-timeout 30 \
                --access-logfile - \
                --error-logfile - \
                --capture-output
            ;;

        worker)
            wait_for_db
            wait_for_redis

            echo -e "${GREEN}Starting Celery worker...${NC}"
            exec celery -A orchestration.celery_app.app worker \
                --loglevel=${CELERY_LOG_LEVEL:-info} \
                --concurrency=${CELERY_CONCURRENCY:-4}
            ;;

        beat)
            wait_for_db
            wait_for_redis

            echo -e "${GREEN}Starting Celery beat scheduler...${NC}"
            exec celery -A orchestration.celery_app.app beat \
                --loglevel=${CELERY_LOG_LEVEL:-info}
            ;;

        flower)
            wait_for_redis

            echo -e "${GREEN}Starting Flower monitoring...${NC}"
            exec celery -A orchestration.celery_app.app flower \
                --port=${FLOWER_PORT:-5555} \
                --basic_auth=${FLOWER_USER:-admin}:${FLOWER_PASSWORD:-admin}
            ;;

        migrate)
            wait_for_db
            run_migrations
            ;;

        shell)
            echo -e "${GREEN}Starting interactive shell...${NC}"
            exec /bin/bash
            ;;

        *)
            # Pass through any other command
            exec "$@"
            ;;
    esac
}

main "$@"
