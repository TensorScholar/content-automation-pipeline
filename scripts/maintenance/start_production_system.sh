#!/bin/bash

################################################################################
# Smarlux Content Automation - Production System Startup
# Version: 1.0.0
# Purpose: Start all required services in correct order with health validation
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration — read from .env file so ports are always consistent.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then
    # Source .env safely (only export VAR=value lines, ignore comments/blanks)
    set -a
    source <(grep -E '^[A-Z_]+=.+' "$REPO_ROOT/.env" | sed 's/^/export /')
    set +a
fi

# Use .env values with sensible fallbacks.
API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${UI_PORT:-3001}"
REDIS_PORT="${REDIS_PORT:-6379}"
LOG_DIR="./logs"
PID_DIR="./pids"
PYTHON_BIN="$REPO_ROOT/.venv/bin/python"
CELERY_BIN="$REPO_ROOT/.venv/bin/celery"
UVICORN_BIN="$REPO_ROOT/.venv/bin/uvicorn"
DETACH_HELPER="$REPO_ROOT/scripts/maintenance/detached_process.py"

# Derive API_URL from API_PORT so the frontend always points to the right API.
API_URL="http://127.0.0.1:${API_PORT}"

# Create directories
mkdir -p "$LOG_DIR" "$PID_DIR"

print_config() {
    echo -e "${YELLOW}Configuration (from .env):${NC}"
    echo "  API_PORT      = $API_PORT"
    echo "  DASHBOARD_PORT= $DASHBOARD_PORT"
    echo "  REDIS_PORT    = $REDIS_PORT"
    echo "  API_URL       = $API_URL"
    echo ""
}

################################################################################
# Helper Functions
################################################################################

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║       SMARLUX CONTENT AUTOMATION - PRODUCTION STARTUP v1.0        ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null 2>&1 ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

wait_for_health() {
    local url=$1
    local max_attempts=${2:-30}
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if curl -sf "$url" >/dev/null 2>&1; then
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    return 1
}

json_field() {
    local field="$1"
    python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get(sys.argv[1], ""))' "$field"
}

run_celery() {
    if [ -x "$CELERY_BIN" ]; then
        "$CELERY_BIN" "$@"
    else
        poetry run celery "$@"
    fi
}

process_alive_from_pidfile() {
    local pidfile="$1"
    local pid

    if [ ! -f "$pidfile" ]; then
        return 1
    fi

    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -z "$pid" ]; then
        return 1
    fi

    kill -0 "$pid" 2>/dev/null
}

stop_pidfile_process() {
    local name="$1"
    local pidfile="$2"
    local pid

    if [ ! -f "$pidfile" ]; then
        return 0
    fi

    pid="$(cat "$pidfile" 2>/dev/null || true)"
    if [ -z "$pid" ]; then
        rm -f "$pidfile"
        return 0
    fi

    if kill -0 "$pid" 2>/dev/null; then
        print_step "Stopping existing $name (PID: $pid)..."
        kill "$pid" 2>/dev/null || true
        sleep 2
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
            sleep 1
        fi
    fi

    rm -f "$pidfile"
}

celery_worker_healthy() {
    run_celery -A orchestration.celery_app.app inspect ping --timeout=5 2>/dev/null | grep -q "pong"
}

wait_for_celery_worker() {
    local max_attempts=${1:-30}
    local attempt=1

    while [ $attempt -le $max_attempts ]; do
        if celery_worker_healthy; then
            return 0
        fi
        sleep 1
        ((attempt++))
    done
    return 1
}

shutdown_existing_celery_workers() {
    if run_celery -A orchestration.celery_app.app inspect ping --timeout=2 2>/dev/null | grep -q "pong"; then
        print_step "Requesting existing Celery workers to shut down..."
        run_celery -A orchestration.celery_app.app control shutdown >/dev/null 2>&1 || true
        sleep 3
    fi
}

kill_existing_services() {
    print_step "Checking for existing services..."

    # Kill existing API
    if check_port $API_PORT; then
        print_step "Stopping existing API on port $API_PORT..."
        lsof -ti:$API_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    # Kill existing Frontend
    if check_port $DASHBOARD_PORT; then
        print_step "Stopping existing Frontend on port $DASHBOARD_PORT..."
        lsof -ti:$DASHBOARD_PORT | xargs kill -9 2>/dev/null || true
        sleep 2
    fi

    # Stop Celery processes from pidfiles first. Process-list access can be
    # unavailable in restricted desktop runtimes, but pidfile ownership and
    # Celery's control plane still give us deterministic stop paths.
    stop_pidfile_process "Celery Worker" "$PID_DIR/celery_worker.pid"
    stop_pidfile_process "Celery Beat" "$PID_DIR/celery_beat.pid"
    shutdown_existing_celery_workers
    pkill -f "celery.*orchestration.celery_app" >/dev/null 2>&1 || true
    sleep 2

    print_success "Existing services stopped"
}

################################################################################
# Service Startup Functions
################################################################################

start_redis() {
    print_step "Step 1/6: Checking Redis..."

    if check_port $REDIS_PORT; then
        if redis-cli ping >/dev/null 2>&1; then
            print_success "Redis already running on port $REDIS_PORT"
            return 0
        fi
    fi

    print_step "Starting Redis..."
    if command -v redis-server >/dev/null 2>&1; then
        redis-server --daemonize yes --port $REDIS_PORT
        sleep 2
        if redis-cli ping >/dev/null 2>&1; then
            print_success "Redis started successfully"
        else
            print_error "Redis failed to start"
            return 1
        fi
    else
        print_error "redis-server not found. Install with: brew install redis"
        return 1
    fi
}

start_celery_worker() {
    print_step "Step 2/6: Starting Celery Worker..."

    run_celery -A orchestration.celery_app.app worker \
        --loglevel=info \
        --queues=critical,high,medium,default,low \
        --logfile="$LOG_DIR/celery_worker.log" \
        --pidfile="$PID_DIR/celery_worker.pid" \
        --detach

    if wait_for_celery_worker 30; then
        print_success "Celery Worker started (PID: $(cat $PID_DIR/celery_worker.pid 2>/dev/null || echo 'unknown'))"
    else
        print_error "Celery Worker failed to start"
        return 1
    fi
}

start_celery_beat() {
    print_step "Step 3/6: Starting Celery Beat (Scheduler)..."

    run_celery -A orchestration.celery_app.app beat \
        --loglevel=info \
        --logfile="$LOG_DIR/celery_beat.log" \
        --pidfile="$PID_DIR/celery_beat.pid" \
        --detach

    sleep 2

    if process_alive_from_pidfile "$PID_DIR/celery_beat.pid"; then
        print_success "Celery Beat started (PID: $(cat $PID_DIR/celery_beat.pid 2>/dev/null || echo 'unknown'))"
    else
        print_error "Celery Beat failed to start"
        return 1
    fi
}

start_api() {
    print_step "Step 4/6: Starting Content Automation API on port $API_PORT..."

    if [ -x "$UVICORN_BIN" ]; then
        api_cmd=("$UVICORN_BIN" api.main:app --host 0.0.0.0 --port "$API_PORT")
    else
        api_cmd=(poetry run uvicorn api.main:app --host 0.0.0.0 --port "$API_PORT")
    fi

    "$PYTHON_BIN" "$DETACH_HELPER" \
        --cwd "$REPO_ROOT" \
        --pidfile "$PID_DIR/api.pid" \
        --logfile "$LOG_DIR/api.log" \
        -- "${api_cmd[@]}"

    print_step "Waiting for API health check..."
    if wait_for_health "http://127.0.0.1:$API_PORT/health" 30; then
        print_success "API started successfully on http://127.0.0.1:$API_PORT"
        print_success "API Docs: http://127.0.0.1:$API_PORT/docs"
    else
        print_error "API failed health check"
        return 1
    fi
}

start_dashboard() {
    print_step "Step 5/6: Starting Next.js Frontend on port $DASHBOARD_PORT..."
    print_step "Frontend will connect to API at: $API_URL"

    if [ ! -f "frontend/.next/BUILD_ID" ]; then
        print_step "Building frontend production bundle..."
        API_PROXY_TARGET="$API_URL" \
        NEXT_PUBLIC_API_URL="$API_URL" \
        bash -lc "cd frontend && ./node_modules/.bin/next build" \
            >> "$LOG_DIR/frontend.log" 2>&1
    fi

    "$PYTHON_BIN" "$DETACH_HELPER" \
        --cwd "$REPO_ROOT/frontend" \
        --pidfile "$PID_DIR/dashboard.pid" \
        --logfile "$LOG_DIR/frontend.log" \
        --env "API_PROXY_TARGET=$API_URL" \
        --env "NEXT_PUBLIC_API_URL=$API_URL" \
        -- ./node_modules/.bin/next start -p "$DASHBOARD_PORT"

    sleep 5

    if check_port $DASHBOARD_PORT; then
        print_success "Frontend started successfully"
        print_success "Frontend URL: http://127.0.0.1:$DASHBOARD_PORT"
    else
        print_error "Frontend failed to start"
        return 1
    fi
}

verify_system() {
    print_step "Step 6/6: System Verification..."

    local all_healthy=true

    # Check Redis
    if redis-cli ping >/dev/null 2>&1; then
        print_success "Redis: HEALTHY"
    else
        print_error "Redis: DOWN"
        all_healthy=false
    fi

    # Check Database
    api_health="$(curl -sf "http://127.0.0.1:$API_PORT/health" 2>/dev/null || true)"
    db_status="$(printf "%s" "$api_health" | python3 -c 'import json,sys; data=json.load(sys.stdin); print((data.get("dependencies") or {}).get("database", ""))' 2>/dev/null || true)"
    if [ "$db_status" = "healthy" ]; then
        print_success "PostgreSQL: HEALTHY"
    else
        print_error "PostgreSQL: DOWN"
        all_healthy=false
    fi

    # Check Celery Worker
    if celery_worker_healthy; then
        print_success "Celery Worker: RUNNING"
    else
        print_error "Celery Worker: DOWN"
        all_healthy=false
    fi

    # Check Celery Beat
    if process_alive_from_pidfile "$PID_DIR/celery_beat.pid"; then
        print_success "Celery Beat: RUNNING"
    else
        print_error "Celery Beat: DOWN"
        all_healthy=false
    fi

    # Check API
    if curl -sf "http://127.0.0.1:$API_PORT/health" >/dev/null; then
        print_success "API: HEALTHY (Port $API_PORT)"
    else
        print_error "API: DOWN"
        all_healthy=false
    fi

    # Check Frontend
    if check_port $DASHBOARD_PORT; then
        print_success "Frontend: RUNNING (Port $DASHBOARD_PORT)"
    else
        print_error "Frontend: DOWN"
        all_healthy=false
    fi

    echo ""
    if [ "$all_healthy" = true ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                    ALL SYSTEMS OPERATIONAL ✓                       ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${BLUE}📊 Frontend:${NC}  http://127.0.0.1:$DASHBOARD_PORT"
        echo -e "${BLUE}🔧 API Docs:${NC}  http://127.0.0.1:$API_PORT/docs"
        echo ""
        echo -e "${YELLOW}📝 Logs:${NC}"
        echo "   - API:      tail -f $LOG_DIR/api.log"
        echo "   - Worker:   tail -f $LOG_DIR/celery_worker.log"
        echo "   - Beat:     tail -f $LOG_DIR/celery_beat.log"
        echo "   - Frontend: tail -f $LOG_DIR/frontend.log"
        echo ""
        echo -e "${YELLOW}🛑 Stop all:${NC} ./scripts/maintenance/stop_production_system.sh"
        echo ""
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║               SYSTEM STARTUP FAILED - CHECK LOGS                   ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo "Check logs in: $LOG_DIR/"
        return 1
    fi
}

################################################################################
# Main Execution
################################################################################

main() {
    print_header

    echo "Starting Smarlux Content Automation System..."
    echo ""

    # Show resolved configuration
    print_config

    # Step 0: Clean up
    kill_existing_services

    # Step 1: Redis
    start_redis || { print_error "Failed to start Redis"; exit 1; }

    # Step 2: Celery Worker
    start_celery_worker || { print_error "Failed to start Celery Worker"; exit 1; }

    # Step 3: Celery Beat
    start_celery_beat || { print_error "Failed to start Celery Beat"; exit 1; }

    # Step 4: API
    start_api || { print_error "Failed to start API"; exit 1; }

    # Step 5: Frontend
    start_dashboard || { print_error "Failed to start Frontend"; exit 1; }

    # Step 6: Verification
    echo ""
    verify_system
}

# Run main function
main "$@"
