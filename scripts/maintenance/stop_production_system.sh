#!/bin/bash

################################################################################
# Smarlux Content Automation - Production System Shutdown
# Version: 1.0.0
# Purpose: Gracefully stop all services
################################################################################

set -e

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Configuration - Read from .env to stay in sync with start script
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"
if [ -f "$REPO_ROOT/.env" ]; then
    source <(grep -E '^(API_PORT|DASHBOARD_PORT|UI_PORT)=' "$REPO_ROOT/.env" 2>/dev/null || true)
fi
API_PORT="${API_PORT:-8000}"
DASHBOARD_PORT="${DASHBOARD_PORT:-${UI_PORT:-3001}}"
PID_DIR="./pids"
CELERY_BIN="$REPO_ROOT/.venv/bin/celery"

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

run_celery() {
    if [ -x "$CELERY_BIN" ]; then
        "$CELERY_BIN" "$@"
    else
        poetry run celery "$@"
    fi
}

stop_pid() {
    local service_name=$1
    local pid=$2

    if [ -z "$pid" ]; then
        return 0
    fi

    print_step "Stopping $service_name (PID: $pid)..."
    if ! kill "$pid" 2>/dev/null; then
        print_error "Unable to signal $service_name PID $pid"
        return 1
    fi

    sleep 2
    if kill -0 "$pid" >/dev/null 2>&1; then
        print_error "$service_name still running, forcing PID $pid..."
        kill -9 "$pid" 2>/dev/null || true
        sleep 1
    fi

    if kill -0 "$pid" >/dev/null 2>&1; then
        return 1
    fi
    return 0
}

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════════╗"
    echo "║       SMARLUX CONTENT AUTOMATION - SYSTEM SHUTDOWN                ║"
    echo "╚════════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

stop_service_by_pid() {
    local service_name=$1
    local pid_file="$PID_DIR/$2.pid"

    if [ -f "$pid_file" ]; then
        local pid=$(cat "$pid_file")
        if kill -0 "$pid" >/dev/null 2>&1; then
            if stop_pid "$service_name" "$pid"; then
                print_success "$service_name stopped"
            else
                print_error "$service_name did not stop cleanly"
                rm -f "$pid_file"
                return 1
            fi
        else
            print_step "$service_name not running (stale PID file)"
        fi
        rm -f "$pid_file"
    else
        print_step "$service_name PID file not found"
    fi
}

stop_service_by_port() {
    local service_name=$1
    local port=$2

    print_step "Stopping $service_name on port $port..."
    local pids=$(lsof -ti:$port 2>/dev/null || true)
    if [ -n "$pids" ]; then
        local failed=false
        for pid in $pids; do
            stop_pid "$service_name" "$pid" || failed=true
        done
        local remaining=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$remaining" ]; then
            print_error "$service_name still listening on port $port"
            return 1
        elif [ "$failed" = true ]; then
            print_error "$service_name stop had signaling errors"
            return 1
        else
            print_success "$service_name stopped"
        fi
    else
        print_step "$service_name not running on port $port"
    fi
}

stop_service_by_pattern() {
    local service_name=$1
    local pattern=$2

    print_step "Stopping $service_name..."
    local pids=$(pgrep -f "$pattern" || true)
    if [ -n "$pids" ]; then
        local failed=false
        for pid in $pids; do
            stop_pid "$service_name" "$pid" || failed=true
        done
        pids=$(pgrep -f "$pattern" || true)
        if [ -n "$pids" ]; then
            print_error "$service_name still has matching processes"
            return 1
        elif [ "$failed" = true ]; then
            print_error "$service_name stop had signaling errors"
            return 1
        fi
        print_success "$service_name stopped"
    else
        print_step "$service_name not running"
    fi
}

main() {
    print_header

    echo "Shutting down all Smarlux services..."
    echo ""
    local all_stopped=true

    # Stop Frontend
    stop_service_by_pid "Frontend" "dashboard" || all_stopped=false
    stop_service_by_port "Frontend" "$DASHBOARD_PORT" || all_stopped=false

    # Stop API
    stop_service_by_pid "API" "api" || all_stopped=false
    stop_service_by_port "API" "$API_PORT" || all_stopped=false

    # Stop Celery Beat
    stop_service_by_pid "Celery Beat" "celery_beat" || all_stopped=false
    stop_service_by_pattern "Celery Beat" "celery.*beat" || all_stopped=false

    # Stop Celery Worker
    if run_celery -A orchestration.celery_app.app inspect ping --timeout=2 2>/dev/null | grep -q "pong"; then
        print_step "Requesting Celery workers to shut down..."
        run_celery -A orchestration.celery_app.app control shutdown >/dev/null 2>&1 || true
        sleep 3
    fi
    stop_service_by_pid "Celery Worker" "celery_worker" || all_stopped=false
    stop_service_by_pattern "Celery Worker" "celery.*worker" || all_stopped=false

    # Clean up PID directory
    if [ -d "$PID_DIR" ]; then
        rm -f "$PID_DIR"/*.pid
    fi

    echo ""
    if [ "$all_stopped" = true ]; then
        echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${GREEN}║                    ALL SERVICES STOPPED ✓                          ║${NC}"
        echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        echo -e "${YELLOW}Note:${NC} Redis left running (system service)"
        echo "      Stop manually if needed: redis-cli shutdown"
        echo ""
    else
        echo -e "${RED}╔════════════════════════════════════════════════════════════════════╗${NC}"
        echo -e "${RED}║               SHUTDOWN INCOMPLETE - CHECK LOGS/PIDS                ║${NC}"
        echo -e "${RED}╚════════════════════════════════════════════════════════════════════╝${NC}"
        echo ""
        return 1
    fi
}

main "$@"
