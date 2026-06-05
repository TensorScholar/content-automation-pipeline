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

print_step() {
    echo -e "${YELLOW}▶ $1${NC}"
}

print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
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
        if ps -p "$pid" > /dev/null 2>&1; then
            print_step "Stopping $service_name (PID: $pid)..."
            kill "$pid" 2>/dev/null || kill -9 "$pid" 2>/dev/null || true
            sleep 2
            if ps -p "$pid" > /dev/null 2>&1; then
                print_error "$service_name still running, forcing..."
                kill -9 "$pid" 2>/dev/null || true
            else
                print_success "$service_name stopped"
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
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
        pids=$(lsof -ti:$port 2>/dev/null || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
        fi
        print_success "$service_name stopped"
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
        echo "$pids" | xargs kill 2>/dev/null || true
        sleep 2
        pids=$(pgrep -f "$pattern" || true)
        if [ -n "$pids" ]; then
            echo "$pids" | xargs kill -9 2>/dev/null || true
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

    # Stop Frontend
    stop_service_by_pid "Frontend" "dashboard"
    stop_service_by_port "Frontend" "$DASHBOARD_PORT"

    # Stop API
    stop_service_by_pid "API" "api"
    stop_service_by_port "API" "$API_PORT"

    # Stop Celery Beat
    stop_service_by_pid "Celery Beat" "celery_beat"
    stop_service_by_pattern "Celery Beat" "celery.*beat"

    # Stop Celery Worker
    stop_service_by_pid "Celery Worker" "celery_worker"
    stop_service_by_pattern "Celery Worker" "celery.*worker"

    # Clean up PID directory
    if [ -d "$PID_DIR" ]; then
        rm -f "$PID_DIR"/*.pid
    fi

    echo ""
    echo -e "${GREEN}╔════════════════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║                    ALL SERVICES STOPPED ✓                          ║${NC}"
    echo -e "${GREEN}╚════════════════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${YELLOW}Note:${NC} Redis left running (system service)"
    echo "      Stop manually if needed: redis-cli shutdown"
    echo ""
}

main "$@"
