#!/bin/bash
# Health Check Script for Smarlux Services
# Reads ports from .env to stay in sync with all other scripts

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# ============================================================================
# Read configuration from .env (single source of truth)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
if [ -f "$REPO_ROOT/.env" ]; then
    set -a
    source "$REPO_ROOT/.env"
    set +a
fi

# Extract port from API_URL, fallback to 8000
API_URL="${API_URL:-http://127.0.0.1:8000}"
API_PORT=$(echo "$API_URL" | grep -oE ':[0-9]+' | tail -1 | tr -d ':')
API_PORT="${API_PORT:-8000}"

FRONTEND_PORT="${UI_PORT:-3001}"
REDIS_PORT="${REDIS_PORT:-6379}"

json_field() {
    local field="$1"
    python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get(sys.argv[1], ""))' "$field"
}

echo "🏥 Smarlux Health Check"
echo "======================="
echo ""

# Check API
echo "1️⃣  API Server (Port $API_PORT)"
API_HEALTH="$(curl -s --max-time 5 "http://127.0.0.1:${API_PORT}/health" 2>/dev/null || true)"
API_STATUS="$(printf "%s" "$API_HEALTH" | json_field status 2>/dev/null || true)"
if [ "$API_STATUS" = "healthy" ] || [ "$API_STATUS" = "degraded" ]; then
    echo -e "${GREEN}   ✅ Status: HEALTHY${NC}"
    echo "   URL: http://localhost:${API_PORT}"
    echo "   Docs: http://localhost:${API_PORT}/docs"
else
    echo -e "${RED}   ❌ Status: UNHEALTHY${NC}"
    [ -n "$API_HEALTH" ] && echo "   Response: $API_HEALTH"
    echo "   Try: ./scripts/maintenance/start_production_system.sh"
fi
echo ""

# Check Frontend
echo "2️⃣  Frontend (Port $FRONTEND_PORT)"
if lsof -Pi :"$FRONTEND_PORT" -sTCP:LISTEN -t >/dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Status: RUNNING${NC}"
    echo "   URL: http://localhost:${FRONTEND_PORT}"
else
    echo -e "${RED}   ❌ Status: NOT RUNNING${NC}"
    echo "   Try: ./scripts/maintenance/start_production_system.sh"
fi
echo ""

# Check Redis
echo "3️⃣  Redis (Port $REDIS_PORT)"
if redis-cli -p "$REDIS_PORT" ping >/dev/null 2>&1; then
    echo -e "${GREEN}   ✅ Status: RUNNING${NC}"
else
    echo -e "${RED}   ❌ Status: NOT RUNNING${NC}"
    echo "   Try: brew services start redis"
fi
echo ""

# Check Celery Worker
echo "4️⃣  Celery Worker"
if pgrep -f "celery.*worker" > /dev/null; then
    WORKER_COUNT=$(pgrep -f "celery.*worker" | wc -l)
    echo -e "${GREEN}   ✅ Status: RUNNING ($WORKER_COUNT workers)${NC}"
else
    echo -e "${RED}   ❌ Status: NOT RUNNING${NC}"
    echo "   Try: ./scripts/maintenance/start_production_system.sh"
fi
echo ""

# Check Celery Beat
echo "5️⃣  Celery Beat"
if pgrep -f "celery.*beat" > /dev/null; then
    echo -e "${GREEN}   ✅ Status: RUNNING${NC}"
else
    echo -e "${RED}   ❌ Status: NOT RUNNING${NC}"
    echo "   Try: ./scripts/maintenance/start_production_system.sh"
fi
echo ""

# Check Database
echo "6️⃣  PostgreSQL"
if poetry run python -c "
import asyncio
from infrastructure.database import DatabaseManager
async def check():
    db = DatabaseManager()
    try:
        await db.initialize()
        print('CONNECTED')
        await db.close()
    except Exception:
        print('FAILED')
asyncio.run(check())
" 2>/dev/null | grep -q "CONNECTED"; then
    echo -e "${GREEN}   ✅ Status: CONNECTED${NC}"
else
    echo -e "${RED}   ❌ Status: CONNECTION FAILED${NC}"
    echo "   Check DATABASE_URL in .env"
fi
echo ""

echo "======================="
echo -e "${YELLOW}📋 Config (from .env):${NC}"
echo "   API_URL:        $API_URL"
echo "   Frontend Port:  $FRONTEND_PORT"
echo ""
echo "💡 Quick Commands:"
echo "   Start All:  ./scripts/maintenance/start_production_system.sh"
echo "   Stop All:   ./scripts/maintenance/stop_production_system.sh"
echo "   View Logs:  tail -f logs/api.log"
echo "======================="
