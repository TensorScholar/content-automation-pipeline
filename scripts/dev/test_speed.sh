#!/usr/bin/env bash
set -u

API_URL="${API_URL:-http://127.0.0.1:8000}"
UI_URL="${UI_URL:-http://127.0.0.1:3001}"

measure() {
  local label="$1"
  local url="$2"

  printf "%s\n" "$label"
  curl -sS --max-time 10 -o /dev/null -w "  status=%{http_code} time=%{time_total}s\n" "$url" || true
  printf "\n"
}

printf "Smarlux local response-time check\n"
printf "API: %s\n" "$API_URL"
printf "UI:  %s\n\n" "$UI_URL"

measure "1. FastAPI /health" "$API_URL/health"
measure "2. Next.js root" "$UI_URL"
measure "3. Next.js API proxy /api/health" "$UI_URL/api/health"

printf "Process status\n"
pgrep -fl "uvicorn api.main|next dev|next start|celery.*worker" || true
