#!/usr/bin/env bash
set -u

API_URL="${API_URL:-http://127.0.0.1:8000}"
UI_URL="${UI_URL:-http://127.0.0.1:3001}"
REDIS_URL="${REDIS_URL:-}"

failures=0

pass() {
  printf "PASS: %s\n" "$1"
}

fail() {
  printf "FAIL: %s\n" "$1"
  failures=$((failures + 1))
}

warn() {
  printf "WARN: %s\n" "$1"
}

json_field() {
  local field="$1"
  python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get(sys.argv[1], ""))' "$field"
}

printf "Smarlux quick reliability check\n"
printf "API: %s\n" "$API_URL"
printf "UI:  %s\n\n" "$UI_URL"

if command -v pg_isready >/dev/null 2>&1; then
  if pg_isready >/dev/null 2>&1; then
    pass "PostgreSQL accepts connections"
  else
    fail "PostgreSQL is not accepting connections"
  fi
else
  warn "pg_isready not found; skipping PostgreSQL socket check"
fi

if command -v redis-cli >/dev/null 2>&1; then
  if [[ -n "$REDIS_URL" ]]; then
    redis-cli -u "$REDIS_URL" ping >/dev/null 2>&1 && pass "Redis responds to ping" || fail "Redis ping failed"
  else
    redis-cli ping >/dev/null 2>&1 && pass "Redis responds to ping" || fail "Redis ping failed"
  fi
else
  warn "redis-cli not found; skipping Redis socket check"
fi

api_health="$(curl -sS --max-time 5 "$API_URL/health" 2>/dev/null || true)"
if [[ -z "$api_health" ]]; then
  fail "FastAPI /health is not reachable"
else
  api_status="$(printf "%s" "$api_health" | json_field status 2>/dev/null || true)"
  if [[ "$api_status" == "healthy" || "$api_status" == "degraded" ]]; then
    pass "FastAPI /health returned $api_status"
  else
    fail "FastAPI /health returned unexpected status: ${api_status:-unparseable}"
    printf "%s\n" "$api_health"
  fi
fi

ui_code="$(curl -sS --max-time 5 -o /dev/null -w "%{http_code}" "$UI_URL" 2>/dev/null || true)"
if [[ "$ui_code" == "200" ]]; then
  pass "Next.js UI root is reachable"
else
  fail "Next.js UI root returned HTTP ${ui_code:-000}"
fi

proxy_health="$(curl -sS --max-time 5 "$UI_URL/api/health" 2>/dev/null || true)"
if [[ -z "$proxy_health" ]]; then
  fail "Next.js API proxy /api/health is not reachable"
else
  proxy_status="$(printf "%s" "$proxy_health" | json_field status 2>/dev/null || true)"
  if [[ "$proxy_status" == "healthy" || "$proxy_status" == "degraded" ]]; then
    pass "Next.js API proxy returned $proxy_status"
  else
    fail "Next.js API proxy returned unexpected status: ${proxy_status:-unparseable}"
    printf "%s\n" "$proxy_health"
  fi
fi

printf "\n"
if [[ "$failures" -eq 0 ]]; then
  printf "Quick check passed.\n"
  exit 0
fi

printf "Quick check failed with %s issue(s).\n" "$failures"
exit 1
