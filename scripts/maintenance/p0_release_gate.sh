#!/usr/bin/env bash
# P0 release gate for WordPress, Search Console, frontend, migrations, and deployment config.
# Usage:
#   scripts/maintenance/p0_release_gate.sh --static-only
#   scripts/maintenance/p0_release_gate.sh --full

set -uo pipefail

MODE="${1:---full}"
if [[ "$MODE" != "--static-only" && "$MODE" != "--full" ]]; then
  echo "Usage: $0 [--static-only|--full]" >&2
  exit 2
fi

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

if [[ -x "$ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$ROOT/.venv/bin/python"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python3)"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="$(command -v python)"
else
  printf '[FAIL] No supported Python interpreter found\n' >&2
  exit 1
fi

PASS_COUNT=0
WARN_COUNT=0
FAIL_COUNT=0

pass() { printf '[PASS] %s\n' "$1"; PASS_COUNT=$((PASS_COUNT + 1)); }
warn() { printf '[WARN] %s\n' "$1"; WARN_COUNT=$((WARN_COUNT + 1)); }
fail() { printf '[FAIL] %s\n' "$1" >&2; FAIL_COUNT=$((FAIL_COUNT + 1)); }
run_check() {
  local label="$1"; shift
  if "$@"; then pass "$label"; else fail "$label"; fi
}
require_nonempty() {
  local name="$1"
  if [[ -n "${!name:-}" ]]; then pass "$name is set"; else fail "$name is required"; fi
}

printf 'Smarlux P0 release gate (%s)\n' "$MODE"
printf 'Repository: %s\n\n' "$ROOT"

if [[ -d .git ]]; then
  run_check "Git diff whitespace validation" git diff --check
else
  warn "No .git directory; Git diff validation skipped"
fi

run_check "Python syntax compilation" "$PYTHON_BIN" -m compileall -q \
  api config core execution infrastructure intelligence knowledge orchestration services tests alembic
run_check "Alembic graph has one expected head" "$PYTHON_BIN" scripts/maintenance/validate_migration_graph.py

if "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)
PY
then
  pass "Supported Python runtime (3.11 or 3.12)"
elif [[ "$MODE" == "--static-only" ]]; then
  warn "Current Python is outside supported 3.11-3.12; static checks continue"
else
  fail "Production validation requires Python 3.11 or 3.12"
fi

if "$PYTHON_BIN" - <<'PY'
from pathlib import Path
import yaml
for path in (Path('docker-compose.yml'), Path('docker-compose.prod.yml'), Path('docker-compose.prod.https.yml')):
    payload = yaml.safe_load(path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict) or not isinstance(payload.get('services'), dict):
        raise SystemExit(f'invalid Compose YAML: {path}')
print('compose YAML parsed')
PY
then
  pass "Compose YAML parsing"
else
  fail "Compose YAML parsing"
fi

if grep -q 'Queue("integrations"' orchestration/celery_app.py \
  && grep -Eq '(^|[ ,])-Q[= ]?[^\n]*integrations|critical,high,medium,default,integrations,low' docker-compose.yml docker-compose.prod.yml entrypoint.sh 2>/dev/null; then
  pass "Integration queue declared and consumed"
else
  fail "Celery integrations queue is not both declared and consumed"
fi

run_check "P0 architectural invariants" "$PYTHON_BIN" scripts/maintenance/p0_static_invariants.py
run_check "Production configuration contract (static)" "$PYTHON_BIN" scripts/maintenance/validate_production_config.py --static

if grep -q 'uq_publishing_success_idempotency' infrastructure/schema.py \
  && grep -q '_verify_wordpress_post' execution/distributer.py \
  && grep -q 'reconcile_wordpress_publishes' orchestration/celery_app.py; then
  pass "WordPress P0 persistence, verification, and reconciliation hooks present"
else
  fail "WordPress P0 reliability hooks are incomplete"
fi

if [[ -d frontend/node_modules ]] \
  && [[ -f frontend/node_modules/typescript/package.json ]] \
  && [[ -f frontend/node_modules/react/package.json ]] \
  && [[ -f frontend/node_modules/next/package.json ]]; then
  (
    cd frontend
    npm run typecheck && npm run lint && npm run build
  )
  if [[ $? -eq 0 ]]; then pass "Frontend typecheck, lint, and production build"; else fail "Frontend validation"; fi
else
  if [[ "$MODE" == "--static-only" ]]; then
    if NODE_PATH="$(npm root -g 2>/dev/null)" node - <<'JS'
const fs = require('fs');
const path = require('path');
const ts = require('typescript');
const files = [];
function walk(dir) {
  if (!fs.existsSync(dir)) return;
  for (const entry of fs.readdirSync(dir, {withFileTypes: true})) {
    const full = path.join(dir, entry.name);
    if (entry.isDirectory()) {
      if (!['node_modules', '.next', 'target'].includes(entry.name)) walk(full);
    } else if (/\.(ts|tsx)$/.test(entry.name)) files.push(full);
  }
}
walk('frontend/src'); walk('frontend/app');
const errors = [];
for (const file of files) {
  const result = ts.transpileModule(fs.readFileSync(file, 'utf8'), {
    fileName: file,
    reportDiagnostics: true,
    compilerOptions: {jsx: ts.JsxEmit.ReactJSX, target: ts.ScriptTarget.ES2022, module: ts.ModuleKind.ESNext},
  });
  for (const diagnostic of result.diagnostics || []) {
    if (diagnostic.category === ts.DiagnosticCategory.Error) {
      errors.push(`${file}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, ' ')}`);
    }
  }
}
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log(`transpiled ${files.length} files`);
JS
    then
      pass "Frontend TypeScript/TSX syntax fallback"
      warn "Full frontend dependency-backed typecheck/lint/build not available"
    else
      fail "Frontend TypeScript/TSX syntax fallback"
    fi
  else
    fail "Complete frontend node_modules is required for full validation"
  fi
fi

if [[ "$MODE" == "--full" ]]; then
  require_nonempty SECRET_KEY
  require_nonempty CREDENTIAL_ENCRYPTION_KEY
  require_nonempty DATABASE_URL
  require_nonempty REDIS_URL
  require_nonempty CELERY_BROKER_URL
  require_nonempty CELERY_RESULT_BACKEND
  require_nonempty POSTGRES_PASSWORD
  require_nonempty REDIS_PASSWORD

  if "$PYTHON_BIN" - <<'PY'
import os
from cryptography.fernet import Fernet
key = os.environ.get('CREDENTIAL_ENCRYPTION_KEY', '')
try:
    Fernet(key.encode('ascii'))
except Exception:
    raise SystemExit(1)
PY
  then pass "Credential encryption key is a valid Fernet key"; else fail "Invalid CREDENTIAL_ENCRYPTION_KEY"; fi

  gsc_names=(
    GOOGLE_SEARCH_CONSOLE_CLIENT_ID
    GOOGLE_SEARCH_CONSOLE_CLIENT_SECRET
    GOOGLE_SEARCH_CONSOLE_REDIRECT_URI
    GOOGLE_SEARCH_CONSOLE_FRONTEND_RETURN_URL
  )
  gsc_any=false
  for name in "${gsc_names[@]}"; do [[ -n "${!name:-}" ]] && gsc_any=true; done
  if [[ "$gsc_any" == true ]]; then
    for name in "${gsc_names[@]}"; do require_nonempty "$name"; done
    if [[ "${ENVIRONMENT:-production}" == "production" ]]; then
      if [[ "${GOOGLE_SEARCH_CONSOLE_REDIRECT_URI:-}" == https://* ]] \
        && [[ "${GOOGLE_SEARCH_CONSOLE_FRONTEND_RETURN_URL:-}" == https://* ]]; then
        pass "Search Console production URLs use HTTPS"
      else
        fail "Search Console production redirect and return URLs must use HTTPS"
      fi
    fi
  else
    warn "Search Console is disabled because OAuth variables are not set"
  fi

  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    if docker compose -f docker-compose.prod.yml config -q; then
      pass "Docker Compose production interpolation"
    else
      fail "Docker Compose production interpolation"
    fi
  else
    fail "Docker and Docker Compose are required for full release validation"
  fi

  if command -v poetry >/dev/null 2>&1; then
    TEST=(poetry run pytest)
  else
    TEST=("$PYTHON_BIN" -m pytest)
  fi
  if "${TEST[@]}" -q \
      tests/test_p0_integration_reliability.py \
      tests/test_performance_feedback_service.py \
      tests/test_publishing_safety.py \
      tests/integration/test_wordpress_distribution.py \
      --maxfail=1; then
    pass "Focused P0 automated tests"
  else
    fail "Focused P0 automated tests"
  fi
fi

if grep -RIE --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.next \
  --exclude-dir=.venv --exclude-dir=.venv_clean \
  --exclude='*.lock' --exclude='P0-VALIDATION.md' \
  '(AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)' \
  . >/tmp/smarlux-p0-secret-scan.txt 2>/dev/null; then
  fail "Potential committed secret material detected"
else
  pass "High-confidence secret pattern scan"
fi
rm -f /tmp/smarlux-p0-secret-scan.txt

printf '\nP0 gate summary: %d passed, %d warnings, %d failed\n' "$PASS_COUNT" "$WARN_COUNT" "$FAIL_COUNT"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
  exit 1
fi
printf 'P0_RELEASE_GATE_PASS\n'
