#!/usr/bin/env bash
# Unified launch gate for the P0+P1+P2 Smarlux production candidate.
# Usage:
#   scripts/maintenance/launch_release_gate.sh --static-only
#   scripts/maintenance/launch_release_gate.sh --full

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
run_check() { local label="$1"; shift; if "$@"; then pass "$label"; else fail "$label"; fi; }

printf 'Smarlux launch release gate (%s)\nRepository: %s\n\n' "$MODE" "$ROOT"

if [[ -d .git ]]; then
  run_check "Git diff whitespace validation" git diff --check
  if [[ "$MODE" == "--full" ]]; then
    if [[ -z "$(git status --porcelain)" ]]; then pass "Clean release working tree"; else fail "Release working tree is dirty"; fi
  fi
else
  warn "No .git directory; immutable-commit checks skipped"
fi

run_check "P0 static release gate" scripts/maintenance/p0_release_gate.sh --static-only
run_check "P1/P2 architectural invariants" "$PYTHON_BIN" scripts/maintenance/p1_p2_static_invariants.py
run_check "Python syntax compilation" "$PYTHON_BIN" -m compileall -q \
  api config core execution infrastructure intelligence knowledge orchestration services tests alembic scripts
run_check "Alembic graph validation" "$PYTHON_BIN" scripts/maintenance/validate_migration_graph.py

if "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path
import yaml
for path in (
    Path("docker-compose.yml"),
    Path("docker-compose.prod.yml"),
    Path("docker-compose.prod.https.yml"),
    Path("monitoring/alert_rules.yml"),
    Path("monitoring/prometheus.yml"),
):
    yaml.safe_load(path.read_text(encoding="utf-8"))
json.loads(Path("grafana/dashboards/content-automation-dashboard.json").read_text(encoding="utf-8"))
print("configuration parsed")
PY
then pass "YAML/JSON configuration parsing"; else fail "YAML/JSON configuration parsing"; fi

if [[ -d frontend/node_modules ]] \
  && [[ -f frontend/node_modules/typescript/package.json ]] \
  && [[ -f frontend/node_modules/react/package.json ]] \
  && [[ -f frontend/node_modules/next/package.json ]]; then
  if (cd frontend && npm run typecheck && npm run lint && npm run build); then
    pass "Frontend typecheck, lint, and production build"
  else
    fail "Frontend dependency-backed validation"
  fi
elif [[ "$MODE" == "--static-only" ]]; then
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
    compilerOptions: {
      jsx: ts.JsxEmit.ReactJSX,
      target: ts.ScriptTarget.ES2022,
      module: ts.ModuleKind.ESNext,
    },
  });
  for (const diagnostic of result.diagnostics || []) {
    if (diagnostic.category === ts.DiagnosticCategory.Error) {
      errors.push(`${file}: ${ts.flattenDiagnosticMessageText(diagnostic.messageText, ' ')}`);
    }
  }
}
if (errors.length) { console.error(errors.join('\n')); process.exit(1); }
console.log(`transpiled ${files.length} TypeScript/TSX files`);
JS
  then
    pass "Frontend TypeScript/TSX syntax fallback"
    warn "Full frontend dependency-backed build remains an environment gate"
  else
    fail "Frontend TypeScript/TSX syntax fallback"
  fi
else
  fail "Full launch gate requires complete frontend/node_modules"
fi

if [[ "$MODE" == "--full" ]]; then
  if "$PYTHON_BIN" - <<'PY'
import sys
raise SystemExit(0 if sys.version_info[:2] in {(3, 11), (3, 12)} else 1)
PY
  then pass "Supported Python runtime (3.11 or 3.12)"; else fail "Full gate requires Python 3.11 or 3.12"; fi

  if command -v poetry >/dev/null 2>&1; then TEST=(poetry run pytest); else TEST=("$PYTHON_BIN" -m pytest); fi
  if "${TEST[@]}" -q \
      tests/test_p0_integration_reliability.py \
      tests/test_p1_p2_launch_quality.py \
      tests/test_performance_feedback_service.py \
      tests/test_publishing_safety.py \
      tests/integration/test_wordpress_distribution.py \
      --maxfail=1; then
    pass "Focused P0/P1/P2 automated tests"
  else
    fail "Focused P0/P1/P2 automated tests"
  fi

  if command -v docker >/dev/null 2>&1 && docker compose version >/dev/null 2>&1; then
    run_check "Production Compose interpolation" docker compose -f docker-compose.prod.yml config -q
  else
    fail "Docker Compose is required for full launch validation"
  fi

  if [[ "${RUN_BACKUP_RESTORE_GATE:-0}" == "1" ]]; then
    run_check "Disposable backup/restore drill" scripts/maintenance/verify_backup_restore.sh
  else
    fail "Set RUN_BACKUP_RESTORE_GATE=1 and execute the disposable backup/restore drill"
  fi

  if [[ "${RUN_BROWSER_GATE:-0}" == "1" ]]; then
    run_check "Critical browser canary" "$PYTHON_BIN" scripts/maintenance/browser_release_canary.py
  else
    fail "Set RUN_BROWSER_GATE=1 and execute the critical browser canary"
  fi
fi

if grep -RIE --exclude-dir=.git --exclude-dir=node_modules --exclude-dir=.next \
  --exclude-dir=.venv --exclude-dir=.venv_clean \
  --exclude='*.lock' --exclude='*VALIDATION*.md' \
  '(AIza[0-9A-Za-z_-]{30,}|sk-[A-Za-z0-9]{20,}|-----BEGIN (RSA |EC |OPENSSH )?PRIVATE KEY-----)' \
  . >/tmp/smarlux-launch-secret-scan.txt 2>/dev/null; then
  fail "Potential committed secret material detected"
else
  pass "High-confidence secret pattern scan"
fi
rm -f /tmp/smarlux-launch-secret-scan.txt

printf '\nLaunch gate summary: %d passed, %d warnings, %d failed\n' "$PASS_COUNT" "$WARN_COUNT" "$FAIL_COUNT"
if [[ "$FAIL_COUNT" -gt 0 ]]; then exit 1; fi
printf 'SMARLUX_LAUNCH_RELEASE_GATE_PASS\n'
