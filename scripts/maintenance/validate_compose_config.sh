#!/bin/bash
# ==============================================================================
# Compose Configuration Static Validation
# ==============================================================================
# Safe, reproducible validation that every Docker Compose file renders a valid
# config --quiet output WITHOUT creating a real .env or weakening the
# required-secret checks that use ${VAR:?Required}.
#
# The script exports temporary, clearly non-production placeholder values in
# its own process only. It never:
#   - writes a .env file
#   - prints secret values
#   - starts containers or builds images
#
# Required placeholders (per compose file, derived from ${VAR:?required} usage):
#   POSTGRES_PASSWORD, REDIS_PASSWORD, SECRET_KEY         (docker-compose.yml)
#   REDIS_PASSWORD, SECRET_KEY, CREDENTIAL_ENCRYPTION_KEY,
#   FLOWER_USER, FLOWER_PASSWORD                           (prod)
#   SERVER_NAME                                            (prod https overlay)
#
# docker-compose.prod.https.yml is an overlay and is validated together with
# docker-compose.prod.yml, matching its documented usage:
#   docker compose -f docker-compose.prod.yml -f docker-compose.prod.https.yml
#
# Usage:
#   scripts/maintenance/validate_compose_config.sh
# ==============================================================================

set -euo pipefail

# --- Resolve repository root from any working directory ----------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# --- Temporary, explicit non-production placeholder values -------------------
# These satisfy interpolation only. They are never used to configure a service,
# never written to disk, and never printed below.
export POSTGRES_PASSWORD="validate-postgres-pw-placeholder"
export REDIS_PASSWORD="validate-redis-pw-placeholder"
export SECRET_KEY="validate-secret-key-placeholder"
export CREDENTIAL_ENCRYPTION_KEY="validate-credential-encryption-key-placeholder"
export FLOWER_USER="validate-flower-user-placeholder"
export FLOWER_PASSWORD="validate-flower-pw-placeholder"
export SERVER_NAME="validate.example.internal"

# --- Backward API compatibility: no real env-based required checks weakened ----
# The ${VAR:?required} protections in the compose files remain untouched.

failures=0

# --- Validate each compose configuration ---------------------------------------
label_and_files=(
    "docker-compose.yml|${REPO_ROOT}/docker-compose.yml"
    "docker-compose.prod.yml|${REPO_ROOT}/docker-compose.prod.yml"
    "docker-compose.prod.https.yml (overlay on prod)|${REPO_ROOT}/docker-compose.prod.yml|${REPO_ROOT}/docker-compose.prod.https.yml"
)

for entry in "${label_and_files[@]}"; do
    IFS='|' read -r label base_file extra_file <<<"${entry}"
    if docker compose -f "${base_file}" ${extra_file:+-f ${extra_file}} config --quiet >/dev/null 2>&1; then
        printf 'PASS: %s\n' "${label}"
    else
        printf 'FAIL: %s\n' "${label}"
        failures=$((failures + 1))
    fi
done

if [ "${failures}" -ne 0 ]; then
    printf 'COMPOSE CONFIG VALIDATION FAILED (%d file(s))\n' "${failures}"
    exit 1
fi

printf 'COMPOSE CONFIG VALIDATION PASSED (no containers started, no .env written, no secrets printed)\n'
exit 0