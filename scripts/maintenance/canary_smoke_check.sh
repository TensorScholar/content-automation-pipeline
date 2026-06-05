#!/usr/bin/env bash
# Minimal production canary smoke check.
# Validates auth, project CRUD basics, async generation submission, and task terminal state.

set -euo pipefail

API_URL="${CANARY_API_URL:-http://127.0.0.1:8000}"
EMAIL="${CANARY_EMAIL:-}"
PASSWORD="${CANARY_PASSWORD:-}"
TIMEOUT_SECONDS="${CANARY_TIMEOUT_SECONDS:-240}"
POLL_SECONDS="${CANARY_POLL_SECONDS:-4}"

if [[ -z "$EMAIL" || -z "$PASSWORD" ]]; then
  echo "ERROR: Set CANARY_EMAIL and CANARY_PASSWORD."
  exit 1
fi

echo "Canary smoke started"
echo "API: $API_URL"

json_get() {
  local key="$1"
  python3 -c 'import json,sys; data=json.load(sys.stdin); print(data.get(sys.argv[1], ""))' "$key"
}

auth_resp="$(curl -sS -X POST "$API_URL/auth/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  --data-urlencode "username=$EMAIL" \
  --data-urlencode "password=$PASSWORD")"

token="$(echo "$auth_resp" | json_get "access_token")"
if [[ -z "$token" ]]; then
  echo "ERROR: auth/token failed"
  echo "$auth_resp"
  exit 1
fi
echo "OK: auth/token"

project_name="Canary Smoke $(date +%s)"
create_resp="$(curl -sS -X POST "$API_URL/projects" \
  -H "Authorization: Bearer $token" \
  -H "Content-Type: application/json" \
  -d "{\"name\":\"$project_name\",\"domain\":\"example.com\",\"description\":\"canary smoke\"}")"

project_id="$(echo "$create_resp" | json_get "id")"
if [[ -z "$project_id" ]]; then
  echo "ERROR: project create failed"
  echo "$create_resp"
  exit 1
fi
echo "OK: project create ($project_id)"

task_resp="$(curl -sS -X POST "$API_URL/content/generate/async" \
  -H "Authorization: Bearer $token" \
  -H "Content-Type: application/json" \
  -d "{\"project_id\":\"$project_id\",\"topic\":\"Canary reliability check topic\",\"priority\":\"high\",\"additional_instructions\":\"Return stable, concise output.\"}")"

task_id="$(echo "$task_resp" | json_get "task_id")"
if [[ -z "$task_id" ]]; then
  echo "ERROR: content/generate/async failed"
  echo "$task_resp"
  exit 1
fi
echo "OK: async task submitted ($task_id)"

start_epoch="$(date +%s)"
end_epoch=$((start_epoch + TIMEOUT_SECONDS))
last_state=""

while [[ "$(date +%s)" -lt "$end_epoch" ]]; do
  status_resp="$(curl -sS -X GET "$API_URL/content/task/$task_id" \
    -H "Authorization: Bearer $token")"
  state="$(echo "$status_resp" | json_get "state")"
  ready="$(echo "$status_resp" | json_get "ready")"
  if [[ -n "$state" && "$state" != "$last_state" ]]; then
    echo "Task state: $state"
    last_state="$state"
  fi

  if [[ "$ready" == "True" || "$ready" == "true" ]]; then
    if [[ "$state" == "SUCCESS" ]]; then
      article_id="$(echo "$status_resp" | python3 -c 'import json,sys; d=json.load(sys.stdin); print(((d.get("result") or {}).get("article_id")) or "")')"
      if [[ -z "$article_id" ]]; then
        echo "ERROR: SUCCESS task missing article_id"
        echo "$status_resp"
        exit 1
      fi
      echo "OK: task success ($article_id)"
      echo "Canary smoke PASSED"
      exit 0
    fi
    echo "ERROR: task reached terminal non-success state: $state"
    echo "$status_resp"
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

echo "ERROR: task did not reach terminal state within timeout ($TIMEOUT_SECONDS s)"
exit 1
