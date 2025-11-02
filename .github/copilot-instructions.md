<!-- .github/copilot-instructions.md - guidance for AI coding agents -->
# Content Automation Pipeline — AI assistant instructions

Short, actionable notes to help a coding AI become productive in this repository.

- Project purpose: a FastAPI + Celery content automation service that coordinates research, planning, and LLM-driven content generation. Key integrations: Redis (broker/cache), PostgreSQL, OpenAI/Anthropic via `infrastructure/llm_client.py`.

- High-level architecture (see `README.md` and `api/main.py`):
  - HTTP/API layer: `api/` (routes in `api/routes/*`, main app in `api/main.py`).
  - Orchestration: `orchestration/` (Celery app in `orchestration/celery_app.py`, tasks in `orchestration/tasks.py`, agent in `orchestration/content_agent.py`).
  - Execution pipeline: `execution/` (keyword research, planning, generation).
  - Intelligence & optimization: `intelligence/`, `optimization/` (decision engine, prompt compression, token budgets).
  - Persistence and infra clients: `infrastructure/` (Postgres, Redis, LLM client).

- Local dev & commands (discoverable from `README.md` / `pyproject.toml`):
  - Create virtualenv + install: `python -m venv venv && source venv/bin/activate && poetry install --no-root`.
  - Run API locally: `uvicorn api.main:app --reload` (interactive docs at `/docs`).
  - Celery worker (development): use the `orchestration.celery_app:app` Celery instance. Example launch (adjust env vars):
    - `CELERY_BROKER_URL=redis://localhost:6379/0 celery -A orchestration.celery_app.app worker --loglevel=info -Q default,high,low`
  - Tests: `pytest` (configured in `pyproject.toml` to use `tests/` paths).
  - Load test script available as `poetry run load-test` (defined in `tool.poetry.scripts`).

- Key patterns & project-specific conventions (examples):
  - Dependency Injection: a central `container` is used (`container.py`) and wired in `api/main.py` and Celery worker init (`orchestration/celery_app.py`) — prefer to fetch services via container helpers (e.g., `get_content_service`, `get_redis`).
  - LLM calls funnel through `infrastructure/llm_client.py` — it implements provider routing, circuit breakers (Redis-backed), caching hooks (`cache_manager`), and daily token accounting. When changing LLM behavior, update this file and any DI wiring.
  - Celery integration: workers initialize the DI container in `worker_process_init` (see `orchestration/celery_app.py`) — avoid heavy global state at import-time; use async container initialization instead.
  - Caching and Redis: Redis is used for multiple roles (rate-limiting, circuit-state, cache). Look for keys like `llm_cache:`, `breaker:openai`, `rate_limit:` to understand cross-cutting state.
  - Metrics: optional `metrics_collector` hooks are present in LLM and orchestration code to emit Prometheus-style metrics. Preserve metric emission points when refactoring.

- Files to inspect first when working on features or bugs:
  - `api/main.py` — app wiring, middleware (tracing/rate limiting/security), route registration.
  - `infrastructure/llm_client.py` — primary LLM abstraction and provider adapters.
  - `orchestration/celery_app.py` & `orchestration/tasks.py` — Celery config and task wiring.
  - `container.py` — dependency injection registration and helpers used across app and workers.
  - `execution/`, `intelligence/`, `optimization/` — pipeline stages (small modules; read top-level docstrings).

- Testing & quick checks for PRs:
  - Run unit tests: `pytest tests/unit -q`.
  - Run a focused integration test: `pytest tests/integration/test_caching.py::TestCaching -q`.
  - Linting/formatting: `black .` (project Black config in `pyproject.toml`).

- When making changes, watch out for these gotchas:
  - Imports that trigger container initialization: many modules expect DI wiring; avoid importing modules that call `container_manager.initialize()` at import time.
  - Async vs sync boundaries: Celery worker processes initialize an asyncio loop explicitly; ensure async functions are awaited and sync wrappers exist for blocking calls.
  - Persistent state keys: Redis keys are used by multiple components — changing naming conventions must be coordinated.
  - LLM cost/accounting logic: `LLMClient` updates daily token/cost counters in Redis; tests may assume these keys exist or are zeroed.

- Minimal PR checklist for the AI assistant to create edits safely:
  1. Run unit tests locally for touched modules (`pytest tests/unit -q`).
 2. Run `pytest -q` for any integration changes affecting Celery/Redis (may require local Redis/Postgres).
 3. Preserve DI usage — prefer adding new services to `container.py` and wiring rather than creating globals.
 4. Add or update tests under `tests/unit` (happy path + 1 edge case) when changing core behavior.

If anything above is unclear or you'd like the file to include more examples (code snippets or common endpoints), tell me which area to expand and I'll iterate.
