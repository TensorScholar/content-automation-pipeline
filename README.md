# Content Automation Pipeline

**Production-grade NLP-driven content automation with adaptive intelligence**

A zero-fault production refactor implementing enterprise-grade reliability, security, observability, and performance optimization for automated content generation workflows.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-1a1a1a.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2d2d2d.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)
[![Code Quality](https://img.shields.io/badge/code%20quality-Ruff%20%7C%20Bandit%20%7C%20MyPy-black)](https://docs.astral.sh/ruff/)

---

## Overview

A comprehensive content automation platform that orchestrates research, planning, and LLM-driven content generation through a multi-layered architecture with production-grade reliability guarantees.

### Enterprise Features

- **Zero-Trust Security**: Environment-only secrets, JWT authentication, Bandit scanning
- **Fault Tolerance**: Tenacity retry with exponential backoff, circuit breakers, dead letter queues
- **Idempotency**: Redis-based deduplication with atomic SET NX EX operations
- **Observability**: Structlog JSON logging, Prometheus metrics, comprehensive audit trails
- **Performance**: Redis query caching, database indexes, token budget management
- **Quality**: Automated CI/CD with Ruff, MyPy, Pytest, Docker build validation
- **Maintainability**: Dependency injection, typed interfaces, pre-commit hooks

### Production Stack

- **API Layer**: FastAPI 0.120+ with async/await, Pydantic validation
- **Task Queue**: Celery 5.3+ with Redis broker, priority queues, acks_late
- **Data Storage**: PostgreSQL 16 with SQLAlchemy Core, Alembic migrations, composite indexes
- **Caching**: Redis 7 with query caching, idempotency keys, circuit breaker state
- **LLM Integrations**: OpenAI GPT-4, Anthropic Claude 3, LiteLLM routing
- **Monitoring**: Prometheus metrics, Structlog JSON, task result audit trail
- **CI/CD**: GitHub Actions with lint, security scan, type check, tests, Docker build

---

## Architecture

### System Topology

A six-layer architecture with dependency injection, idempotent task execution, and comprehensive observability:

```
┌──────────────────────────────────────────────────────────────────────┐
│                        API LAYER (FastAPI)                           │
│  JWT Auth │ Rate Limiting │ Tracing │ Error Handling │ OpenAPI     │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────────────┐
│                   ORCHESTRATION (Celery + Redis)                     │
│  ContentAgent │ Task Queues │ Idempotency │ DLQ │ Audit Trail      │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────────────┐
│                   KNOWLEDGE (PostgreSQL + Cache)                     │
│  ProjectRepo │ RulebookMgr │ PatternExtractor │ Query Cache         │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────────────┐
│                  INTELLIGENCE (Decision Engine)                      │
│  SemanticAnalyzer │ ContextSynthesizer │ BestPracticesKB            │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────────────┐
│               OPTIMIZATION (Caching + Budget + Routing)              │
│  CacheManager │ TokenBudget │ PromptCompressor │ ModelRouter        │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────────────┐
│           EXECUTION (Research → Planning → Generation)               │
│  KeywordResearcher │ ContentPlanner │ ContentGenerator              │
└────────────────────────┬─────────────────────────────────────────────┘
                         │
┌────────────────────────┴─────────────────────────────────────────────┐
│               INFRASTRUCTURE (External Services)                     │
│  PostgreSQL │ Redis │ LLM Client │ Monitoring │ Logging             │
└──────────────────────────────────────────────────────────────────────┘
```

### Key Design Patterns

- **Dependency Injection**: `container.py` with Dependency Injector wires all components
- **Repository Pattern**: Knowledge layer abstracts database operations
- **Circuit Breaker**: LLM client with Redis-backed failure tracking
- **Idempotency**: Task deduplication with atomic Redis operations
- **CQRS**: Read-optimized query cache, write-through to PostgreSQL
- **Retry with Backoff**: Tenacity library for external API resilience
- **Dead Letter Queue**: Failed tasks routed for manual review
- **Audit Trail**: All task executions persisted to `task_results` table

---

## Zero-Fault Production Refactor

This project underwent a comprehensive 7-phase refactor to achieve production-grade reliability:

### Phase 1: Environment & Dependency Hardening ✅

**Consolidated Tooling & Reproducible Builds**

- Replaced Black/Flake8/isort with **Ruff 0.5.7** (10-100x faster)
- Multi-stage **Dockerfile** with Poetry caching layers
- Production **docker-compose.prod.yml** with health checks
- Removed hard-pinned versions, Poetry manages lockfile

**Changes:**
- `pyproject.toml`: Added Ruff configuration ([tool.ruff])
- `Dockerfile`: Multi-stage build with poetry export + slim runtime
- `docker-compose.prod.yml`: PostgreSQL + Redis + API + Celery
- Removed: `.flake8`, `.pylintrc`, `black.toml`

### Phase 2: Zero-Trust Configuration & Security ✅

**Environment-Only Secrets & JWT Authentication**

- **Pydantic Settings** with env-only validation (no defaults)
- JWT authentication on ALL routes (except health/metrics)
- Passwords hashed with **bcrypt** (passlib)
- Rate limiting middleware on API endpoints

**Changes:**
- `config/settings.py`: Pydantic Settings BaseSettings with env-only secrets
- `security.py`: JWT token generation/validation, password hashing
- `api/routes/auth.py`: Login endpoint returning JWT
- `api/main.py`: JWT dependency on protected routes

### Phase 3: Resilient Infrastructure & Observability ✅

**Database Migrations, Retry Logic, Structured Logging**

- **Alembic** for schema migrations with asyncpg→psycopg2 conversion
- **Tenacity** retry with exponential backoff (multiplier=1, min=2s, max=60s)
- **Structlog** JSON logging with ISO 8601 timestamps
- Replaced print statements with structured logs

**Changes:**
- `alembic/env.py`: PostgreSQL migration support with psycopg2
- `infrastructure/llm_client.py`: @retry decorator with exponential backoff
- `api/main.py`: Structlog initialization with JSON processor
- `pyproject.toml`: Added alembic ^1.13.1, tenacity ^8.2.3, structlog ^24.1.0

### Phase 4: Performant Data Access ✅

**Database Indexes & Redis Query Caching**

- **17+ indexes** (9 composite, 8+ individual) on hot query paths
- Redis **query cache decorator** with TTL-based invalidation
- Cache warming on application startup
- Indexed: project_id, status, created_at, rulebook_id, priority

**Changes:**
- `infrastructure/schema.py`: Added Index() for composite and single-column indexes
- `optimization/query_cache.py`: @cached_query decorator with MD5 key generation
- `knowledge/project_repository.py`: Added redis_client, @cached_query(ttl=600)
- `knowledge/article_repository.py`: Added redis_client, @cached_query(ttl=300)

### Phase 5: Idempotent & Atomic Task Execution ✅

**Redis Deduplication & Task Audit Trail**

- **Idempotency keys** with Redis SET NX EX atomic operations
- **TaskResultRepository** with 14-column audit trail
- Task lifecycle hooks: on_success, on_failure, on_retry
- **Dead Letter Queue** for permanently failed tasks
- Database migration for `task_results` table

**Changes:**
- `orchestration/tasks.py`: generate_idempotency_key(), check_idempotency(), route_to_dlq()
- `orchestration/task_persistence.py`: TaskResultRepository with CRUD + querying
- `orchestration/celery_app.py`: Added "dead_letter" queue
- `alembic/versions/001_*.py`: Migration for task_results table
- `tests/phase5_validation.py`: Comprehensive validation tests

### Phase 6: Automated Quality & Security Pipeline ✅

**CI/CD with GitHub Actions**

- **GitHub Actions workflow** with 8 parallel/sequential jobs
- Ruff linting + formatting (blocking)
- **Bandit security scan** (warning)
- **MyPy type checking** (warning)
- Pytest with PostgreSQL + Redis services (blocking)
- **Safety dependency audit** (warning)
- Docker build validation (blocking)
- **Pre-commit hooks** for local enforcement

**Changes:**
- `.github/workflows/ci.yml`: 8-stage pipeline with caching
- `.bandit`: Configuration excluding false positives
- `.pre-commit-config.yaml`: Ruff, Bandit, standard hooks
- `docs/CODE_QUALITY.md`: Comprehensive quality documentation

### Phase 7: Final README & Knowledge Transfer ✅

**Production Deployment Documentation**

- Updated README with refactoring phases
- Deployment guide (Docker Compose + Kubernetes)
- Monitoring setup (Prometheus + Grafana)
- Troubleshooting runbook
- Performance tuning guidelines

---

---

## Quickstart

### Local Development

```bash
# Clone repository
git clone https://github.com/TensorScholar/content-automation-pipeline.git
cd content-automation-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install Poetry
pip install poetry

# Install dependencies
poetry install --no-root

# Set environment variables (create .env file)
cp .env.example .env
# Edit .env with your credentials:
# - DATABASE_URL
# - REDIS_URL
# - OPENAI_API_KEY
# - JWT_SECRET_KEY

# Run database migrations
poetry run alembic upgrade head

# Start API server
poetry run uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# In another terminal, start Celery worker
poetry run celery -A orchestration.celery_app.app worker --loglevel=info -Q default,high,low
```

**Interactive API Docs**: http://localhost:8000/docs  
**OpenAPI Spec**: http://localhost:8000/openapi.json  
**Health Check**: http://localhost:8000/health  
**Metrics**: http://localhost:8000/metrics

### Docker Compose (Recommended)

```bash
# Production deployment
docker-compose -f docker-compose.prod.yml up -d

# View logs
docker-compose -f docker-compose.prod.yml logs -f

# Stop services
docker-compose -f docker-compose.prod.yml down
```

**Services Started:**
- PostgreSQL 16 (port 5432)
- Redis 7 (port 6379)
- FastAPI (port 8000)
- Celery Worker (background)

---

## Production Deployment

### Environment Variables

Create `.env` file with required secrets:

```bash
# Database
DATABASE_URL=postgresql+asyncpg://user:password@postgres:5432/content_automation

# Redis
REDIS_URL=redis://redis:6379/0
CELERY_BROKER_URL=redis://redis:6379/0
CELERY_RESULT_BACKEND=redis://redis:6379/0

# Security
JWT_SECRET_KEY=<generate-with-openssl-rand-hex-32>

# LLM Providers (at least one required)
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
LITELLM_API_KEY=...

# Optional: Monitoring
PROMETHEUS_PUSHGATEWAY_URL=http://pushgateway:9091
```

### Docker Deployment

```bash
# Build production image
docker build -t content-automation-pipeline:latest .

# Run with docker-compose
docker-compose -f docker-compose.prod.yml up -d

# Scale workers
docker-compose -f docker-compose.prod.yml up -d --scale celery_worker=4

# View logs
docker-compose logs -f api
docker-compose logs -f celery_worker
```

### Kubernetes Deployment

```yaml
# Example k8s deployment (simplified)
apiVersion: apps/v1
kind: Deployment
metadata:
  name: content-automation-api
spec:
  replicas: 3
  selector:
    matchLabels:
      app: content-automation-api
  template:
    metadata:
      labels:
        app: content-automation-api
    spec:
      containers:
      - name: api
        image: content-automation-pipeline:latest
        ports:
        - containerPort: 8000
        env:
        - name: DATABASE_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: database-url
        - name: REDIS_URL
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: redis-url
        - name: JWT_SECRET_KEY
          valueFrom:
            secretKeyRef:
              name: content-automation-secrets
              key: jwt-secret
        livenessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 30
          periodSeconds: 10
        readinessProbe:
          httpGet:
            path: /health
            port: 8000
          initialDelaySeconds: 5
          periodSeconds: 5
        resources:
          requests:
            memory: "512Mi"
            cpu: "500m"
          limits:
            memory: "2Gi"
            cpu: "2000m"
```

---

## Monitoring & Observability

### Prometheus Metrics

Metrics endpoint: `GET /metrics`

**Key Metrics:**
- `http_requests_total{method, endpoint, status}`: Request counts
- `http_request_duration_seconds{method, endpoint}`: Latency histogram
- `celery_task_duration_seconds{task_name}`: Task execution time
- `llm_requests_total{provider, model}`: LLM API calls
- `llm_tokens_used{provider, model}`: Token consumption
- `llm_cost_usd{provider, model}`: API costs

### Grafana Dashboards

```yaml
# Recommended panels:
1. Request Rate: rate(http_requests_total[5m])
2. Error Rate: rate(http_requests_total{status=~"5.."}[5m])
3. P95 Latency: histogram_quantile(0.95, http_request_duration_seconds)
4. Celery Queue Depth: celery_queue_length
5. LLM Cost per Hour: rate(llm_cost_usd[1h]) * 3600
6. Cache Hit Rate: redis_cache_hits / (redis_cache_hits + redis_cache_misses)
```

### Structured Logging

All logs output as JSON for easy parsing:

```json
{
  "event": "content_generation_completed",
  "timestamp": "2024-01-15T10:30:45.123456Z",
  "level": "info",
  "task_id": "abc123",
  "project_id": "proj-456",
  "article_id": "art-789",
  "word_count": 1500,
  "total_cost": 0.0234,
  "duration_seconds": 12.5
}
```

Query with `jq`:
```bash
# View all errors in the last hour
docker-compose logs api --since 1h | jq 'select(.level == "error")'

# Calculate average task duration
docker-compose logs celery_worker | jq -s 'map(select(.duration_seconds)) | map(.duration_seconds) | add / length'
```

### Health Checks

**API Health**: `GET /health`
```json
{
  "status": "healthy",
  "database": "connected",
  "redis": "connected",
  "celery": "active_workers:4"
}
```

**Celery Inspection**:
```bash
# Check active workers
poetry run celery -A orchestration.celery_app.app inspect active

# View task stats
poetry run celery -A orchestration.celery_app.app inspect stats

# Check failed tasks (dead letter queue)
# Query PostgreSQL: SELECT * FROM task_results WHERE status = 'REVOKED' ORDER BY created_at DESC
```

---

## Troubleshooting

### Common Issues

#### 1. Database Connection Errors

```bash
# Check PostgreSQL is running
docker-compose ps postgres

# Verify DATABASE_URL format
# Correct: postgresql+asyncpg://user:pass@host:5432/dbname
# For migrations: Use psycopg2 URL (Alembic auto-converts)

# Run migrations
poetry run alembic upgrade head
```

#### 2. Redis Connection Errors

```bash
# Test Redis connectivity
redis-cli -u redis://localhost:6379 ping

# Clear Redis cache if stale
redis-cli -u redis://localhost:6379 FLUSHDB
```

#### 3. Celery Tasks Stuck

```bash
# Check Celery worker logs
docker-compose logs celery_worker

# Purge all queues (USE WITH CAUTION)
poetry run celery -A orchestration.celery_app.app purge

# Restart workers
docker-compose restart celery_worker
```

#### 4. Idempotency Key Conflicts

```bash
# Check Redis idempotency keys
redis-cli --scan --pattern "idempotency:*"

# Clear specific idempotency key
redis-cli DEL idempotency:<key>

# Idempotency keys expire after 1 hour by default
```

#### 5. LLM API Rate Limits

```python
# Circuit breaker state stored in Redis
# Check: redis-cli GET breaker:openai

# Reset circuit breaker manually
redis-cli DEL breaker:openai
redis-cli DEL breaker:anthropic
```

### Performance Tuning

#### Database Optimization

```sql
-- Check slow queries
SELECT query, mean_exec_time, calls
FROM pg_stat_statements
ORDER BY mean_exec_time DESC
LIMIT 10;

-- Check index usage
SELECT schemaname, tablename, indexname, idx_scan
FROM pg_stat_user_indexes
WHERE idx_scan = 0;

-- Vacuum and analyze
VACUUM ANALYZE;
```

#### Redis Memory Optimization

```bash
# Check memory usage
redis-cli INFO memory

# Set maxmemory policy (evict LRU when full)
redis-cli CONFIG SET maxmemory-policy allkeys-lru
redis-cli CONFIG SET maxmemory 2gb
```

#### Celery Concurrency

```bash
# Increase workers for high throughput
docker-compose up -d --scale celery_worker=8

# Adjust prefetch multiplier (default: 1)
# Set worker_prefetch_multiplier=4 in celery_app.py for batching

# Monitor queue length
poetry run celery -A orchestration.celery_app.app inspect active_queues
```

---

## API Reference

### Authentication

All endpoints (except `/health`, `/metrics`) require JWT token:

```bash
# Login to get token
curl -X POST http://localhost:8000/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "password"}'

# Response:
{
  "access_token": "eyJ0eXAiOiJKV1QiLCJhbGc...",
  "token_type": "bearer"
}

# Use token in subsequent requests
curl http://localhost:8000/projects \
  -H "Authorization: Bearer eyJ0eXAiOiJKV1QiLCJhbGc..."
```

### Core Endpoints

**Projects**
- `GET /projects` - List all projects
- `POST /projects` - Create new project
- `GET /projects/{id}` - Get project by ID
- `PUT /projects/{id}` - Update project
- `DELETE /projects/{id}` - Delete project

**Content Generation**
- `POST /content/generate` - Generate content (async)
  ```json
  {
    "project_id": "uuid",
    "topic": "Python FastAPI Best Practices",
    "priority": "high",
    "custom_instructions": "Focus on production deployment"
  }
  ```
- `GET /content/tasks/{task_id}` - Check task status
- `GET /content/articles/{id}` - Get generated article

**System**
- `GET /health` - Health check
- `GET /metrics` - Prometheus metrics
- `GET /system/info` - System information

### Interactive Docs

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **OpenAPI JSON**: http://localhost:8000/openapi.json

---

## Development Guide

### Running Tests

```bash
# Run all tests
poetry run pytest -v

# Unit tests only
poetry run pytest tests/unit -v

# Integration tests (requires PostgreSQL + Redis)
poetry run pytest tests/integration -v

# With coverage report
poetry run pytest --cov=. --cov-report=html --cov-report=term-missing

# Run specific test
poetry run pytest tests/unit/test_llm_client.py::TestLLMClient::test_retry_on_failure -v
```

### Code Quality Checks

```bash
# Install pre-commit hooks
poetry run pre-commit install

# Run all quality checks
poetry run pre-commit run --all-files

# Individual checks
poetry run ruff check .                    # Linting
poetry run ruff format .                   # Formatting
poetry run bandit -r . -c .bandit          # Security scan
poetry run mypy . --ignore-missing-imports # Type checking
```

### Database Migrations

```bash
# Create new migration
poetry run alembic revision --autogenerate -m "add_new_table"

# Apply migrations
poetry run alembic upgrade head

# Rollback migration
poetry run alembic downgrade -1

# View migration history
poetry run alembic history
```

### Load Testing

```bash
# Run Locust load tests
poetry run locust -f locustfile.py --host=http://localhost:8000

# Open browser: http://localhost:8089
# Configure users and spawn rate
```

---

## Project Status

**Production Ready** ✅

All 7 phases of zero-fault refactoring completed:
- ✅ Phase 1: Environment & Dependency Hardening
- ✅ Phase 2: Zero-Trust Configuration & Security
- ✅ Phase 3: Resilient Infrastructure & Observability
- ✅ Phase 4: Performant Data Access
- ✅ Phase 5: Idempotent & Atomic Task Execution
- ✅ Phase 6: Automated Quality & Security Pipeline
- ✅ Phase 7: Final README & Knowledge Transfer

**Test Coverage**: Unit + Integration tests with PostgreSQL/Redis fixtures  
**CI/CD**: GitHub Actions with 8-stage pipeline  
**Documentation**: Comprehensive API docs, deployment guides, troubleshooting  
**Monitoring**: Prometheus metrics, Structlog JSON, audit trails

---

## Development

Local setup uses Poetry (see Quickstart). Adjust or pin versions as needed for your environment. Tests and tooling are evolving.

### Project Structure

```
content-automation-pipeline/
├── api/                    # FastAPI application layer
│   ├── routes/            # HTTP endpoint definitions
│   ├── middleware/        # Request/response processing
│   └── dependencies/      # Dependency injection
├── core/                  # Domain models and type definitions
│   ├── models/           # Pydantic schemas
│   ├── enums/            # Enumeration types
│   └── exceptions/       # Custom exception hierarchy
├── infrastructure/        # External service integrations
│   ├── database/         # PostgreSQL client
│   ├── cache/            # Redis operations
│   └── llm/              # LLM provider abstractions
├── knowledge/            # Knowledge management layer
│   ├── repositories/     # Data access objects
│   ├── rulebooks/        # Rule storage and retrieval
│   └── patterns/         # Pattern extraction logic
├── intelligence/         # Decision-making engine
│   ├── decision/         # Three-tier decision logic
│   ├── semantic/         # Semantic analysis utilities
│   └── context/          # Context synthesis
├── optimization/         # Performance optimization
│   ├── cache/            # Caching strategies
│   ├── tokens/           # Token budget management
│   └── compression/      # Prompt compression
├── execution/            # Content generation pipeline
│   ├── research/         # Keyword research
│   ├── planning/         # Content outline generation
│   └── generation/       # Article synthesis
├── orchestration/        # Workflow coordination
│   ├── agent/            # Agent orchestration
│   ├── workflow/         # Workflow state machine
│   └── tasks/            # Async task management
├── tests/                # Comprehensive test suite
│   ├── unit/            # Unit tests
│   ├── integration/     # Integration tests
│   └── e2e/             # End-to-end tests
└── scripts/              # Utility scripts
    ├── setup_database.py
    └── seed_best_practices.py
```

---

## Contributing

Contributions are welcome. Please open an issue or pull request with a clear description and scope.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for full terms.

---

## Author

**Mohammad Atashi**  
*Principal Engineer*

- Email: mohammadaliatashi@icloud.com
- GitHub: [@TensorScholar](https://github.com/TensorScholar)
- LinkedIn: [Mohammad Atashi](https://linkedin.com/in/mohammadaliatashi)

---
