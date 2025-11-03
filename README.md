# Content Automation Pipeline

**Production-grade NLP-driven content automation with adaptive intelligence**

Enterprise-grade automated content generation platform powered by FastAPI, Celery, PostgreSQL, Redis, and multiple LLM providers with comprehensive reliability, security, and observability.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-1a1a1a.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.120+-009688.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2d2d2d.svg)](https://opensource.org/licenses/MIT)
[![CI/CD](https://img.shields.io/badge/CI%2FCD-GitHub%20Actions-2088FF.svg)](https://github.com/features/actions)

---

## Features

- **Smart Content Generation**: Multi-stage pipeline with keyword research, content planning, and LLM-powered generation
- **JWT Authentication**: Secure API access with token-based authentication
- **Async Task Processing**: Celery-based distributed task queue with priority queues and dead letter handling
- **Idempotent Operations**: Redis-backed request deduplication prevents duplicate task execution
- **Comprehensive Audit Trail**: Full task execution history with `task_results` table
- **Intelligent Caching**: Multi-layer caching with Redis query cache and LLM response cache
- **Circuit Breakers**: Graceful handling of external service failures with automatic recovery
- **Production Monitoring**: Prometheus metrics, Structlog JSON logging, health checks
- **Automated Quality Gates**: GitHub Actions CI/CD with linting, security scanning, and tests

## Technology Stack

- **API**: FastAPI 0.120+ with async/await, Pydantic validation, OpenAPI docs
- **Task Queue**: Celery 5.3+ with Redis broker, exponential backoff, acks_late
- **Database**: PostgreSQL 16 with SQLAlchemy Core, Alembic migrations, composite indexes
- **Cache**: Redis 7 (query caching, idempotency keys, circuit breaker state)
- **LLM Providers**: OpenAI GPT-4, Anthropic Claude 3, LiteLLM multi-provider routing
- **Monitoring**: Prometheus metrics, Structlog JSON logging
- **CI/CD**: GitHub Actions (Ruff, Bandit, MyPy, Pytest, Docker build)
- **Containerization**: Docker with multi-stage builds, docker-compose for local dev

## Architecture

```
┌──────────────────────────────────────────────┐
│           API Layer (FastAPI)                │
│   JWT Auth │ Async Routes │ OpenAPI         │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────┴───────────────────────────┐
│      Orchestration (Celery + Redis)          │
│   Task Queues │ Idempotency │ DLQ           │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────┴───────────────────────────┐
│    Execution Pipeline (3-stage)              │
│   Research → Planning → Generation           │
└──────────────────┬───────────────────────────┘
                   │
┌──────────────────┴───────────────────────────┐
│    Infrastructure (External Services)        │
│   PostgreSQL │ Redis │ LLM Providers        │
└──────────────────────────────────────────────┘
```

**Design Patterns:**
- Dependency Injection (container.py)
- Repository Pattern (data access layer)
- Circuit Breaker (LLM client resilience)
- CQRS (query cache + write-through)
- Idempotency (Redis atomic operations)

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
