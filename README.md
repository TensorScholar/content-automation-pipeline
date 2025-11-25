# Content Automation Pipeline

**Intelligent SEO content generation platform powered by advanced NLP and distributed computing.**

A sophisticated content automation engine that leverages large language models (Claude, GPT-4) with adaptive intelligence layers for enterprise-grade SEO content generation. Built with async-first architecture, fault-tolerant task orchestration, and comprehensive observability.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

Content Automation Pipeline is designed to solve the challenge of generating high-quality, SEO-optimized content at scale. It combines:

- **Multi-Provider LLM Integration** - Seamlessly switch between OpenAI and Anthropic models with automatic failover
- **Intelligent Content Planning** - Semantic analysis and keyword research for strategic content creation
- **Quality Assurance** - Built-in readability scoring, keyword density analysis, and content validation
- **Distributed Processing** - Celery-based task queue for handling large content generation workloads

## Key Features

### Content Intelligence
- **Semantic Analysis** - Deep understanding of content topics using sentence transformers
- **Keyword Research Engine** - Graph-theoretic clustering and PageRank-based keyword scoring
- **Website Pattern Inference** - Automatically learns writing style from existing content
- **Quality Evaluation** - Readability metrics, SEO scoring, and content coherence checks

### Architecture Highlights
- **Circuit Breaker Pattern** - Fault isolation for external API calls with automatic recovery
- **Idempotent Task Execution** - Redis-backed deduplication prevents duplicate content generation
- **Dead Letter Queue** - Failed tasks are captured for manual review and replay
- **Token Budget Management** - Track and control LLM API costs with configurable limits

### Observability
- **Distributed Tracing** - OpenTelemetry integration for request flow visualization
- **Structured Logging** - JSON-formatted logs with correlation IDs
- **Prometheus Metrics** - Request latency, token usage, and cost tracking
- **Health Endpoints** - Comprehensive dependency health checks

## Architecture

\`\`\`
┌─────────────────────────────────────────────────────────────────┐
│                        API Gateway                               │
│                   (FastAPI + Rate Limiting)                      │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   Content     │ │   Project     │ │   Auth        │
│   Service     │ │   Service     │ │   Service     │
└───────┬───────┘ └───────┬───────┘ └───────────────┘
        │                 │
        ▼                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Orchestration Layer                           │
│              (Celery + Redis Task Queue)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              │
│  │ Content     │  │ Website     │  │ Batch       │              │
│  │ Generation  │  │ Analysis    │  │ Processing  │              │
│  └─────────────┘  └─────────────┘  └─────────────┘              │
└─────────────────────────┬───────────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
┌───────────────┐ ┌───────────────┐ ┌───────────────┐
│   LLM Client  │ │   Semantic    │ │   Pattern     │
│ (Claude/GPT)  │ │   Analyzer    │ │   Extractor   │
└───────────────┘ └───────────────┘ └───────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Infrastructure Layer                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐        │
│  │PostgreSQL│  │  Redis   │  │Prometheus│  │  Jaeger  │        │
│  │(asyncpg) │  │ (Cache)  │  │(Metrics) │  │(Tracing) │        │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘        │
└─────────────────────────────────────────────────────────────────┘
\`\`\`

## Quick Start

### Prerequisites

- Python 3.12+
- PostgreSQL 14+
- Redis 7+
- Docker & Docker Compose (optional)

### Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/TensorScholar/content-automation-pipeline.git
cd content-automation-pipeline

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e ".[dev]"
# or using Poetry
poetry install
\`\`\`

### Configuration

\`\`\`bash
# Copy environment template
cp .env.example .env

# Edit with your credentials
nano .env
\`\`\`

Required environment variables:

\`\`\`bash
# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost:5432/content_db

# Redis
REDIS_URL=redis://localhost:6379/0

# LLM Providers (at least one required)
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...

# Security
SECRET_KEY=your-secure-secret-key-min-32-chars
\`\`\`

### Database Setup

\`\`\`bash
# Run migrations
alembic upgrade head

# Seed initial data (optional)
python scripts/setup_database.py
\`\`\`

### Running the Application

**Development Mode:**
\`\`\`bash
# Start API server
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

# Start Celery worker (in another terminal)
celery -A orchestration.celery_app worker --loglevel=info
\`\`\`

**Docker Compose (Recommended):**
\`\`\`bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f api

# Scale workers for higher throughput
docker-compose up -d --scale celery-worker=4
\`\`\`

## API Documentation

Once the server is running, access interactive documentation:

- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc
- **OpenAPI Schema:** http://localhost:8000/openapi.json

### Core Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| \`/api/v1/content/generate\` | POST | Generate content for a topic |
| \`/api/v1/content/batch\` | POST | Batch generate multiple articles |
| \`/api/v1/projects\` | GET/POST | Manage content projects |
| \`/api/v1/projects/{id}/analyze\` | POST | Analyze website patterns |
| \`/health\` | GET | Health check with dependency status |
| \`/system/metrics\` | GET | Prometheus metrics |

### Example: Generate Content

\`\`\`bash
curl -X POST http://localhost:8000/api/v1/content/generate \\
  -H "Content-Type: application/json" \\
  -H "Authorization: Bearer <token>" \\
  -d '{
    "project_id": "123e4567-e89b-12d3-a456-426614174000",
    "topic": "Best practices for cloud-native applications",
    "priority": "high"
  }'
\`\`\`

## Project Structure

\`\`\`
content-automation-pipeline/
├── api/                      # FastAPI application layer
│   ├── main.py              # Application entry point
│   ├── routes/              # API endpoint handlers
│   │   ├── auth.py          # Authentication endpoints
│   │   ├── content.py       # Content generation endpoints
│   │   ├── projects.py      # Project management
│   │   └── system.py        # Health & metrics
│   ├── schemas.py           # Pydantic request/response models
│   └── exceptions.py        # Custom exception handlers
│
├── config/                   # Configuration management
│   ├── settings.py          # Pydantic settings with validation
│   └── constants.py         # Application constants
│
├── core/                     # Domain models and business logic
│   ├── models.py            # SQLAlchemy ORM models
│   ├── enums.py             # Domain enumerations
│   └── exceptions.py        # Domain-specific exceptions
│
├── execution/                # Content generation executors
│   ├── content_generator.py # LLM-based content generation
│   ├── content_planner.py   # Content strategy planning
│   ├── keyword_researcher.py# Keyword analysis engine
│   └── distributer.py       # Content distribution
│
├── infrastructure/           # External service integrations
│   ├── database.py          # Async database management
│   ├── llm_client.py        # Multi-provider LLM client
│   ├── redis_client.py      # Redis connection pooling
│   └── monitoring.py        # Logging and metrics
│
├── intelligence/             # AI/ML intelligence layer
│   ├── semantic_analyzer.py # Sentence transformer analysis
│   ├── quality_evaluator.py # Content quality scoring
│   ├── decision_engine.py   # Adaptive decision making
│   └── context_synthesizer.py# Context aggregation
│
├── knowledge/                # Knowledge management
│   ├── website_analyzer.py  # Website pattern inference
│   ├── pattern_extractor.py # Linguistic pattern extraction
│   └── article_repository.py# Article storage
│
├── optimization/             # Performance optimization
│   ├── cache_manager.py     # Multi-tier caching
│   ├── model_router.py      # Intelligent model selection
│   ├── token_budget_manager.py# Cost control
│   └── prompt_compressor.py # Prompt optimization
│
├── orchestration/            # Task orchestration
│   ├── celery_app.py        # Celery configuration
│   ├── tasks.py             # Async task definitions
│   ├── content_agent.py     # Content workflow orchestration
│   └── task_persistence.py  # Task result tracking
│
├── services/                 # Application services
│   ├── content_service.py   # Content business logic
│   ├── project_service.py   # Project management
│   └── user_service.py      # User management
│
├── tests/                    # Test suite
│   ├── unit/                # Unit tests
│   ├── integration/         # Integration tests
│   └── verification/        # System verification
│
├── scripts/                  # Utility scripts
│   ├── setup_database.py    # Database initialization
│   └── benchmark.py         # Performance benchmarking
│
├── alembic/                  # Database migrations
├── container.py             # Dependency injection container
├── security.py              # Security utilities
├── docker-compose.yml       # Service orchestration
├── Dockerfile               # Container build
└── pyproject.toml           # Project dependencies
\`\`\`

## Development

### Running Tests

\`\`\`bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=. --cov-report=html

# Run specific test categories
pytest tests/unit/ -v
pytest tests/integration/ -v
\`\`\`

### Code Quality

\`\`\`bash
# Linting
ruff check .

# Type checking
mypy .

# Security scan
bandit -r . -x tests

# Format code
ruff format .
\`\`\`

### Load Testing

\`\`\`bash
# Run Locust load tests
locust -f locustfile.py --host=http://localhost:8000
\`\`\`

## Monitoring

### Health Check

\`\`\`bash
curl http://localhost:8000/health
\`\`\`

Response:
\`\`\`json
{
  "status": "healthy",
  "timestamp": "2025-11-25T10:30:00Z",
  "version": "1.0.0",
  "dependencies": {
    "database": "healthy",
    "redis": "healthy"
  }
}
\`\`\`

### Prometheus Metrics

Key metrics exposed at \`/system/metrics\`:

- \`llm_api_calls_total\` - LLM API call count by provider/status
- \`llm_tokens_used_total\` - Token consumption by model
- \`content_generation_duration_seconds\` - Generation latency histogram
- \`workflow_completion_total\` - Workflow success/failure counts

## Deployment

See [DEPLOYMENT.md](DEPLOYMENT.md) for detailed deployment instructions including:

- Docker Compose setup
- Kubernetes deployment
- Environment configuration
- SSL/TLS setup
- Scaling considerations

## Contributing

Contributions are welcome! Please read our contributing guidelines before submitting PRs.

1. Fork the repository
2. Create a feature branch (\`git checkout -b feature/amazing-feature\`)
3. Commit your changes (\`git commit -m 'Add amazing feature'\`)
4. Push to the branch (\`git push origin feature/amazing-feature\`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Author

**Mohammad Atashi** - [GitHub](https://github.com/TensorScholar)

---

Built with ❤️ using FastAPI, Celery, and cutting-edge NLP technology.
