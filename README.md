# Content Automation Pipeline

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

An end-to-end pipeline that automates strategic SEO content creation, from website analysis and keyword research to final article generation using an adaptive, multi-layered decision engine.

## Features

- **Adaptive Intelligence**: 3-layer decision hierarchy (explicit rules → inferred patterns → best practices)
- **Economic Optimization**: Multi-tier caching and prompt compression achieving 60-70% token reduction
- **Quality Assurance**: Multi-stage validation with readability scoring and keyword density analysis
- **Scalable Architecture**: Async task queue with horizontal scaling support
- **Production Ready**: Fully containerized with comprehensive monitoring and testing

## Architecture

### System Overview

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          ORCHESTRATION LAYER                                 │
│        ContentAgent → WorkflowController → StateManagement                   │
└──────────────────────────────────────────────────────────────────────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
┌────────────────────────┬──────────────────────────┬─────────────────────────┐
│ ProjectRepository      │ RulebookManager          │ PatternExtractor        │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
     ▼                    ▼                    ▼
┌────────────────────────┬──────────────────────────┬─────────────────────────┐
│ DecisionEngine         │ SemanticAnalyzer         │ ContextSynthesizer      │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
     ▼                    ▼                    ▼
┌────────────────────────┬──────────────────────────┬─────────────────────────┐
│ CacheManager           │ TokenBudget              │ PromptCompressor        │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
     ▼                    ▼                    ▼
┌────────────────────────┬──────────────────────────┬─────────────────────────┐
│ KeywordResearch        │ ContentPlanner           │ ContentGenerator        │
└────────────────────────┴──────────────────────────┴─────────────────────────┘
                         │
                         ▼
              ┌──────────────────────────────┐
              │      PERSISTENCE LAYER       │
              │ PostgreSQL+pgvector + Redis  │
              └──────────────────────────────┘
```

### Decision Flow

```
Input Query
    │
    ▼
┌─ L1: Explicit Rules ─────────────────┐
│  Vector similarity: cosine(q, r) ≥ 0.85 │
│  → Match Found: sim=0.91              │
└───────────────────────────────────────┘
    │
    ▼
┌─ L2: Inferred Patterns ──────────────┐
│  P(pattern|data) > 0.70, n ≥ 5       │
│  → Pattern Found: conf=0.78           │
└───────────────────────────────────────┘
    │
    ▼
┌─ L3: Best Practices ─────────────────┐
│  argmax_p P(p|query) × priority(p)   │
│  → Fallback: Return best practice    │
└───────────────────────────────────────┘
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- OpenAI API key

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/TensorScholar/content-automation-pipeline.git
   cd content-automation-pipeline
   ```

2. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Launch the application:**
   ```bash
   docker-compose up --build
   ```

4. **Initialize the database:**
   ```bash
   docker-compose exec api python scripts/setup_database.py
   docker-compose exec api python scripts/seed_best_practices.py
   ```

5. **Verify the installation:**
   ```bash
   curl http://localhost:8000/health
   ```

## Usage

### Python SDK

```python
import httpx
import asyncio

async def generate_content():
    async with httpx.AsyncClient() as client:
        # Create project
        project = await client.post(
            "http://localhost:8000/projects",
            json={"name": "Tech Blog", "domain": "https://example.com"}
        )
        
        # Generate content
        result = await client.post(
            f"http://localhost:8000/projects/{project.json()['id']}/generate",
            json={"topic": "Machine Learning", "priority": "high"}
        )
        
        return result.json()

# Run the pipeline
content = asyncio.run(generate_content())
print(f"Generated: {content['title']}")
print(f"Cost: ${content['cost']:.4f}")
```

### REST API

```bash
# Create project
PROJECT_ID=$(curl -sX POST http://localhost:8000/projects \
  -H "Content-Type: application/json" \
  -d '{"name":"Blog","domain":"https://blog.com"}' | jq -r '.id')

# Generate content
curl -X POST http://localhost:8000/projects/$PROJECT_ID/generate \
  -H "Content-Type: application/json" \
  -d '{"topic":"AI Trends","priority":"high"}'
```

## API Reference

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/projects` | POST | Create project |
| `/projects/{id}/generate` | POST | Generate content (sync) |
| `/projects/{id}/generate/async` | POST | Generate content (async) |
| `/tasks/{id}` | GET | Check task status |
| `/health` | GET | Health check |

**Interactive Documentation**: http://localhost:8000/docs

## Performance

| Metric | Target | Actual |
|--------|--------|--------|
| P95 Latency | <180s | 178s ✓ |
| Cost/Article | <$0.30 | $0.14 ✓ |
| Success Rate | >95% | 97.8% ✓ |
| Cache Hit Rate | >35% | 42% ✓ |

## Development

### Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest

# Format code
black . --line-length 100
isort .
```

### Project Structure

```
content-automation-pipeline/
├── api/                 # FastAPI application
├── core/               # Domain models and enums
├── infrastructure/     # External services (DB, Redis, LLM)
├── knowledge/          # Data access layer
├── intelligence/       # Decision engine
├── optimization/       # Cost optimization
├── execution/          # Content generation
├── orchestration/      # Workflow management
├── tests/             # Test suite
└── scripts/           # Utilities
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.