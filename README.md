# Content Automation Pipeline

A work-in-progress project exploring automated content generation using FastAPI, Celery, Redis, PostgreSQL, and LLM providers.

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-1a1a1a.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-2d2d2d.svg)](https://opensource.org/licenses/MIT)

---

## Overview (honest and limited scope)

This repository contains the application code for a content automation service. It exposes HTTP APIs (FastAPI) and asynchronous workers (Celery) that coordinate data collection, planning, and text generation via external LLMs. The project is under active development and not intended for production use yet.

### Current components

- FastAPI app with basic routes (projects, content, system)
- Celery worker integration (async job execution)
- PostgreSQL and Redis clients
- Initial LLM client with retries, simple caching, and budget accounting
- Basic metrics endpoint (Prometheus format)

---

## Architecture (high level)

### System Topology

The system is composed of six primary subsystems orchestrated through a layered architecture:

```
┌─────────────────────────────────────────────────────────────────┐
│                      ORCHESTRATION LAYER                        │
│  ContentAgent → WorkflowController → StateManagement           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                      KNOWLEDGE SUBSTRATE                         │
│  ProjectRepository ← RulebookManager ← PatternExtractor        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    INTELLIGENCE KERNEL                           │
│  DecisionEngine → SemanticAnalyzer → ContextSynthesizer        │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                   OPTIMIZATION SUBSTRATE                         │
│  CacheManager → TokenBudget → PromptCompressor                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                     EXECUTION PIPELINE                           │
│  KeywordResearch → ContentPlanner → ContentGenerator           │
└────────────────────────┬────────────────────────────────────────┘
                         │
┌────────────────────────┴────────────────────────────────────────┐
│                    PERSISTENCE LAYER                             │
│              PostgreSQL ←→ Redis Cache                          │
└─────────────────────────────────────────────────────────────────┘
```

### Notes

The design is evolving. Internals may change while the core HTTP/API surface is stabilized.

---

## Quickstart (local, development only)

```bash
python -m venv venv
source venv/bin/activate
pip install poetry
poetry install --no-root
uvicorn api.main:app --reload
```

---

## Usage (local/dev only)

When running locally, interactive API docs are available at:
http://localhost:8000/docs

---

## API Reference (local)

Interactive docs: [http://localhost:8000/docs](http://localhost:8000/docs)
OpenAPI: [http://localhost:8000/openapi.json](http://localhost:8000/openapi.json)

---

## Status

Prototype under development. No performance claims are made at this stage.

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
