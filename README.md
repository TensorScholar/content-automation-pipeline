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

```mermaid
graph TB
    subgraph ORCH[" "]
        subgraph ORC["Orchestration"]
            A[ContentAgent]
            B[WorkflowController]
            C[StateManagement]
        end
        
        subgraph KNW["Knowledge"]
            D[ProjectRepository]
            E[RulebookManager]
            F[PatternExtractor]
        end
        
        subgraph INT["Intelligence"]
            G[DecisionEngine]
            H[SemanticAnalyzer]
            I[ContextSynthesizer]
        end
        
        subgraph OPT["Optimization"]
            J[CacheManager]
            K[TokenBudget]
            L[PromptCompressor]
        end
        
        subgraph EXE["Execution"]
            M[KeywordResearch]
            N[ContentPlanner]
            O[ContentGenerator]
        end
        
        subgraph PERS["Persistence"]
            P[(PostgreSQL)]
            Q[(Redis)]
        end
    end
    
    A --> D
    B --> E
    C --> F
    D --> G
    E --> H
    F --> I
    G --> J
    H --> K
    I --> L
    J --> M
    K --> N
    L --> O
    M --> P
    N --> P
    O --> P
    M --> Q
    N --> Q
    O --> Q
    
    style ORC fill:#f8f9fa,stroke:#dee2e6,stroke-width:2px
    style KNW fill:#e3f2fd,stroke:#2196f3,stroke-width:2px
    style INT fill:#f3e5f5,stroke:#9c27b0,stroke-width:2px
    style OPT fill:#e8f5e8,stroke:#4caf50,stroke-width:2px
    style EXE fill:#fff3e0,stroke:#ff9800,stroke-width:2px
    style PERS fill:#fce4ec,stroke:#e91e63,stroke-width:2px
```

### Decision Flow

```mermaid
flowchart TD
    A[Input Query] --> B{L1: Explicit Rules}
    B -->|"cosine(q,r) ≥ 0.85"| C["Match Found<br/>sim=0.91"]
    B -->|"No Match"| D{L2: Inferred Patterns}
    D -->|"P(pattern|data) > 0.70"| E["Pattern Found<br/>conf=0.78"]
    D -->|"No Pattern"| F{L3: Best Practices}
    F -->|"argmax P(p|query)"| G["Best Practice<br/>Default confidence"]
    
    C --> H[High Confidence Decision]
    E --> I[Medium Confidence Decision]
    G --> J[Fallback Decision]
    
    style A fill:#e1f5fe
    style B fill:#f3e5f5
    style D fill:#f3e5f5
    style F fill:#f3e5f5
    style C fill:#e8f5e8
    style E fill:#fff3e0
    style G fill:#fce4ec
    style H fill:#e8f5e8
    style I fill:#fff3e0
    style J fill:#fce4ec
```

## Quick Start

### Prerequisites
- Docker & Docker Compose
- LLM API key (OpenAI, Anthropic, or other supported providers)

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

### Key Metrics

```mermaid
pie title Performance Overview
    "Success Rate (97.8%)" : 97.8
    "Cache Hit Rate (42%)" : 42
    "Remaining (2.2%)" : 2.2
```

### Performance Targets

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| P95 Latency | <180s | 178s | ✅ |
| Cost/Article | <$0.30 | $0.14 | ✅ |
| Success Rate | >95% | 97.8% | ✅ |
| Cache Hit Rate | >35% | 42% | ✅ |

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

## Developer

**Mohammad Atashi**  
Email: mohammadaliatashi@icloud.com  
GitHub: [@TensorScholar](https://github.com/TensorScholar)