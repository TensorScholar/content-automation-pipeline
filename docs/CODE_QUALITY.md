# Code Quality & Security Pipeline Documentation

## Overview

This document describes the automated quality gates and security checks enforced through CI/CD.

## Local Development Setup

### Install Pre-commit Hooks

```bash
# Install hooks into your local git repository
poetry run pre-commit install

# Run hooks manually on all files
poetry run pre-commit run --all-files

# Run specific hook
poetry run pre-commit run ruff --all-files
```

### Run Quality Checks Locally

```bash
# Linting with Ruff (fast)
poetry run ruff check .

# Auto-fix linting issues
poetry run ruff check . --fix

# Format code
poetry run ruff format .

# Security scan with Bandit
poetry run bandit -r . -c .bandit

# Type checking with MyPy
poetry run mypy . --ignore-missing-imports

# Run unit tests with coverage
poetry run pytest tests/unit -v --cov=. --cov-report=term-missing

# Run integration tests
poetry run pytest tests/integration -v

# Run all tests
poetry run pytest -v
```

## CI/CD Pipeline

### GitHub Actions Workflow

The pipeline runs automatically on:
- **Push to main/develop branches**
- **Pull requests to main/develop branches**

### Pipeline Stages

#### 1. Lint & Format Check
- **Tool**: Ruff
- **Purpose**: Enforce code style and catch common errors
- **Configuration**: `pyproject.toml` → `[tool.ruff]`
- **Rules**: PEP 8 compliance, import sorting, unused imports
- **Failure**: Blocks PR merge

#### 2. Security Vulnerability Scan
- **Tool**: Bandit
- **Purpose**: Detect security vulnerabilities in code
- **Configuration**: `.bandit`
- **Checks**: SQL injection, shell injection, unsafe deserialization, weak crypto
- **Output**: JSON report artifact
- **Failure**: Warning only (manual review required)

#### 3. Type Checking
- **Tool**: MyPy
- **Purpose**: Static type analysis
- **Configuration**: `pyproject.toml` → `[tool.mypy]`
- **Checks**: Type consistency, missing return types, unused variables
- **Failure**: Warning only (gradually enforcing types)

#### 4. Unit Tests
- **Tool**: Pytest
- **Purpose**: Validate business logic in isolation
- **Configuration**: `pyproject.toml` → `[tool.pytest.ini_options]`
- **Coverage**: Minimum not enforced (tracked for visibility)
- **Services**: None required (mocked dependencies)
- **Failure**: Blocks PR merge

#### 5. Integration Tests
- **Tool**: Pytest
- **Purpose**: Validate component interactions
- **Services**: PostgreSQL 16 + Redis 7 (Docker containers)
- **Environment Variables**: Test credentials injected
- **Failure**: Warning only (known flaky tests tolerated)

#### 6. Dependency Security Audit
- **Tool**: Safety
- **Purpose**: Check for known vulnerabilities in dependencies
- **Input**: `poetry.lock` exported to `requirements.txt`
- **Output**: JSON report
- **Failure**: Warning only (manual review required)

#### 7. Docker Build
- **Purpose**: Validate production image builds successfully
- **Base Image**: python:3.12-slim
- **Layers**: Multi-stage with poetry
- **Cache**: GitHub Actions cache for faster builds
- **Failure**: Blocks PR merge

#### 8. Code Quality Report
- **Purpose**: Aggregate all quality metrics
- **Output**: GitHub Actions summary with job status
- **Artifacts**: Coverage reports, Bandit reports, test results

## Quality Gate Matrix

| Stage | Tool | Blocking | Environment | Artifacts |
|-------|------|----------|-------------|-----------|
| Lint & Format | Ruff | ✅ Yes | Ubuntu Latest | None |
| Security Scan | Bandit | ⚠️ No | Ubuntu Latest | JSON report |
| Type Check | MyPy | ⚠️ No | Ubuntu Latest | None |
| Unit Tests | Pytest | ✅ Yes | Ubuntu Latest | Coverage XML/HTML |
| Integration Tests | Pytest | ⚠️ No | Ubuntu + PG + Redis | Test results |
| Dependency Audit | Safety | ⚠️ No | Ubuntu Latest | JSON report |
| Docker Build | Docker | ✅ Yes | Ubuntu + Buildx | None |

## Metrics & Reporting

### Coverage Reports

Coverage reports are uploaded as artifacts and can be downloaded from:
- **GitHub Actions** → **Workflow Run** → **Artifacts** → `coverage-report`

View HTML report:
```bash
# After downloading artifact
cd coverage-report/htmlcov
open index.html
```

### Security Reports

Bandit security scans generate JSON reports:
- **Artifact Name**: `bandit-security-report`
- **Format**: JSON with severity levels (LOW, MEDIUM, HIGH)

Review critical issues:
```bash
jq '.results[] | select(.issue_severity == "HIGH")' bandit-report.json
```

### Test Results

Test results are output to console and can be viewed in:
- **GitHub Actions** → **Workflow Run** → **Test Job** → **Run Unit Tests** logs

## Troubleshooting

### Pre-commit Hook Failures

Skip specific hooks temporarily:
```bash
SKIP=bandit git commit -m "WIP: Testing changes"
```

Bypass all hooks (use sparingly):
```bash
git commit --no-verify -m "Emergency hotfix"
```

### CI/CD Pipeline Failures

#### Ruff Formatting Errors
```bash
# Auto-fix locally and push
poetry run ruff format .
git add -u
git commit --amend --no-edit
git push --force-with-lease
```

#### MyPy Type Errors
```bash
# Add type ignores where needed
# type: ignore[error-code]

# Or suppress specific errors in pyproject.toml
[tool.mypy]
warn_return_any = false
```

#### Flaky Integration Tests
```bash
# Run tests with retries locally
poetry run pytest tests/integration --maxfail=3 --tb=short
```

## Security Best Practices

### Secrets Management

❌ **Never commit secrets to git**
✅ **Use environment variables from `.env`**

```python
# Bad
api_key = "sk-1234567890abcdef"

# Good
api_key = os.getenv("OPENAI_API_KEY")
```

### SQL Injection Prevention

❌ **String concatenation in queries**
✅ **Parameterized queries with SQLAlchemy**

```python
# Bad
query = f"SELECT * FROM users WHERE id = {user_id}"

# Good
query = users_table.select().where(users_table.c.id == user_id)
```

### Dependency Updates

```bash
# Check for outdated packages
poetry show --outdated

# Update specific package
poetry update fastapi

# Update all packages (test thoroughly!)
poetry update
```

## Performance Optimization

### Caching Strategy

The pipeline uses GitHub Actions caching:
- **Poetry virtualenv**: Cached between runs
- **Docker layers**: Cached with BuildKit

Cache keys:
- `venv-${{ runner.os }}-${{ python-version }}-${{ poetry.lock }}`
- Docker: `type=gha` (GitHub Actions cache)

### Parallel Execution

Jobs run in parallel when possible:
- Lint, Security, Type Check, Tests → **Parallel**
- Docker Build → **After Tests** (depends on)
- Code Quality Report → **After All** (summary)

## Maintenance

### Update GitHub Actions

```bash
# Check for action updates
# .github/workflows/ci.yml

# Update to latest versions
actions/checkout@v4 → actions/checkout@v5
```

### Update Pre-commit Hooks

```bash
# Update hook versions in .pre-commit-config.yaml
poetry run pre-commit autoupdate

# Test updated hooks
poetry run pre-commit run --all-files
```

## Contact & Support

For questions about quality checks:
- **GitHub Issues**: Open an issue with label `quality` or `ci-cd`
- **Email**: mohammadaliatashi@icloud.com
