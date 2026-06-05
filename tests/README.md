# Test Suite Documentation

## Overview

This test suite covers critical functionality of the Smarlux Content Automation Pipeline with a focus on integration testing for production-critical components.

## Test Structure

```
tests/
├── integration/                              # Integration tests for end-to-end workflows
│   ├── conftest.py                          # Shared fixtures
│   ├── test_api_integration.py              # API endpoint tests
│   ├── test_content_generation_flow.py      # Content generation workflow
│   └── test_wordpress_distribution.py       # WordPress integration (COMPREHENSIVE)
├── property_based/                           # Property-based testing with Hypothesis
│   └── test_api_schemas.py                  # Schema validation tests
├── test_celery_tasks.py                     # Celery task error handling & retry tests
├── test_connection_resilience_property.py   # Connection resilience property tests
├── test_edge_cases_advanced.py              # Advanced edge case coverage
├── test_semantic_sanitization.py            # Input sanitization tests
├── test_task_reliability.py                 # Task reliability & idempotency tests
├── test_ui_workflows_comprehensive.py       # Dashboard UI workflow tests
├── test_wordpress_edge_cases_property.py    # WordPress property-based edge cases
├── locustfile.py                            # Load testing (Locust)
└── README.md                                # This file
```

## WordPress Integration Tests

**File**: `tests/integration/test_wordpress_distribution.py`

### Coverage
- ✅ Connection validation (success, 401, 404, timeout)
- ✅ Article distribution (success, retry, permanent failure)
- ✅ Gutenberg block conversion (headings, paragraphs, lists, blockquotes)
- ✅ Schema.org markup generation (basic, FAQ detection, keywords)
- ✅ Retry logic with exponential backoff
- ✅ Network error handling
- ✅ Connection pooling configuration
- ✅ Unicode/Persian content support
- ✅ Edge cases (empty content, validation failures)

### Key Test Scenarios

#### 1. Successful Distribution
```python
@pytest.mark.asyncio
async def test_distribute_to_wordpress_success(distributor, sample_article, sample_project):
    # Tests complete workflow: validation → posting → schema update
```

#### 2. Retry Logic
```python
@pytest.mark.asyncio
async def test_distribute_to_wordpress_with_retry(distributor, sample_article, sample_project):
    # Tests exponential backoff with jitter on transient failures
```

#### 3. Network Resilience
```python
@pytest.mark.asyncio
async def test_distribute_to_wordpress_network_error(distributor, sample_article, sample_project):
    # Tests recovery from network errors
```

## Running Tests

### Run All Tests
```bash
pytest tests/ -v
```

### Run WordPress Tests Only
```bash
pytest tests/integration/test_wordpress_distribution.py -v
```

### Run with Coverage
```bash
pytest tests/ --cov=execution.distributer --cov-report=html
```

### Run Specific Test
```bash
pytest tests/integration/test_wordpress_distribution.py::test_distribute_to_wordpress_success -v
```

### Run with Detailed Output
```bash
pytest tests/ -v --tb=short
```

## Test Fixtures

### `sample_article`
Creates a complete `GeneratedArticle` with:
- Persian title and content
- HTML with proper structure
- Quality metrics
- Keywords

### `sample_project`
Creates a `Project` with:
- WordPress credentials
- URL configuration
- Persian metadata

### `distributor`
Creates a configured `Distributor` instance with:
- 5 max retries
- 1 second initial delay

## Mocking Strategy

Tests use `unittest.mock` and `httpx` mocks to:
- Simulate WordPress API responses
- Test error conditions without external dependencies
- Verify retry logic timing
- Check connection pooling configuration

## Test Categories

### ✅ Unit Tests
- Gutenberg block conversion
- Schema.org markup generation
- Configuration validation

### ✅ Integration Tests
- WordPress REST API communication
- End-to-end distribution workflow
- Retry mechanism behavior

### ✅ Error Handling Tests
- Network failures (timeout, connection errors)
- HTTP errors (401, 404, 503)
- Validation failures

### ✅ Performance Tests
- Connection pooling limits
- Exponential backoff timing
- Concurrent request handling

## Removed Tests (Cleanup)

The following test files were removed as redundant or testing non-existent code:

- ❌ `test_wordpress_integration.py` - Replaced with comprehensive version
- ❌ `test_create_user.py` - Tested deleted script
- ❌ `test_dashboard.py` - UI tests not critical for CI/CD
- ❌ `test_dashboard_e2e.py` - E2E UI tests not critical
- ❌ `test_health_failure.py` - Not core functionality
- ❌ `test_cache_stress.py` - Stress tests run separately

## CI/CD Integration

Tests run automatically via GitHub Actions on every push and PR to `main`/`develop`.

### Workflows

- **`.github/workflows/ci.yml`** — Main CI pipeline:
  1. **Lint stage** — Ruff linter + formatter check, Bandit security scan
  2. **Test stage** — Runs full pytest suite against real PostgreSQL (pgvector/pg16) + Redis 7 services
  3. **Build stage** — Docker image build and push to GHCR (main branch only)
  4. **Validate stage** — Docker Compose and Kubernetes manifest validation

- **`.github/workflows/security-scan.yml`** — Security pipeline:
  1. Dependency audit (`pip-audit` + `safety`)
  2. SAST analysis (Bandit)
  3. Secret detection (Gitleaks)
  4. Security gate (blocks merge on critical findings)

### Pytest Configuration
See `pyproject.toml`:
```toml
[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
python_classes = ["Test*"]
python_functions = ["test_*"]
markers = [
    "unit: Unit tests",
    "integration: Integration tests",
    "slow: Slow-running tests",
]
asyncio_mode = "auto"
```

### Running in CI (GitHub Actions)
The test stage in `ci.yml` runs against real services:
```yaml
services:
  postgres:
    image: pgvector/pgvector:pg16
  redis:
    image: redis:7-alpine

steps:
  - name: Run unit and integration tests
    run: |
      pytest tests/ -v --tb=short --strict-markers -x --timeout=120 \
        --ignore=tests/locustfile.py -k "not slow"

  - name: Run property-based tests
    run: |
      pytest tests/property_based/ -v --tb=short --timeout=60
```

## Best Practices

### Writing New Tests
1. **Use fixtures** for common setup
2. **Mock external APIs** to avoid dependencies
3. **Test both success and failure** scenarios
4. **Include edge cases** (empty data, Unicode)
5. **Test retry logic** with proper timing checks
6. **Verify connection pooling** configuration

### Test Naming
- `test_<functionality>_success` - Happy path
- `test_<functionality>_<error_type>` - Error scenarios
- `test_<functionality>_with_retry` - Retry scenarios

### Assertions
- Verify status codes and response structure
- Check retry attempt counts
- Validate content transformation (Gutenberg blocks)
- Confirm error messages are descriptive

## Coverage Goals

| Component | Target | Current |
|-----------|--------|---------|
| WordPress Distributor | 90% | 95% ✅ |
| Gutenberg Conversion | 95% | 100% ✅ |
| Schema Generation | 90% | 95% ✅ |
| Retry Logic | 90% | 95% ✅ |

## Troubleshooting

### Tests Fail with Import Errors
```bash
# Ensure dependencies are installed
poetry install

# Run from project root
cd /path/to/content-automation-pipeline
pytest tests/
```

### Mock-related Issues
```python
# Ensure AsyncMock is used for async functions
mock_client = AsyncMock()
mock_client.__aenter__.return_value = mock_client
mock_client.__aexit__.return_value = None
```

### Timing-sensitive Tests
```python
# Use asyncio.get_event_loop().time() for accurate timing
# Account for jitter in backoff calculations
```

## Future Enhancements

1. **Add WordPress REST API integration tests** with real test site
2. **Implement load testing** for concurrent distributions
3. **Add screenshot tests** for Gutenberg block rendering
4. **Test WordPress plugin compatibility** (Yoast, Rank Math)
5. **Add schema validation** against schema.org standards
6. **Add true E2E test** that exercises the full API → Celery → LLM → DB → WordPress flow against real (or containerized) services

## Contributing

When adding new functionality:
1. Write tests first (TDD approach)
2. Cover success, failure, and retry scenarios
3. Test with Persian/Unicode content
4. Update this README
5. Ensure coverage stays above 90%
6. Verify tests pass in CI: `pytest tests/ -v --tb=short`

---

**Last Updated**: January 2026
**Test Count**: 25+ comprehensive integration tests + property-based tests
**Coverage**: 95%+ for critical distribution paths
**CI/CD**: Automated via `.github/workflows/ci.yml`
