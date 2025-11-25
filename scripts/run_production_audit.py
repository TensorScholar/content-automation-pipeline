#!/usr/bin/env python3
"""
Production Readiness Audit - Standalone Runner
===============================================

Runs comprehensive production readiness checks without pytest dependencies.
"""

import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Tuple

try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    print("⚠️  PyYAML not installed - some checks will be skipped")


class ProductionAuditor:
    """Comprehensive production readiness audit."""

    def __init__(self, project_root: Path):
        self.project_root = project_root
        self.results: Dict[str, List[Tuple[str, bool, str]]] = {}
        self.total_checks = 0
        self.passed_checks = 0
        self.failed_checks = 0
        self.skipped_checks = 0

    def run_all_checks(self):
        """Run all production readiness checks."""
        print("=" * 80)
        print("PRODUCTION READINESS AUDIT")
        print("=" * 80)
        print()

        self.check_security()
        self.check_performance()
        self.check_error_handling()
        self.check_monitoring()
        self.check_configuration()
        self.check_database()
        self.check_api_standards()
        self.check_infrastructure()

        self.print_summary()

    def add_result(self, category: str, check: str, passed: bool, details: str = ""):
        """Add a check result."""
        if category not in self.results:
            self.results[category] = []

        self.results[category].append((check, passed, details))
        self.total_checks += 1

        if passed:
            self.passed_checks += 1
        elif details.startswith("SKIP"):
            self.skipped_checks += 1
        else:
            self.failed_checks += 1

    def check_security(self):
        """Security checks."""
        category = "🔒 Security"
        print(f"\n{category}")
        print("-" * 80)

        # 1. No hardcoded secrets
        check = "No hardcoded secrets"
        secret_patterns = [
            r'api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
            r'password\s*=\s*["\'][^$][^"\']+["\']',  # Exclude password=${VAR}
            r'sk-[a-zA-Z0-9]{48}',  # OpenAI key pattern
        ]

        issues = []
        for py_file in self.project_root.rglob("*.py"):
            if "venv" in str(py_file) or ".git" in str(py_file) or "test" in str(py_file):
                continue
            # Skip this audit script itself (contains regex patterns)
            if py_file.name == "run_production_audit.py":
                continue

            try:
                with open(py_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    for pattern in secret_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues.append(f"{py_file.name}")
                            break
            except:
                pass

        self.add_result(category, check, len(issues) == 0,
                       f"Found {len(issues)} potential issues" if issues else "Clean")
        print(f"  {'✅' if len(issues) == 0 else '❌'} {check}: {len(issues)} potential issues")

        # 2. Environment variables documented
        check = ".env.example exists"
        env_example = self.project_root / ".env.example"
        exists = env_example.exists()
        self.add_result(category, check, exists, "Present" if exists else "Missing")
        print(f"  {'✅' if exists else '❌'} {check}")

        # 3. CORS configuration
        check = "CORS middleware configured"
        main_py = self.project_root / "api" / "main.py"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_cors = "CORSMiddleware" in content
            self.add_result(category, check, has_cors)
            print(f"  {'✅' if has_cors else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: main.py not found")
            print(f"  ⏭️  {check}: main.py not found")

        # 4. Security headers
        check = "Security headers configured"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_security = any(x in content for x in ["SecurityHeadersMiddleware", "X-Frame-Options"])
            self.add_result(category, check, has_security)
            print(f"  {'✅' if has_security else '❌'} {check}")

        # 5. Password hashing
        check = "Password hashing implemented"
        security_file = self.project_root / "security.py"
        if security_file.exists():
            with open(security_file) as f:
                content = f.read()
            has_hashing = any(x in content for x in ["bcrypt", "passlib", "argon2"])
            self.add_result(category, check, has_hashing)
            print(f"  {'✅' if has_hashing else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: security.py not found")
            print(f"  ⏭️  {check}: security.py not found")

        # 6. Authentication dependencies
        check = "Authentication implemented"
        routes_dir = self.project_root / "api" / "routes"
        if routes_dir.exists():
            has_auth = False
            for route_file in routes_dir.glob("*.py"):
                with open(route_file) as f:
                    if "Depends(get_current_user)" in f.read():
                        has_auth = True
                        break
            self.add_result(category, check, has_auth)
            print(f"  {'✅' if has_auth else '❌'} {check}")

    def check_performance(self):
        """Performance checks."""
        category = "⚡ Performance"
        print(f"\n{category}")
        print("-" * 80)

        # 1. Async database
        check = "Async database client"
        db_file = self.project_root / "infrastructure" / "database.py"
        if db_file.exists():
            with open(db_file) as f:
                content = f.read()
            is_async = "asyncpg" in content or "AsyncSession" in content
            self.add_result(category, check, is_async)
            print(f"  {'✅' if is_async else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: database.py not found")
            print(f"  ⏭️  {check}: database.py not found")

        # 2. Connection pooling
        check = "Database connection pooling"
        if db_file.exists():
            with open(db_file) as f:
                content = f.read()
            has_pool = any(x in content for x in ["pool_size", "create_async_engine"])
            self.add_result(category, check, has_pool)
            print(f"  {'✅' if has_pool else '❌'} {check}")

        # 3. Response compression
        check = "Response compression (GZip)"
        main_py = self.project_root / "api" / "main.py"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_gzip = "GZipMiddleware" in content
            self.add_result(category, check, has_gzip)
            print(f"  {'✅' if has_gzip else '❌'} {check}")

        # 4. Celery configuration
        check = "Celery worker configured"
        celery_file = self.project_root / "orchestration" / "celery_app.py"
        if celery_file.exists():
            with open(celery_file) as f:
                content = f.read()
            is_configured = all(x in content for x in ["worker_prefetch_multiplier", "task_acks_late"])
            self.add_result(category, check, is_configured)
            print(f"  {'✅' if is_configured else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: celery_app.py not found")
            print(f"  ⏭️  {check}: celery_app.py not found")

        # 5. Redis pooling
        check = "Redis connection pooling"
        redis_files = list(self.project_root.rglob("*redis*.py"))
        has_redis_pool = False
        for redis_file in redis_files:
            if "venv" not in str(redis_file):
                try:
                    with open(redis_file) as f:
                        if "ConnectionPool" in f.read():
                            has_redis_pool = True
                            break
                except:
                    pass
        self.add_result(category, check, has_redis_pool)
        print(f"  {'✅' if has_redis_pool else '❌'} {check}")

    def check_error_handling(self):
        """Error handling checks."""
        category = "🛡️  Error Handling"
        print(f"\n{category}")
        print("-" * 80)

        # 1. Exception handlers
        check = "Global exception handlers"
        main_py = self.project_root / "api" / "main.py"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_handlers = "add_exception_handlers" in content or "@app.exception_handler" in content
            self.add_result(category, check, has_handlers)
            print(f"  {'✅' if has_handlers else '❌'} {check}")

        # 2. Custom exceptions
        check = "Custom exception classes"
        exceptions_file = self.project_root / "core" / "exceptions.py"
        if exceptions_file.exists():
            with open(exceptions_file) as f:
                content = f.read()
            has_custom = "class" in content and "Exception" in content
            self.add_result(category, check, has_custom)
            print(f"  {'✅' if has_custom else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: exceptions.py not found")
            print(f"  ⏭️  {check}: exceptions.py not found")

        # 3. Retry logic
        check = "Retry logic for external services"
        llm_client = self.project_root / "infrastructure" / "llm_client.py"
        if llm_client.exists():
            with open(llm_client) as f:
                content = f.read()
            has_retry = any(x in content for x in ["@retry", "tenacity", "max_retries"])
            self.add_result(category, check, has_retry)
            print(f"  {'✅' if has_retry else '❌'} {check}")

        # 4. Circuit breaker
        check = "Circuit breaker pattern"
        if llm_client.exists():
            with open(llm_client) as f:
                content = f.read()
            has_breaker = "CircuitBreaker" in content
            self.add_result(category, check, has_breaker)
            print(f"  {'✅' if has_breaker else '❌'} {check}")

        # 5. Timeout configuration
        check = "Timeout configuration"
        timeout_found = False
        for py_file in self.project_root.rglob("*.py"):
            if "venv" not in str(py_file) and py_file.name == "llm_client.py":
                try:
                    with open(py_file) as f:
                        if "timeout=" in f.read():
                            timeout_found = True
                            break
                except:
                    pass
        self.add_result(category, check, timeout_found)
        print(f"  {'✅' if timeout_found else '❌'} {check}")

    def check_monitoring(self):
        """Monitoring checks."""
        category = "📊 Monitoring"
        print(f"\n{category}")
        print("-" * 80)

        # 1. Structured logging
        check = "Structured logging (structlog)"
        monitoring_file = self.project_root / "infrastructure" / "monitoring.py"
        if monitoring_file.exists():
            with open(monitoring_file) as f:
                content = f.read()
            has_structlog = "structlog" in content
            self.add_result(category, check, has_structlog)
            print(f"  {'✅' if has_structlog else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: monitoring.py not found")
            print(f"  ⏭️  {check}: monitoring.py not found")

        # 2. Metrics collection
        check = "Metrics collection (Prometheus)"
        has_metrics = False
        for py_file in self.project_root.rglob("*.py"):
            if "venv" not in str(py_file):
                try:
                    with open(py_file) as f:
                        if "prometheus_client" in f.read():
                            has_metrics = True
                            break
                except:
                    pass
        self.add_result(category, check, has_metrics)
        print(f"  {'✅' if has_metrics else '❌'} {check}")

        # 3. Health check endpoint
        check = "Health check endpoint"
        main_py = self.project_root / "api" / "main.py"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_health = "/health" in content
            self.add_result(category, check, has_health)
            print(f"  {'✅' if has_health else '❌'} {check}")

        # 4. OpenTelemetry
        check = "OpenTelemetry tracing"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_otel = "opentelemetry" in content or "FastAPIInstrumentation" in content
            self.add_result(category, check, has_otel)
            print(f"  {'✅' if has_otel else '❌'} {check}")

    def check_configuration(self):
        """Configuration checks."""
        category = "⚙️  Configuration"
        print(f"\n{category}")
        print("-" * 80)

        # 1. Settings class
        check = "Pydantic settings class"
        settings_file = self.project_root / "config" / "settings.py"
        if settings_file.exists():
            with open(settings_file) as f:
                content = f.read()
            has_pydantic = "BaseSettings" in content or "pydantic" in content
            self.add_result(category, check, has_pydantic)
            print(f"  {'✅' if has_pydantic else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: settings.py not found")
            print(f"  ⏭️  {check}: settings.py not found")

        # 2. Environment detection
        check = "Environment detection"
        if settings_file.exists():
            with open(settings_file) as f:
                content = f.read()
            has_env = any(x in content for x in ["is_production", "environment", "ENV"])
            self.add_result(category, check, has_env)
            print(f"  {'✅' if has_env else '❌'} {check}")

        # 3. Secret management
        check = "SecretStr for sensitive values"
        if settings_file.exists():
            with open(settings_file) as f:
                content = f.read()
            uses_secret_str = "SecretStr" in content
            self.add_result(category, check, uses_secret_str)
            print(f"  {'✅' if uses_secret_str else '❌'} {check}")

        # 4. Docker compose
        check = "Docker Compose configuration"
        compose_file = self.project_root / "docker-compose.yml"
        if compose_file.exists() and HAS_YAML:
            with open(compose_file) as f:
                compose_data = yaml.safe_load(f)
            has_services = "services" in compose_data
            self.add_result(category, check, has_services)
            print(f"  {'✅' if has_services else '❌'} {check}")
        elif not HAS_YAML:
            self.add_result(category, check, False, "SKIP: PyYAML not installed")
            print(f"  ⏭️  {check}: PyYAML not installed")
        else:
            self.add_result(category, check, False, "SKIP: docker-compose.yml not found")
            print(f"  ⏭️  {check}: docker-compose.yml not found")

    def check_database(self):
        """Database checks."""
        category = "🗄️  Database"
        print(f"\n{category}")
        print("-" * 80)

        # 1. Alembic migrations
        check = "Alembic migrations configured"
        alembic_ini = self.project_root / "alembic.ini"
        migrations_dir = self.project_root / "migrations"
        has_migrations = alembic_ini.exists() or migrations_dir.exists()
        self.add_result(category, check, has_migrations)
        print(f"  {'✅' if has_migrations else '❌'} {check}")

        # 2. Database models
        check = "Database models defined"
        models_locations = [
            self.project_root / "models",
            self.project_root / "database" / "models",
            self.project_root / "core" / "models.py",
        ]
        model_files = []
        for location in models_locations:
            if location.is_file():
                model_files.append(location)
            elif location.is_dir():
                model_files.extend(list(location.glob("*.py")))
        has_models = len(model_files) > 0
        self.add_result(category, check, has_models, f"{len(model_files)} model files")
        print(f"  {'✅' if has_models else '❌'} {check}: {len(model_files)} files")

        # 3. Async engine
        check = "SQLAlchemy async engine"
        db_file = self.project_root / "infrastructure" / "database.py"
        if db_file.exists():
            with open(db_file) as f:
                content = f.read()
            has_async = "create_async_engine" in content
            self.add_result(category, check, has_async)
            print(f"  {'✅' if has_async else '❌'} {check}")

    def check_api_standards(self):
        """API standards checks."""
        category = "🌐 API Standards"
        print(f"\n{category}")
        print("-" * 80)

        # 1. OpenAPI docs
        check = "OpenAPI documentation"
        main_py = self.project_root / "api" / "main.py"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_docs = all(x in content for x in ["FastAPI", "title=", "description="])
            self.add_result(category, check, has_docs)
            print(f"  {'✅' if has_docs else '❌'} {check}")

        # 2. Response models
        check = "Response models (Pydantic)"
        schemas_file = self.project_root / "api" / "schemas.py"
        if schemas_file.exists():
            with open(schemas_file) as f:
                content = f.read()
            has_models = "BaseModel" in content
            self.add_result(category, check, has_models)
            print(f"  {'✅' if has_models else '❌'} {check}")
        else:
            self.add_result(category, check, False, "SKIP: schemas.py not found")
            print(f"  ⏭️  {check}: schemas.py not found")

        # 3. API versioning
        check = "API versioning strategy"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_versioning = "version=" in content or "/v1/" in content
            self.add_result(category, check, has_versioning, "Recommended" if not has_versioning else "Present")
            print(f"  {'✅' if has_versioning else '⚠️ '} {check}")

        # 4. Rate limiting
        check = "Rate limiting configured"
        if main_py.exists():
            with open(main_py) as f:
                content = f.read()
            has_rate_limit = any(x in content for x in ["RateLimitMiddleware", "fastapi-limiter", "FastAPILimiter"])
            self.add_result(category, check, has_rate_limit)
            print(f"  {'✅' if has_rate_limit else '❌'} {check}")

    def check_infrastructure(self):
        """Infrastructure checks."""
        category = "🏗️  Infrastructure"
        print(f"\n{category}")
        print("-" * 80)

        # 1. Dependencies specified
        check = "Dependencies specified"
        pyproject = self.project_root / "pyproject.toml"
        requirements = self.project_root / "requirements.txt"
        has_deps = pyproject.exists() or requirements.exists()
        self.add_result(category, check, has_deps)
        print(f"  {'✅' if has_deps else '❌'} {check}")

        # 2. Critical dependencies
        check = "Production dependencies present"
        if pyproject.exists():
            with open(pyproject) as f:
                content = f.read()
            critical_deps = ["fastapi", "uvicorn", "redis", "celery", "sqlalchemy", "pydantic"]
            has_all = all(dep in content.lower() for dep in critical_deps)
            self.add_result(category, check, has_all)
            print(f"  {'✅' if has_all else '❌'} {check}")

        # 3. Dev dependencies separated
        check = "Dev dependencies separated"
        if pyproject.exists():
            with open(pyproject) as f:
                content = f.read()
            has_dev = "dev" in content.lower() and "dependencies" in content
            self.add_result(category, check, has_dev)
            print(f"  {'✅' if has_dev else '❌'} {check}")

        # 4. Dockerfile
        check = "Dockerfile exists"
        dockerfile = self.project_root / "Dockerfile"
        has_dockerfile = dockerfile.exists()
        self.add_result(category, check, has_dockerfile, "Recommended" if not has_dockerfile else "Present")
        print(f"  {'✅' if has_dockerfile else '⚠️ '} {check}")

        # 5. Multi-stage build
        check = "Multi-stage Docker build"
        if dockerfile.exists():
            with open(dockerfile) as f:
                content = f.read()
            is_multi_stage = content.count("FROM") > 1
            self.add_result(category, check, is_multi_stage, "Recommended" if not is_multi_stage else "Present")
            print(f"  {'✅' if is_multi_stage else '⚠️ '} {check}")
        else:
            self.add_result(category, check, False, "SKIP: Dockerfile not found")
            print(f"  ⏭️  {check}: Dockerfile not found")

    def print_summary(self):
        """Print audit summary."""
        print("\n" + "=" * 80)
        print("AUDIT SUMMARY")
        print("=" * 80)

        for category, results in self.results.items():
            passed = sum(1 for _, p, _ in results if p)
            total = len(results)
            skipped = sum(1 for _, _, d in results if d.startswith("SKIP"))
            failed = total - passed - skipped

            status = "✅" if failed == 0 else "❌"
            print(f"{status} {category}: {passed}/{total - skipped} passed", end="")
            if skipped > 0:
                print(f" ({skipped} skipped)", end="")
            print()

        print("\n" + "-" * 80)
        percentage = (self.passed_checks / max(self.total_checks - self.skipped_checks, 1)) * 100
        print(f"TOTAL: {self.passed_checks}/{self.total_checks - self.skipped_checks} checks passed ({percentage:.1f}%)")

        if self.skipped_checks > 0:
            print(f"       {self.skipped_checks} checks skipped")

        print("=" * 80)

        # Production readiness assessment
        if percentage >= 90:
            print("\n🎉 PRODUCTION READY")
            print("   System meets production requirements.")
        elif percentage >= 70:
            print("\n⚠️  PRODUCTION CANDIDATE")
            print("   Address remaining issues before deploying to production.")
        else:
            print("\n❌ NOT PRODUCTION READY")
            print("   Critical issues must be resolved before production deployment.")

        print()


def main():
    """Run production readiness audit."""
    project_root = Path(__file__).parent.parent

    auditor = ProductionAuditor(project_root)
    auditor.run_all_checks()

    # Return exit code based on results
    if auditor.failed_checks > 0:
        sys.exit(1)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
