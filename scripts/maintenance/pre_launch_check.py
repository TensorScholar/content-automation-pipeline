#!/usr/bin/env python3
"""
Pre-Launch Reliability Check Script
Comprehensive validation of all system components for production readiness

This script validates:
- Infrastructure (PostgreSQL, Redis)
- API health and endpoints
- Database schema and migrations
- Environment configuration
- Security settings
- Error handling and resilience
- Performance benchmarks
- Dashboard connectivity
"""

import sys
import os
import time
import json
import subprocess
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import asyncio


# Color codes for terminal output
class Colors:
    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{text:^80}{Colors.ENDC}")
    print(f"{Colors.BOLD}{Colors.HEADER}{'=' * 80}{Colors.ENDC}\n")


def print_success(text: str):
    print(f"{Colors.OKGREEN}✓ {text}{Colors.ENDC}")


def print_warning(text: str):
    print(f"{Colors.WARNING}⚠ {text}{Colors.ENDC}")


def print_error(text: str):
    print(f"{Colors.FAIL}✗ {text}{Colors.ENDC}")


def print_info(text: str):
    print(f"{Colors.OKCYAN}ℹ {text}{Colors.ENDC}")


def dependency_is_healthy(raw_status: str) -> bool:
    """Avoid treating values like 'unhealthy' as healthy via substring checks."""
    normalized = raw_status.lower()
    return "healthy" in normalized and "unhealthy" not in normalized


class ReliabilityChecker:
    def __init__(self):
        self.results: Dict[str, Dict] = {}
        self.critical_failures: List[str] = []
        self.warnings: List[str] = []
        self.api_url = os.getenv("API_URL", "http://127.0.0.1:8000")
        self.ui_url = os.getenv("UI_URL", "http://127.0.0.1:3001")

    def run_command(self, cmd: str, shell: bool = True) -> Tuple[int, str, str]:
        """Execute shell command and return exit code, stdout, stderr"""
        try:
            result = subprocess.run(cmd, shell=shell, capture_output=True, text=True, timeout=30)
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "Command timed out"
        except Exception as e:
            return -1, "", str(e)

    def check_redis(self) -> bool:
        """Check Redis connectivity and health"""
        print_info("Checking Redis...")

        # Check if Redis is running
        code, stdout, stderr = self.run_command("redis-cli ping")
        if code == 0 and "PONG" in stdout:
            print_success("Redis is running and responding")

            # Check memory usage
            code, stdout, _ = self.run_command("redis-cli info memory | grep used_memory_human")
            if code == 0:
                memory = stdout.strip()
                print_info(f"Redis memory usage: {memory}")

            # Check connected clients
            code, stdout, _ = self.run_command("redis-cli client list | wc -l")
            if code == 0:
                clients = stdout.strip()
                print_info(f"Redis connected clients: {clients}")

            self.results["redis"] = {
                "status": "healthy",
                "details": memory if "memory" in locals() else "N/A",
            }
            return True
        else:
            print_error("Redis is not responding")
            self.critical_failures.append("Redis is not running or not accessible")
            self.results["redis"] = {"status": "failed", "error": stderr}
            return False

    def check_postgresql(self) -> bool:
        """Check PostgreSQL connectivity and health"""
        print_info("Checking PostgreSQL...")

        # Check if PostgreSQL is running
        code, stdout, stderr = self.run_command("pg_isready")
        if code == 0:
            print_success("PostgreSQL is running and accepting connections")

            # Check database exists
            code, stdout, _ = self.run_command("psql -l | grep -c content_automation")
            if code == 0 and int(stdout.strip()) > 0:
                print_success("Database 'content_automation' exists")

                # Check database size
                code, stdout, _ = self.run_command(
                    "psql -d content_automation -t -c \"SELECT pg_size_pretty(pg_database_size('content_automation'));\""
                )
                if code == 0:
                    size = stdout.strip()
                    print_info(f"Database size: {size}")

                # Check for pgvector extension
                code, stdout, _ = self.run_command(
                    "psql -d content_automation -t -c \"SELECT COUNT(*) FROM pg_extension WHERE extname='vector';\""
                )
                if code == 0 and int(stdout.strip()) > 0:
                    print_success("pgvector extension is installed")
                else:
                    print_warning("pgvector extension not found - vector operations may fail")
                    self.warnings.append("pgvector extension not installed")

                self.results["postgresql"] = {
                    "status": "healthy",
                    "details": size if "size" in locals() else "N/A",
                }
                return True
            else:
                print_error("Database 'content_automation' not found")
                self.critical_failures.append("Database 'content_automation' does not exist")
                self.results["postgresql"] = {"status": "failed", "error": "Database not found"}
                return False
        else:
            print_error(f"PostgreSQL is not running: {stderr}")
            self.critical_failures.append("PostgreSQL is not running or not accessible")
            self.results["postgresql"] = {"status": "failed", "error": stderr}
            return False

    def check_api_health(self) -> bool:
        """Check API server health"""
        print_info(f"Checking API health at {self.api_url}...")

        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.api_url}/health")

                if response.status_code == 200:
                    data = response.json()
                    print_success(f"API is healthy (version: {data.get('version', 'unknown')})")

                    # Check dependencies
                    deps = data.get("dependencies", {})
                    db_dep = str(deps.get("database", "unknown"))
                    redis_dep = str(deps.get("redis", "unknown"))
                    worker_dep = str(deps.get("celery_workers", "unknown"))

                    if dependency_is_healthy(db_dep):
                        print_success("API -> Database connection: healthy")
                    else:
                        print_error("API -> Database connection: unhealthy")
                        self.critical_failures.append("API cannot connect to database")

                    if dependency_is_healthy(redis_dep):
                        print_success("API -> Redis connection: healthy")
                    else:
                        print_error("API -> Redis connection: unhealthy")
                        self.critical_failures.append("API cannot connect to Redis")

                    if dependency_is_healthy(worker_dep):
                        print_success("API -> Celery workers: healthy")
                    else:
                        print_warning(f"API -> Celery workers: {worker_dep or 'unknown'}")
                        self.warnings.append("Celery workers not reported healthy")

                    self.results["api"] = {
                        "status": "healthy",
                        "version": data.get("version"),
                        "dependencies": deps,
                    }
                    return True
                else:
                    print_error(f"API health check failed with status {response.status_code}")
                    self.critical_failures.append(
                        f"API health endpoint returned {response.status_code}"
                    )
                    self.results["api"] = {"status": "failed", "status_code": response.status_code}
                    return False

        except ImportError:
            print_error("httpx library not installed - cannot check API")
            self.critical_failures.append("httpx library missing")
            return False
        except Exception as e:
            print_error(f"API is not accessible: {str(e)}")
            self.critical_failures.append(f"API not accessible: {str(e)}")
            self.results["api"] = {"status": "failed", "error": str(e)}
            return False

    def check_api_endpoints(self) -> bool:
        """Check critical API endpoints"""
        print_info("Checking critical API endpoints...")

        endpoints = [
            ("/docs", "API Documentation", {200}),
            ("/health", "Root health", {200}),
            ("/auth/token", "Authentication", {405, 422}),
            ("/projects", "Projects endpoint", {401}),
            ("/system/health", "System health", {401}),
            ("/content/task/test/events", "SSE task stream route", {401, 404}),
        ]

        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                for endpoint, description, expected_statuses in endpoints:
                    try:
                        response = client.get(f"{self.api_url}{endpoint}")
                        if response.status_code in expected_statuses:
                            print_success(f"{description}: accessible")
                        else:
                            print_warning(
                                f"{description}: status {response.status_code}, expected {sorted(expected_statuses)}"
                            )
                    except Exception as e:
                        print_warning(f"{description}: {str(e)}")

                self.results["api_endpoints"] = {"status": "checked"}
                return True

        except ImportError:
            print_error("httpx library not installed")
            return False
        except Exception as e:
            print_error(f"Failed to check endpoints: {str(e)}")
            return False

    def check_dashboard(self) -> bool:
        """Check frontend dashboard availability"""
        print_info(f"Checking Dashboard at {self.ui_url}...")

        try:
            import httpx

            with httpx.Client(timeout=10.0) as client:
                response = client.get(self.ui_url)

                if response.status_code == 200:
                    print_success("Dashboard is accessible")
                    # Check Next.js API proxy wiring for backend availability.
                    health_response = client.get(f"{self.ui_url}/api/health")
                    if health_response.status_code == 200:
                        print_success("Dashboard API proxy health: OK")
                    else:
                        print_warning(
                            f"Dashboard API proxy returned status {health_response.status_code}"
                        )
                        self.warnings.append("Frontend API proxy not healthy")

                    self.results["dashboard"] = {"status": "healthy"}
                    return True
                else:
                    print_error(f"Dashboard returned status {response.status_code}")
                    self.critical_failures.append(
                        f"Dashboard not accessible: {response.status_code}"
                    )
                    self.results["dashboard"] = {
                        "status": "failed",
                        "status_code": response.status_code,
                    }
                    return False

        except ImportError:
            print_error("httpx library not installed")
            return False
        except Exception as e:
            print_error(f"Dashboard is not accessible: {str(e)}")
            self.critical_failures.append(f"Dashboard not accessible: {str(e)}")
            self.results["dashboard"] = {"status": "failed", "error": str(e)}
            return False

    def check_environment(self) -> bool:
        """Check environment variables and configuration"""
        print_info("Checking environment configuration...")

        critical_vars = [
            "DATABASE_URL",
            "REDIS_URL",
            "SECRET_KEY",
        ]

        optional_vars = [
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "LOCAL_LLM_URL",
            "LLM_PROVIDER",
            "ENVIRONMENT",
            "API_URL",
        ]

        all_present = True
        for var in critical_vars:
            if os.getenv(var):
                print_success(f"{var}: configured")
            else:
                print_error(f"{var}: NOT SET (CRITICAL)")
                self.critical_failures.append(f"Critical environment variable {var} is not set")
                all_present = False

        for var in optional_vars:
            if os.getenv(var):
                # Don't print the actual value for security
                if "KEY" in var or "SECRET" in var:
                    print_success(f"{var}: configured (***)")
                else:
                    value = os.getenv(var)
                    print_success(f"{var}: {value}")
            else:
                print_warning(f"{var}: not set (optional)")

        # Check SECRET_KEY strength
        secret_key = os.getenv("SECRET_KEY", "")
        if len(secret_key) < 32:
            print_warning("SECRET_KEY is too short (should be at least 32 characters)")
            self.warnings.append("SECRET_KEY should be longer for production use")

        self.results["environment"] = {"status": "checked", "all_critical_set": all_present}
        return all_present

    def check_database_migrations(self) -> bool:
        """Check database migrations status"""
        print_info("Checking database migrations...")

        try:
            # Check if alembic is configured
            if os.path.exists("alembic.ini"):
                print_success("Alembic configuration found")

                # Check current migration version
                code, stdout, stderr = self.run_command("poetry run alembic current")
                if code == 0:
                    print_success(f"Current migration: {stdout.strip()}")

                    # Check if migrations are up to date
                    code, stdout, stderr = self.run_command("poetry run alembic check")
                    if code == 0 or "No new upgrade operations detected" in stdout:
                        print_success("Database schema is up to date")
                        self.results["migrations"] = {"status": "up_to_date"}
                        return True
                    else:
                        print_warning("Pending migrations detected")
                        print_info("Run: poetry run alembic upgrade head")
                        self.warnings.append("Database migrations pending")
                        self.results["migrations"] = {"status": "pending"}
                        return True
                else:
                    print_warning(f"Could not check migration status: {stderr}")
                    self.results["migrations"] = {"status": "unknown"}
                    return True
            else:
                print_warning("Alembic not configured - skipping migration check")
                self.results["migrations"] = {"status": "not_configured"}
                return True

        except Exception as e:
            print_warning(f"Migration check failed: {str(e)}")
            self.results["migrations"] = {"status": "check_failed", "error": str(e)}
            return True  # Non-critical

    def check_security(self) -> bool:
        """Check security configurations"""
        print_info("Checking security settings...")

        # Check if running in production mode
        env = os.getenv("ENVIRONMENT", "development")
        if env.lower() == "production":
            print_info("Running in PRODUCTION mode")

            # Check DEBUG is disabled
            debug = os.getenv("DEBUG", "false").lower()
            if debug == "false":
                print_success("DEBUG mode: disabled")
            else:
                print_error("DEBUG mode is enabled in production!")
                self.critical_failures.append("DEBUG mode must be disabled in production")

            # Check HTTPS in API_URL
            if self.api_url.startswith("https://"):
                print_success("API URL uses HTTPS")
            else:
                print_warning("API URL does not use HTTPS in production")
                self.warnings.append("Consider using HTTPS for production API")
        else:
            print_info(f"Running in {env.upper()} mode")
            print_success("Security checks relaxed for development")

        # Check for .env file
        if os.path.exists(".env"):
            print_success(".env file found")

            # Check .env is in .gitignore
            if os.path.exists(".gitignore"):
                with open(".gitignore", "r") as f:
                    gitignore = f.read()
                    if ".env" in gitignore:
                        print_success(".env is in .gitignore")
                    else:
                        print_warning(".env should be in .gitignore")
                        self.warnings.append(".env file should be in .gitignore")

        self.results["security"] = {"status": "checked", "environment": env}
        return True

    def check_performance(self) -> bool:
        """Run basic performance checks"""
        print_info("Running performance benchmarks...")

        # Check API response time
        try:
            import httpx

            start = time.time()
            with httpx.Client(timeout=10.0) as client:
                response = client.get(f"{self.api_url}/health")
            elapsed = time.time() - start

            if elapsed < 0.5:
                print_success(f"API response time: {elapsed:.3f}s (excellent)")
            elif elapsed < 1.0:
                print_success(f"API response time: {elapsed:.3f}s (good)")
            elif elapsed < 2.0:
                print_warning(f"API response time: {elapsed:.3f}s (acceptable)")
            else:
                print_warning(f"API response time: {elapsed:.3f}s (slow)")
                self.warnings.append(f"API response time is slow: {elapsed:.3f}s")

            self.results["performance"] = {"api_response_time": elapsed}
            return True

        except Exception as e:
            print_warning(f"Performance check failed: {str(e)}")
            return True  # Non-critical

    def check_logging(self) -> bool:
        """Check logging configuration"""
        print_info("Checking logging configuration...")

        # Check logs directory
        if os.path.exists("logs"):
            print_success("Logs directory exists")

            # Check if logs are being written
            log_files = [f for f in os.listdir("logs") if f.endswith(".log")]
            if log_files:
                print_success(f"Found {len(log_files)} log file(s)")
                # Check latest log file
                latest_log = max([os.path.join("logs", f) for f in log_files], key=os.path.getmtime)
                stat = os.stat(latest_log)
                print_info(f"Latest log: {os.path.basename(latest_log)} ({stat.st_size} bytes)")
            else:
                print_warning("No log files found")
        else:
            print_warning("Logs directory does not exist")
            os.makedirs("logs", exist_ok=True)
            print_info("Created logs directory")

        # Check logging_config.py
        if os.path.exists("logging_config.py"):
            print_success("logging_config.py found")
        else:
            print_warning("logging_config.py not found")

        self.results["logging"] = {"status": "checked"}
        return True

    def run_all_checks(self) -> bool:
        """Run all reliability checks"""
        print_header("🔍 PRE-LAUNCH RELIABILITY CHECK")
        print_info(
            f"Starting comprehensive system validation at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )

        checks = [
            ("Environment Configuration", self.check_environment),
            ("PostgreSQL", self.check_postgresql),
            ("Redis", self.check_redis),
            ("Database Migrations", self.check_database_migrations),
            ("API Health", self.check_api_health),
            ("API Endpoints", self.check_api_endpoints),
            ("Dashboard", self.check_dashboard),
            ("Security", self.check_security),
            ("Performance", self.check_performance),
            ("Logging", self.check_logging),
        ]

        passed = 0
        failed = 0

        for name, check_func in checks:
            print_header(f"CHECK: {name}")
            try:
                result = check_func()
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print_error(f"Check crashed: {str(e)}")
                failed += 1
            time.sleep(0.5)  # Brief pause between checks

        # Print summary
        self._print_summary(passed, failed)

        # Return True only if no critical failures
        return len(self.critical_failures) == 0

    def _print_summary(self, passed: int, failed: int):
        """Print final summary"""
        print_header("📊 SUMMARY")

        total = passed + failed
        print(f"\n{Colors.BOLD}Checks Completed:{Colors.ENDC}")
        print(f"  Total:  {total}")
        print(f"  {Colors.OKGREEN}Passed: {passed}{Colors.ENDC}")
        print(f"  {Colors.FAIL}Failed: {failed}{Colors.ENDC}")

        if self.critical_failures:
            print(
                f"\n{Colors.BOLD}{Colors.FAIL}CRITICAL FAILURES ({len(self.critical_failures)}):{Colors.ENDC}"
            )
            for i, failure in enumerate(self.critical_failures, 1):
                print(f"  {i}. {failure}")

        if self.warnings:
            print(f"\n{Colors.BOLD}{Colors.WARNING}WARNINGS ({len(self.warnings)}):{Colors.ENDC}")
            for i, warning in enumerate(self.warnings, 1):
                print(f"  {i}. {warning}")

        print(f"\n{Colors.BOLD}System Status:{Colors.ENDC}")
        if len(self.critical_failures) == 0:
            if len(self.warnings) == 0:
                print(
                    f"  {Colors.OKGREEN}{Colors.BOLD}✓ EXCELLENT - System is ready for launch!{Colors.ENDC}"
                )
            else:
                print(
                    f"  {Colors.WARNING}{Colors.BOLD}⚠ GOOD - System is functional but has warnings{Colors.ENDC}"
                )
        else:
            print(
                f"  {Colors.FAIL}{Colors.BOLD}✗ CRITICAL ISSUES - System is NOT ready for launch{Colors.ENDC}"
            )

        # Service URLs
        print(f"\n{Colors.BOLD}Service URLs:{Colors.ENDC}")
        print(f"  API:       {self.api_url}")
        print(f"  Dashboard: {self.ui_url}")
        print(f"  API Docs:  {self.api_url}/docs")

        # Save results to file
        self._save_results()

    def _save_results(self):
        """Save results to JSON file"""
        results_file = "logs/pre_launch_check.json"
        report = {
            "timestamp": datetime.now().isoformat(),
            "critical_failures": self.critical_failures,
            "warnings": self.warnings,
            "results": self.results,
        }

        try:
            os.makedirs("logs", exist_ok=True)
            with open(results_file, "w") as f:
                json.dump(report, f, indent=2)
            print(f"\n{Colors.OKCYAN}Full report saved to: {results_file}{Colors.ENDC}")
        except Exception as e:
            print(f"\n{Colors.WARNING}Could not save report: {str(e)}{Colors.ENDC}")


def main():
    """Main entry point"""
    checker = ReliabilityChecker()

    try:
        success = checker.run_all_checks()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n\n{Colors.WARNING}Check interrupted by user{Colors.ENDC}")
        sys.exit(2)
    except Exception as e:
        print(f"\n{Colors.FAIL}Fatal error: {str(e)}{Colors.ENDC}")
        sys.exit(3)


if __name__ == "__main__":
    main()
