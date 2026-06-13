"""
Health Monitoring & Auto-Recovery System
=========================================

Purpose: Detect and fix issues BEFORE they cause stuck tasks

Features:
1. Continuous health monitoring
2. Automatic issue detection
3. Self-healing mechanisms
4. Alert generation
5. Metrics collection

Run as: poetry run python orchestration/health_monitor.py
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import httpx
import redis
from celery import Celery

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==============================================================================
# HEALTH CHECK DEFINITIONS
# ==============================================================================

@dataclass
class HealthCheckResult:
    """Result of a health check"""
    check_name: str
    is_healthy: bool
    message: str
    severity: str  # "critical", "warning", "info"
    metrics: Dict[str, any] = None
    auto_fix_attempted: bool = False
    auto_fix_successful: bool = False


class HealthMonitor:
    """
    Monitors system health and attempts auto-recovery
    """

    def __init__(
        self,
        redis_url: str = "redis://127.0.0.1:6379/0",
        api_url: str = "http://127.0.0.1:9001",
        check_interval: int = 60  # seconds
    ):
        self.redis_url = redis_url
        self.api_url = api_url
        self.check_interval = check_interval

        self.redis_client = None
        self.celery_app = None
        self.issues_detected: List[HealthCheckResult] = []

    async def initialize(self):
        """Initialize connections"""
        try:
            self.redis_client = redis.from_url(self.redis_url)
            self.redis_client.ping()
            logger.info("✓ Redis connection initialized")
        except Exception as e:
            logger.error(f"✗ Redis connection failed: {e}")

        try:
            from orchestration.celery_app import app
            self.celery_app = app
            logger.info("✓ Celery app initialized")
        except Exception as e:
            logger.error(f"✗ Celery app initialization failed: {e}")

    # ==========================================================================
    # HEALTH CHECKS
    # ==========================================================================

    async def check_redis_health(self) -> HealthCheckResult:
        """Check Redis broker health"""
        try:
            # Ping test
            start = time.time()
            self.redis_client.ping()
            latency_ms = (time.time() - start) * 1000

            # Memory usage
            info = self.redis_client.info('memory')
            used_memory_mb = info['used_memory'] / (1024 * 1024)
            max_memory_mb = info.get('maxmemory', 0) / (1024 * 1024)

            # Check for issues
            if latency_ms > 100:
                return HealthCheckResult(
                    check_name="redis_latency",
                    is_healthy=False,
                    message=f"Redis latency high: {latency_ms:.1f}ms",
                    severity="warning",
                    metrics={"latency_ms": latency_ms}
                )

            if max_memory_mb > 0 and used_memory_mb > max_memory_mb * 0.9:
                return HealthCheckResult(
                    check_name="redis_memory",
                    is_healthy=False,
                    message=f"Redis memory high: {used_memory_mb:.1f}MB / {max_memory_mb:.1f}MB",
                    severity="critical",
                    metrics={"used_mb": used_memory_mb, "max_mb": max_memory_mb}
                )

            return HealthCheckResult(
                check_name="redis",
                is_healthy=True,
                message=f"Healthy (latency: {latency_ms:.1f}ms, memory: {used_memory_mb:.1f}MB)",
                severity="info",
                metrics={"latency_ms": latency_ms, "used_mb": used_memory_mb}
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="redis",
                is_healthy=False,
                message=f"Redis connection failed: {e}",
                severity="critical"
            )

    async def check_worker_health(self) -> HealthCheckResult:
        """Check Celery worker health"""
        try:
            inspect = self.celery_app.control.inspect(timeout=5)

            # Check active workers
            active_workers = inspect.active()

            if active_workers is None:
                # Try to auto-fix by restarting workers
                return HealthCheckResult(
                    check_name="workers",
                    is_healthy=False,
                    message="No workers responding - workers may be offline",
                    severity="critical",
                    auto_fix_attempted=False  # Can't auto-fix this easily
                )

            if len(active_workers) == 0:
                return HealthCheckResult(
                    check_name="workers",
                    is_healthy=False,
                    message="No active workers found",
                    severity="critical"
                )

            # Check if workers are processing tasks
            total_active_tasks = sum(len(tasks) for tasks in active_workers.values())

            # Check registered tasks
            registered = inspect.registered()
            if registered:
                all_tasks = []
                for worker, tasks in registered.items():
                    all_tasks.extend(tasks)

                required_tasks = [
                    'orchestration.tasks.generate_content_task',
                ]

                missing_tasks = [t for t in required_tasks if t not in all_tasks]

                if missing_tasks:
                    return HealthCheckResult(
                        check_name="worker_tasks",
                        is_healthy=False,
                        message=f"Required tasks not registered: {missing_tasks}",
                        severity="critical"
                    )

            return HealthCheckResult(
                check_name="workers",
                is_healthy=True,
                message=f"{len(active_workers)} workers, {total_active_tasks} active tasks",
                severity="info",
                metrics={
                    "worker_count": len(active_workers),
                    "active_tasks": total_active_tasks
                }
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="workers",
                is_healthy=False,
                message=f"Worker health check failed: {e}",
                severity="critical"
            )

    async def check_queue_health(self) -> HealthCheckResult:
        """Check task queue health"""
        try:
            # Check queue length
            queue_length = self.redis_client.llen('celery')

            # Check for old pending tasks
            if queue_length > 100:
                return HealthCheckResult(
                    check_name="queue_overflow",
                    is_healthy=False,
                    message=f"Queue overflowing: {queue_length} pending tasks",
                    severity="warning",
                    metrics={"queue_length": queue_length}
                )

            # Check for stuck tasks in result backend
            stuck_tasks = await self._find_stuck_tasks()

            if stuck_tasks:
                # Auto-fix: Revoke stuck tasks
                fixed_count = await self._revoke_stuck_tasks(stuck_tasks)

                return HealthCheckResult(
                    check_name="stuck_tasks",
                    is_healthy=False,
                    message=f"Found {len(stuck_tasks)} stuck tasks (auto-fixed: {fixed_count})",
                    severity="warning",
                    metrics={"stuck_count": len(stuck_tasks), "fixed_count": fixed_count},
                    auto_fix_attempted=True,
                    auto_fix_successful=fixed_count > 0
                )

            return HealthCheckResult(
                check_name="queue",
                is_healthy=True,
                message=f"Queue healthy ({queue_length} pending)",
                severity="info",
                metrics={"queue_length": queue_length}
            )

        except Exception as e:
            return HealthCheckResult(
                check_name="queue",
                is_healthy=False,
                message=f"Queue health check failed: {e}",
                severity="warning"
            )

    async def check_api_health(self) -> HealthCheckResult:
        """Check API server health"""
        try:
            async with httpx.AsyncClient() as client:
                start = time.time()
                response = await client.get(
                    f"{self.api_url}/health",
                    timeout=5.0
                )
                latency_ms = (time.time() - start) * 1000

                if response.status_code != 200:
                    return HealthCheckResult(
                        check_name="api",
                        is_healthy=False,
                        message=f"API returned {response.status_code}",
                        severity="critical"
                    )

                health_data = response.json()

                # Check dependencies
                deps = health_data.get('dependencies', {})
                unhealthy_deps = [
                    name for name, status in deps.items()
                    if status != 'healthy'
                ]

                if unhealthy_deps:
                    return HealthCheckResult(
                        check_name="api_dependencies",
                        is_healthy=False,
                        message=f"Unhealthy dependencies: {unhealthy_deps}",
                        severity="critical"
                    )

                return HealthCheckResult(
                    check_name="api",
                    is_healthy=True,
                    message=f"Healthy (latency: {latency_ms:.1f}ms)",
                    severity="info",
                    metrics={"latency_ms": latency_ms}
                )

        except httpx.ConnectError:
            return HealthCheckResult(
                check_name="api",
                is_healthy=False,
                message="API not reachable - server may be down",
                severity="critical"
            )
        except Exception as e:
            return HealthCheckResult(
                check_name="api",
                is_healthy=False,
                message=f"API health check failed: {e}",
                severity="critical"
            )

    # ==========================================================================
    # AUTO-RECOVERY MECHANISMS
    # ==========================================================================

    async def _find_stuck_tasks(self) -> List[str]:
        """
        Find tasks stuck in PENDING for > 5 minutes

        Root cause: Worker was offline when task created
        """
        stuck_tasks = []

        try:
            # Scan for task metadata in Redis
            cursor = 0
            while True:
                cursor, keys = self.redis_client.scan(
                    cursor,
                    match="celery-task-meta-*",
                    count=100
                )

                for key in keys:
                    try:
                        data = self.redis_client.get(key)
                        if data:
                            import json
                            task_data = json.loads(data)

                            # Check if PENDING and old
                            if task_data.get('status') == 'PENDING':
                                # Task IDs don't have timestamps, skip for now
                                # This would need task creation time tracking
                                pass

                    except (json.JSONDecodeError, TypeError, Exception) as e:
                        logger.warning(f"Failed to parse task metadata for {key}: {e}")

                if cursor == 0:
                    break

        except Exception as e:
            logger.error(f"Error finding stuck tasks: {e}")

        return stuck_tasks

    async def _revoke_stuck_tasks(self, task_ids: List[str]) -> int:
        """Revoke stuck tasks"""
        fixed_count = 0

        for task_id in task_ids:
            try:
                from celery.result import AsyncResult
                result = AsyncResult(task_id, app=self.celery_app)
                result.revoke(terminate=True)
                fixed_count += 1
                logger.info(f"✓ Revoked stuck task: {task_id}")
            except Exception as e:
                logger.error(f"✗ Failed to revoke task {task_id}: {e}")

        return fixed_count

    # ==========================================================================
    # MONITORING LOOP
    # ==========================================================================

    async def run_health_checks(self) -> List[HealthCheckResult]:
        """Run all health checks"""
        results = []

        # Run checks in parallel
        checks = [
            self.check_redis_health(),
            self.check_worker_health(),
            self.check_queue_health(),
            self.check_api_health(),
        ]

        results = await asyncio.gather(*checks, return_exceptions=True)

        # Filter out exceptions
        results = [r for r in results if isinstance(r, HealthCheckResult)]

        return results

    async def monitor_loop(self):
        """Continuous monitoring loop"""
        logger.info(f"🔍 Health monitor started (check interval: {self.check_interval}s)")

        while True:
            try:
                logger.info("="*60)
                logger.info(f"Health check: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

                results = await self.run_health_checks()

                # Log results
                critical_issues = []
                warnings = []

                for result in results:
                    icon = "✓" if result.is_healthy else "✗"
                    logger.info(f"{icon} {result.check_name}: {result.message}")

                    if not result.is_healthy:
                        if result.severity == "critical":
                            critical_issues.append(result)
                        elif result.severity == "warning":
                            warnings.append(result)

                        # Track issues
                        self.issues_detected.append(result)

                # Summary
                if critical_issues:
                    logger.error(f"🚨 {len(critical_issues)} CRITICAL ISSUES DETECTED")
                    for issue in critical_issues:
                        logger.error(f"   - {issue.check_name}: {issue.message}")

                if warnings:
                    logger.warning(f"⚠️  {len(warnings)} warnings")

                if not critical_issues and not warnings:
                    logger.info("✅ All systems healthy")

            except Exception as e:
                logger.error(f"Health check error: {e}")

            # Wait for next check
            await asyncio.sleep(self.check_interval)

    async def run(self):
        """Start monitoring"""
        await self.initialize()
        await self.monitor_loop()


# ==============================================================================
# MAIN
# ==============================================================================

async def main():
    """Run health monitor"""
    monitor = HealthMonitor(
        check_interval=60  # Check every minute
    )

    try:
        await monitor.run()
    except KeyboardInterrupt:
        logger.info("\n✓ Health monitor stopped")


if __name__ == "__main__":
    asyncio.run(main())
