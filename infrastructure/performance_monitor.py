"""
Performance Monitoring and Alerting

Tracks key performance metrics and sends alerts when thresholds are exceeded.
"""

import asyncio
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, Optional

from loguru import logger

from infrastructure.database import DatabaseManager
from infrastructure.redis_client import RedisClient


class PerformanceMonitor:
    """Monitor system performance and trigger alerts."""

    def __init__(self, database: DatabaseManager, redis: RedisClient):
        self.database = database
        self.redis = redis

        # Alert thresholds
        self.SLOW_QUERY_THRESHOLD_MS = 5000  # 5 seconds
        self.DAILY_COST_ALERT_THRESHOLD = 10.0  # $10
        self.CONNECTION_POOL_WARNING_THRESHOLD = 0.8  # 80% utilization

    async def check_slow_queries(self) -> Dict[str, any]:
        """Check for slow database queries."""
        try:
            async with self.database.read_session() as session:
                from sqlalchemy.sql import text

                # First check if pg_stat_statements extension is installed
                # This is a fast catalog query that won't block
                ext_check = await session.execute(text(
                    "SELECT 1 FROM pg_extension WHERE extname = 'pg_stat_statements'"
                ))
                if not ext_check.fetchone():
                    return {"status": "unavailable", "error": "pg_stat_statements extension not installed"}

                # Extension exists — safe to query
                result = await session.execute(text("""
                    SELECT
                        query,
                        calls,
                        mean_exec_time,
                        max_exec_time
                    FROM pg_stat_statements
                    WHERE mean_exec_time > :threshold
                    ORDER BY mean_exec_time DESC
                    LIMIT 10
                """), {"threshold": self.SLOW_QUERY_THRESHOLD_MS})

                slow_queries = result.fetchall()

                if slow_queries:
                    logger.warning(f"Found {len(slow_queries)} slow queries")
                    return {
                        "status": "warning",
                        "slow_queries_count": len(slow_queries),
                        "queries": [
                            {
                                "query": q[0][:100],
                                "calls": q[1],
                                "mean_time_ms": q[2],
                                "max_time_ms": q[3]
                            }
                            for q in slow_queries
                        ]
                    }

                return {"status": "ok", "slow_queries_count": 0}

        except Exception as e:
            logger.debug(f"pg_stat_statements not available: {e}")
            return {"status": "unavailable", "error": str(e)}

    async def check_daily_costs(self) -> Dict[str, any]:
        """Check LLM API costs for the current day."""
        try:
            async with self.database.read_session() as session:
                from sqlalchemy.sql import text

                # Use SQL's date_trunc to avoid timezone issues
                result = await session.execute(text("""
                    SELECT
                        COUNT(*) as article_count,
                        SUM(total_cost) as total_cost
                    FROM generated_articles
                    WHERE created_at >= date_trunc('day', NOW())
                """))

                row = result.fetchone()
                article_count = row[0] or 0
                total_cost = float(row[1] or 0)

                status = "ok"
                if total_cost >= self.DAILY_COST_ALERT_THRESHOLD:
                    status = "warning"
                    logger.warning(
                        f"Daily cost threshold exceeded: ${total_cost:.2f} "
                        f"(threshold: ${self.DAILY_COST_ALERT_THRESHOLD})"
                    )

                return {
                    "status": status,
                    "date": datetime.now(timezone.utc).date().isoformat(),
                    "article_count": article_count,
                    "total_cost_usd": round(total_cost, 4),
                    "threshold_usd": self.DAILY_COST_ALERT_THRESHOLD,
                }
        except Exception as e:
            logger.error(f"Failed to check daily costs: {e}")
            return {"status": "error", "error": str(e)}

    async def check_connection_pool(self) -> Dict[str, any]:
        """Check database connection pool utilization."""
        try:
            writer_engine = self.database._writer_engine

            if writer_engine:
                pool = writer_engine.pool
                pool_size = pool.size()
                checked_out = pool.checkedout()
                overflow = pool.overflow()

                utilization = checked_out / (pool_size + overflow) if (pool_size + overflow) > 0 else 0

                status = "ok"
                if utilization >= self.CONNECTION_POOL_WARNING_THRESHOLD:
                    status = "warning"
                    logger.warning(
                        f"Connection pool utilization high: {utilization*100:.1f}% "
                        f"({checked_out}/{pool_size + overflow})"
                    )

                return {
                    "status": status,
                    "pool_size": pool_size,
                    "checked_out": checked_out,
                    "overflow": overflow,
                    "utilization_percent": round(utilization * 100, 1),
                }

            return {"status": "unavailable", "error": "Engine not initialized"}

        except Exception as e:
            logger.error(f"Failed to check connection pool: {e}")
            return {"status": "error", "error": str(e)}

    async def get_system_health(self) -> Dict[str, any]:
        """Get comprehensive system health metrics."""
        try:
            slow_queries = await self.check_slow_queries()
            daily_costs = await self.check_daily_costs()
            connection_pool = await self.check_connection_pool()

            # Check Redis health
            redis_ok = await self.redis.ping()

            overall_status = "healthy"
            if any(
                m.get("status") == "warning"
                for m in [slow_queries, daily_costs, connection_pool]
            ):
                overall_status = "degraded"

            if not redis_ok or any(
                m.get("status") == "error"
                for m in [slow_queries, daily_costs, connection_pool]
            ):
                overall_status = "unhealthy"

            return {
                "overall_status": overall_status,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "metrics": {
                    "slow_queries": slow_queries,
                    "daily_costs": daily_costs,
                    "connection_pool": connection_pool,
                    "redis": {"status": "ok" if redis_ok else "error"}
                }
            }
        except Exception as e:
            logger.error(f"Failed to get system health: {e}")
            return {
                "overall_status": "error",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "error": str(e)
            }


async def monitor_performance_loop(
    database: DatabaseManager,
    redis: RedisClient,
    interval_seconds: int = 300  # Check every 5 minutes
):
    """Background task that continuously monitors performance."""
    monitor = PerformanceMonitor(database, redis)

    while True:
        try:
            health = await monitor.get_system_health()

            # Log status
            status = health["overall_status"]
            if status == "degraded":
                logger.warning(f"System performance degraded: {health}")
            elif status == "unhealthy":
                logger.error(f"System unhealthy: {health}")
            else:
                logger.info("System health check: OK")

            # Store metrics in Redis for dashboard
            await redis.set(
                "system:health:latest",
                health,
                ttl=interval_seconds * 2
            )

        except Exception as e:
            logger.error(f"Performance monitoring error: {e}")

        await asyncio.sleep(interval_seconds)
