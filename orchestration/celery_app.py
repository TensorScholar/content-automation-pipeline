"""
Celery Application Configuration
=================================

Configures Celery distributed task queue for asynchronous operations with:
- Redis as message broker and result backend
- Task auto-discovery
- Production-ready settings for reliability
- Monitoring and metrics integration

Design Pattern: Distributed Task Queue with Result Backend
"""

# CRITICAL: Set spawn method FIRST to prevent fork() issues with PyTorch/MPS
# This must be done before any other imports that might use multiprocessing
import multiprocessing

try:
    multiprocessing.set_start_method("spawn", force=True)
except RuntimeError as e:
    # Already set - this is expected in some environments (e.g., Jupyter, tests)
    import logging

    logging.getLogger(__name__).debug(f"Multiprocessing start method already set: {e}")

import asyncio
import os
from typing import Optional

from celery import Celery
from celery.schedules import crontab
from celery.signals import worker_process_init, worker_process_shutdown
from kombu import Queue
from loguru import logger

from api.dependencies import get_database
from config.settings import get_settings
from infrastructure.error_tracking import initialize_sentry

_worker_loop: asyncio.AbstractEventLoop | None = None

# Build Celery app lazily so importing this module (e.g. shared by the API
# import graph) never requires runtime configuration. Settings are resolved
# only the first time the Celery runtime actually needs them.
_built_app: Optional[Celery] = None


def _build_celery_app() -> Celery:
    """Construct the singleton Celery application using lazily-resolved settings."""
    global _built_app
    if _built_app is not None:
        return _built_app

    settings = get_settings()
    initialize_sentry("celery")

    # Create Celery application instance
    app = Celery(
        "content_automation",
        broker=settings.celery.broker_url,
        backend=settings.celery.result_backend,
        include=[
            "orchestration.tasks",  # Auto-discover tasks from this module
            "orchestration.cleanup_tasks",  # Periodic cleanup and maintenance tasks
        ],
    )

    # Configure Celery for production reliability and performance
    app.conf.update(
        # Serialization
        task_serializer="json",
        accept_content=["json"],
        result_serializer="json",
        # Timezone
        timezone="UTC",
        enable_utc=True,
        # Task execution
        task_track_started=True,
        task_time_limit=3600,  # 1 hour hard limit
        task_soft_time_limit=3000,  # 50 minutes soft limit
        # Worker configuration
        worker_prefetch_multiplier=1,  # Fair task distribution
        worker_max_tasks_per_child=50,  # Restart workers periodically to prevent memory leaks
        worker_disable_rate_limits=False,
        # Result backend
        result_expires=86400,  # Results expire after 24 hours
        result_backend_transport_options={
            "retry_policy": {
                "timeout": 5.0,
            },
        },
        # Task acknowledgment (critical for reliability)
        task_acks_late=True,  # Acknowledge after completion, not before
        task_reject_on_worker_lost=True,  # Re-queue tasks if worker dies
        # Queue configuration with priority support and dead letter queue
        task_queues=(
            Queue("critical", routing_key="critical", priority=10),
            Queue("high", routing_key="high", priority=7),
            Queue("medium", routing_key="medium", priority=5),
            Queue("default", routing_key="default", priority=5),
            Queue("integrations", routing_key="integrations", priority=6),
            Queue("low", routing_key="low", priority=3),
            Queue(
                "dead_letter", routing_key="dead_letter", priority=0
            ),  # DLQ for permanently failed tasks
        ),
        task_default_queue="default",
        task_default_routing_key="default",
        # Monitoring
        worker_send_task_events=True,
        task_send_sent_event=True,
        # Retry configuration
        task_autoretry_for=(Exception,),
        task_max_retries=3,
        task_default_retry_delay=300,  # 5 minutes
        # Result compression
        result_compression="gzip",
        task_compression="gzip",
    )

    # Configure Celery Beat for periodic tasks
    # Supports both file-based and Redis-based (redbeat) scheduling
    app.conf.beat_schedule = {
        "cleanup-old-tasks-daily": {
            "task": "cleanup_old_task_results",
            "schedule": crontab(hour=2, minute=0),  # 2 AM daily
            "options": {"queue": "low"},  # Low priority maintenance task
        },
        "cleanup-cache-hourly": {
            "task": "cleanup_expired_cache",
            "schedule": crontab(minute=0),  # Top of every hour
            "options": {"queue": "low"},
        },
        "search-console-sync-weekly": {
            "task": "content_automation.sync_all_search_console",
            "schedule": crontab(day_of_week=1, hour=4, minute=0),
            "options": {"queue": "integrations"},
        },
        "search-console-sync-reconciliation": {
            "task": "content_automation.reconcile_search_console_syncs",
            "schedule": crontab(minute="*/15"),
            "options": {"queue": "integrations"},
        },
        "search-console-oauth-state-cleanup-hourly": {
            "task": "content_automation.cleanup_search_console_oauth_states",
            "schedule": crontab(minute=15),
            "options": {"queue": "low"},
        },
        "wordpress-publish-reconciliation": {
            "task": "content_automation.reconcile_wordpress_publishes",
            "schedule": crontab(minute="*/15"),
            "options": {"queue": "integrations"},
        },
        "integration-operational-metrics": {
            "task": "content_automation.refresh_integration_operational_metrics",
            "schedule": crontab(minute="*/5"),
            "options": {"queue": "integrations"},
        },
        "vacuum-database-weekly": {
            "task": "vacuum_analyze_database",
            "schedule": crontab(day_of_week=0, hour=3, minute=0),  # Sunday 3 AM
            "options": {"queue": "low"},
        },
    }

    # RedBeat configuration for distributed beat scheduling (eliminates SPOF)
    # Enable by running: celery -A orchestration.celery_app.app beat -S redbeat.RedBeatScheduler
    app.conf.update(
        # Redbeat settings - stores schedule in Redis for efficiency
        redbeat_redis_url=settings.celery.broker_url,
        redbeat_key_prefix="redbeat:",
        redbeat_lock_timeout=300,  # 5 minute lock timeout
    )

    _built_app = app
    return app


class _LazyCeleryProxy:
    """Module-level Celery stand-in that builds the real app on first use."""

    __slots__ = ()

    def __getattr__(self, name: str):
        return getattr(_build_celery_app(), name)

    def __setattr__(self, name: str, value) -> None:
        setattr(_build_celery_app(), name, value)

    def __delattr__(self, name: str) -> None:
        delattr(_build_celery_app(), name)


app = _LazyCeleryProxy()


@worker_process_init.connect
def on_worker_init(**kwargs):
    """Initialize DI container and resources when a Celery worker process starts."""
    global _worker_loop
    loop = None
    try:
        logger.info(f"Celery worker process initializing... (PID: {os.getpid()})")

        # Log available hardware acceleration
        try:
            import torch

            if torch.cuda.is_available():
                logger.info(f"CUDA available: {torch.cuda.get_device_name(0)}")
            elif torch.backends.mps.is_available():
                logger.info("MPS (Apple Metal) backend available and enabled")
            else:
                logger.info("Running on CPU (no GPU acceleration available)")
        except ImportError:
            logger.debug("PyTorch not installed, skipping device detection")

        # Create a new event loop for initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        _worker_loop = loop

        # Initialize database
        db = get_database()
        loop.run_until_complete(db.initialize())

        logger.success(f"Resources initialized successfully for worker (PID: {os.getpid()}).")
    except Exception as e:
        logger.critical(f"Failed to initialize resources for worker (PID: {os.getpid()}): {e}")
        # Exit the worker process if initialization fails
        os._exit(1)
    finally:
        # Keep the loop running for the worker lifecycle
        if loop and not loop.is_closed():
            # Don't close the loop here - it's needed for async tasks
            pass


@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Cleanup resources when a Celery worker process shuts down."""
    global _worker_loop
    loop = _worker_loop
    try:
        logger.info(f"Celery worker process shutting down... (PID: {os.getpid()})")

        # Clear per-process registries to release resources
        from api.dependencies import clear_redis_registry, reset_semantic_analyzer

        clear_redis_registry()
        reset_semantic_analyzer()

        if loop is None or loop.is_closed():
            loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        # Close database connections properly
        try:
            db = get_database()
            loop.run_until_complete(db.close())
            get_database.cache_clear()

            from infrastructure.database import reset_db_manager

            reset_db_manager()
        except Exception as db_error:
            logger.warning(f"Database cleanup error (may already be closed): {db_error}")

        logger.success(f"Resources cleaned up for worker (PID: {os.getpid()}).")
    except Exception as e:
        logger.error(f"Failed to cleanup resources for worker (PID: {os.getpid()}): {e}")
    finally:
        # Always close the event loop to prevent memory leaks
        if loop and not loop.is_closed():
            try:
                # Cancel all running tasks
                pending = asyncio.all_tasks(loop)
                for task in pending:
                    task.cancel()

                # Run loop one final time to process cancellations
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))

                # Close the loop
                loop.close()
                _worker_loop = None
                logger.debug(f"Event loop closed for worker (PID: {os.getpid()})")
            except Exception as loop_error:
                logger.warning(f"Error closing event loop: {loop_error}")


# Autodiscover tasks from Django apps (if using Django integration)
# app.autodiscover_tasks()

if __name__ == "__main__":
    app.start()
