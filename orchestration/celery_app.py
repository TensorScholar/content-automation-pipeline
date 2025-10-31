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

import os
import asyncio

from celery import Celery
from celery.signals import worker_process_init, worker_process_shutdown
from kombu import Queue
from loguru import logger

from container import container_manager

# Read Redis configuration from environment or use defaults
REDIS_HOST = os.getenv("REDIS_HOST", "localhost")
REDIS_PORT = os.getenv("REDIS_PORT", "6379")
REDIS_DB = os.getenv("REDIS_DB", "0")
REDIS_PASSWORD = os.getenv("REDIS_PASSWORD", "")

# Construct Redis URL
if REDIS_PASSWORD:
    REDIS_URL = f"redis://:{REDIS_PASSWORD}@{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"
else:
    REDIS_URL = f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}"

# Alternative: Use REDIS_URL environment variable directly if provided
REDIS_URL = os.getenv("REDIS_URL", REDIS_URL)

# Create Celery application instance
app = Celery(
    "content_automation",
    broker=REDIS_URL,
    backend=REDIS_URL,
    include=[
        "orchestration.tasks",  # Auto-discover tasks from this module
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
    # Queue configuration with priority support
    task_queues=(
        Queue("critical", routing_key="critical", priority=10),
        Queue("high", routing_key="high", priority=7),
        Queue("default", routing_key="default", priority=5),
        Queue("low", routing_key="low", priority=3),
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

# Optional: Configure Celery Beat for periodic tasks
app.conf.beat_schedule = {
    # Example periodic task - can be customized as needed
    "cleanup-old-results-daily": {
        "task": "orchestration.tasks.cleanup_old_results",
        "schedule": 86400.0,  # Every 24 hours
    },
}


@worker_process_init.connect
def on_worker_init(**kwargs):
    """Initialize DI container and resources when a Celery worker process starts."""
    try:
        logger.info(f"Celery worker process initializing... (PID: {os.getpid()})")
        # Create a new event loop for initialization
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Run the async initialization
        loop.run_until_complete(container_manager.initialize())
        loop.close()
        logger.success(f"Container initialized successfully for worker (PID: {os.getpid()}).")
    except Exception as e:
        logger.critical(f"Failed to initialize container for worker (PID: {os.getpid()}): {e}")
        # Exit the worker process if initialization fails
        os._exit(1)


@worker_process_shutdown.connect
def on_worker_shutdown(**kwargs):
    """Cleanup DI container and resources when a Celery worker process shuts down."""
    try:
        logger.info(f"Celery worker process shutting down... (PID: {os.getpid()})")
        # Create a new event loop for cleanup
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        # Run the async cleanup
        loop.run_until_complete(container_manager.cleanup())
        loop.close()
        logger.success(f"Container resources cleaned up for worker (PID: {os.getpid()}).")
    except Exception as e:
        logger.error(f"Failed to cleanup container for worker (PID: {os.getpid()}): {e}")


# Autodiscover tasks from Django apps (if using Django integration)
# app.autodiscover_tasks()

if __name__ == "__main__":
    app.start()
