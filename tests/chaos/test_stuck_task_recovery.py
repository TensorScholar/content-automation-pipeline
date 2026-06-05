import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest
from celery.result import AsyncResult

from orchestration.celery_app import app
from orchestration.tasks import generate_social_posts_task


MTTR_SLA_SECONDS = int(os.getenv("CHAOS_MTTR_SLA_SECONDS", "120"))
STALE_SECONDS = int(os.getenv("CHAOS_STALE_SECONDS", "45"))

SKILL_DIR = Path(
    os.getenv(
        "HEALER_SKILL_DIR",
        str(Path.home() / ".codex" / "skills" / "celery-stuck-task-healer"),
    )
)
SCAN_SCRIPT = Path(
    os.getenv(
        "HEALER_SCAN_SCRIPT", str(SKILL_DIR / "scripts" / "scan_stuck_tasks.py")
    )
)
HEAL_SCRIPT = Path(
    os.getenv(
        "HEALER_HEAL_SCRIPT", str(SKILL_DIR / "scripts" / "heal_stuck_tasks.py")
    )
)

KILL_WORKER_CMD = os.getenv(
    "CHAOS_KILL_WORKER_CMD",
    'pkill -f "celery -A orchestration.celery_app.app worker"',
)
START_WORKER_CMD = os.getenv(
    "CHAOS_START_WORKER_CMD",
    "celery -A orchestration.celery_app.app worker --loglevel=info --concurrency=2 --detach --pidfile=/tmp/celery-chaos.pid",
)


def _run_shell(cmd: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        shell=True,
        check=check,
        capture_output=True,
        text=True,
    )


def _worker_up() -> bool:
    inspect = app.control.inspect(timeout=2)
    try:
        ping = inspect.ping()
    except Exception:
        return False
    return bool(ping)


def _run_scan(output_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(SCAN_SCRIPT),
        "--stale-seconds",
        str(STALE_SECONDS),
        "--output",
        str(output_path),
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(output_path.read_text(encoding="utf-8"))


def _run_heal(scan_input: Path, output_path: Path) -> dict:
    cmd = [
        sys.executable,
        str(HEAL_SCRIPT),
        "--mode",
        "guarded-auto",
        "--scan-input",
        str(scan_input),
        "--output",
        str(output_path),
        "--restart-worker-cmd",
        START_WORKER_CMD,
    ]
    proc = subprocess.run(cmd, check=True, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr or proc.stdout)
    return json.loads(output_path.read_text(encoding="utf-8"))


@pytest.mark.chaos
@pytest.mark.integration
def test_worker_down_and_pending_stale_recovery_within_mttr(tmp_path: Path):
    """
    Chaos proof:
    1) Inject WORKER_DOWN by stopping Celery worker.
    2) Enqueue a lightweight task to force pending backlog.
    3) Verify detector classifies WORKER_DOWN_QUEUE_GROWING and PENDING_STALE.
    4) Run guarded healer and prove task completes within MTTR SLA.
    """
    if not SCAN_SCRIPT.exists() or not HEAL_SCRIPT.exists():
        pytest.skip("Healer scripts not found. Install skill first.")

    # Inject worker-down condition
    _run_shell(KILL_WORKER_CMD, check=False)
    time.sleep(2)
    assert not _worker_up(), "Chaos injection failed: worker still appears up"

    # Enqueue a lightweight task that should complete quickly once worker returns
    task = generate_social_posts_task.apply_async(
        kwargs={
            "article_id": str(uuid.uuid4()),
            "title": "Chaos MTTR Proof",
            "topic": "Self-healing reliability",
            "language": "en",
        },
        queue="default",
        routing_key="default",
    )
    t_inject = time.time()

    scan_path = tmp_path / "scan.json"
    heal_path = tmp_path / "heal.json"

    seen_worker_down = False
    seen_pending_stale = False

    # Wait until detector has enough evidence for stale classification
    detection_deadline = t_inject + STALE_SECONDS + 60
    while time.time() < detection_deadline:
        scan = _run_scan(scan_path)
        classes = {issue.get("class") for issue in scan.get("issues", [])}
        seen_worker_down = seen_worker_down or ("WORKER_DOWN_QUEUE_GROWING" in classes)
        seen_pending_stale = seen_pending_stale or ("PENDING_STALE" in classes)
        if seen_worker_down and seen_pending_stale:
            break
        time.sleep(5)

    assert seen_worker_down, "Detector did not classify WORKER_DOWN_QUEUE_GROWING"
    assert seen_pending_stale, "Detector did not classify PENDING_STALE"

    heal = _run_heal(scan_path, heal_path)
    assert heal.get("mode") == "guarded-auto"

    # Recovery proof
    result = AsyncResult(task.id, app=app)
    recover_deadline = t_inject + MTTR_SLA_SECONDS
    while time.time() < recover_deadline:
        if result.state == "SUCCESS":
            break
        time.sleep(2)

    elapsed = time.time() - t_inject
    assert result.state == "SUCCESS", f"Task did not recover. Final state={result.state}"
    assert elapsed <= MTTR_SLA_SECONDS, (
        f"MTTR SLA breach: elapsed={elapsed:.2f}s > {MTTR_SLA_SECONDS}s"
    )

    # Cleanup/ensure worker is running for subsequent tests
    if not _worker_up():
        _run_shell(START_WORKER_CMD, check=False)
