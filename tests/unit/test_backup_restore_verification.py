from __future__ import annotations

import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "maintenance" / "verify_backup_restore.sh"


def _install_fake_docker(tmp_path: Path) -> tuple[Path, Path]:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    log_file = tmp_path / "docker.log"
    docker = bin_dir / "docker"
    docker.write_text(
        r'''#!/usr/bin/env bash
set -euo pipefail

printf '%s\n' "$*" >> "$FAKE_DOCKER_LOG"
joined="$*"

if [[ "$joined" == "compose version" ]]; then
  exit 0
fi

if [[ " $joined " == *" ps "* ]]; then
  printf '%s\n' "${POSTGRES_SERVICE:-postgres}"
  exit 0
fi

if [[ "$joined" == *" alembic heads"* ]]; then
  printf '%s\n' "${FAKE_ALEMBIC_HEADS:-20260903_001 (head)}"
  exit 0
fi

if [[ " $joined " == *" pg_dump "* ]]; then
  printf 'FAKE_CUSTOM_DUMP\n'
  exit 0
fi

if [[ "$joined" == *"pg_restore --list"* || "$joined" == *"pg_restore --exit-on-error"* ]]; then
  exit 0
fi

if [[ " $joined " == *" createdb "* || " $joined " == *" dropdb "* ]]; then
  exit 0
fi

if [[ " $joined " == *" psql "* ]]; then
  database=""
  sql=""
  for arg in "$@"; do
    case "$arg" in
      --dbname=*) database="${arg#--dbname=}" ;;
      --command=*) sql="${arg#--command=}" ;;
    esac
  done

  if [[ "$sql" == *"FROM alembic_version"* && "$sql" == *"COUNT(*) = 1"* ]]; then
    if [[ "$database" == smarlux_restore_verify_* ]]; then
      printf '%s\n' "${FAKE_RESTORE_HEAD:-20260903_001}"
    else
      printf '%s\n' "${FAKE_SOURCE_HEAD:-20260903_001}"
    fi
    exit 0
  fi

  if [[ -n "${FAKE_FAIL_GUARD:-}" && "$sql" == *"$FAKE_FAIL_GUARD"* ]]; then
    printf 'f\n'
    exit 0
  fi

  if [[ "$sql" == *"to_regclass"* || "$sql" == *"SELECT EXISTS"* || "$sql" == *"NOT EXISTS"* ]]; then
    printf 't\n'
    exit 0
  fi

  printf 't\n'
  exit 0
fi

printf 'unexpected fake docker invocation: %s\n' "$joined" >&2
exit 9
''',
        encoding="utf-8",
    )
    docker.chmod(0o755)
    return bin_dir, log_file


def _run(
    tmp_path: Path,
    *args: str,
    source_head: str = "20260903_001",
    restore_head: str = "20260903_001",
    alembic_heads: str = "20260903_001 (head)",
    extra_env: dict[str, str] | None = None,
) -> tuple[subprocess.CompletedProcess[str], str]:
    bin_dir, log_file = _install_fake_docker(tmp_path)
    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{bin_dir}:{env['PATH']}",
            "FAKE_DOCKER_LOG": str(log_file),
            "FAKE_SOURCE_HEAD": source_head,
            "FAKE_RESTORE_HEAD": restore_head,
            "FAKE_ALEMBIC_HEADS": alembic_heads,
        }
    )
    if extra_env:
        env.update(extra_env)

    result = subprocess.run(
        ["bash", str(SCRIPT), *args],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    return result, log_file.read_text(encoding="utf-8")


def test_temporary_backup_captures_source_head_without_repository_lookup(tmp_path: Path) -> None:
    result, docker_log = _run(tmp_path, "--confirm-disposable-restore")

    assert result.returncode == 0, result.stderr
    assert "BACKUP_FORMAT_PASS" in result.stdout
    assert "RESTORE_REVISION_SCHEMA_GUARDS_PASS" in result.stdout
    assert "RESTORE_REVISION_INTEGRITY_PASS" in result.stdout
    assert "alembic_head=20260903_001" in result.stdout
    assert "alembic heads" not in docker_log


def test_restore_fails_when_restored_head_differs_from_backup_source(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        "--confirm-disposable-restore",
        source_head="20260903_001",
        restore_head="20260801_001",
    )

    assert result.returncode != 0
    assert "Alembic head '20260801_001' != '20260903_001'" in result.stderr


def test_external_backup_resolves_single_repository_head(tmp_path: Path) -> None:
    backup = tmp_path / "external.dump"
    backup.write_bytes(b"fake-backup")

    result, docker_log = _run(tmp_path, str(backup), "--confirm-disposable-restore")

    assert result.returncode == 0, result.stderr
    assert "alembic heads" in docker_log
    assert "alembic_head=20260903_001" in result.stdout


def test_external_backup_rejects_multiple_repository_heads(tmp_path: Path) -> None:
    backup = tmp_path / "external.dump"
    backup.write_bytes(b"fake-backup")

    result, _ = _run(
        tmp_path,
        str(backup),
        "--confirm-disposable-restore",
        alembic_heads="20260903_001 (head)\nbranch_b_001 (head)",
    )

    assert result.returncode != 0
    assert "expected exactly one repository Alembic head, found 2" in result.stderr


def test_revision_guard_failure_is_not_accepted(tmp_path: Path) -> None:
    result, _ = _run(
        tmp_path,
        "--confirm-disposable-restore",
        extra_env={"FAKE_FAIL_GUARD": "trg_article_revisions_prevent_update"},
    )

    assert result.returncode != 0
    assert "missing or disabled trigger trg_article_revisions_prevent_update" in result.stderr
