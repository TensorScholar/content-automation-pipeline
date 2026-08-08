#!/usr/bin/env python3
"""Validate the Alembic revision graph without importing application dependencies."""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

EXPECTED_HEAD = "20260801_001"
ROOT = Path(__file__).resolve().parents[2]
VERSIONS = ROOT / "alembic" / "versions"


def literal_assignment(tree: ast.Module, name: str) -> Any:
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id == name:
                return ast.literal_eval(node.value)
    raise ValueError(f"Missing {name!r} assignment")


def normalize_parents(value: Any) -> set[str]:
    if value is None:
        return set()
    if isinstance(value, str):
        return {value}
    if isinstance(value, (tuple, list, set)):
        return {str(item) for item in value if item}
    raise ValueError(f"Unsupported down_revision value: {value!r}")


def main() -> int:
    revisions: dict[str, tuple[Path, set[str]]] = {}
    failures: list[str] = []
    for path in sorted(VERSIONS.glob("*.py")):
        if path.name.startswith("__"):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            revision = str(literal_assignment(tree, "revision"))
            parents = normalize_parents(literal_assignment(tree, "down_revision"))
        except Exception as exc:
            failures.append(f"{path.name}: {type(exc).__name__}: {exc}")
            continue
        if revision in revisions:
            failures.append(f"Duplicate revision {revision}: {revisions[revision][0].name}, {path.name}")
            continue
        revisions[revision] = (path, parents)

    for revision, (path, parents) in revisions.items():
        for parent in parents:
            if parent not in revisions:
                failures.append(f"{path.name}: missing parent revision {parent}")

    referenced = {parent for _, parents in revisions.values() for parent in parents}
    heads = sorted(set(revisions) - referenced)
    if len(heads) != 1:
        failures.append(f"Expected one Alembic head; found {heads}")
    elif heads[0] != EXPECTED_HEAD:
        failures.append(f"Expected head {EXPECTED_HEAD}; found {heads[0]}")

    if failures:
        print("MIGRATION_GRAPH_FAIL", file=sys.stderr)
        for failure in failures:
            print(f"- {failure}", file=sys.stderr)
        return 1
    print(f"MIGRATION_GRAPH_PASS revisions={len(revisions)} head={heads[0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
