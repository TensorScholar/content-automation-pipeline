#!/usr/bin/env python3
"""Evaluate one persisted article without modifying application data."""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from uuid import UUID

# Allow direct execution from the repository root without package installation.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from execution.article_quality_gate import evaluate_article_quality
from infrastructure.database import get_db_manager
from knowledge.article_repository import ArticleRepository

BENCHMARK_VERSION = "phase2-v1"
RUBRIC_VERSION = "article-quality-rubric-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--article-id", required=True, type=UUID)
    parser.add_argument("--language", default="fa")
    parser.add_argument("--target-words", required=True, type=int)
    return parser.parse_args()


async def run(args: argparse.Namespace) -> dict:
    database = get_db_manager()
    await database.initialize()
    try:
        article = await ArticleRepository(database).get_by_id(args.article_id)
        if article is None:
            raise SystemExit(f"Article {args.article_id} was not found.")

        result = evaluate_article_quality(
            article.get("content") or "",
            language=args.language,
            target_word_count=args.target_words,
        )
        return {
            "benchmark_version": BENCHMARK_VERSION,
            "rubric_version": RUBRIC_VERSION,
            "evaluation_scope": "deterministic_release_gate_only",
            "article_id": str(args.article_id),
            "stored_word_count": article.get("word_count"),
            "quality_gate": result.to_dict(),
            "limitations": [
                "This report does not provide human-review dimension scores.",
                "Language and target word count are explicit CLI inputs because they are not safely inferable from every persisted article.",
                "A passing deterministic gate is not production release approval.",
            ],
        }
    finally:
        await database.close()


def main() -> None:
    args = parse_args()
    print(json.dumps(asyncio.run(run(args)), ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
