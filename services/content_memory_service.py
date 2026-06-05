"""Project content memory without new infrastructure.

The first useful version intentionally reuses generated article metadata. It
gives planning a concise memory of recent titles and keywords so new articles
can avoid repetition without adding a vector store, migrations, or background
indexing.
"""

from __future__ import annotations

import re
from collections import Counter
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import UUID

from loguru import logger

if TYPE_CHECKING:
    from knowledge.article_repository import ArticleRepository


class ContentMemoryService:
    """Summarize project-level content history for planning and manager review."""

    def __init__(self, article_repository: "ArticleRepository"):
        self.articles = article_repository

    async def get_project_memory(self, project_id: UUID, limit: int = 12) -> dict[str, Any]:
        recent = await self.articles.get_recent_project_articles(project_id, limit=limit)
        titles = [str(row.get("title") or "").strip() for row in recent if row.get("title")]
        word_counts = [
            int(row["word_count"])
            for row in recent
            if isinstance(row.get("word_count"), int) and row["word_count"] > 0
        ]
        keywords = self._collect_keywords(recent)
        repeated_keywords = [
            {"keyword": keyword, "count": count}
            for keyword, count in Counter(keywords).most_common(12)
            if count > 1
        ]
        last_article_at = self._serialize_datetime(recent[0].get("created_at")) if recent else None

        return {
            "project_id": str(project_id),
            "article_count": len(recent),
            "recent_titles": titles,
            "repeated_keywords": repeated_keywords,
            "average_word_count": round(sum(word_counts) / len(word_counts)) if word_counts else None,
            "last_article_at": last_article_at,
            "planning_guidance": self._build_guidance(titles, repeated_keywords),
        }

    async def build_planning_guidance(
        self, project_id: UUID, topic: str, limit: int = 8
    ) -> str | None:
        """Return concise prompt guidance for avoiding duplicate content angles."""
        try:
            memory = await self.get_project_memory(project_id, limit=limit)
        except Exception as exc:
            logger.warning(f"Content memory lookup failed for project {project_id}: {exc}")
            return None

        titles = memory["recent_titles"][:limit]
        if not titles:
            return None

        similar_titles = self._similar_titles(topic, titles)
        lines = [
            "Avoid duplicating recently generated project content.",
            "Recent titles:",
            *[f"- {title}" for title in titles[:6]],
        ]
        if similar_titles:
            lines.extend([
                "The requested topic overlaps with these recent titles; choose a distinct angle:",
                *[f"- {title}" for title in similar_titles[:3]],
            ])
        if memory["repeated_keywords"]:
            keywords = ", ".join(item["keyword"] for item in memory["repeated_keywords"][:6])
            lines.append(f"Repeated keywords to use carefully, not mechanically: {keywords}.")
        return "\n".join(lines)

    @staticmethod
    def _collect_keywords(rows: list[dict[str, Any]]) -> list[str]:
        keywords: list[str] = []
        for row in rows:
            raw = row.get("keywords")
            if isinstance(raw, list):
                keywords.extend(str(item).strip().lower() for item in raw if str(item).strip())
            elif isinstance(raw, dict):
                keywords.extend(str(key).strip().lower() for key in raw.keys() if str(key).strip())
            elif isinstance(raw, str) and raw.strip():
                keywords.append(raw.strip().lower())
        return keywords

    @staticmethod
    def _similar_titles(topic: str, titles: list[str]) -> list[str]:
        stopwords = {
            "the", "and", "for", "with", "from", "your", "this", "that", "how",
            "را", "به", "از", "در", "با", "برای", "این", "آن",
            "في", "من", "على", "إلى", "هذا", "هذه", "عن",
        }
        topic_terms = {
            term.lower()
            for term in re.findall(r"\w+", topic)
            if len(term) > 2 and term.lower() not in stopwords
        }
        if not topic_terms:
            return []
        similar = []
        for title in titles:
            title_terms = {term.lower() for term in re.findall(r"\w+", title)}
            if topic_terms & title_terms:
                similar.append(title)
        return similar

    @staticmethod
    def _build_guidance(
        titles: list[str], repeated_keywords: list[dict[str, Any]]
    ) -> list[str]:
        guidance: list[str] = []
        if titles:
            guidance.append("Review recent titles before approving another article on a similar angle.")
        if repeated_keywords:
            guidance.append("Repeated keywords are available; use them deliberately instead of stuffing.")
        if not guidance:
            guidance.append("No historical content patterns yet. Generate the first article normally.")
        return guidance

    @staticmethod
    def _serialize_datetime(value: Any) -> str | None:
        if isinstance(value, datetime):
            return value.isoformat()
        if value is None:
            return None
        return str(value)
