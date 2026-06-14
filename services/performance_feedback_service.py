"""Read-only content performance feedback service.

This phase intentionally uses deterministic rules and manual snapshots only.
It does not call LLM providers, rewrite content, or publish anything.
"""

import csv
import io
from datetime import date, datetime
from typing import Any, Optional
from urllib.parse import urlparse, urlunparse
from uuid import UUID

from fastapi import HTTPException, status

from knowledge.performance_repository import PerformanceRepository

LOW_CTR_HIGH_IMPRESSIONS_MIN = 1000
LOW_CTR_THRESHOLD = 0.01
STRIKING_DISTANCE_MIN = 8.0
STRIKING_DISTANCE_MAX = 20.0
DECLINING_CLICKS_RATIO = 0.70
REQUIRED_COLUMNS = {
    "url",
    "clicks",
    "impressions",
    "ctr",
    "average_position",
    "date_from",
    "date_to",
}


class PerformanceFeedbackService:
    """Manual performance snapshot import and opportunity detection."""

    def __init__(self, repository: PerformanceRepository):
        self.repository = repository

    async def import_csv(
        self,
        *,
        project_id: UUID,
        csv_text: str,
        source: str = "manual_csv",
    ) -> dict[str, Any]:
        if source != "manual_csv":
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Only manual_csv performance imports are supported in this phase",
            )
        if not await self.repository.project_exists(project_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        records = await self._parse_csv(project_id=project_id, csv_text=csv_text, source=source)
        articles = await self.repository.list_project_articles(project_id)
        article_by_id = {article["id"]: article for article in articles}
        article_by_url = {}
        for article in articles:
            normalized = self._safe_normalize_url(article.get("wordpress_post_url"))
            if normalized:
                article_by_url[normalized] = article

        imported_snapshots: list[dict[str, Any]] = []
        opportunities: list[dict[str, Any]] = []
        for record in records:
            if record.get("article_id"):
                if record["article_id"] not in article_by_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="CSV article_id does not belong to this project",
                    )
            else:
                article = article_by_url.get(record["url"])
                if article:
                    record["article_id"] = article["id"]

            snapshot = await self.repository.upsert_snapshot(record)
            imported_snapshots.append(snapshot)
            if snapshot.get("article_id"):
                await self.repository.resolve_opportunity(
                    project_id=project_id,
                    article_id=snapshot["article_id"],
                    opportunity_type="missing_performance_data",
                )
            opportunities.extend(await self._detect_snapshot_opportunities(snapshot))

        opportunities.extend(await self._detect_missing_performance_data(project_id, articles))
        current = await self.get_project_performance(project_id)

        return {
            "project_id": str(project_id),
            "imported_count": len(imported_snapshots),
            "snapshot_count": len(current["snapshots"]),
            "opportunity_count": len(current["opportunities"]),
            "opportunities_created_or_updated": len(opportunities),
        }

    async def get_project_performance(self, project_id: UUID) -> dict[str, Any]:
        if not await self.repository.project_exists(project_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        snapshots = await self.repository.list_snapshots(project_id)
        opportunities = await self.repository.list_opportunities(project_id)
        latest_imported_at = max(
            (snapshot.get("imported_at") for snapshot in snapshots if snapshot.get("imported_at")),
            default=None,
        )
        return {
            "project_id": str(project_id),
            "summary": {
                "snapshot_count": len(snapshots),
                "opportunity_count": len(opportunities),
                "high_priority_count": sum(1 for item in opportunities if item.get("severity") == "high"),
                "latest_imported_at": latest_imported_at,
            },
            "snapshots": [self._public_snapshot(snapshot) for snapshot in snapshots],
            "opportunities": [self._public_opportunity(item) for item in opportunities],
        }

    async def dismiss_opportunity(self, *, project_id: UUID, opportunity_id: UUID) -> dict[str, Any]:
        if not await self.repository.project_exists(project_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")
        dismissed = await self.repository.dismiss_opportunity(
            project_id=project_id,
            opportunity_id=opportunity_id,
        )
        if not dismissed:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Opportunity not found")
        return {"id": str(opportunity_id), "status": "dismissed"}

    async def _parse_csv(
        self,
        *,
        project_id: UUID,
        csv_text: str,
        source: str,
    ) -> list[dict[str, Any]]:
        if not csv_text.strip():
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV payload is empty")

        reader = csv.DictReader(io.StringIO(csv_text.strip()))
        headers = {header.strip() for header in (reader.fieldnames or [])}
        missing = sorted(REQUIRED_COLUMNS - headers)
        if missing:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"CSV is missing required columns: {', '.join(missing)}",
            )

        records = []
        for line_number, row in enumerate(reader, start=2):
            try:
                date_from = self._parse_date(row["date_from"], "date_from")
                date_to = self._parse_date(row["date_to"], "date_to")
                if date_to < date_from:
                    raise ValueError("date_to must be on or after date_from")

                records.append({
                    "project_id": project_id,
                    "article_id": self._optional_uuid(row.get("article_id")),
                    "url": self._normalize_url(row["url"]),
                    "date_from": date_from,
                    "date_to": date_to,
                    "clicks": self._parse_int(row["clicks"], "clicks"),
                    "impressions": self._parse_int(row["impressions"], "impressions"),
                    "ctr": self._parse_ctr(row["ctr"]),
                    "average_position": self._parse_float(
                        row["average_position"],
                        "average_position",
                    ),
                    "source": source,
                })
            except ValueError as exc:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid CSV row {line_number}: {exc}",
                ) from exc

        if not records:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="CSV has no data rows")
        return records

    async def _detect_snapshot_opportunities(self, snapshot: dict[str, Any]) -> list[dict[str, Any]]:
        opportunities: list[dict[str, Any]] = []
        if snapshot["impressions"] >= LOW_CTR_HIGH_IMPRESSIONS_MIN and snapshot["ctr"] < LOW_CTR_THRESHOLD:
            opportunities.append(await self._store_opportunity(
                snapshot,
                opportunity_type="low_ctr_high_impressions",
                severity="high" if snapshot["impressions"] >= 5000 else "medium",
                reason=(
                    f"{snapshot['impressions']} impressions with "
                    f"{snapshot['ctr'] * 100:.2f}% CTR."
                ),
                suggested_action="Review the title and meta description to improve click-through rate.",
                supporting_metrics={
                    "impressions": snapshot["impressions"],
                    "ctr": snapshot["ctr"],
                },
            ))

        position = float(snapshot["average_position"])
        if STRIKING_DISTANCE_MIN <= position <= STRIKING_DISTANCE_MAX:
            opportunities.append(await self._store_opportunity(
                snapshot,
                opportunity_type="striking_distance_position",
                severity="high" if position <= 12 else "medium",
                reason=f"Average position is {position:.1f}, close to first-page visibility.",
                suggested_action="Refresh and expand the article section that targets this query/page.",
                supporting_metrics={"average_position": position},
            ))

        previous = await self.repository.latest_previous_snapshot(
            project_id=snapshot["project_id"],
            url=snapshot["url"],
            date_from=snapshot["date_from"],
        )
        if previous and previous.get("clicks", 0) > 0:
            current_clicks = int(snapshot["clicks"])
            previous_clicks = int(previous["clicks"])
            if current_clicks <= previous_clicks * DECLINING_CLICKS_RATIO:
                opportunities.append(await self._store_opportunity(
                    snapshot,
                    opportunity_type="declining_clicks",
                    severity="high" if current_clicks <= previous_clicks * 0.5 else "medium",
                    reason=f"Clicks declined from {previous_clicks} to {current_clicks}.",
                    suggested_action="Inspect the article for freshness, SERP intent drift, or outdated sections.",
                    supporting_metrics={
                        "previous_clicks": previous_clicks,
                        "current_clicks": current_clicks,
                    },
                ))

        if not snapshot.get("article_id"):
            opportunities.append(await self._store_opportunity(
                snapshot,
                opportunity_type="unmapped_url",
                severity="low",
                reason="Performance data exists, but the URL is not mapped to a generated article.",
                suggested_action="Map this URL to an article before using it for editorial prioritization.",
                supporting_metrics={"clicks": snapshot["clicks"], "impressions": snapshot["impressions"]},
            ))

        return opportunities

    async def _detect_missing_performance_data(
        self,
        project_id: UUID,
        articles: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        article_ids_with_data = await self.repository.list_article_ids_with_snapshots(project_id)
        opportunities = []
        for article in articles:
            article_id = article["id"]
            if article_id in article_ids_with_data:
                continue
            url = article.get("wordpress_post_url") or f"article:{article_id}"
            opportunities.append(await self.repository.upsert_opportunity({
                "project_id": project_id,
                "article_id": article_id,
                "snapshot_id": None,
                "url": url,
                "type": "missing_performance_data",
                "severity": "low",
                "reason": "This article has no imported performance snapshot yet.",
                "suggested_action": "Import a Search Console CSV snapshot before making performance decisions.",
                "supporting_metrics": {},
            }))
        return opportunities

    async def _store_opportunity(
        self,
        snapshot: dict[str, Any],
        *,
        opportunity_type: str,
        severity: str,
        reason: str,
        suggested_action: str,
        supporting_metrics: dict[str, Any],
    ) -> dict[str, Any]:
        return await self.repository.upsert_opportunity({
            "project_id": snapshot["project_id"],
            "article_id": snapshot.get("article_id"),
            "snapshot_id": snapshot["id"],
            "url": snapshot["url"],
            "type": opportunity_type,
            "severity": severity,
            "reason": reason,
            "suggested_action": suggested_action,
            "supporting_metrics": supporting_metrics,
        })

    @staticmethod
    def _normalize_url(value: str | None) -> str:
        raw = (value or "").strip()
        parsed = urlparse(raw)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("url must be an http(s) URL")
        normalized = parsed._replace(
            scheme=parsed.scheme.lower(),
            netloc=parsed.netloc.lower(),
            fragment="",
        )
        return urlunparse(normalized).rstrip("/")

    @classmethod
    def _safe_normalize_url(cls, value: str | None) -> Optional[str]:
        try:
            return cls._normalize_url(value)
        except ValueError:
            return None

    @staticmethod
    def _parse_date(value: str | None, field_name: str) -> date:
        try:
            return datetime.strptime((value or "").strip(), "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(f"{field_name} must use YYYY-MM-DD") from exc

    @staticmethod
    def _parse_int(value: str | None, field_name: str) -> int:
        try:
            parsed = int(str(value or "").strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be an integer") from exc
        if parsed < 0:
            raise ValueError(f"{field_name} must be non-negative")
        return parsed

    @staticmethod
    def _parse_float(value: str | None, field_name: str) -> float:
        try:
            parsed = float(str(value or "").strip())
        except ValueError as exc:
            raise ValueError(f"{field_name} must be numeric") from exc
        if parsed < 0:
            raise ValueError(f"{field_name} must be non-negative")
        return parsed

    @classmethod
    def _parse_ctr(cls, value: str | None) -> float:
        raw = str(value or "").strip()
        is_percent = raw.endswith("%")
        parsed = cls._parse_float(raw.rstrip("%"), "ctr")
        if is_percent or parsed > 1:
            parsed = parsed / 100
        if parsed > 1:
            raise ValueError("ctr must be a decimal ratio or percentage")
        return parsed

    @staticmethod
    def _optional_uuid(value: str | None) -> Optional[UUID]:
        normalized = (value or "").strip()
        return UUID(normalized) if normalized else None

    @staticmethod
    def _public_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(snapshot["id"]),
            "project_id": str(snapshot["project_id"]),
            "article_id": str(snapshot["article_id"]) if snapshot.get("article_id") else None,
            "url": snapshot["url"],
            "date_from": snapshot["date_from"],
            "date_to": snapshot["date_to"],
            "clicks": snapshot["clicks"],
            "impressions": snapshot["impressions"],
            "ctr": snapshot["ctr"],
            "average_position": snapshot["average_position"],
            "source": snapshot["source"],
            "imported_at": snapshot.get("imported_at"),
        }

    @staticmethod
    def _public_opportunity(opportunity: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(opportunity["id"]),
            "project_id": str(opportunity["project_id"]),
            "article_id": str(opportunity["article_id"]) if opportunity.get("article_id") else None,
            "snapshot_id": str(opportunity["snapshot_id"]) if opportunity.get("snapshot_id") else None,
            "url": opportunity["url"],
            "type": opportunity["type"],
            "severity": opportunity["severity"],
            "reason": opportunity["reason"],
            "suggested_action": opportunity["suggested_action"],
            "supporting_metrics": opportunity.get("supporting_metrics") or {},
            "status": opportunity["status"],
            "article_title": opportunity.get("article_title"),
            "created_at": opportunity.get("created_at"),
            "updated_at": opportunity.get("updated_at"),
        }
