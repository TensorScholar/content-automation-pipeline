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
                detail="CSV imports must use the manual_csv source",
            )
        records = await self._parse_csv(project_id=project_id, csv_text=csv_text, source=source)
        return await self.import_records(project_id=project_id, records=records, source=source)

    async def import_records(
        self,
        *,
        project_id: UUID,
        records: list[dict[str, Any]],
        source: str,
    ) -> dict[str, Any]:
        """Import normalized snapshots using bounded bulk, idempotent database writes."""
        if source not in {"manual_csv", "search_console_api"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Unsupported performance snapshot source",
            )
        if not await self.repository.project_exists(project_id):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Project not found")

        articles = await self.repository.list_project_articles(project_id)
        article_by_id = {article["id"]: article for article in articles}
        article_by_url: dict[str, dict[str, Any]] = {}
        for article in articles:
            normalized = self._safe_normalize_url(article.get("wordpress_post_url"))
            if normalized:
                article_by_url[normalized] = article

        # PostgreSQL cannot update the same conflict key twice in one INSERT.
        # Collapse duplicates deterministically before issuing bounded bulk upserts.
        normalized_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        for raw_record in records:
            record = dict(raw_record)
            record["project_id"] = project_id
            record["source"] = source
            record["url"] = self._normalize_url(str(record.get("url") or ""))
            article_id = record.get("article_id")
            if article_id and not isinstance(article_id, UUID):
                try:
                    article_id = UUID(str(article_id))
                except ValueError as exc:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Performance article_id is invalid",
                    ) from exc
                record["article_id"] = article_id
            if article_id:
                if article_id not in article_by_id:
                    raise HTTPException(
                        status_code=status.HTTP_400_BAD_REQUEST,
                        detail="Performance article_id does not belong to this project",
                    )
            else:
                article = article_by_url.get(record["url"])
                if article:
                    record["article_id"] = article["id"]

            date_from = record.get("date_from")
            date_to = record.get("date_to")
            if not isinstance(date_from, date) or not isinstance(date_to, date):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Performance date_from and date_to must be dates",
                )
            if date_from > date_to:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="Performance date_from must be on or before date_to",
                )
            record["clicks"] = max(0, int(record.get("clicks") or 0))
            record["impressions"] = max(0, int(record.get("impressions") or 0))
            record["ctr"] = max(0.0, min(float(record.get("ctr") or 0), 1.0))
            record["average_position"] = max(0.0, float(record.get("average_position") or 0))
            key = (project_id, record["url"], date_from, date_to, source)
            normalized_by_key[key] = record

        normalized_records = list(normalized_by_key.values())
        imported_snapshots = await self.repository.bulk_upsert_snapshots(normalized_records)

        mapped_article_ids = {
            snapshot["article_id"]
            for snapshot in imported_snapshots
            if snapshot.get("article_id")
        }
        await self.repository.resolve_missing_performance_opportunities(
            project_id=project_id,
            article_ids=mapped_article_ids,
        )

        previous_by_period: dict[date, dict[str, dict[str, Any]]] = {}
        urls_by_period: dict[date, set[str]] = {}
        for snapshot in imported_snapshots:
            urls_by_period.setdefault(snapshot["date_from"], set()).add(snapshot["url"])
        for period_start, urls in urls_by_period.items():
            previous_by_period[period_start] = await self.repository.latest_previous_snapshots(
                project_id=project_id,
                urls=urls,
                before_date=period_start,
            )

        opportunity_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
        opportunity_sort_dates: dict[tuple[Any, ...], date] = {}
        for snapshot in imported_snapshots:
            previous = previous_by_period.get(snapshot["date_from"], {}).get(snapshot["url"])
            for opportunity in self._build_snapshot_opportunities(snapshot, previous=previous):
                key = (opportunity["project_id"], opportunity["url"], opportunity["type"])
                sort_date = snapshot["date_to"]
                if key not in opportunity_by_key or sort_date >= opportunity_sort_dates[key]:
                    opportunity_by_key[key] = opportunity
                    opportunity_sort_dates[key] = sort_date

        article_ids_with_data = await self.repository.list_article_ids_with_snapshots(project_id)
        for article in articles:
            article_id = article["id"]
            if article_id in article_ids_with_data:
                continue
            url = article.get("wordpress_post_url") or f"article:{article_id}"
            opportunity = {
                "project_id": project_id,
                "article_id": article_id,
                "snapshot_id": None,
                "url": url,
                "type": "missing_performance_data",
                "severity": "low",
                "reason": "This article has no imported performance snapshot yet.",
                "suggested_action": "Connect Search Console or import a snapshot before making performance decisions.",
                "supporting_metrics": {},
            }
            opportunity_by_key[(project_id, url, "missing_performance_data")] = opportunity

        persisted_opportunities = await self.repository.bulk_upsert_opportunities(
            list(opportunity_by_key.values())
        )
        current = await self.get_project_performance(project_id)
        return {
            "project_id": str(project_id),
            "imported_count": len(imported_snapshots),
            "deduplicated_input_count": len(records) - len(normalized_records),
            "snapshot_count": len(current["snapshots"]),
            "opportunity_count": len(current["opportunities"]),
            "opportunities_created_or_updated": len(persisted_opportunities),
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

    @staticmethod
    def _build_snapshot_opportunities(
        snapshot: dict[str, Any],
        *,
        previous: dict[str, Any] | None,
    ) -> list[dict[str, Any]]:
        """Build deterministic opportunity rows without database round trips."""
        opportunities: list[dict[str, Any]] = []

        def add(
            opportunity_type: str,
            severity: str,
            reason: str,
            suggested_action: str,
            supporting_metrics: dict[str, Any],
        ) -> None:
            opportunities.append(
                {
                    "project_id": snapshot["project_id"],
                    "article_id": snapshot.get("article_id"),
                    "snapshot_id": snapshot["id"],
                    "url": snapshot["url"],
                    "type": opportunity_type,
                    "severity": severity,
                    "reason": reason,
                    "suggested_action": suggested_action,
                    "supporting_metrics": supporting_metrics,
                }
            )

        if snapshot["impressions"] >= LOW_CTR_HIGH_IMPRESSIONS_MIN and snapshot["ctr"] < LOW_CTR_THRESHOLD:
            add(
                "low_ctr_high_impressions",
                "high" if snapshot["impressions"] >= 5000 else "medium",
                f"{snapshot['impressions']} impressions with {snapshot['ctr'] * 100:.2f}% CTR.",
                "Review the title and meta description to improve click-through rate.",
                {"impressions": snapshot["impressions"], "ctr": snapshot["ctr"]},
            )

        position = float(snapshot["average_position"])
        if STRIKING_DISTANCE_MIN <= position <= STRIKING_DISTANCE_MAX:
            add(
                "striking_distance_position",
                "high" if position <= 12 else "medium",
                f"Average position is {position:.1f}, close to first-page visibility.",
                "Refresh and expand the article section that targets this query/page.",
                {"average_position": position},
            )

        if previous and previous.get("clicks", 0) > 0:
            current_clicks = int(snapshot["clicks"])
            previous_clicks = int(previous["clicks"])
            if current_clicks <= previous_clicks * DECLINING_CLICKS_RATIO:
                add(
                    "declining_clicks",
                    "high" if current_clicks <= previous_clicks * 0.5 else "medium",
                    f"Clicks declined from {previous_clicks} to {current_clicks}.",
                    "Inspect the article for freshness, SERP intent drift, or outdated sections.",
                    {"previous_clicks": previous_clicks, "current_clicks": current_clicks},
                )

        if not snapshot.get("article_id"):
            add(
                "unmapped_url",
                "low",
                "Performance data exists, but the URL is not mapped to a generated article.",
                "Map this URL to an article before using it for editorial prioritization.",
                {"clicks": snapshot["clicks"], "impressions": snapshot["impressions"]},
            )
        return opportunities

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
