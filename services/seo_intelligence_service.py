"""Deterministic SEO portfolio intelligence built from first-party performance data.

The engine deliberately avoids LLM calls and external network requests. It turns
Search Console/manual snapshots and existing improvement opportunities into a
bounded, explainable prioritization model suitable for production dashboards.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import TYPE_CHECKING, Any, Iterable
from uuid import UUID

from fastapi import HTTPException, status

if TYPE_CHECKING:
    from knowledge.performance_repository import PerformanceRepository
    from knowledge.search_console_repository import SearchConsoleRepository

ENGINE_VERSION = "seo-intelligence-v2.0"
MAX_SNAPSHOTS = 2_000
MAX_OPPORTUNITIES = 250
MAX_RANKED_OPPORTUNITIES = 100
FRESH_DAYS = 21
STALE_DAYS = 45
VERY_STALE_DAYS = 90


@dataclass(frozen=True)
class ScoreBreakdown:
    priority_score: int
    confidence: float
    priority: str
    estimated_impact: str
    estimated_effort: str
    freshness_factor: float
    factors: dict[str, float]


class SeoIntelligenceService:
    """Create low-overhead, explainable SEO portfolio recommendations."""

    def __init__(
        self,
        *,
        performance_repository: "PerformanceRepository",
        search_console_repository: "SearchConsoleRepository",
    ) -> None:
        self.performance = performance_repository
        self.search_console = search_console_repository

    async def get_project_intelligence(self, project_id: UUID) -> dict[str, Any]:
        if not await self.performance.project_exists(project_id):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND, detail="Project not found"
            )

        articles, snapshots, opportunities, connection, sync_runs = await asyncio.gather(
            self.performance.list_project_articles(project_id),
            self.performance.list_snapshots(project_id, limit=MAX_SNAPSHOTS),
            self.performance.list_opportunities(
                project_id,
                status="open",
                limit=MAX_OPPORTUNITIES,
            ),
            self.search_console.get_connection(project_id),
            self.search_console.list_sync_runs(project_id, limit=10),
        )

        snapshots_by_url = self._group_snapshots(snapshots)
        latest_snapshots = [items[0] for items in snapshots_by_url.values() if items]
        latest_by_url = {str(item.get("url")): item for item in latest_snapshots}

        ranked = [
            self._rank_opportunity(
                opportunity,
                latest=latest_by_url.get(str(opportunity.get("url") or "")),
                history=snapshots_by_url.get(str(opportunity.get("url") or ""), []),
            )
            for opportunity in opportunities
        ]
        ranked.sort(
            key=lambda item: (
                int(item["priority_score"]),
                float(item["confidence"]),
                self._date_sort_value(item.get("updated_at")),
            ),
            reverse=True,
        )

        portfolio = self._portfolio_summary(
            articles=articles,
            latest_snapshots=latest_snapshots,
            snapshots_by_url=snapshots_by_url,
            ranked_opportunities=ranked,
        )
        data_quality = self._data_quality(
            articles=articles,
            latest_snapshots=latest_snapshots,
            connection=connection,
            sync_runs=sync_runs,
        )
        recommended_queue = self._recommended_queue(ranked)

        return {
            "project_id": str(project_id),
            "engine_version": ENGINE_VERSION,
            "generated_at": datetime.now(timezone.utc),
            "method": "deterministic_first_party_data",
            "portfolio": portfolio,
            "data_quality": data_quality,
            "recommended_queue": recommended_queue,
            "opportunities": ranked[:MAX_RANKED_OPPORTUNITIES],
            "guardrails": {
                "uses_llm": False,
                "performs_network_requests": False,
                "rewrites_content": False,
                "publishes_content": False,
                "explanation_available": True,
            },
        }

    @staticmethod
    def _group_snapshots(snapshots: Iterable[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for snapshot in snapshots:
            url = str(snapshot.get("url") or "")
            if not url:
                continue
            grouped[url].append(snapshot)
        for items in grouped.values():
            items.sort(
                key=lambda item: (
                    SeoIntelligenceService._date_sort_value(item.get("date_to")),
                    SeoIntelligenceService._date_sort_value(item.get("imported_at")),
                ),
                reverse=True,
            )
        return dict(grouped)

    def _rank_opportunity(
        self,
        opportunity: dict[str, Any],
        *,
        latest: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> dict[str, Any]:
        breakdown = self._score_opportunity(opportunity, latest=latest, history=history)
        action_plan = self._action_plan(opportunity, latest=latest, history=history)
        supporting_metrics = dict(opportunity.get("supporting_metrics") or {})
        if latest:
            supporting_metrics.update(
                {
                    "latest_clicks": int(latest.get("clicks") or 0),
                    "latest_impressions": int(latest.get("impressions") or 0),
                    "latest_ctr": float(latest.get("ctr") or 0),
                    "latest_average_position": float(latest.get("average_position") or 0),
                    "latest_period_end": self._serialize_date(latest.get("date_to")),
                    "source": str(latest.get("source") or "unknown"),
                }
            )
        return {
            "id": str(opportunity.get("id")),
            "project_id": str(opportunity.get("project_id")),
            "article_id": str(opportunity.get("article_id")) if opportunity.get("article_id") else None,
            "article_title": opportunity.get("article_title"),
            "url": str(opportunity.get("url") or ""),
            "type": str(opportunity.get("type") or "unknown"),
            "severity": str(opportunity.get("severity") or "low"),
            "status": str(opportunity.get("status") or "open"),
            "reason": str(opportunity.get("reason") or ""),
            "suggested_action": str(opportunity.get("suggested_action") or ""),
            "priority_score": breakdown.priority_score,
            "confidence": breakdown.confidence,
            "priority": breakdown.priority,
            "estimated_impact": breakdown.estimated_impact,
            "estimated_effort": breakdown.estimated_effort,
            "freshness_factor": breakdown.freshness_factor,
            "score_factors": breakdown.factors,
            "supporting_metrics": supporting_metrics,
            "action_plan": action_plan,
            "created_at": opportunity.get("created_at"),
            "updated_at": opportunity.get("updated_at"),
        }

    def _score_opportunity(
        self,
        opportunity: dict[str, Any],
        *,
        latest: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> ScoreBreakdown:
        severity = str(opportunity.get("severity") or "low")
        opportunity_type = str(opportunity.get("type") or "unknown")
        metrics = dict(opportunity.get("supporting_metrics") or {})

        severity_score = {"high": 78.0, "medium": 56.0, "low": 30.0}.get(severity, 35.0)
        type_score = {
            "declining_clicks": 18.0,
            "low_ctr_high_impressions": 17.0,
            "striking_distance_position": 15.0,
            "unmapped_url": 6.0,
            "missing_performance_data": 3.0,
        }.get(opportunity_type, 8.0)

        impressions = float(
            (latest or {}).get("impressions")
            or metrics.get("impressions")
            or 0
        )
        clicks = float((latest or {}).get("clicks") or metrics.get("current_clicks") or 0)
        position = float((latest or {}).get("average_position") or metrics.get("average_position") or 0)
        ctr = float((latest or {}).get("ctr") or metrics.get("ctr") or 0)

        traffic_score = min(18.0, math.log10(max(1.0, impressions) + 1.0) * 4.0)
        conversion_score = 0.0
        if opportunity_type == "low_ctr_high_impressions":
            conversion_score = min(14.0, max(0.0, (0.03 - ctr) / 0.03 * 14.0))
        elif opportunity_type == "striking_distance_position" and position > 0:
            conversion_score = min(14.0, max(0.0, (21.0 - position) / 13.0 * 14.0))
        elif opportunity_type == "declining_clicks":
            previous_clicks = float(metrics.get("previous_clicks") or 0)
            if previous_clicks > 0:
                decline = max(0.0, 1.0 - clicks / previous_clicks)
                conversion_score = min(14.0, decline * 18.0)

        mapped_bonus = 5.0 if opportunity.get("article_id") else 0.0
        freshness_factor = self._freshness_factor(latest)
        raw_score = severity_score * 0.64 + type_score + traffic_score + conversion_score + mapped_bonus
        priority_score = int(round(max(0.0, min(100.0, raw_score * freshness_factor))))

        source = str((latest or {}).get("source") or "")
        source_confidence = 0.62 if source == "search_console_api" else 0.50
        volume_confidence = min(0.22, math.log10(max(1.0, impressions) + 1.0) / 20.0)
        history_confidence = 0.12 if len(history) >= 2 else 0.0
        mapping_confidence = 0.08 if opportunity.get("article_id") else -0.08
        if opportunity_type in {"missing_performance_data", "unmapped_url"}:
            source_confidence = 0.38
            volume_confidence = 0.0
            history_confidence = 0.0
        confidence = round(
            max(0.2, min(0.98, (source_confidence + volume_confidence + history_confidence + mapping_confidence) * freshness_factor)),
            2,
        )

        if priority_score >= 82:
            priority = "critical"
        elif priority_score >= 68:
            priority = "high"
        elif priority_score >= 45:
            priority = "medium"
        else:
            priority = "low"

        estimated_impact = "high" if priority_score >= 68 else "medium" if priority_score >= 45 else "low"
        estimated_effort = {
            "low_ctr_high_impressions": "low",
            "striking_distance_position": "medium",
            "declining_clicks": "medium",
            "unmapped_url": "low",
            "missing_performance_data": "low",
        }.get(opportunity_type, "medium")

        return ScoreBreakdown(
            priority_score=priority_score,
            confidence=confidence,
            priority=priority,
            estimated_impact=estimated_impact,
            estimated_effort=estimated_effort,
            freshness_factor=round(freshness_factor, 2),
            factors={
                "severity": round(severity_score * 0.64, 2),
                "opportunity_type": round(type_score, 2),
                "traffic": round(traffic_score, 2),
                "conversion_headroom": round(conversion_score, 2),
                "article_mapping": round(mapped_bonus, 2),
            },
        )

    @staticmethod
    def _action_plan(
        opportunity: dict[str, Any],
        *,
        latest: dict[str, Any] | None,
        history: list[dict[str, Any]],
    ) -> list[dict[str, str]]:
        opportunity_type = str(opportunity.get("type") or "unknown")
        plans: dict[str, list[tuple[str, str, str]]] = {
            "low_ctr_high_impressions": [
                ("Inspect the Search Console page/query context", "Confirm intent before changing copy.", "analysis"),
                ("Rewrite title and meta description", "Increase relevance and differentiation without clickbait.", "editorial"),
                ("Measure the next completed window", "Keep the change only if CTR improves without ranking loss.", "measurement"),
            ],
            "striking_distance_position": [
                ("Map the dominant search intent", "Identify the missing subtopic or answer depth.", "analysis"),
                ("Refresh the highest-value section", "Improve completeness, internal links, and evidence.", "editorial"),
                ("Recheck position and clicks", "Evaluate against the next comparable period.", "measurement"),
            ],
            "declining_clicks": [
                ("Compare current and previous periods", "Separate demand decline from page-specific deterioration.", "analysis"),
                ("Audit freshness and SERP drift", "Update obsolete claims, examples, and intent coverage.", "editorial"),
                ("Verify recovery before further edits", "Avoid stacking changes without attribution.", "measurement"),
            ],
            "unmapped_url": [
                ("Map the URL to the correct article", "Enable article-level attribution and safe prioritization.", "data_quality"),
                ("Confirm canonical URL consistency", "Prevent duplicate or fragmented page metrics.", "technical"),
            ],
            "missing_performance_data": [
                ("Connect or sync Search Console", "Collect first-party evidence before editing.", "data_quality"),
                ("Wait for a complete comparison window", "Avoid decisions from incomplete or partial data.", "measurement"),
            ],
        }
        selected = plans.get(
            opportunity_type,
            [
                ("Review the supporting evidence", "Validate the opportunity before making a change.", "analysis"),
                ("Apply one bounded change", "Keep the outcome measurable and reversible.", "editorial"),
            ],
        )
        result = [
            {"order": str(index), "title": title, "rationale": rationale, "kind": kind}
            for index, (title, rationale, kind) in enumerate(selected, start=1)
        ]
        if latest and len(history) < 2 and opportunity_type not in {"unmapped_url", "missing_performance_data"}:
            result.append(
                {
                    "order": str(len(result) + 1),
                    "title": "Treat the recommendation as provisional",
                    "rationale": "Only one comparable performance period is available.",
                    "kind": "guardrail",
                }
            )
        return result

    def _portfolio_summary(
        self,
        *,
        articles: list[dict[str, Any]],
        latest_snapshots: list[dict[str, Any]],
        snapshots_by_url: dict[str, list[dict[str, Any]]],
        ranked_opportunities: list[dict[str, Any]],
    ) -> dict[str, Any]:
        latest_mapped_article_ids = {
            str(item.get("article_id")) for item in latest_snapshots if item.get("article_id")
        }
        article_count = len(articles)
        coverage = len(latest_mapped_article_ids) / article_count if article_count else 0.0
        totals = self._aggregate_metrics(latest_snapshots)

        comparable_current: list[dict[str, Any]] = []
        comparable_previous: list[dict[str, Any]] = []
        for history in snapshots_by_url.values():
            if len(history) < 2:
                continue
            comparable_current.append(history[0])
            comparable_previous.append(history[1])
        current_totals = self._aggregate_metrics(comparable_current)
        previous_totals = self._aggregate_metrics(comparable_previous)

        trend = {
            "comparable_url_count": len(comparable_current),
            "clicks_change_percent": self._percent_change(current_totals["clicks"], previous_totals["clicks"]),
            "impressions_change_percent": self._percent_change(current_totals["impressions"], previous_totals["impressions"]),
            "ctr_change_points": round(current_totals["ctr"] - previous_totals["ctr"], 4),
            "average_position_change": round(
                current_totals["average_position"] - previous_totals["average_position"],
                2,
            ),
        }

        high_priority = sum(1 for item in ranked_opportunities if item["priority"] in {"critical", "high"})
        health_score = self._portfolio_health_score(
            coverage=coverage,
            high_priority=high_priority,
            article_count=article_count,
            trend=trend,
        )
        return {
            "health_score": health_score,
            "health_status": "strong" if health_score >= 80 else "watch" if health_score >= 55 else "at_risk",
            "article_count": article_count,
            "measured_article_count": len(latest_mapped_article_ids),
            "coverage_ratio": round(coverage, 3),
            "latest_url_count": len(latest_snapshots),
            "clicks": totals["clicks"],
            "impressions": totals["impressions"],
            "ctr": totals["ctr"],
            "average_position": totals["average_position"],
            "open_opportunity_count": len(ranked_opportunities),
            "high_priority_count": high_priority,
            "trend": trend,
        }

    @staticmethod
    def _aggregate_metrics(snapshots: list[dict[str, Any]]) -> dict[str, Any]:
        clicks = sum(max(0, int(item.get("clicks") or 0)) for item in snapshots)
        impressions = sum(max(0, int(item.get("impressions") or 0)) for item in snapshots)
        weighted_position_numerator = sum(
            max(0.0, float(item.get("average_position") or 0))
            * max(0, int(item.get("impressions") or 0))
            for item in snapshots
        )
        ctr = clicks / impressions if impressions else 0.0
        average_position = weighted_position_numerator / impressions if impressions else 0.0
        return {
            "clicks": clicks,
            "impressions": impressions,
            "ctr": round(ctr, 4),
            "average_position": round(average_position, 2),
        }

    def _data_quality(
        self,
        *,
        articles: list[dict[str, Any]],
        latest_snapshots: list[dict[str, Any]],
        connection: dict[str, Any] | None,
        sync_runs: list[dict[str, Any]],
    ) -> dict[str, Any]:
        warnings: list[dict[str, str]] = []
        latest_date = max(
            (self._coerce_date(item.get("date_to")) for item in latest_snapshots),
            default=None,
        )
        age_days = (date.today() - latest_date).days if latest_date else None
        mapped = sum(1 for item in latest_snapshots if item.get("article_id"))
        unmapped = max(0, len(latest_snapshots) - mapped)
        truncated_runs = sum(1 for run in sync_runs if bool(run.get("truncated")))
        failed_runs = sum(1 for run in sync_runs if str(run.get("status")) == "failed")

        if not connection or str(connection.get("status")) != "connected":
            warnings.append(
                {
                    "code": "search_console_disconnected",
                    "severity": "warning",
                    "message": "Search Console is not connected; intelligence may rely on manual or stale data.",
                }
            )
        if age_days is None:
            warnings.append(
                {
                    "code": "no_performance_data",
                    "severity": "blocking",
                    "message": "No performance snapshot is available.",
                }
            )
        elif age_days > VERY_STALE_DAYS:
            warnings.append(
                {
                    "code": "performance_data_very_stale",
                    "severity": "blocking",
                    "message": f"Latest performance data is {age_days} days old.",
                }
            )
        elif age_days > STALE_DAYS:
            warnings.append(
                {
                    "code": "performance_data_stale",
                    "severity": "warning",
                    "message": f"Latest performance data is {age_days} days old.",
                }
            )
        if unmapped:
            warnings.append(
                {
                    "code": "unmapped_urls",
                    "severity": "warning",
                    "message": f"{unmapped} measured URL(s) are not mapped to generated articles.",
                }
            )
        if truncated_runs:
            warnings.append(
                {
                    "code": "truncated_sync_runs",
                    "severity": "warning",
                    "message": f"{truncated_runs} recent Search Console sync run(s) were truncated.",
                }
            )
        if failed_runs:
            warnings.append(
                {
                    "code": "failed_sync_runs",
                    "severity": "warning",
                    "message": f"{failed_runs} recent Search Console sync run(s) failed.",
                }
            )

        status_value = "good"
        if any(item["severity"] == "blocking" for item in warnings):
            status_value = "insufficient"
        elif warnings:
            status_value = "limited"
        return {
            "status": status_value,
            "latest_period_end": latest_date,
            "age_days": age_days,
            "source_count": len({str(item.get("source") or "unknown") for item in latest_snapshots}),
            "measured_url_count": len(latest_snapshots),
            "mapped_url_count": mapped,
            "unmapped_url_count": unmapped,
            "article_count": len(articles),
            "recent_sync_run_count": len(sync_runs),
            "truncated_run_count": truncated_runs,
            "failed_run_count": failed_runs,
            "warnings": warnings,
        }

    @staticmethod
    def _recommended_queue(ranked: list[dict[str, Any]]) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        seen_articles: set[str] = set()
        for item in ranked:
            article_key = str(item.get("article_id") or item.get("url") or item.get("id"))
            if article_key in seen_articles:
                continue
            seen_articles.add(article_key)
            selected.append(
                {
                    "rank": len(selected) + 1,
                    "opportunity_id": item["id"],
                    "article_id": item.get("article_id"),
                    "article_title": item.get("article_title"),
                    "url": item["url"],
                    "type": item["type"],
                    "priority": item["priority"],
                    "priority_score": item["priority_score"],
                    "confidence": item["confidence"],
                    "estimated_impact": item["estimated_impact"],
                    "estimated_effort": item["estimated_effort"],
                    "next_action": item["action_plan"][0] if item.get("action_plan") else None,
                }
            )
            if len(selected) >= 10:
                break
        return selected

    @staticmethod
    def _portfolio_health_score(
        *,
        coverage: float,
        high_priority: int,
        article_count: int,
        trend: dict[str, Any],
    ) -> int:
        score = 50.0 + min(35.0, coverage * 35.0)
        denominator = max(1, article_count)
        score -= min(30.0, high_priority / denominator * 100.0)
        clicks_change = trend.get("clicks_change_percent")
        if isinstance(clicks_change, (int, float)):
            score += max(-10.0, min(10.0, float(clicks_change) / 5.0))
        return int(round(max(0.0, min(100.0, score))))

    @staticmethod
    def _freshness_factor(snapshot: dict[str, Any] | None) -> float:
        if not snapshot:
            return 0.65
        snapshot_date = SeoIntelligenceService._coerce_date(snapshot.get("date_to"))
        if snapshot_date is None:
            return 0.65
        age = max(0, (date.today() - snapshot_date).days)
        if age <= FRESH_DAYS:
            return 1.0
        if age <= STALE_DAYS:
            return 0.88
        if age <= VERY_STALE_DAYS:
            return 0.68
        return 0.45

    @staticmethod
    def _percent_change(current: int | float, previous: int | float) -> float | None:
        previous_value = float(previous)
        if previous_value == 0:
            return None
        return round((float(current) - previous_value) / previous_value * 100.0, 2)

    @staticmethod
    def _coerce_date(value: Any) -> date | None:
        if isinstance(value, datetime):
            return value.date()
        if isinstance(value, date):
            return value
        if isinstance(value, str):
            try:
                return date.fromisoformat(value[:10])
            except ValueError:
                return None
        return None

    @staticmethod
    def _serialize_date(value: Any) -> str | None:
        parsed = SeoIntelligenceService._coerce_date(value)
        return parsed.isoformat() if parsed else None

    @staticmethod
    def _date_sort_value(value: Any) -> float:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                value = value.replace(tzinfo=timezone.utc)
            return value.timestamp()
        parsed = SeoIntelligenceService._coerce_date(value)
        if parsed:
            return datetime(parsed.year, parsed.month, parsed.day, tzinfo=timezone.utc).timestamp()
        return 0.0
