"""
Deterministic draft risk assessment.

The v1 risk score avoids another LLM call. It catches practical publishing
risks with transparent rules so managers can trust why a draft is blocked or
needs review.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RiskIssue:
    id: str
    severity: str
    category: str
    message: str
    suggested_fix: str

    def as_dict(self) -> dict[str, str]:
        return {
            "id": self.id,
            "severity": self.severity,
            "category": self.category,
            "message": self.message,
            "suggested_fix": self.suggested_fix,
        }


class DraftRiskService:
    """Score article publish risk using deterministic, explainable checks."""

    CLAIM_PATTERN = re.compile(
        r"\b(guaranteed|guarantee|always|never|100%|proven|risk-free|instant|بدون\s+ریسک|تضمینی|دائماً|أبداً|مضمون)\b",
        re.IGNORECASE,
    )
    HEADING_PATTERN = re.compile(r"(<h[2-4]\b|^\s{0,3}#{2,4}\s+)", re.IGNORECASE | re.MULTILINE)
    FAQ_PATTERN = re.compile(r"\b(faq|frequently asked|سوالات متداول|پرسش‌های متداول|الأسئلة الشائعة)\b", re.IGNORECASE)

    def assess(self, article: dict[str, Any]) -> dict[str, Any]:
        title = str(article.get("title") or "").strip()
        content = str(article.get("content") or article.get("html_content") or "").strip()
        meta_description = str(article.get("meta_description") or "").strip()
        keywords = article.get("keywords") or []

        issues: list[RiskIssue] = []
        score = 100

        if not title:
            issues.append(RiskIssue(
                "missing_title",
                "blocking",
                "content",
                "Article title is missing.",
                "Add a clear publish-ready title.",
            ))
            score -= 30
        elif len(title) < 20:
            issues.append(RiskIssue(
                "short_title",
                "warning",
                "seo",
                "Article title is very short.",
                "Use a descriptive title with the primary topic or keyword.",
            ))
            score -= 10

        if not content:
            issues.append(RiskIssue(
                "missing_content",
                "blocking",
                "content",
                "Article content is missing.",
                "Regenerate or restore the article content before publishing.",
            ))
            score -= 50

        word_count = self._word_count(content)
        if content and word_count < 450:
            issues.append(RiskIssue(
                "low_word_count",
                "warning",
                "content",
                f"Article is short at approximately {word_count} words.",
                "Expand the article or intentionally publish it as a short-form post.",
            ))
            score -= 14

        heading_count = len(self.HEADING_PATTERN.findall(content))
        if content and heading_count < 2:
            issues.append(RiskIssue(
                "weak_structure",
                "warning",
                "structure",
                "Article has fewer than two section headings.",
                "Add scannable H2/H3 sections before publishing.",
            ))
            score -= 12

        if not meta_description:
            issues.append(RiskIssue(
                "missing_meta_description",
                "warning",
                "seo",
                "Meta description is missing.",
                "Add a concise meta description for search previews.",
            ))
            score -= 8
        elif len(meta_description) > 170:
            issues.append(RiskIssue(
                "long_meta_description",
                "warning",
                "seo",
                "Meta description is likely too long for search results.",
                "Keep the meta description near 140-160 characters.",
            ))
            score -= 5

        if not keywords:
            issues.append(RiskIssue(
                "missing_keywords",
                "warning",
                "seo",
                "No keywords are attached to this article.",
                "Attach primary and secondary keywords for tracking.",
            ))
            score -= 8

        if content and self.CLAIM_PATTERN.search(content):
            issues.append(RiskIssue(
                "absolute_claims",
                "warning",
                "trust",
                "Draft contains absolute or guarantee-style claims.",
                "Review claims and add evidence, caveats, or source attribution.",
            ))
            score -= 10

        if content and not self.FAQ_PATTERN.search(content):
            issues.append(RiskIssue(
                "missing_faq",
                "info",
                "seo",
                "No FAQ section was detected.",
                "Consider adding FAQ content when the article targets search traffic.",
            ))
            score -= 3

        score = max(0, min(100, score))
        blocking = [issue for issue in issues if issue.severity == "blocking"]
        warnings = [issue for issue in issues if issue.severity == "warning"]

        if blocking:
            risk_level = "blocked"
        elif score < 60:
            risk_level = "high"
        elif score < 80 or warnings:
            risk_level = "medium"
        else:
            risk_level = "low"

        return {
            "overall_score": score,
            "risk_level": risk_level,
            "blocking_issues": [issue.as_dict() for issue in blocking],
            "warnings": [issue.as_dict() for issue in warnings],
            "issues": [issue.as_dict() for issue in issues],
            "suggested_fixes": [issue.suggested_fix for issue in issues],
        }

    @staticmethod
    def _word_count(content: str) -> int:
        normalized = re.sub(r"<[^>]+>", " ", content)
        return len(re.findall(r"[\w\u0600-\u06FF]+", normalized))
