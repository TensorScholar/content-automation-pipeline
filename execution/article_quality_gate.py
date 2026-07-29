"""Deterministic, locale-aware article quality checks.

This module intentionally evaluates observable output only. It does not use an
LLM, external service, or project configuration, so its results are suitable
for repeatable benchmark and regression checks.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from html import unescape
from typing import Literal

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WORD_RE = re.compile(r"[^\W_]+(?:[\u200c\u200d][^\W_]+)*", re.UNICODE)
_HTML_HEADING_RE = re.compile(
    r"<h([1-6])\b[^>]*>(.*?)</h\1\s*>", re.IGNORECASE | re.DOTALL
)
_MARKDOWN_HEADING_RE = re.compile(
    r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$", re.MULTILINE
)
_HTML_PARAGRAPH_RE = re.compile(r"<p\b[^>]*>(.*?)</p\s*>", re.IGNORECASE | re.DOTALL)
_ARABIC_DIACRITICS_RE = re.compile(r"[\u064B-\u065F\u0670\u06D6-\u06ED]")
_FAQ_HEADINGS = {
    "faq",
    "frequently asked questions",
    "common questions",
    "پرسش های متداول",
    "سوالات متداول",
    "سؤالات متداول",
    "الاسئله الشائعه",
    "الأسئلة الشائعة",
    "اسئله شائعه",
}


@dataclass(frozen=True)
class _Heading:
    level: int
    text: str
    start: int
    end: int


@dataclass(frozen=True)
class QualityFinding:
    """A deterministic quality failure or warning."""

    code: str
    severity: Literal["error", "warning"]
    message: str
    expected: str
    actual: str


@dataclass(frozen=True)
class ArticleQualityResult:
    """Structured outcome for a single article-quality evaluation."""

    language: str
    word_count: int
    target_word_count: int | None
    minimum_word_count: int
    maximum_word_count: int
    heading_count: int
    paragraph_count: int
    findings: tuple[QualityFinding, ...]

    @property
    def passed(self) -> bool:
        return not any(finding.severity == "error" for finding in self.findings)

    def to_dict(self) -> dict:
        return {
            "passed": self.passed,
            "language": self.language,
            "word_count": self.word_count,
            "target_word_count": self.target_word_count,
            "minimum_word_count": self.minimum_word_count,
            "maximum_word_count": self.maximum_word_count,
            "heading_count": self.heading_count,
            "paragraph_count": self.paragraph_count,
            "findings": [asdict(finding) for finding in self.findings],
        }


def _plain_text(content: str) -> str:
    return unescape(_HTML_TAG_RE.sub(" ", content or ""))


def _normalize_heading(value: str) -> str:
    text = _ARABIC_DIACRITICS_RE.sub("", _plain_text(value)).casefold()
    text = text.translate(str.maketrans({"ي": "ی", "ك": "ک", "ة": "ه", "ۀ": "ه"}))
    text = text.replace("\u200c", " ").replace("\u200d", " ")
    return " ".join(re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE).split())


def _extract_headings(content: str) -> list[_Heading]:
    headings = [
        _Heading(int(match.group(1)), _plain_text(match.group(2)).strip(), *match.span())
        for match in _HTML_HEADING_RE.finditer(content)
        if _plain_text(match.group(2)).strip()
    ]
    headings.extend(
        _Heading(
            len(match.group(1)),
            _plain_text(match.group(2)).strip(),
            *match.span(),
        )
        for match in _MARKDOWN_HEADING_RE.finditer(content)
        if _plain_text(match.group(2)).strip()
    )
    return sorted(headings, key=lambda heading: heading.start)


def _duplicate_adjacent_heading(headings: list[_Heading]) -> str | None:
    for previous, current in zip(headings, headings[1:]):
        normalized = _normalize_heading(current.text)
        if normalized and _normalize_heading(previous.text) == normalized:
            return current.text
    return None


def _answered_faq_items(content: str, headings: list[_Heading], faq_index: int) -> int:
    faq_heading = headings[faq_index]
    section_end = len(content)
    for heading in headings[faq_index + 1 :]:
        if heading.level <= faq_heading.level:
            section_end = heading.start
            break

    nested = [
        heading
        for heading in headings[faq_index + 1 :]
        if heading.start < section_end and heading.level > faq_heading.level
    ]
    answered = 0
    for index, question in enumerate(nested):
        answer_end = nested[index + 1].start if index + 1 < len(nested) else section_end
        if _plain_text(content[question.end:answer_end]).strip():
            answered += 1
    if answered:
        return answered

    section = content[faq_heading.end:section_end]
    html_paragraphs = [
        _plain_text(body).strip()
        for body in _HTML_PARAGRAPH_RE.findall(section)
        if _plain_text(body).strip()
    ]
    if html_paragraphs:
        return sum(
            1
            for index, paragraph in enumerate(html_paragraphs[:-1])
            if paragraph.endswith(("?", "؟")) and html_paragraphs[index + 1]
        )

    blocks = [
        block.strip()
        for block in re.split(r"\n\s*\n", _plain_text(section))
        if block.strip()
    ]
    return sum(
        1
        for index, block in enumerate(blocks[:-1])
        if block.endswith(("?", "؟")) and blocks[index + 1]
    )


def _count_paragraphs(content: str) -> int:
    html_paragraphs = _HTML_PARAGRAPH_RE.findall(content)
    if html_paragraphs:
        return sum(1 for body in html_paragraphs if _plain_text(body).strip())

    blocks = [block for block in re.split(r"\n\s*\n", _plain_text(content)) if block.strip()]
    return len(blocks)


def evaluate_article_quality(
    content: str,
    *,
    language: str,
    target_word_count: int | None,
    hard_minimum_words: int = 800,
    hard_maximum_words: int = 3500,
    minimum_headings: int = 2,
    minimum_paragraphs: int = 3,
    require_faq: bool = False,
    minimum_faq_items: int = 2,
) -> ArticleQualityResult:
    """Evaluate hard output requirements without provider-specific heuristics."""

    if hard_minimum_words <= 0 or hard_maximum_words < hard_minimum_words:
        raise ValueError("Invalid hard word-count bounds.")
    if target_word_count is not None and target_word_count <= 0:
        raise ValueError("target_word_count must be positive when provided.")
    if minimum_faq_items <= 0:
        raise ValueError("minimum_faq_items must be positive.")

    text = _plain_text(content)
    word_count = len(_WORD_RE.findall(text))
    headings = _extract_headings(content)
    heading_count = len(headings)
    paragraph_count = _count_paragraphs(content)

    minimum_word_count = hard_minimum_words
    maximum_word_count = hard_maximum_words
    if target_word_count is not None:
        minimum_word_count = max(hard_minimum_words, math.ceil(target_word_count * 0.7))
        maximum_word_count = max(
            minimum_word_count,
            min(hard_maximum_words, math.floor(target_word_count * 2.0)),
        )

    findings: list[QualityFinding] = []
    if word_count < minimum_word_count:
        findings.append(
            QualityFinding(
                code="word_count_below_minimum",
                severity="error",
                message="Article is shorter than the required minimum.",
                expected=f"at least {minimum_word_count} words",
                actual=f"{word_count} words",
            )
        )
    elif word_count > maximum_word_count:
        findings.append(
            QualityFinding(
                code="word_count_above_maximum",
                severity="error",
                message="Article exceeds the allowed length.",
                expected=f"at most {maximum_word_count} words",
                actual=f"{word_count} words",
            )
        )

    if heading_count < minimum_headings:
        findings.append(
            QualityFinding(
                code="insufficient_headings",
                severity="error",
                message="Article does not have enough structural headings.",
                expected=f"at least {minimum_headings} headings",
                actual=f"{heading_count} headings",
            )
        )

    if paragraph_count < minimum_paragraphs:
        findings.append(
            QualityFinding(
                code="insufficient_paragraphs",
                severity="error",
                message="Article does not have enough readable paragraphs.",
                expected=f"at least {minimum_paragraphs} paragraphs",
                actual=f"{paragraph_count} paragraphs",
            )
        )

    duplicate_heading = _duplicate_adjacent_heading(headings)
    if duplicate_heading:
        findings.append(
            QualityFinding(
                code="duplicate_adjacent_headings",
                severity="error",
                message="Article contains consecutive duplicate headings.",
                expected="each adjacent heading to be distinct",
                actual=duplicate_heading,
            )
        )

    if require_faq:
        faq_index = next(
            (
                index
                for index, heading in enumerate(headings)
                if any(
                    _normalize_heading(heading.text) == _normalize_heading(candidate)
                    for candidate in _FAQ_HEADINGS
                )
            ),
            None,
        )
        if faq_index is None:
            findings.append(
                QualityFinding(
                    code="missing_required_faq",
                    severity="error",
                    message="The requested FAQ section is missing.",
                    expected=f"an FAQ section with at least {minimum_faq_items} answered questions",
                    actual="no FAQ section",
                )
            )
        else:
            answered_items = _answered_faq_items(content, headings, faq_index)
            if answered_items < minimum_faq_items:
                findings.append(
                    QualityFinding(
                        code="incomplete_required_faq",
                        severity="error",
                        message="The requested FAQ section does not contain enough answered questions.",
                        expected=f"at least {minimum_faq_items} answered questions",
                        actual=f"{answered_items} answered questions",
                    )
                )

    return ArticleQualityResult(
        language=language,
        word_count=word_count,
        target_word_count=target_word_count,
        minimum_word_count=minimum_word_count,
        maximum_word_count=maximum_word_count,
        heading_count=heading_count,
        paragraph_count=paragraph_count,
        findings=tuple(findings),
    )
