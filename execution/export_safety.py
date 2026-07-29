"""Shared safety helpers for article HTML and JSON-LD export surfaces."""

import html
import json
import re
from typing import Any

import bleach  # type: ignore[import-untyped]

_ALLOWED_TAGS = {
    "a",
    "blockquote",
    "br",
    "code",
    "em",
    "h1",
    "h2",
    "h3",
    "h4",
    "h5",
    "h6",
    "hr",
    "li",
    "ol",
    "p",
    "pre",
    "span",
    "strong",
    "table",
    "tbody",
    "td",
    "th",
    "thead",
    "tr",
    "ul",
}
_ALLOWED_ATTRIBUTES = {
    "a": ["href", "title"],
    "blockquote": ["class"],
    "code": ["class"],
    "span": ["class"],
}


def sanitize_html_fragment(content: str) -> str:
    """Return a publishable HTML fragment without executable markup."""
    return str(
        bleach.clean(
            content or "",
            tags=_ALLOWED_TAGS,
            attributes=_ALLOWED_ATTRIBUTES,
            protocols={"http", "https", "mailto"},
            strip=True,
            strip_comments=True,
        )
    ).strip()


def _render_inline_markdown(value: str) -> str:
    """Render the small inline subset used by the dependency-free fallback."""
    rendered = html.escape(value, quote=False)
    rendered = re.sub(r"`([^`]+)`", r"<code>\1</code>", rendered)
    rendered = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", rendered)
    rendered = re.sub(r"(?<!\*)\*([^*]+)\*(?!\*)", r"<em>\1</em>", rendered)
    return rendered


def _fallback_markdown_to_html(content: str) -> str:
    """Convert common article Markdown constructs without an optional package."""
    output: list[str] = []
    paragraph: list[str] = []
    list_tag: str | None = None

    def flush_paragraph() -> None:
        if paragraph:
            output.append(f"<p>{' '.join(paragraph)}</p>")
            paragraph.clear()

    def close_list() -> None:
        nonlocal list_tag
        if list_tag:
            output.append(f"</{list_tag}>")
            list_tag = None

    for raw_line in content.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = raw_line.strip()
        if not line:
            flush_paragraph()
            close_list()
            continue

        heading = re.match(r"^(#{1,6})\s+(.+)$", line)
        unordered_item = re.match(r"^[-*+]\s+(.+)$", line)
        ordered_item = re.match(r"^\d+[.)]\s+(.+)$", line)
        html_block = re.fullmatch(
            r"<([a-zA-Z][\w:-]*)\b[^>]*>.*</\1>",
            line,
        )

        if html_block:
            flush_paragraph()
            close_list()
            output.append(line)
        elif heading:
            flush_paragraph()
            close_list()
            level = len(heading.group(1))
            output.append(
                f"<h{level}>{_render_inline_markdown(heading.group(2))}</h{level}>"
            )
        elif unordered_item or ordered_item:
            flush_paragraph()
            required_tag = "ul" if unordered_item else "ol"
            if list_tag != required_tag:
                close_list()
                list_tag = required_tag
                output.append(f"<{list_tag}>")
            item_match = unordered_item if unordered_item is not None else ordered_item
            assert item_match is not None
            item = item_match.group(1)
            output.append(f"<li>{_render_inline_markdown(item)}</li>")
        elif line.startswith("> "):
            flush_paragraph()
            close_list()
            output.append(
                f"<blockquote>{_render_inline_markdown(line[2:])}</blockquote>"
            )
        elif re.fullmatch(r"[-*_]{3,}", line):
            flush_paragraph()
            close_list()
            output.append("<hr>")
        else:
            close_list()
            paragraph.append(_render_inline_markdown(line))

    flush_paragraph()
    close_list()
    return "\n".join(output)


def render_safe_article_html(content: str) -> str:
    """Convert article Markdown or HTML to one sanitized semantic fragment."""
    normalized = (content or "").strip()
    if not normalized:
        return ""

    nonempty_lines = [line.strip() for line in normalized.splitlines() if line.strip()]
    is_html_fragment = bool(nonempty_lines) and all(
        re.match(r"^</?[a-zA-Z][^>]*>", line) for line in nonempty_lines
    )
    if is_html_fragment:
        return sanitize_html_fragment(normalized)

    try:
        import markdown  # type: ignore[import-untyped]

        rendered = markdown.markdown(
            normalized,
            extensions=["extra", "nl2br", "sane_lists"],
        )
    except (ImportError, RuntimeError, ValueError):
        rendered = _fallback_markdown_to_html(normalized)

    return sanitize_html_fragment(rendered)


def json_for_html_script(payload: dict[str, Any], *, indent: int | None = None) -> str:
    """Serialize JSON safely for an inline application/ld+json script."""
    return json.dumps(payload, ensure_ascii=False, indent=indent).replace("</", "<\\/")


def infer_article_language(content: str, explicit_language: object = None) -> str:
    """Resolve the supported export locale without requiring a schema change."""
    if isinstance(explicit_language, str):
        normalized = explicit_language.strip().lower().replace("_", "-")
        if normalized.startswith("fa"):
            return "fa"
        if normalized.startswith("en"):
            return "en"

    sample = re.sub(r"<[^>]+>", " ", content or "")[:4000]
    rtl_characters = len(re.findall(r"[\u0600-\u06ff]", sample))
    latin_characters = len(re.findall(r"[A-Za-z]", sample))
    return "fa" if rtl_characters > latin_characters else "en"
