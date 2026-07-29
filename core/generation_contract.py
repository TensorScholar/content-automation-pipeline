"""Shared product constraints for article generation requests."""

import re
from typing import Optional

ARTICLE_WORD_COUNT_MIN = 800
ARTICLE_WORD_COUNT_MAX = 3500


def normalize_word_count_range(value: Optional[str]) -> Optional[str]:
    """Validate and normalize an optional article word-count range."""
    if value is None:
        return None

    match = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", value)
    if not match:
        raise ValueError("word_count_range must use 'min-max', for example '800-1000'.")

    minimum, maximum = (int(part) for part in match.groups())
    if minimum < ARTICLE_WORD_COUNT_MIN or maximum > ARTICLE_WORD_COUNT_MAX:
        raise ValueError(
            f"Article word counts must be between {ARTICLE_WORD_COUNT_MIN} "
            f"and {ARTICLE_WORD_COUNT_MAX}."
        )
    if maximum < minimum:
        raise ValueError("Maximum word count must be greater than or equal to the minimum word count.")

    return f"{minimum}-{maximum}"


def normalize_target_word_count(value: Optional[int]) -> Optional[int]:
    """Validate the legacy single article word-count target."""
    if value is None:
        return None
    if not ARTICLE_WORD_COUNT_MIN <= value <= ARTICLE_WORD_COUNT_MAX:
        raise ValueError(
            f"Target word count must be between {ARTICLE_WORD_COUNT_MIN} "
            f"and {ARTICLE_WORD_COUNT_MAX}."
        )
    return value
