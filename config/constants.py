"""
System Constants & Invariants
==============================
Immutable domain constants defining system behavior boundaries,
cost models, and semantic thresholds.

Architecture: Value Objects + Namespace Organization
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

# =============================================================================
# SEMANTIC THRESHOLDS
# =============================================================================


@dataclass(frozen=True)
class SemanticThresholds:
    """
    Cosine similarity thresholds for NLP-based decision making.

    Derived from empirical analysis of sentence-transformers performance
    on semantic textual similarity benchmarks (STS-B dataset).
    """

    # Decision hierarchy layer confidence thresholds
    EXPLICIT_RULE_MATCH: float = 0.85  # High confidence: direct match
    INFERRED_PATTERN_MATCH: float = 0.70  # Medium confidence: statistical pattern
    BEST_PRACTICE_MATCH: float = 0.50  # Low confidence: general guidance

    # Cache similarity threshold (fuzzy matching)
    CACHE_HIT_THRESHOLD: float = 0.92  # Very high similarity for cache reuse

    # Content similarity detection (plagiarism/uniqueness)
    DUPLICATE_CONTENT_THRESHOLD: float = 0.95  # Near-identical content
    SIMILAR_CONTENT_THRESHOLD: float = 0.80  # Significantly similar

    # Semantic clustering thresholds
    TIGHT_CLUSTER_THRESHOLD: float = 0.75  # Keywords in same semantic space
    LOOSE_CLUSTER_THRESHOLD: float = 0.60  # Related but distinct concepts


SIMILARITY_THRESHOLDS: Final = SemanticThresholds()


# =============================================================================
# LLM COST MODELS (USD per 1K tokens)
# =============================================================================


@dataclass(frozen=True)
class ModelCosts:
    """OpenAI API pricing as of January 2025."""

    input_cost: float
    output_cost: float
    context_window: int

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate total cost for a completion."""
        return (input_tokens / 1000) * self.input_cost + (output_tokens / 1000) * self.output_cost


@dataclass(frozen=True)
class LLMCostModel:
    """Cost structure for all supported models."""

    GPT_4_TURBO: ModelCosts = ModelCosts(input_cost=0.01, output_cost=0.03, context_window=128_000)

    GPT_4: ModelCosts = ModelCosts(input_cost=0.03, output_cost=0.06, context_window=8_192)

    GPT_3_5_TURBO: ModelCosts = ModelCosts(
        input_cost=0.0005, output_cost=0.0015, context_window=16_385
    )

    # Anthropic fallback
    CLAUDE_3_SONNET: ModelCosts = ModelCosts(
        input_cost=0.003, output_cost=0.015, context_window=200_000
    )


COST_MODEL: Final = LLMCostModel()


# =============================================================================
# TOKEN BUDGETS & OPTIMIZATION
# =============================================================================


@dataclass(frozen=True)
class TokenBudgets:
    """Per-task token allocation guidelines."""

    # Keyword research
    KEYWORD_RESEARCH_INPUT: int = 500
    KEYWORD_RESEARCH_OUTPUT: int = 300

    # Content planning
    OUTLINE_GENERATION_INPUT: int = 1_000
    OUTLINE_GENERATION_OUTPUT: int = 500

    # Content generation (per section)
    SECTION_GENERATION_INPUT: int = 1_500
    SECTION_GENERATION_OUTPUT: int = 400

    # Full article generation
    FULL_ARTICLE_INPUT: int = 2_000
    FULL_ARTICLE_OUTPUT: int = 2_500

    # Maximum context size (safety margin)
    MAX_CONTEXT_TOKENS: int = 120_000  # Leave headroom for 128K models


TOKEN_BUDGETS: Final = TokenBudgets()


@dataclass(frozen=True)
class CompressionTargets:
    """Context compression ratio targets."""

    AGGRESSIVE: float = 0.9  # 90% reduction
    STANDARD: float = 0.7  # 70% reduction
    LIGHT: float = 0.5  # 50% reduction
    NONE: float = 0.0  # No compression


COMPRESSION_TARGETS: Final = CompressionTargets()


# =============================================================================
# CONTENT QUALITY METRICS
# =============================================================================


@dataclass(frozen=True)
class QualityThresholds:
    """Content quality acceptance criteria."""

    # Readability (Flesch-Kincaid grade level)
    MIN_READABILITY: float = 8.0  # 8th grade
    MAX_READABILITY: float = 12.0  # 12th grade (high school)
    TARGET_READABILITY: float = 10.0  # Optimal

    # Keyword density
    MIN_KEYWORD_DENSITY: float = 0.005  # 0.5%
    MAX_KEYWORD_DENSITY: float = 0.02  # 2.0%
    TARGET_KEYWORD_DENSITY: float = 0.01  # 1.0%

    # Content length
    MIN_ARTICLE_WORDS: int = 800
    TARGET_ARTICLE_WORDS: int = 1_500
    MAX_ARTICLE_WORDS: int = 3_000

    # Lexical diversity (type-token ratio)
    MIN_LEXICAL_DIVERSITY: float = 0.40  # 40% unique words
    TARGET_LEXICAL_DIVERSITY: float = 0.60  # 60% unique words


QUALITY_THRESHOLDS: Final = QualityThresholds()


# =============================================================================
# DECISION HIERARCHY
# =============================================================================


class DecisionLayer(Enum):
    """Three-layer adaptive intelligence hierarchy."""

    EXPLICIT_RULE = auto()  # Layer 1: Rulebook directives
    INFERRED_PATTERN = auto()  # Layer 2: Learned from website
    BEST_PRACTICE = auto()  # Layer 3: Universal defaults


@dataclass(frozen=True)
class DecisionLayerWeights:
    """Authority weights for decision resolution."""

    EXPLICIT_RULE: float = 1.0  # Highest priority
    INFERRED_PATTERN: float = 0.7  # Medium priority
    BEST_PRACTICE: float = 0.5  # Lowest priority


DECISION_WEIGHTS: Final = DecisionLayerWeights()


# =============================================================================
# SCRAPING & PATTERN INFERENCE
# =============================================================================


@dataclass(frozen=True)
class ScrapingLimits:
    """Website analysis constraints."""

    MIN_ARTICLES_FOR_INFERENCE: int = 5  # Minimum sample size
    TARGET_ARTICLES_FOR_INFERENCE: int = 15  # Target sample
    MAX_ARTICLES_FOR_INFERENCE: int = 30  # Maximum to prevent over-fitting

    # Statistical confidence
    MIN_PATTERN_CONFIDENCE: float = 0.70  # 70% confidence threshold
    MIN_SAMPLE_SIZE_FOR_CONFIDENCE: int = 10


SCRAPING_LIMITS: Final = ScrapingLimits()


# =============================================================================
# CACHE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class CacheTTL:
    """Time-to-live for cached data (seconds)."""

    EMBEDDING: int = 30 * 24 * 3600  # 30 days (embeddings are stable)
    LLM_RESPONSE: int = 30 * 24 * 3600  # 30 days (content is evergreen)
    KEYWORD_RESEARCH: int = 7 * 24 * 3600  # 7 days (search volumes change)
    INFERRED_PATTERNS: int = 14 * 24 * 3600  # 14 days (re-analyze periodically)
    WEBSITE_CONTENT: int = 24 * 3600  # 1 day (websites update frequently)


CACHE_TTL: Final = CacheTTL()


# =============================================================================
# RATE LIMITING
# =============================================================================


@dataclass(frozen=True)
class RateLimits:
    """API rate limiting parameters."""

    # OpenAI tier limits (adjust based on your tier)
    OPENAI_RPM: int = 500  # Requests per minute
    OPENAI_TPM: int = 90_000  # Tokens per minute

    # Web scraping (respectful limits)
    SCRAPING_REQUESTS_PER_SECOND: float = 0.5  # 1 request per 2 seconds
    SCRAPING_MAX_CONCURRENT: int = 5

    # Internal API
    API_REQUESTS_PER_MINUTE: int = 60


RATE_LIMITS: Final = RateLimits()


# =============================================================================
# VECTOR DATABASE CONFIGURATION
# =============================================================================


@dataclass(frozen=True)
class VectorConfig:
    """pgvector configuration parameters."""

    EMBEDDING_DIMENSION: int = 384  # sentence-transformers dimension

    # IVFFlat index parameters
    IVFFLAT_LISTS: int = 100  # Number of clusters (sqrt of expected rows)

    # HNSW index parameters (alternative, higher performance)
    HNSW_M: int = 16  # Max connections per layer
    HNSW_EF_CONSTRUCTION: int = 64  # Size of dynamic candidate list
    HNSW_EF_SEARCH: int = 40  # Size of dynamic candidate list for search


VECTOR_CONFIG: Final = VectorConfig()


# =============================================================================
# TASK PRIORITIES
# =============================================================================


class TaskPriority(Enum):
    """Celery task priority levels."""

    CRITICAL = 10  # Immediate processing
    HIGH = 7  # Expedited processing
    MEDIUM = 5  # Normal processing
    LOW = 3  # Background processing


# =============================================================================
# CONTENT STRUCTURE PATTERNS
# =============================================================================


class ContentStructureType(Enum):
    """Common content structure archetypes."""

    LISTICLE = "listicle"
    HOW_TO = "how_to"
    PROBLEM_SOLUTION = "problem_solution"
    COMPARISON = "comparison"
    GUIDE = "guide"
    NEWS = "news"
    OPINION = "opinion"
    CASE_STUDY = "case_study"


# =============================================================================
# KEYWORD INTENT CLASSIFICATION
# =============================================================================


class KeywordIntent(Enum):
    """Search intent taxonomy (based on Google's categorization)."""

    INFORMATIONAL = "informational"  # Seeking knowledge
    NAVIGATIONAL = "navigational"  # Finding specific page
    COMMERCIAL = "commercial"  # Research before purchase
    TRANSACTIONAL = "transactional"  # Ready to buy/convert


# =============================================================================
# ERROR CODES
# =============================================================================


class ErrorCode(Enum):
    """System error taxonomy."""

    # LLM errors
    LLM_RATE_LIMIT = "LLM_RATE_LIMIT"
    LLM_TIMEOUT = "LLM_TIMEOUT"
    LLM_INVALID_RESPONSE = "LLM_INVALID_RESPONSE"
    LLM_COST_EXCEEDED = "LLM_COST_EXCEEDED"

    # Database errors
    DB_CONNECTION_FAILED = "DB_CONNECTION_FAILED"
    DB_QUERY_TIMEOUT = "DB_QUERY_TIMEOUT"

    # Cache errors
    CACHE_MISS = "CACHE_MISS"
    CACHE_WRITE_FAILED = "CACHE_WRITE_FAILED"

    # Content errors
    CONTENT_QUALITY_FAILED = "CONTENT_QUALITY_FAILED"
    CONTENT_TOO_SHORT = "CONTENT_TOO_SHORT"
    CONTENT_TOO_LONG = "CONTENT_TOO_LONG"

    # Scraping errors
    SCRAPING_TIMEOUT = "SCRAPING_TIMEOUT"
    SCRAPING_BLOCKED = "SCRAPING_BLOCKED"

    # Project errors
    PROJECT_NOT_FOUND = "PROJECT_NOT_FOUND"
    RULEBOOK_PARSE_FAILED = "RULEBOOK_PARSE_FAILED"


# =============================================================================
# RETRY STRATEGIES
# =============================================================================


@dataclass(frozen=True)
class RetryStrategy:
    """Exponential backoff configuration."""

    max_attempts: int
    initial_delay: float  # seconds
    max_delay: float  # seconds
    exponential_base: float


@dataclass(frozen=True)
class RetryStrategies:
    """Domain-specific retry configurations."""

    LLM_API: RetryStrategy = RetryStrategy(
        max_attempts=3, initial_delay=1.0, max_delay=10.0, exponential_base=2.0
    )

    DATABASE: RetryStrategy = RetryStrategy(
        max_attempts=3, initial_delay=0.5, max_delay=5.0, exponential_base=2.0
    )

    SCRAPING: RetryStrategy = RetryStrategy(
        max_attempts=5, initial_delay=2.0, max_delay=30.0, exponential_base=2.0
    )


RETRY_STRATEGIES: Final = RetryStrategies()


# =============================================================================
# REGEX PATTERNS
# =============================================================================


@dataclass(frozen=True)
class RegexPatterns:
    """Compiled regex patterns for text processing."""

    URL: str = r"https?://(?:[-\w.]|(?:%[\da-fA-F]{2}))+"
    EMAIL: str = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"
    PHONE: str = r"\+?1?\d{9,15}"

    # HTML cleaning
    HTML_TAG: str = r"<[^>]+>"
    MULTIPLE_SPACES: str = r"\s+"
    MULTIPLE_NEWLINES: str = r"\n{3,}"


REGEX_PATTERNS: Final = RegexPatterns()


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Thresholds
    "SIMILARITY_THRESHOLDS",
    "QUALITY_THRESHOLDS",
    "DECISION_WEIGHTS",
    "SCRAPING_LIMITS",
    # Cost & Budgets
    "COST_MODEL",
    "TOKEN_BUDGETS",
    "COMPRESSION_TARGETS",
    # Configuration
    "CACHE_TTL",
    "RATE_LIMITS",
    "VECTOR_CONFIG",
    "RETRY_STRATEGIES",
    "REGEX_PATTERNS",
    # Enums
    "DecisionLayer",
    "TaskPriority",
    "ContentStructureType",
    "KeywordIntent",
    "ErrorCode",
]
