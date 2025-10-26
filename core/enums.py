"""
Domain Enumerations & Type Taxonomy
====================================
Exhaustive type-safe enumerations for domain modeling with
first-class support for pattern matching and serialization.

Architecture: Type-Driven Design + ADT (Algebraic Data Types)
"""

from enum import Enum, IntEnum, auto
from typing import Optional


class RuleType(str, Enum):
    """
    Rulebook rule categorization taxonomy.

    String enum for JSON serialization compatibility and
    database storage without integer mapping fragility.
    """

    TONE = "tone"
    STRUCTURE = "structure"
    TOPIC = "topic"
    STYLE = "style"
    FORMATTING = "formatting"
    AUDIENCE = "audience"
    SEO = "seo"
    BRAND_VOICE = "brand_voice"
    CONTENT_TYPE = "content_type"
    PROHIBITED = "prohibited"

    def __str__(self) -> str:
        """Human-readable representation."""
        return self.value.replace("_", " ").title()


class SectionIntent(str, Enum):
    """
    Semantic role of content sections in hierarchical structure.

    Maps to narrative arc and rhetorical functions.
    """

    INTRODUCE = "introduce"  # Hook, context setting
    EXPLAIN = "explain"  # Core information delivery
    DEMONSTRATE = "demonstrate"  # Examples, case studies
    COMPARE = "compare"  # Alternatives, contrasts
    ANALYZE = "analyze"  # Deep dive, implications
    SYNTHESIZE = "synthesize"  # Integration of ideas
    CONCLUDE = "conclude"  # Summary, call-to-action
    TRANSITION = "transition"  # Bridging sections

    @property
    def typical_word_count_range(self) -> tuple[int, int]:
        """Expected word count range for this intent."""
        ranges = {
            self.INTRODUCE: (100, 200),
            self.EXPLAIN: (200, 400),
            self.DEMONSTRATE: (150, 300),
            self.COMPARE: (200, 350),
            self.ANALYZE: (250, 450),
            self.SYNTHESIZE: (150, 300),
            self.CONCLUDE: (100, 200),
            self.TRANSITION: (50, 100),
        }
        return ranges[self]


class KeywordIntent(str, Enum):
    """
    Search intent classification (Google's taxonomy).

    Enables intent-aware content optimization and structure planning.
    """

    INFORMATIONAL = "informational"  # "how to", "what is"
    NAVIGATIONAL = "navigational"  # Brand/product searches
    COMMERCIAL = "commercial"  # "best", "review", "compare"
    TRANSACTIONAL = "transactional"  # "buy", "price", "discount"

    @property
    def typical_content_structure(self) -> str:
        """Recommended structure for this intent."""
        structures = {
            self.INFORMATIONAL: "guide",
            self.NAVIGATIONAL: "landing_page",
            self.COMMERCIAL: "comparison",
            self.TRANSACTIONAL: "product_page",
        }
        return structures[self]

    @property
    def conversion_priority(self) -> int:
        """Priority for conversion optimization (1-10)."""
        priorities = {
            self.TRANSACTIONAL: 10,
            self.COMMERCIAL: 7,
            self.NAVIGATIONAL: 5,
            self.INFORMATIONAL: 3,
        }
        return priorities[self]


class ContentStructureType(str, Enum):
    """
    Content archetype taxonomy for structural templates.

    Each type implies specific section ordering and rhetorical patterns.
    """

    LISTICLE = "listicle"  # "10 Ways to..."
    HOW_TO = "how_to"  # Step-by-step guide
    PROBLEM_SOLUTION = "problem_solution"  # Agitate, then resolve
    COMPARISON = "comparison"  # A vs B analysis
    GUIDE = "guide"  # Comprehensive reference
    NEWS = "news"  # Timely announcement
    OPINION = "opinion"  # Editorial, perspective
    CASE_STUDY = "case_study"  # Real-world example
    PILLAR = "pillar"  # Exhaustive topic authority

    @property
    def typical_section_count(self) -> tuple[int, int]:
        """Expected section count range (min, max)."""
        counts = {
            self.LISTICLE: (5, 15),
            self.HOW_TO: (4, 10),
            self.PROBLEM_SOLUTION: (3, 6),
            self.COMPARISON: (3, 7),
            self.GUIDE: (6, 12),
            self.NEWS: (3, 5),
            self.OPINION: (3, 6),
            self.CASE_STUDY: (4, 8),
            self.PILLAR: (10, 20),
        }
        return counts[self]


class DecisionLayer(IntEnum):
    """
    Adaptive intelligence hierarchy with ordinal priority.

    IntEnum for natural ordering comparisons: EXPLICIT > INFERRED > BEST_PRACTICE
    """

    EXPLICIT_RULE = 3  # Highest authority: direct rulebook match
    INFERRED_PATTERN = 2  # Medium authority: learned from analysis
    BEST_PRACTICE = 1  # Lowest authority: universal defaults

    @property
    def confidence_threshold(self) -> float:
        """Minimum similarity threshold for this layer."""
        from config.constants import SIMILARITY_THRESHOLDS

        thresholds = {
            self.EXPLICIT_RULE: SIMILARITY_THRESHOLDS.EXPLICIT_RULE_MATCH,
            self.INFERRED_PATTERN: SIMILARITY_THRESHOLDS.INFERRED_PATTERN_MATCH,
            self.BEST_PRACTICE: SIMILARITY_THRESHOLDS.BEST_PRACTICE_MATCH,
        }
        return thresholds[self]

    def __str__(self) -> str:
        """Human-readable layer description."""
        return self.name.replace("_", " ").title()


class TaskPriority(IntEnum):
    """
    Task execution priority for queue management.

    Higher values = higher priority in Celery routing.
    """

    CRITICAL = 10  # Immediate: user-facing requests
    HIGH = 7  # Expedited: time-sensitive operations
    MEDIUM = 5  # Normal: background content generation
    LOW = 3  # Deferred: analytics, cleanup

    @property
    def timeout_seconds(self) -> int:
        """Task timeout based on priority."""
        timeouts = {
            self.CRITICAL: 120,  # 2 minutes
            self.HIGH: 300,  # 5 minutes
            self.MEDIUM: 600,  # 10 minutes
            self.LOW: 1800,  # 30 minutes
        }
        return timeouts[self]


class ModelTier(str, Enum):
    """
    LLM model capability tiers for hierarchical routing.

    Enables cost-optimal model selection based on task complexity.
    """

    PREMIUM = "premium"  # GPT-4, Claude Opus: Complex reasoning
    STANDARD = "standard"  # GPT-3.5-turbo: Most tasks
    ECONOMIC = "economic"  # Smaller models: Simple classification
    LOCAL = "local"  # On-premise: Embeddings, NER

    @property
    def typical_models(self) -> list[str]:
        """Default models for this tier."""
        models = {
            self.PREMIUM: ["gpt-4-turbo-preview", "claude-3-opus"],
            self.STANDARD: ["gpt-3.5-turbo", "claude-3-sonnet"],
            self.ECONOMIC: ["gpt-3.5-turbo"],
            self.LOCAL: ["sentence-transformers", "spacy"],
        }
        return models[self]

    def estimate_cost_per_1k_tokens(self) -> tuple[float, float]:
        """(input_cost, output_cost) per 1K tokens."""
        from config.constants import COST_MODEL

        costs = {
            self.PREMIUM: (COST_MODEL.GPT_4_TURBO.input_cost, COST_MODEL.GPT_4_TURBO.output_cost),
            self.STANDARD: (
                COST_MODEL.GPT_3_5_TURBO.input_cost,
                COST_MODEL.GPT_3_5_TURBO.output_cost,
            ),
            self.ECONOMIC: (
                COST_MODEL.GPT_3_5_TURBO.input_cost,
                COST_MODEL.GPT_3_5_TURBO.output_cost,
            ),
            self.LOCAL: (0.0, 0.0),
        }
        return costs[self]


class CacheStrategy(str, Enum):
    """
    Cache invalidation and refresh strategies.

    Defines temporal stability characteristics of cached data.
    """

    PERMANENT = "permanent"  # Never invalidate (embeddings)
    LONG_LIVED = "long_lived"  # 30 days (LLM responses)
    MEDIUM_LIVED = "medium_lived"  # 7 days (keyword data)
    SHORT_LIVED = "short_lived"  # 1 day (website content)
    NO_CACHE = "no_cache"  # Always fresh

    @property
    def ttl_seconds(self) -> Optional[int]:
        """Time-to-live in seconds, or None for permanent."""
        from config.constants import CACHE_TTL

        ttls = {
            self.PERMANENT: None,
            self.LONG_LIVED: CACHE_TTL.LLM_RESPONSE,
            self.MEDIUM_LIVED: CACHE_TTL.KEYWORD_RESEARCH,
            self.SHORT_LIVED: CACHE_TTL.WEBSITE_CONTENT,
            self.NO_CACHE: 0,
        }
        return ttls[self]


class GenerationStatus(str, Enum):
    """
    Content generation lifecycle states.

    Finite state machine for tracking article production pipeline.
    """

    PENDING = "pending"  # Queued, not started
    RESEARCHING = "researching"  # Keyword discovery phase
    PLANNING = "planning"  # Outline creation
    GENERATING = "generating"  # LLM content creation
    VALIDATING = "validating"  # Quality checks
    COMPLETED = "completed"  # Ready for distribution
    FAILED = "failed"  # Unrecoverable error
    CANCELLED = "cancelled"  # User-initiated abort

    @property
    def is_terminal(self) -> bool:
        """Check if this is a final state."""
        return self in {self.COMPLETED, self.FAILED, self.CANCELLED}

    @property
    def is_active(self) -> bool:
        """Check if actively processing."""
        return self in {self.RESEARCHING, self.PLANNING, self.GENERATING, self.VALIDATING}

    def can_transition_to(self, target: "GenerationStatus") -> bool:
        """Validate state transition legality."""
        valid_transitions = {
            self.PENDING: {self.RESEARCHING, self.CANCELLED},
            self.RESEARCHING: {self.PLANNING, self.FAILED, self.CANCELLED},
            self.PLANNING: {self.GENERATING, self.FAILED, self.CANCELLED},
            self.GENERATING: {self.VALIDATING, self.FAILED, self.CANCELLED},
            self.VALIDATING: {self.COMPLETED, self.GENERATING, self.FAILED},
            self.COMPLETED: set(),
            self.FAILED: set(),
            self.CANCELLED: set(),
        }
        return target in valid_transitions.get(self, set())


class DistributionChannel(str, Enum):
    """
    Content distribution channel taxonomy.

    Extensible for multi-channel publishing strategies.
    """

    TELEGRAM = "telegram"
    WORDPRESS = "wordpress"
    MEDIUM = "medium"
    EMAIL = "email"
    WEBHOOK = "webhook"
    FILE_SYSTEM = "file_system"

    @property
    def requires_authentication(self) -> bool:
        """Check if channel needs API credentials."""
        return self in {self.TELEGRAM, self.WORDPRESS, self.MEDIUM, self.EMAIL}

    @property
    def supports_html(self) -> bool:
        """Check if channel accepts HTML formatting."""
        return self in {self.TELEGRAM, self.WORDPRESS, self.MEDIUM, self.EMAIL}


class ErrorSeverity(IntEnum):
    """
    Error classification by impact severity.

    Determines alerting, retry, and recovery strategies.
    """

    CRITICAL = 5  # System failure, immediate intervention required
    ERROR = 4  # Operation failed, automatic retry possible
    WARNING = 3  # Degraded performance, monitoring needed
    INFO = 2  # Notable event, no action required
    DEBUG = 1  # Diagnostic information

    @property
    def should_alert(self) -> bool:
        """Determine if severity warrants immediate alert."""
        return self >= self.ERROR

    @property
    def should_retry(self) -> bool:
        """Determine if operation should auto-retry."""
        return self == self.ERROR


class ValidationResult(str, Enum):
    """
    Content quality validation outcomes.

    Tri-state validation: pass, warn (acceptable with caveats), fail.
    """

    PASS = "pass"  # Meets all quality criteria  # nosec B105
    PASS_WITH_WARNINGS = "pass_with_warnings"  # Acceptable but suboptimal  # nosec B105
    FAIL = "fail"  # Below quality threshold

    @property
    def is_acceptable(self) -> bool:
        """Check if content can proceed to distribution."""
        return self in {self.PASS, self.PASS_WITH_WARNINGS}


class CompressionLevel(IntEnum):
    """
    Context compression aggressiveness levels.

    Trade-off between token budget and information preservation.
    """

    NONE = 0  # No compression (full context)
    LIGHT = 1  # 50% reduction (extractive summarization)
    STANDARD = 2  # 70% reduction (semantic distillation)
    AGGRESSIVE = 3  # 90% reduction (essence extraction)

    @property
    def compression_ratio(self) -> float:
        """Target compression ratio (0.0-1.0)."""
        ratios = {
            self.NONE: 0.0,
            self.LIGHT: 0.5,
            self.STANDARD: 0.7,
            self.AGGRESSIVE: 0.9,
        }
        return ratios[self]

    @property
    def quality_preservation_estimate(self) -> float:
        """Estimated information preservation (0.0-1.0)."""
        preservation = {
            self.NONE: 1.0,
            self.LIGHT: 0.95,
            self.STANDARD: 0.85,
            self.AGGRESSIVE: 0.70,
        }
        return preservation[self]


class LanguageCode(str, Enum):
    """
    ISO 639-1 language codes for multi-lingual support.

    Currently supporting English, extensible for internationalization.
    """

    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    PORTUGUESE = "pt"

    @property
    def spacy_model(self) -> str:
        """Corresponding spaCy model identifier."""
        models = {
            self.ENGLISH: "en_core_web_sm",
            self.SPANISH: "es_core_news_sm",
            self.FRENCH: "fr_core_news_sm",
            self.GERMAN: "de_core_news_sm",
            self.PORTUGUESE: "pt_core_news_sm",
        }
        return models[self]


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================


def get_enum_by_value(enum_class: type[Enum], value: str) -> Optional[Enum]:
    """
    Safe enum lookup by value with None fallback.

    Args:
        enum_class: The enum class to search
        value: The string value to find

    Returns:
        Matching enum member or None if not found
    """
    try:
        return enum_class(value)
    except (ValueError, KeyError):
        return None


def enum_to_choices(enum_class: type[Enum]) -> list[tuple[str, str]]:
    """
    Convert enum to Django/form choices format.

    Args:
        enum_class: The enum to convert

    Returns:
        List of (value, label) tuples
    """
    return [(member.value, member.name.replace("_", " ").title()) for member in enum_class]


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Content & Structure
    "RuleType",
    "SectionIntent",
    "ContentStructureType",
    "KeywordIntent",
    # Intelligence & Decision
    "DecisionLayer",
    "ValidationResult",
    # Execution & Operations
    "TaskPriority",
    "GenerationStatus",
    "ModelTier",
    "CompressionLevel",
    # Infrastructure
    "CacheStrategy",
    "DistributionChannel",
    "ErrorSeverity",
    "LanguageCode",
    # Utilities
    "get_enum_by_value",
    "enum_to_choices",
]
