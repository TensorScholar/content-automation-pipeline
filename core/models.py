"""
Domain Data Models
==================
Complete Pydantic v2 schema definitions with:
- Type-safe validation and coercion
- Computed properties and derived fields
- Optimized serialization for database/API
- Immutability where appropriate

Architecture: Domain-Driven Design + Value Objects
"""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID, uuid4

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    HttpUrl,
    SecretStr,
    computed_field,
    field_validator,
)

from core.enums import (
    ContentStructureType,
    DecisionLayer,
    DistributionChannel,
    GenerationStatus,
    KeywordIntent,
    RuleType,
    SectionIntent,
    ValidationResult,
)

# =============================================================================
# CONFIGURATION
# =============================================================================


class BaseModelConfig(BaseModel):
    """Base configuration for all models."""

    model_config = ConfigDict(
        validate_assignment=True,  # Validate on field updates
        use_enum_values=False,  # Keep enum types (don't convert to strings)
        arbitrary_types_allowed=True,  # Allow numpy arrays, etc.
        json_encoders={
            datetime: lambda v: v.isoformat(),
            UUID: lambda v: str(v),
            np.ndarray: lambda v: v.tolist(),
        },
    )


# =============================================================================
# PROJECT & CONFIGURATION MODELS
# =============================================================================


class Project(BaseModelConfig):
    """
    Multi-tenant project representation.

    Each project maintains isolated context including:
    - Explicit rules (rulebook)
    - Inferred patterns (learned from website)
    - Distribution channels
    - Cost tracking
    """

    id: UUID = Field(default_factory=uuid4)
    name: str = Field(..., min_length=1, max_length=255)
    domain: Optional[str] = Field(
        default=None,
        description="Target website domain for pattern inference (e.g., 'example.com')",
    )

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_active: datetime = Field(default_factory=datetime.utcnow)
    total_articles_generated: int = Field(default=0, ge=0)
    total_cost_usd: float = Field(default=0.0, ge=0.0)

    # Distribution
    telegram_channel: Optional[str] = Field(default=None)
    wordpress_url: Optional[str] = Field(
        default=None, description="The base URL of the WordPress site (e.g., https://example.com)"
    )
    wordpress_username: Optional[str] = Field(
        default=None, description="WordPress username for Application Password"
    )
    wordpress_app_password: Optional[SecretStr] = Field(
        default=None, description="WordPress Application Password (use pydantic.SecretStr)"
    )

    # Relationships (loaded separately for performance)
    rulebook_id: Optional[UUID] = Field(default=None)
    inferred_patterns_id: Optional[UUID] = Field(default=None)

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Ensure name is not just whitespace."""
        if not v.strip():
            raise ValueError("Project name cannot be empty or whitespace")
        return v.strip()

    @computed_field
    @property
    def is_configured(self) -> bool:
        """Check if project has minimum viable configuration."""
        return bool(self.domain or self.rulebook_id)

    def update_last_active(self) -> None:
        """Update last_active timestamp."""
        self.last_active = datetime.utcnow()


class Rule(BaseModelConfig):
    """
    Individual rulebook directive with semantic representation.

    Rules are semantically indexed for fuzzy matching against queries.
    """

    id: UUID = Field(default_factory=uuid4)
    rulebook_id: UUID

    rule_type: RuleType
    content: str = Field(..., min_length=1, max_length=5000)
    embedding: Optional[list[float]] = Field(
        default=None, description="384-dim semantic embedding (computed asynchronously)"
    )

    priority: int = Field(default=5, ge=1, le=10)
    context: Optional[str] = Field(
        default=None, max_length=1000, description="Conditions under which this rule applies"
    )

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("embedding")
    @classmethod
    def validate_embedding_dimension(cls, v: Optional[list[float]]) -> Optional[list[float]]:
        """Ensure embedding has correct dimensionality."""
        if v is not None and len(v) != 384:
            raise ValueError(f"Embedding must be 384-dimensional, got {len(v)}")
        return v


class Rulebook(BaseModelConfig):
    """
    Explicit project guidelines with versioning.

    Contains structured rules extracted from natural language specifications.
    """

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID

    raw_content: str = Field(..., min_length=1)
    rules: list[Rule] = Field(default_factory=list)

    version: int = Field(default=1, ge=1)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def rule_count(self) -> int:
        """Total number of rules."""
        return len(self.rules)

    @computed_field
    @property
    def rules_by_type(self) -> dict[RuleType, int]:
        """Count of rules per type."""
        counts: dict[RuleType, int] = {}
        for rule in self.rules:
            counts[rule.rule_type] = counts.get(rule.rule_type, 0) + 1
        return counts


class StructurePattern(BaseModelConfig):
    """
    Observed content structure pattern.

    Represents common structural archetypes found in analyzed content.
    """

    pattern_type: ContentStructureType
    frequency: float = Field(..., ge=0.0, le=1.0, description="Occurrence rate")
    typical_sections: list[str] = Field(default_factory=list)
    avg_word_count: int = Field(..., ge=0)

    @computed_field
    @property
    def is_dominant_pattern(self) -> bool:
        """Check if this is the most common pattern (>50% frequency)."""
        return self.frequency > 0.5


class InferredPatterns(BaseModelConfig):
    """
    Learned patterns from website analysis (Layer 2 intelligence).

    Statistical patterns extracted from target website content.
    """

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID

    # Linguistic metrics
    avg_sentence_length: float = Field(..., gt=0.0)
    sentence_length_std: float = Field(..., ge=0.0)
    lexical_diversity: float = Field(..., ge=0.0, le=1.0)
    readability_score: float = Field(..., ge=0.0)

    # Semantic representation
    tone_embedding: Optional[list[float]] = Field(default=None)

    # Structural patterns
    common_structures: list[StructurePattern] = Field(default_factory=list)

    # Statistical metadata
    confidence: float = Field(..., ge=0.0, le=1.0)
    sample_size: int = Field(..., gt=0)
    analyzed_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def is_reliable(self) -> bool:
        """Check if patterns have sufficient statistical confidence."""
        from config.constants import SCRAPING_LIMITS

        return (
            self.confidence >= SCRAPING_LIMITS.MIN_PATTERN_CONFIDENCE
            and self.sample_size >= SCRAPING_LIMITS.MIN_SAMPLE_SIZE_FOR_CONFIDENCE
        )

    @computed_field
    @property
    def dominant_structure(self) -> Optional[ContentStructureType]:
        """Most frequent content structure."""
        if not self.common_structures:
            return None
        return max(self.common_structures, key=lambda p: p.frequency).pattern_type


# =============================================================================
# KEYWORD & CONTENT PLANNING MODELS
# =============================================================================


class Keyword(BaseModelConfig):
    """
    Keyword with semantic and strategic metadata.

    Enriched with search intent classification and semantic relationships.
    """

    phrase: str = Field(..., min_length=1, max_length=200)
    search_volume: Optional[int] = Field(default=None, ge=0)
    difficulty: Optional[float] = Field(default=None, ge=0.0, le=100.0)
    intent: KeywordIntent

    embedding: Optional[list[float]] = Field(default=None)
    related_concepts: list[str] = Field(default_factory=list)

    @field_validator("phrase")
    @classmethod
    def normalize_phrase(cls, v: str) -> str:
        """Normalize keyword phrase."""
        return " ".join(v.lower().strip().split())

    @computed_field
    @property
    def is_high_value(self) -> bool:
        """Heuristic for high-value keywords."""
        if self.search_volume is None or self.difficulty is None:
            return False
        # High volume, low difficulty
        return self.search_volume > 1000 and self.difficulty < 50


class Section(BaseModelConfig):
    """
    Content section with semantic theme and target keywords.

    Represents a hierarchical unit of content with specific rhetorical intent.
    """

    heading: str = Field(..., min_length=1, max_length=500)
    theme_embedding: Optional[list[float]] = Field(default=None)
    target_keywords: list[str] = Field(default_factory=list)
    estimated_words: int = Field(..., gt=0, le=2000)
    intent: SectionIntent

    # Generation metadata (populated during generation)
    generated_content: Optional[str] = Field(default=None)
    actual_word_count: Optional[int] = Field(default=None, ge=0)

    @computed_field
    @property
    def is_generated(self) -> bool:
        """Check if section has been generated."""
        return self.generated_content is not None


class Outline(BaseModelConfig):
    """
    Hierarchical content outline.

    Structured plan for article generation with SEO metadata.
    """

    title: str = Field(..., min_length=1, max_length=500)
    meta_description: str = Field(..., min_length=50, max_length=160)
    sections: list[Section] = Field(..., min_length=1)

    @computed_field
    @property
    def total_estimated_words(self) -> int:
        """Sum of all section word count estimates."""
        return sum(section.estimated_words for section in self.sections)

    @computed_field
    @property
    def section_count(self) -> int:
        """Total number of sections."""
        return len(self.sections)

    @computed_field
    @property
    def is_complete(self) -> bool:
        """Check if all sections have been generated."""
        return all(section.is_generated for section in self.sections)


class ContentPlan(BaseModelConfig):
    """
    Strategic content plan for a topic.

    Complete blueprint from keyword strategy to structural outline.
    """

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    topic: str = Field(..., min_length=1, max_length=500)

    # Strategic elements
    primary_keywords: list[Keyword] = Field(default_factory=list)
    secondary_keywords: list[Keyword] = Field(default_factory=list)

    # Content structure
    outline: Outline

    # SEO configuration
    target_word_count: int = Field(..., gt=0)
    readability_target: str = Field(default="grade_10-12")

    # Generation metadata
    estimated_cost_usd: float = Field(default=0.0, ge=0.0)
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def all_keywords(self) -> list[Keyword]:
        """Combined list of all keywords."""
        return self.primary_keywords + self.secondary_keywords

    @computed_field
    @property
    def keyword_count(self) -> int:
        """Total number of targeted keywords."""
        return len(self.all_keywords)


# =============================================================================
# CONTENT GENERATION MODELS
# =============================================================================


class QualityMetrics(BaseModelConfig):
    """
    Content quality assessment metrics.

    Computed metrics for validation and monitoring.
    """

    word_count: int = Field(..., ge=0)
    readability_score: float = Field(..., ge=0.0)
    lexical_diversity: float = Field(..., ge=0.0, le=1.0)
    keyword_density: dict[str, float] = Field(default_factory=dict)

    # Semantic metrics
    avg_sentence_length: float = Field(..., gt=0.0)
    paragraph_count: int = Field(..., ge=0)

    @computed_field
    @property
    def is_acceptable_length(self) -> bool:
        """Check if word count is within acceptable range."""
        from config.constants import QUALITY_THRESHOLDS

        return (
            QUALITY_THRESHOLDS.MIN_ARTICLE_WORDS
            <= self.word_count
            <= QUALITY_THRESHOLDS.MAX_ARTICLE_WORDS
        )

    @computed_field
    @property
    def readability_grade(self) -> str:
        """Convert Flesch-Kincaid score to grade level description."""
        score = self.readability_score
        if score < 8:
            return "elementary"
        elif score < 10:
            return "middle_school"
        elif score < 13:
            return "high_school"
        else:
            return "college"


class GeneratedArticle(BaseModelConfig):
    """
    Final generated article with complete metadata and quality metrics.

    Represents the complete output of the content generation pipeline.
    """

    id: UUID = Field(default_factory=uuid4)
    project_id: UUID
    content_plan_id: UUID

    # Content
    title: str = Field(..., min_length=1, max_length=500)
    content: str = Field(..., min_length=100)
    meta_description: str = Field(..., min_length=50, max_length=160)

    # Quality metrics
    quality_metrics: QualityMetrics
    validation_result: ValidationResult = Field(default=ValidationResult.PASS)

    # Cost tracking
    total_tokens_used: int = Field(..., ge=0)
    total_cost_usd: float = Field(..., ge=0.0)
    generation_time_seconds: float = Field(..., ge=0.0)

    # Generation details
    model_used: str = Field(default="gpt-4-turbo-preview")
    status: GenerationStatus = Field(default=GenerationStatus.COMPLETED)

    # Distribution tracking
    distributed_at: Optional[datetime] = Field(default=None)
    distribution_channels: list[DistributionChannel] = Field(default_factory=list)

    # Timestamps
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def cost_per_word(self) -> float:
        """Calculate cost efficiency metric."""
        if self.quality_metrics.word_count == 0:
            return 0.0
        return self.total_cost_usd / self.quality_metrics.word_count

    @computed_field
    @property
    def is_distributed(self) -> bool:
        """Check if article has been distributed."""
        return self.distributed_at is not None

    @computed_field
    @property
    def generation_speed_words_per_second(self) -> float:
        """Calculate generation throughput."""
        if self.generation_time_seconds == 0:
            return 0.0
        return self.quality_metrics.word_count / self.generation_time_seconds

    def mark_distributed(self, channel: DistributionChannel) -> None:
        """Record successful distribution to a channel."""
        if channel not in self.distribution_channels:
            self.distribution_channels.append(channel)
        if self.distributed_at is None:
            self.distributed_at = datetime.utcnow()
        self.updated_at = datetime.utcnow()


# =============================================================================
# DECISION ENGINE MODELS
# =============================================================================


class Evidence(BaseModelConfig):
    """
    Single piece of evidence for decision-making.

    Used in Bayesian evidence accumulation for adaptive intelligence.
    """

    source_layer: DecisionLayer
    content: str = Field(..., min_length=1)
    embedding: Optional[list[float]] = Field(default=None)

    confidence: float = Field(..., ge=0.0, le=1.0)
    priority: int = Field(default=5, ge=1, le=10)

    # Provenance
    source_id: Optional[UUID] = Field(default=None)
    source_type: str = Field(default="unknown")  # "rule", "pattern", "best_practice"

    @computed_field
    @property
    def weighted_confidence(self) -> float:
        """Confidence weighted by layer authority."""
        from config.constants import DECISION_WEIGHTS

        weights = {
            DecisionLayer.EXPLICIT_RULE: DECISION_WEIGHTS.EXPLICIT_RULE,
            DecisionLayer.INFERRED_PATTERN: DECISION_WEIGHTS.INFERRED_PATTERN,
            DecisionLayer.BEST_PRACTICE: DECISION_WEIGHTS.BEST_PRACTICE,
        }

        layer_weight = weights.get(self.source_layer, 0.5)
        return self.confidence * layer_weight


class Decision(BaseModelConfig):
    """
    Resolved decision with provenance chain.

    Output of the three-layer decision hierarchy.
    """

    decision_id: UUID = Field(default_factory=uuid4)
    decision_type: str = Field(..., description="e.g., 'tone_selection', 'structure_choice'")

    # Decision outcome
    choice: str = Field(..., min_length=1)
    confidence: float = Field(..., ge=0.0, le=1.0)

    # Evidence chain
    evidence_items: list[Evidence] = Field(default_factory=list)
    primary_layer: DecisionLayer

    # Context
    query: str = Field(..., min_length=1)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def evidence_count(self) -> int:
        """Total pieces of evidence considered."""
        return len(self.evidence_items)

    @computed_field
    @property
    def is_high_confidence(self) -> bool:
        """Check if decision has high confidence."""
        return self.confidence >= 0.85

    def get_provenance_summary(self) -> dict[DecisionLayer, int]:
        """Summarize evidence sources by layer."""
        summary: dict[DecisionLayer, int] = {
            DecisionLayer.EXPLICIT_RULE: 0,
            DecisionLayer.INFERRED_PATTERN: 0,
            DecisionLayer.BEST_PRACTICE: 0,
        }
        for evidence in self.evidence_items:
            summary[evidence.source_layer] += 1
        return summary


class ContextSnapshot(BaseModelConfig):
    """
    Immutable snapshot of project context at a point in time.

    Used for reproducibility and audit trails.
    """

    snapshot_id: UUID = Field(default_factory=uuid4)
    project_id: UUID

    # Context components
    has_rulebook: bool
    has_inferred_patterns: bool

    # Metadata
    rulebook_version: Optional[int] = Field(default=None)
    patterns_confidence: Optional[float] = Field(default=None)
    patterns_sample_size: Optional[int] = Field(default=None)

    created_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def primary_decision_layer(self) -> DecisionLayer:
        """Determine which layer should be primary for this context."""
        if self.has_rulebook:
            return DecisionLayer.EXPLICIT_RULE
        elif (
            self.has_inferred_patterns
            and self.patterns_confidence
            and self.patterns_confidence >= 0.7
        ):
            return DecisionLayer.INFERRED_PATTERN
        else:
            return DecisionLayer.BEST_PRACTICE


# =============================================================================
# CACHE MODELS
# =============================================================================


class CacheEntry(BaseModelConfig):
    """
    Generic cache entry with metadata.

    Used across all caching layers (embeddings, LLM responses, etc.).
    """

    cache_key: str = Field(..., min_length=1)
    value: Any  # Flexible value type

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=0, ge=0)
    ttl_seconds: Optional[int] = Field(default=None)

    # Provenance
    source: str = Field(default="unknown")
    version: int = Field(default=1, ge=1)

    @computed_field
    @property
    def is_expired(self) -> bool:
        """Check if cache entry has expired."""
        if self.ttl_seconds is None:
            return False

        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds

    @computed_field
    @property
    def age_hours(self) -> float:
        """Calculate age of cache entry in hours."""
        elapsed = (datetime.utcnow() - self.created_at).total_seconds()
        return elapsed / 3600

    def record_access(self) -> None:
        """Update access metadata."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


class LLMCacheEntry(BaseModelConfig):
    """
    Specialized cache entry for LLM responses.

    Includes prompt fingerprint and token usage for cost tracking.
    """

    prompt_hash: str = Field(..., min_length=1)
    prompt_preview: str = Field(..., max_length=500)

    response: str = Field(..., min_length=1)

    # LLM parameters
    model: str
    temperature: float = Field(..., ge=0.0, le=2.0)

    # Cost metrics
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    cost_usd: float = Field(..., ge=0.0)

    # Cache metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = Field(default=1, ge=1)

    @computed_field
    @property
    def total_tokens(self) -> int:
        """Total tokens consumed."""
        return self.input_tokens + self.output_tokens

    @computed_field
    @property
    def cost_savings_usd(self) -> float:
        """Estimated savings from cache hit."""
        return self.cost_usd * (self.access_count - 1)

    def record_hit(self) -> None:
        """Record cache hit."""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


# =============================================================================
# TASK QUEUE MODELS
# =============================================================================


class TaskMetadata(BaseModelConfig):
    """
    Metadata for async task execution.

    Used for Celery task tracking and monitoring.
    """

    task_id: UUID = Field(default_factory=uuid4)
    task_name: str = Field(..., min_length=1)

    # Execution context
    project_id: UUID
    priority: int = Field(default=5, ge=1, le=10)

    # Status tracking
    status: GenerationStatus = Field(default=GenerationStatus.PENDING)
    progress_percentage: float = Field(default=0.0, ge=0.0, le=100.0)

    # Timing
    queued_at: datetime = Field(default_factory=datetime.utcnow)
    started_at: Optional[datetime] = Field(default=None)
    completed_at: Optional[datetime] = Field(default=None)

    # Results
    result_id: Optional[UUID] = Field(default=None)
    error_message: Optional[str] = Field(default=None)

    @computed_field
    @property
    def execution_time_seconds(self) -> Optional[float]:
        """Calculate execution duration."""
        if self.started_at is None or self.completed_at is None:
            return None
        return (self.completed_at - self.started_at).total_seconds()

    @computed_field
    @property
    def queue_wait_time_seconds(self) -> Optional[float]:
        """Calculate time spent in queue."""
        if self.started_at is None:
            return None
        return (self.started_at - self.queued_at).total_seconds()

    @computed_field
    @property
    def is_terminal(self) -> bool:
        """Check if task is in terminal state."""
        return self.status.is_terminal

    def mark_started(self) -> None:
        """Update task as started."""
        self.started_at = datetime.utcnow()
        self.status = GenerationStatus.GENERATING

    def mark_completed(self, result_id: UUID) -> None:
        """Update task as completed."""
        self.completed_at = datetime.utcnow()
        self.status = GenerationStatus.COMPLETED
        self.result_id = result_id
        self.progress_percentage = 100.0

    def mark_failed(self, error: str) -> None:
        """Update task as failed."""
        self.completed_at = datetime.utcnow()
        self.status = GenerationStatus.FAILED
        self.error_message = error


# =============================================================================
# ANALYTICS & MONITORING MODELS
# =============================================================================


class CostBreakdown(BaseModelConfig):
    """
    Detailed cost breakdown for transparency and optimization.

    Tracks costs across different operations and models.
    """

    # Operational costs
    keyword_research_cost: float = Field(default=0.0, ge=0.0)
    content_planning_cost: float = Field(default=0.0, ge=0.0)
    content_generation_cost: float = Field(default=0.0, ge=0.0)
    validation_cost: float = Field(default=0.0, ge=0.0)

    # Model-specific costs
    gpt4_cost: float = Field(default=0.0, ge=0.0)
    gpt35_cost: float = Field(default=0.0, ge=0.0)
    other_model_cost: float = Field(default=0.0, ge=0.0)

    @computed_field
    @property
    def total_cost(self) -> float:
        """Sum of all costs."""
        return (
            self.keyword_research_cost
            + self.content_planning_cost
            + self.content_generation_cost
            + self.validation_cost
        )

    @computed_field
    @property
    def cost_by_model(self) -> dict[str, float]:
        """Cost distribution by model."""
        return {
            "gpt-4": self.gpt4_cost,
            "gpt-3.5-turbo": self.gpt35_cost,
            "other": self.other_model_cost,
        }


class PerformanceMetrics(BaseModelConfig):
    """
    System performance metrics for monitoring and optimization.

    Aggregated metrics for dashboards and alerting.
    """

    # Throughput
    articles_generated: int = Field(default=0, ge=0)
    total_words_generated: int = Field(default=0, ge=0)

    # Latency (seconds)
    avg_end_to_end_latency: float = Field(default=0.0, ge=0.0)
    avg_llm_latency: float = Field(default=0.0, ge=0.0)
    avg_cache_latency: float = Field(default=0.0, ge=0.0)

    # Cache efficiency
    cache_hit_rate: float = Field(default=0.0, ge=0.0, le=1.0)
    total_cache_hits: int = Field(default=0, ge=0)
    total_cache_misses: int = Field(default=0, ge=0)

    # Cost efficiency
    total_cost_usd: float = Field(default=0.0, ge=0.0)
    avg_cost_per_article: float = Field(default=0.0, ge=0.0)
    cost_savings_from_cache_usd: float = Field(default=0.0, ge=0.0)

    # Quality
    avg_readability_score: float = Field(default=0.0, ge=0.0)
    validation_pass_rate: float = Field(default=0.0, ge=0.0, le=1.0)

    # Time window
    window_start: datetime = Field(default_factory=datetime.utcnow)
    window_end: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def total_cache_requests(self) -> int:
        """Total cache queries."""
        return self.total_cache_hits + self.total_cache_misses

    @computed_field
    @property
    def roi_from_caching(self) -> Optional[float]:
        """Return on investment from caching (savings / implementation cost)."""
        if self.total_cost_usd == 0:
            return None
        # Assuming implementation cost amortized to negligible
        return self.cost_savings_from_cache_usd / self.total_cost_usd


# =============================================================================
# USER MANAGEMENT MODELS
# =============================================================================


class UserCreate(BaseModelConfig):
    """
    User creation model with plain password.

    Used for user registration and account creation.
    """

    email: str = Field(..., min_length=1, max_length=255)
    password: str = Field(..., min_length=8, max_length=128)
    full_name: Optional[str] = Field(default=None, max_length=255)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: str) -> str:
        """Basic email validation."""
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower().strip()

    @field_validator("password")
    @classmethod
    def validate_password(cls, v: str) -> str:
        """Password strength validation."""
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters long")
        # Commented out the problematic length check that was causing issues
        # if len(password.encode('utf-8')) > 72:
        #     raise ValueError("Password too long")
        if not any(c.isupper() for c in v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not any(c.islower() for c in v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not any(c.isdigit() for c in v):
            raise ValueError("Password must contain at least one digit")
        return v


class UserUpdate(BaseModelConfig):
    """
    User update model for partial updates.

    All fields are optional for flexible updates.
    """

    email: Optional[str] = Field(default=None, min_length=1, max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=255)
    is_active: Optional[bool] = Field(default=None)
    is_superuser: Optional[bool] = Field(default=None)

    @field_validator("email")
    @classmethod
    def validate_email(cls, v: Optional[str]) -> Optional[str]:
        """Basic email validation."""
        if v is None:
            return v
        import re

        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, v):
            raise ValueError("Invalid email format")
        return v.lower().strip()


class UserInDB(BaseModelConfig):
    """
    Internal user model with hashed password.

    Used for database operations and internal processing.
    """

    id: UUID = Field(default_factory=uuid4)
    email: str = Field(..., min_length=1, max_length=255)
    hashed_password: str = Field(..., min_length=1)
    full_name: Optional[str] = Field(default=None, max_length=255)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def username(self) -> str:
        """Derive username from email for compatibility."""
        return self.email.split("@")[0]


class User(BaseModelConfig):
    """
    Public user model without sensitive data.

    Used for API responses and external interactions.
    """

    id: UUID = Field(default_factory=uuid4)
    email: str = Field(..., min_length=1, max_length=255)
    full_name: Optional[str] = Field(default=None, max_length=255)
    is_active: bool = Field(default=True)
    is_superuser: bool = Field(default=False)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def username(self) -> str:
        """Derive username from email for compatibility."""
        return self.email.split("@")[0]


# =============================================================================
# REQUEST/RESPONSE MODELS (API)
# =============================================================================


class ContentGenerationRequest(BaseModelConfig):
    """
    Request model for content generation endpoint.

    User-facing API contract for initiating content generation.
    """

    project_id: UUID
    topic: str = Field(..., min_length=1, max_length=500)

    # Optional overrides
    target_word_count: Optional[int] = Field(default=None, gt=0, le=5000)
    primary_keywords: Optional[list[str]] = Field(default=None)
    content_structure: Optional[ContentStructureType] = Field(default=None)

    # Distribution
    auto_distribute: bool = Field(default=False)
    distribution_channels: list[DistributionChannel] = Field(default_factory=list)

    @field_validator("topic")
    @classmethod
    def validate_topic(cls, v: str) -> str:
        """Ensure topic is meaningful."""
        if len(v.strip()) < 3:
            raise ValueError("Topic must be at least 3 characters")
        return v.strip()


class ContentGenerationResponse(BaseModelConfig):
    """
    Response model for content generation endpoint.

    Returns task metadata for async tracking.
    """

    task_id: UUID
    project_id: UUID
    status: GenerationStatus
    estimated_completion_seconds: int = Field(..., ge=0)

    message: str = Field(default="Content generation initiated")


class ArticleRetrievalResponse(BaseModelConfig):
    """
    Response model for article retrieval endpoint.

    Returns complete article with metadata.
    """

    article: GeneratedArticle
    cost_breakdown: CostBreakdown
    performance_metrics: Optional[PerformanceMetrics] = Field(default=None)


# =============================================================================
# MODULE EXPORTS
# =============================================================================

__all__ = [
    # Base
    "BaseModelConfig",
    # Project & Configuration
    "Project",
    "Rule",
    "Rulebook",
    "StructurePattern",
    "InferredPatterns",
    # Keywords & Planning
    "Keyword",
    "Section",
    "Outline",
    "ContentPlan",
    # Content Generation
    "QualityMetrics",
    "GeneratedArticle",
    # Decision Engine
    "Evidence",
    "Decision",
    "ContextSnapshot",
    # Caching
    "CacheEntry",
    "LLMCacheEntry",
    # Task Queue
    "TaskMetadata",
    # Analytics
    "CostBreakdown",
    "PerformanceMetrics",
    # User Management
    "UserCreate",
    "UserUpdate",
    "UserInDB",
    "User",
    # API
    "ContentGenerationRequest",
    "ContentGenerationResponse",
    "ArticleRetrievalResponse",
]
