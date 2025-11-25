"""
Best Practices Knowledge Base - Layer 3 Intelligence
====================================================

Universal knowledge base of content marketing and SEO best practices.
Provides fallback when project-specific knowledge unavailable.

Knowledge Organization:
- Category-based (SEO, Writing, Structure, Tone, etc.)
- Semantic indexed (vector similarity search)
- Confidence scored (evidence-based reliability)
- Versioned (allows knowledge evolution)

Design: Read-heavy workload optimized with caching.

Note for future development:
Currently, the knowledge base is managed in-memory. For enhanced scalability
and dynamic management, consider migrating this data to a dedicated table
in the PostgreSQL database in a future version.
"""

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from loguru import logger

from core.exceptions import ValidationError
from core.models import RuleType
from infrastructure.redis_client import RedisClient
from intelligence.semantic_analyzer import SemanticAnalyzer


class PracticeCategory(str, Enum):
    """Categories of best practices."""

    SEO_TECHNICAL = "seo_technical"
    SEO_CONTENT = "seo_content"
    WRITING_STYLE = "writing_style"
    STRUCTURE = "structure"
    READABILITY = "readability"
    TONE = "tone"
    ENGAGEMENT = "engagement"
    GENERAL = "general"


@dataclass
class BestPractice:
    """Single best practice entry."""

    id: str
    category: PracticeCategory
    content: str
    embedding: np.ndarray
    confidence: float  # 0-1, based on evidence/consensus
    source: str  # Citation or authority
    applicable_contexts: List[str]  # When to apply this practice
    version: int = 1

    def __post_init__(self):
        """Validate practice."""
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be in [0,1], got {self.confidence}")


class BestPracticesKB:
    """
    Knowledge base of universal content best practices.

    Provides Layer 3 intelligence when explicit rules and
    inferred patterns are unavailable or insufficient.
    """

    def __init__(self, redis_client: RedisClient, semantic_analyzer: SemanticAnalyzer):
        """
        Initialize knowledge base.

        Args:
            redis_client: Redis client instance for caching embeddings
            semantic_analyzer: Semantic analyzer instance for embeddings
        """
        self.redis_client = redis_client
        self.semantic_analyzer = semantic_analyzer
        self.practices: Dict[str, BestPractice] = {}
        self.practices_by_category: Dict[PracticeCategory, List[str]] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """
        Initialize knowledge base with curated best practices.

        Seeds system with industry-standard guidance.
        """
        if self._initialized:
            return

        logger.info("Initializing best practices knowledge base...")

        # Load curated practices
        await self._load_curated_practices()

        # Index by category
        self._build_category_index()

        # Cache embeddings
        await self._cache_embeddings()

        self._initialized = True
        logger.info(f"Knowledge base initialized with {len(self.practices)} practices")

    # =========================================================================
    # QUERY INTERFACE
    # =========================================================================

    async def query(
        self,
        query: str,
        query_embedding: Optional[np.ndarray] = None,
        rule_type: Optional[RuleType] = None,
        top_k: int = 3,
        min_confidence: float = 0.7,
    ) -> List[Tuple[BestPractice, float]]:
        """
        Query best practices knowledge base.

        Args:
            query: Natural language query
            query_embedding: Pre-computed query embedding (optional)
            rule_type: Optional filter by rule type
            top_k: Maximum results
            min_confidence: Minimum confidence threshold

        Returns:
            List of (practice, similarity_score) tuples
        """
        if not self._initialized:
            await self.initialize()

        # Generate query embedding if not provided
        if query_embedding is None:
            query_embedding = await self.semantic_analyzer.embed(query, normalize=True)

        # Filter by rule type if specified
        if rule_type:
            category = self._rule_type_to_category(rule_type)
            candidate_ids = self.practices_by_category.get(category, [])
        else:
            candidate_ids = list(self.practices.keys())

        # Compute similarities
        candidates = [self.practices[pid] for pid in candidate_ids]

        similarities = []
        for practice in candidates:
            if practice.confidence < min_confidence:
                continue

            similarity = self.semantic_analyzer.compute_similarity(
                query_embedding, practice.embedding, metric="cosine"
            )

            # Weight similarity by confidence
            weighted_score = similarity * practice.confidence

            similarities.append((practice, weighted_score))

        # Sort by weighted score
        similarities.sort(key=lambda x: x[1], reverse=True)

        results = similarities[:top_k]

        logger.debug(f"Found {len(results)} best practices for query: '{query[:50]}...'")
        return results

    def get_by_category(
        self,
        category: PracticeCategory,
        min_confidence: float = 0.7,
    ) -> List[BestPractice]:
        """
        Retrieve all practices in a category.

        Args:
            category: Practice category
            min_confidence: Minimum confidence threshold

        Returns:
            List of practices
        """
        practice_ids = self.practices_by_category.get(category, [])
        practices = [self.practices[pid] for pid in practice_ids]

        # Filter by confidence
        filtered = [p for p in practices if p.confidence >= min_confidence]

        # Sort by confidence (descending)
        filtered.sort(key=lambda p: p.confidence, reverse=True)

        return filtered

    # =========================================================================
    # KNOWLEDGE BASE POPULATION
    # =========================================================================

    async def _load_curated_practices(self) -> None:
        """Load curated best practices."""

        # SEO Technical Practices
        await self._add_practice(
            category=PracticeCategory.SEO_TECHNICAL,
            content="Include target keywords in the title, ideally near the beginning. "
            "Keep titles between 50-60 characters for optimal display in search results.",
            confidence=0.95,
            source="Google Search Central Guidelines",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.SEO_TECHNICAL,
            content="Write compelling meta descriptions between 150-160 characters. "
            "Include a call-to-action and target keywords naturally.",
            confidence=0.92,
            source="SEO industry consensus",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.SEO_CONTENT,
            content="Maintain keyword density between 0.5-2% for primary keywords. "
            "Use semantic variations and related terms naturally throughout content.",
            confidence=0.88,
            source="Content SEO best practices",
            applicable_contexts=["blog posts", "articles", "guides"],
        )

        await self._add_practice(
            category=PracticeCategory.SEO_CONTENT,
            content="Structure content with clear H1, H2, H3 hierarchy. Use only one H1 per page. "
            "Include keywords in at least one H2 heading.",
            confidence=0.93,
            source="On-page SEO standards",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.SEO_CONTENT,
            content="Target article length of 1500-2500 words for comprehensive topics. "
            "Longer content (2000+ words) tends to rank better for competitive keywords.",
            confidence=0.82,
            source="Content length studies (Backlinko, SEMrush)",
            applicable_contexts=["informational content", "guides"],
        )

        # Writing Style Practices
        await self._add_practice(
            category=PracticeCategory.WRITING_STYLE,
            content="Use active voice whenever possible. Active voice is more engaging, direct, "
            "and easier to read than passive constructions.",
            confidence=0.91,
            source="Writing style guides (AP, Chicago)",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.WRITING_STYLE,
            content="Vary sentence length to maintain reader engagement. Mix short, impactful sentences "
            "with longer, more complex ones. Average 15-20 words per sentence.",
            confidence=0.87,
            source="Readability research",
            applicable_contexts=["articles", "blog posts"],
        )

        await self._add_practice(
            category=PracticeCategory.WRITING_STYLE,
            content="Use concrete, specific examples rather than abstract generalizations. "
            "Show, don't just tell. Include data, statistics, and real-world cases.",
            confidence=0.89,
            source="Content marketing best practices",
            applicable_contexts=["all content types"],
        )

        # Structure Practices
        await self._add_practice(
            category=PracticeCategory.STRUCTURE,
            content="Begin with a compelling introduction that hooks the reader and clearly states "
            "what they'll learn. Use the first 100 words to capture attention.",
            confidence=0.93,
            source="Content engagement studies",
            applicable_contexts=["articles", "blog posts", "guides"],
        )

        await self._add_practice(
            category=PracticeCategory.STRUCTURE,
            content="Break content into scannable sections with descriptive subheadings every 300-500 words. "
            "Most readers scan before committing to read in depth.",
            confidence=0.90,
            source="User behavior studies (Nielsen Norman Group)",
            applicable_contexts=["articles", "blog posts", "long-form content"],
        )

        await self._add_practice(
            category=PracticeCategory.STRUCTURE,
            content="Use bullet points and numbered lists to present information clearly. "
            "Lists improve scannability and information retention.",
            confidence=0.88,
            source="Content formatting research",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.STRUCTURE,
            content="Include a clear conclusion that summarizes key points and provides next steps "
            "or a call-to-action. Never end abruptly.",
            confidence=0.85,
            source="Content structure best practices",
            applicable_contexts=["articles", "blog posts"],
        )

        # Readability Practices
        await self._add_practice(
            category=PracticeCategory.READABILITY,
            content="Target Flesch-Kincaid grade level 8-10 for general audiences. "
            "More technical content can go to grade 12-14, but avoid unnecessary complexity.",
            confidence=0.86,
            source="Readability standards",
            applicable_contexts=["general audience content"],
        )

        await self._add_practice(
            category=PracticeCategory.READABILITY,
            content="Keep paragraphs short, typically 2-4 sentences. Long paragraphs intimidate readers "
            "and reduce engagement, especially on mobile devices.",
            confidence=0.91,
            source="Mobile readability research",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.READABILITY,
            content="Define technical terms and jargon on first use. Write for your least knowledgeable "
            "reader unless creating expert-level content.",
            confidence=0.89,
            source="Technical writing guidelines",
            applicable_contexts=["technical content", "specialized topics"],
        )

        # Tone Practices
        await self._add_practice(
            category=PracticeCategory.TONE,
            content="Match tone to audience and context. B2B content often benefits from professional "
            "yet approachable tone. B2C can be more casual and conversational.",
            confidence=0.87,
            source="Audience engagement research",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.TONE,
            content="Use second person ('you') to create direct connection with readers. "
            "This increases engagement and makes content feel personalized.",
            confidence=0.88,
            source="Engagement studies",
            applicable_contexts=["blog posts", "how-to guides"],
        )

        await self._add_practice(
            category=PracticeCategory.TONE,
            content="Be authentic and human. Avoid corporate jargon and overly formal language "
            "unless required by brand guidelines or industry norms.",
            confidence=0.84,
            source="Brand voice research",
            applicable_contexts=["blog posts", "social content"],
        )

        # Engagement Practices
        await self._add_practice(
            category=PracticeCategory.ENGAGEMENT,
            content="Ask questions throughout the content to prompt reader reflection and engagement. "
            "Rhetorical questions work well to maintain interest.",
            confidence=0.83,
            source="Content engagement tactics",
            applicable_contexts=["articles", "blog posts"],
        )

        await self._add_practice(
            category=PracticeCategory.ENGAGEMENT,
            content="Include relevant images, charts, or graphics every 300-400 words. "
            "Visual content increases engagement and time on page.",
            confidence=0.90,
            source="Visual content research",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.ENGAGEMENT,
            content="Link to authoritative external sources to build credibility. "
            "Also include internal links to related content on your site.",
            confidence=0.87,
            source="SEO and credibility best practices",
            applicable_contexts=["articles", "blog posts"],
        )

        await self._add_practice(
            category=PracticeCategory.ENGAGEMENT,
            content="End with a clear call-to-action (CTA). Tell readers exactly what to do next: "
            "subscribe, download, read more, contact, etc.",
            confidence=0.91,
            source="Conversion optimization research",
            applicable_contexts=["all content types"],
        )

        # General Best Practices
        await self._add_practice(
            category=PracticeCategory.GENERAL,
            content="Edit ruthlessly. Remove unnecessary words, redundant phrases, and filler content. "
            "Every sentence should serve a purpose.",
            confidence=0.93,
            source="Writing craft guidelines",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.GENERAL,
            content="Proofread for grammar, spelling, and punctuation errors. Errors damage credibility "
            "and distract readers from your message.",
            confidence=0.95,
            source="Editorial standards",
            applicable_contexts=["all content types"],
        )

        await self._add_practice(
            category=PracticeCategory.GENERAL,
            content="Update content regularly to maintain accuracy and relevance. "
            "Search engines favor fresh, up-to-date content.",
            confidence=0.85,
            source="SEO freshness factor",
            applicable_contexts=["evergreen content", "guides"],
        )

        await self._add_practice(
            category=PracticeCategory.GENERAL,
            content="Write for humans first, search engines second. Quality content that serves "
            "user intent will naturally perform well in search rankings.",
            confidence=0.94,
            source="Google Quality Guidelines",
            applicable_contexts=["all content types"],
        )

    async def _add_practice(
        self,
        category: PracticeCategory,
        content: str,
        confidence: float,
        source: str,
        applicable_contexts: List[str],
    ) -> None:
        """
        Add a best practice to the knowledge base.

        Args:
            category: Practice category
            content: Practice description
            confidence: Confidence score (0-1)
            source: Citation/authority
            applicable_contexts: When to apply
        """
        # Generate unique ID
        practice_id = f"bp_{uuid4().hex[:12]}"

        # Generate embedding
        embedding = await self.semantic_analyzer.embed(content, normalize=True)

        # Create practice object
        practice = BestPractice(
            id=practice_id,
            category=category,
            content=content,
            embedding=embedding,
            confidence=confidence,
            source=source,
            applicable_contexts=applicable_contexts,
        )

        # Store
        self.practices[practice_id] = practice

    # =========================================================================
    # INDEXING & CACHING
    # =========================================================================

    def _build_category_index(self) -> None:
        """Build category-based index for fast filtering."""
        self.practices_by_category.clear()

        for practice_id, practice in self.practices.items():
            if practice.category not in self.practices_by_category:
                self.practices_by_category[practice.category] = []

            self.practices_by_category[practice.category].append(practice_id)

        logger.debug(f"Built category index: {len(self.practices_by_category)} categories")

    async def _cache_embeddings(self) -> None:
        """Cache all embeddings in Redis for fast access."""
        cache_batch = {}

        for practice_id, practice in self.practices.items():
            cache_key = f"bp_emb:{practice_id}"
            cache_batch[cache_key] = practice.embedding

        if cache_batch:
            await self.redis_client.store_embeddings_batch(
                embeddings=cache_batch, ttl=86400 * 90
            )  # 90 days

            logger.debug(f"Cached {len(cache_batch)} practice embeddings")


@staticmethod
def _rule_type_to_category(rule_type: RuleType) -> PracticeCategory:
    """Map RuleType to PracticeCategory."""
    mapping = {
        RuleType.TONE: PracticeCategory.TONE,
        RuleType.STYLE: PracticeCategory.WRITING_STYLE,
        RuleType.STRUCTURE: PracticeCategory.STRUCTURE,
        RuleType.SEO: PracticeCategory.SEO_CONTENT,
        RuleType.TOPIC: PracticeCategory.GENERAL,
        RuleType.GENERAL: PracticeCategory.GENERAL,
    }

    return mapping.get(rule_type, PracticeCategory.GENERAL)


# =========================================================================
# KNOWLEDGE BASE MANAGEMENT
# =========================================================================


def get_statistics(self) -> Dict[str, any]:
    """Get knowledge base statistics."""
    total = len(self.practices)

    by_category = {cat.value: len(ids) for cat, ids in self.practices_by_category.items()}

    avg_confidence = sum(p.confidence for p in self.practices.values()) / total if total > 0 else 0

    return {
        "total_practices": total,
        "by_category": by_category,
        "average_confidence": avg_confidence,
        "initialized": self._initialized,
    }


async def add_custom_practice(
    self,
    category: PracticeCategory,
    content: str,
    confidence: float,
    source: str,
    applicable_contexts: List[str],
) -> str:
    """
    Add custom best practice (for extending knowledge base).

    Args:
        category: Practice category
        content: Practice description
        confidence: Confidence score
        source: Citation
        applicable_contexts: Application contexts

    Returns:
        Practice ID
    """
    await self._add_practice(
        category=category,
        content=content,
        confidence=confidence,
        source=source,
        applicable_contexts=applicable_contexts,
    )

    # Rebuild index
    self._build_category_index()

    practice_id = list(self.practices.keys())[-1]
    logger.info(f"Added custom practice: {practice_id}")

    return practice_id
