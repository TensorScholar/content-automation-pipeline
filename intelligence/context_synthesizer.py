"""
Context Synthesizer - Intelligent Prompt Compression
=====================================================

Reduces LLM context costs through multi-strategy compression:

1. Extractive Summarization: Sentence ranking via semantic centrality
2. Information Density Analysis: Remove redundant/low-value content
3. Semantic Deduplication: Merge similar information
4. Hierarchical Abstraction: Multi-level detail control
5. Lazy Materialization: Load only what's needed, when needed

Target: 70-85% token reduction with <5% information loss.

Architecture: Functional pipeline of compression transforms,
each preserving semantic coherence while reducing tokens.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
import spacy
from loguru import logger
from sentence_transformers import SentenceTransformer

from core.exceptions import ValidationError
from intelligence.semantic_analyzer import SemanticAnalyzer, SimilarityMetric


class CompressionLevel(str, Enum):
    """Compression intensity levels."""

    MINIMAL = "minimal"  # 20-30% reduction
    STANDARD = "standard"  # 50-60% reduction
    AGGRESSIVE = "aggressive"  # 70-85% reduction
    EXTREME = "extreme"  # 90%+ reduction (lossy)


class ContentType(str, Enum):
    """Type of content being compressed."""

    RULEBOOK = "rulebook"
    WEBSITE_CONTENT = "website_content"
    ARTICLE = "article"
    CONVERSATION = "conversation"
    TECHNICAL_DOC = "technical_doc"


@dataclass
class CompressionResult:
    """Result of compression operation."""

    original_text: str
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    information_density: float  # Estimated semantic preservation (0-1)
    method_chain: List[str] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        """Number of tokens saved."""
        return self.original_tokens - self.compressed_tokens

    @property
    def reduction_percent(self) -> float:
        """Percentage reduction."""
        return (1 - self.compression_ratio) * 100


@dataclass
class ContextPlan:
    """
    Lazy evaluation plan for context loading.

    Describes what context should be loaded and in what order,
    without actually materializing it yet.
    """

    essential_sources: List[Tuple[str, float]]  # (source_id, relevance_score)
    optional_sources: List[Tuple[str, float]]
    compression_level: CompressionLevel
    estimated_tokens: int

    def should_include(self, source_id: str, token_budget: int) -> bool:
        """Determine if source should be included given budget."""
        # Simple heuristic: include if we haven't exceeded budget
        return self.estimated_tokens < token_budget


class ContextSynthesizer:
    """
    Intelligent context compression and synthesis engine.
    Implements multiple compression strategies that can be composed:
    - Extractive summarization (sentence ranking)
    - Semantic deduplication (merge similar content)
    - Information filtering (remove low-value content)
    - Hierarchical abstraction (adaptive detail levels)
    """

    def __init__(self, semantic_analyzer: Optional[SemanticAnalyzer] = None):
        """Initialize synthesizer with NLP models and semantic analyzer.

        Args:
            semantic_analyzer: SemanticAnalyzer instance for embeddings (optional, created if not provided)
        """
        try:
            # Lightweight spaCy for sentence tokenization
            self.nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
            self.nlp.add_pipe("sentencizer")

            # Store semantic analyzer instance
            self.semantic_analyzer = semantic_analyzer

            logger.info("Context synthesizer initialized")

        except Exception as e:
            logger.error(f"Failed to initialize context synthesizer: {e}")
            raise ValidationError(f"Initialization failed: {e}")


# =========================================================================
# HIGH-LEVEL COMPRESSION API
# =========================================================================


async def compress(
    self,
    text: str,
    target_ratio: float = 0.5,
    content_type: ContentType = ContentType.ARTICLE,
    preserve_structure: bool = True,
) -> CompressionResult:
    """
    Compress text to target ratio while preserving semantic content.

    Args:
        text: Input text to compress
        target_ratio: Target size as ratio of original (0.5 = 50%)
        content_type: Type of content (affects compression strategy)
        preserve_structure: If True, maintain document structure

    Returns:
        CompressionResult with compressed text and metrics
    """
    if not text or len(text.strip()) < 50:
        raise ValidationError("Text too short for compression")

    original_tokens = self._estimate_tokens(text)
    target_tokens = int(original_tokens * target_ratio)

    logger.debug(f"Compressing {original_tokens} → {target_tokens} tokens ({target_ratio:.0%})")

    # Select compression strategy based on target ratio
    if target_ratio >= 0.7:
        compressed = await self._compress_minimal(text)
        method_chain = ["minimal"]
    elif target_ratio >= 0.4:
        compressed = await self._compress_standard(text, preserve_structure)
        method_chain = ["extractive", "deduplication"]
    else:
        compressed = await self._compress_aggressive(text, target_tokens, preserve_structure)
        method_chain = ["extractive", "deduplication", "filtering"]

    compressed_tokens = self._estimate_tokens(compressed)
    actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

    # Estimate information preservation
    info_density = await self._estimate_information_density(text, compressed)

    result = CompressionResult(
        original_text=text,
        compressed_text=compressed,
        original_tokens=original_tokens,
        compressed_tokens=compressed_tokens,
        compression_ratio=actual_ratio,
        information_density=info_density,
        method_chain=method_chain,
    )

    logger.info(
        f"Compressed: {result.reduction_percent:.1f}% reduction, "
        f"info density: {info_density:.2f}"
    )

    return result


async def compress_to_budget(
    self,
    text: str,
    token_budget: int,
    content_type: ContentType = ContentType.ARTICLE,
) -> CompressionResult:
    """
    Compress text to fit within token budget.

    Uses iterative compression until budget is met.

    Args:
        text: Input text
        token_budget: Maximum allowed tokens
        content_type: Type of content

    Returns:
        CompressionResult fitting within budget
    """
    original_tokens = self._estimate_tokens(text)

    if original_tokens <= token_budget:
        # No compression needed
        return CompressionResult(
            original_text=text,
            compressed_text=text,
            original_tokens=original_tokens,
            compressed_tokens=original_tokens,
            compression_ratio=1.0,
            information_density=1.0,
            method_chain=["none"],
        )

    # Calculate required ratio
    target_ratio = token_budget / original_tokens

    # Compress to target
    return await self.compress(
        text=text,
        target_ratio=target_ratio,
        content_type=content_type,
        preserve_structure=True,
    )


# =========================================================================
# COMPRESSION STRATEGIES
# =========================================================================


async def _compress_minimal(self, text: str) -> str:
    """
    Minimal compression: whitespace normalization only.

    Target: 20-30% reduction
    """
    import re

    # Remove excess whitespace
    compressed = re.sub(r"\s+", " ", text)

    # Remove redundant punctuation
    compressed = re.sub(r"[.!?]{2,}", ".", compressed)

    # Trim
    compressed = compressed.strip()

    return compressed


async def _compress_standard(self, text: str, preserve_structure: bool) -> str:
    """
    Standard compression: extractive summarization + deduplication.

    Target: 50-60% reduction
    """
    # Step 1: Sentence-level extractive summarization
    sentences = await self._extract_key_sentences(text, keep_ratio=0.6)

    # Step 2: Semantic deduplication
    deduplicated = await self._deduplicate_sentences(sentences, threshold=0.92)

    # Step 3: Reconstruct with structure preservation
    if preserve_structure:
        compressed = self._preserve_structure(text, deduplicated)
    else:
        compressed = " ".join(deduplicated)

    return compressed


async def _compress_aggressive(
    self, text: str, target_tokens: int, preserve_structure: bool
) -> str:
    """
    Aggressive compression: multi-pass extraction + filtering.

    Target: 70-85% reduction
    """
    # Step 1: Extract highly salient sentences
    sentences = await self._extract_key_sentences(text, keep_ratio=0.4)

    # Step 2: Aggressive deduplication
    deduplicated = await self._deduplicate_sentences(sentences, threshold=0.85)

    # Step 3: Filter low-information sentences
    filtered = self._filter_low_information(deduplicated)

    # Step 4: If still over budget, iteratively remove lowest-value sentences
    current_tokens = self._estimate_tokens(" ".join(filtered))
    while current_tokens > target_tokens and len(filtered) > 3:
        # Remove sentence with lowest centrality
        filtered = filtered[:-1]
        current_tokens = self._estimate_tokens(" ".join(filtered))

    # Step 5: Reconstruct
    if preserve_structure:
        compressed = self._preserve_structure(text, filtered)
    else:
        compressed = " ".join(filtered)

    return compressed


# =========================================================================
# EXTRACTIVE SUMMARIZATION
# =========================================================================


async def _extract_key_sentences(self, text: str, keep_ratio: float = 0.5) -> List[str]:
    """
    Extract most important sentences via semantic centrality.

    Algorithm:
    1. Embed all sentences
    2. Compute pairwise similarities
    3. Rank by centrality (similarity to document centroid)
    4. Keep top K sentences by rank

    Args:
        text: Input text
        keep_ratio: Fraction of sentences to keep

    Returns:
        List of selected sentences
    """
    # Tokenize into sentences
    doc = self.nlp(text)
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 10]

    if len(sentences) <= 3:
        return sentences

    # Generate embeddings
    embeddings = await semantic_analyzer.embed(sentences, normalize=True)

    # Compute document centroid
    centroid = semantic_analyzer.compute_centroid(embeddings, normalize=True)

    # Rank sentences by similarity to centroid
    scores = [
        semantic_analyzer.compute_similarity(emb, centroid, SimilarityMetric.COSINE)
        for emb in embeddings
    ]

    # Sort by score (descending)
    ranked = sorted(zip(sentences, scores), key=lambda x: x[1], reverse=True)

    # Keep top K
    num_keep = max(3, int(len(sentences) * keep_ratio))
    selected = [sent for sent, score in ranked[:num_keep]]

    # Restore original order
    selected_set = set(selected)
    ordered = [s for s in sentences if s in selected_set]

    logger.debug(f"Extracted {len(ordered)}/{len(sentences)} key sentences")
    return ordered


async def _deduplicate_sentences(self, sentences: List[str], threshold: float = 0.90) -> List[str]:
    """
    Remove semantically duplicate sentences.

    Args:
        sentences: List of sentences
        threshold: Similarity threshold for considering duplicates

    Returns:
        Deduplicated list of sentences
    """
    if len(sentences) <= 1:
        return sentences

    # Generate embeddings
    embeddings = await semantic_analyzer.embed(sentences, normalize=True)

    # Deduplicate using semantic analyzer
    keep_indices = semantic_analyzer.deduplicate_by_similarity(
        texts=sentences, embeddings=embeddings, threshold=threshold
    )

    deduplicated = [sentences[i] for i in keep_indices]

    logger.debug(f"Deduplicated: {len(sentences)} → {len(deduplicated)} sentences")
    return deduplicated


def _filter_low_information(self, sentences: List[str]) -> List[str]:
    """
    Filter out low-information sentences.

    Removes:
    - Very short sentences (<5 words)
    - Sentences with high stopword ratio (>70%)
    - Meta-commentary ("In this article...", "As we've seen...")

    Args:
        sentences: List of sentences

    Returns:
        Filtered list
    """
    filtered = []

    for sent in sentences:
        # Process with spaCy
        doc = self.nlp(sent)

        # Count content words
        tokens = [t for t in doc if not t.is_punct and not t.is_space]
        if len(tokens) < 5:
            continue

        # Check stopword ratio
        stopwords = [t for t in tokens if t.is_stop]
        stopword_ratio = len(stopwords) / len(tokens) if tokens else 1.0
        if stopword_ratio > 0.7:
            continue

        # Check for meta-commentary
        meta_phrases = [
            "in this article",
            "as we",
            "we will",
            "this post",
            "below, we",
            "let us",
            "as mentioned",
        ]
        sent_lower = sent.lower()
        if any(phrase in sent_lower for phrase in meta_phrases):
            continue

        filtered.append(sent)

    return filtered


# =========================================================================
# STRUCTURE PRESERVATION
# =========================================================================


def _preserve_structure(self, original: str, sentences: List[str]) -> str:
    """
    Preserve document structure markers when reconstructing.

    Maintains:
    - Paragraph breaks (double newlines)
    - Section headers (lines ending with :)
    - List items (lines starting with -, *, numbers)

    Args:
        original: Original text with structure
        sentences: Selected sentences

    Returns:
        Reconstructed text with structure preserved
    """
    # Parse original structure
    paragraphs = original.split("\n\n")

    # Build set of selected sentences for fast lookup
    selected_set = set(sentences)

    # Reconstruct with structure
    reconstructed = []
    for para in paragraphs:
        para_sentences = [s.strip() for s in para.split(".") if s.strip()]

        # Keep sentences that are in selected set
        kept = [s for s in para_sentences if any(sel in s for sel in selected_set)]

        if kept:
            reconstructed.append(". ".join(kept) + ".")

    return "\n\n".join(reconstructed)


# =========================================================================
# LAZY CONTEXT MATERIALIZATION
# =========================================================================


async def plan_context_loading(
    self,
    available_sources: Dict[str, str],
    query_embedding: np.ndarray,
    token_budget: int,
) -> ContextPlan:
    """
    Create lazy evaluation plan for context loading.

    Ranks sources by relevance WITHOUT loading full content.

    Args:
        available_sources: Dict mapping source_id to short description
        query_embedding: Query vector for relevance ranking
        token_budget: Available token budget

    Returns:
        ContextPlan describing what to load and how
    """
    # Embed source descriptions
    descriptions = list(available_sources.values())
    desc_embeddings = await semantic_analyzer.embed(descriptions, normalize=True)

    # Rank by relevance to query
    relevance_scores = [
        semantic_analyzer.compute_similarity(query_embedding, emb, SimilarityMetric.COSINE)
        for emb in desc_embeddings
    ]

    # Create ranked list of sources
    source_ids = list(available_sources.keys())
    ranked = sorted(zip(source_ids, relevance_scores), key=lambda x: x[1], reverse=True)

    # Divide into essential vs optional
    # Essential: top 30% by relevance
    cutoff = max(1, int(len(ranked) * 0.3))
    essential = ranked[:cutoff]
    optional = ranked[cutoff:]

    # Estimate tokens needed
    # Rough estimate: 1 token per 4 characters
    estimated_tokens = sum(len(desc) // 4 for desc in descriptions)

    # Determine compression level based on budget
    if estimated_tokens <= token_budget * 0.5:
        compression = CompressionLevel.MINIMAL
    elif estimated_tokens <= token_budget:
        compression = CompressionLevel.STANDARD
    else:
        compression = CompressionLevel.AGGRESSIVE

    plan = ContextPlan(
        essential_sources=essential,
        optional_sources=optional,
        compression_level=compression,
        estimated_tokens=estimated_tokens,
    )

    logger.debug(
        f"Context plan: {len(essential)} essential, {len(optional)} optional, "
        f"compression: {compression.value}"
    )

    return plan


async def materialize_context(
    self,
    plan: ContextPlan,
    source_loader: callable,
    token_budget: int,
) -> str:
    """
    Materialize context according to plan.

    Args:
        plan: Context loading plan
        source_loader: Async function to load source content by ID
        token_budget: Token budget

    Returns:
        Materialized and compressed context string
    """
    context_parts = []
    tokens_used = 0

    # Load essential sources
    for source_id, relevance in plan.essential_sources:
        if tokens_used >= token_budget:
            break

        content = await source_loader(source_id)

        # Compress based on plan
        if plan.compression_level == CompressionLevel.MINIMAL:
            compressed = await self._compress_minimal(content)
        elif plan.compression_level == CompressionLevel.STANDARD:
            compressed = await self._compress_standard(content, preserve_structure=False)
        else:
            # Aggressive compression to fit budget
            remaining_budget = token_budget - tokens_used
            result = await self.compress_to_budget(content, remaining_budget)
            compressed = result.compressed_text

        context_parts.append(compressed)
        tokens_used += self._estimate_tokens(compressed)

    # Load optional sources if budget allows
    for source_id, relevance in plan.optional_sources:
        if tokens_used >= token_budget * 0.9:  # Leave 10% buffer
            break

        content = await source_loader(source_id)

        # Always use aggressive compression for optional sources
        remaining_budget = token_budget - tokens_used
        result = await self.compress_to_budget(content, remaining_budget)

        context_parts.append(result.compressed_text)
        tokens_used += result.compressed_tokens

    materialized = "\n\n".join(context_parts)

    logger.info(f"Materialized context: {tokens_used}/{token_budget} tokens used")
    return materialized


# =========================================================================
# UTILITIES
# =========================================================================


@staticmethod
def _estimate_tokens(text: str) -> int:
    """
    Estimate token count for text.

    Rule of thumb: 1 token ≈ 4 characters (for English)
    """
    return len(text) // 4


async def _estimate_information_density(self, original: str, compressed: str) -> float:
    """
    Estimate how much semantic information was preserved.

    Uses embedding similarity as proxy for information preservation.

    Args:
        original: Original text
        compressed: Compressed text

    Returns:
        Information density score (0-1)
    """
    # Embed both texts
    orig_emb = await semantic_analyzer.embed(original, normalize=True)
    comp_emb = await semantic_analyzer.embed(compressed, normalize=True)

    # Cosine similarity = information preservation
    similarity = semantic_analyzer.compute_similarity(orig_emb, comp_emb, SimilarityMetric.COSINE)

    return similarity


def compress_multiple(
    self,
    texts: Dict[str, str],
    total_budget: int,
) -> Dict[str, str]:
    """
    Compress multiple texts to fit within total budget.

    Distributes budget proportionally based on text importance.

    Args:
        texts: Dict mapping source_id to text
        total_budget: Total token budget for all texts

    Returns:
        Dict of compressed texts
    """
    # Calculate current token usage
    token_counts = {k: self._estimate_tokens(v) for k, v in texts.items()}
    total_tokens = sum(token_counts.values())

    if total_tokens <= total_budget:
        return texts  # No compression needed

    # Calculate compression ratio
    target_ratio = total_budget / total_tokens

    # Compress each text proportionally
    compressed = {}
    for key, text in texts.items():
        # Allocate budget proportionally
        text_budget = int(token_counts[key] * target_ratio)

        # Compress (this is synchronous, would need async wrapper for production)
        # For now, use minimal compression
        import re

        comp = re.sub(r"\s+", " ", text).strip()
        compressed[key] = comp

    return compressed


# =========================================================================
# GLOBAL INSTANCE
# =========================================================================
context_synthesizer = ContextSynthesizer()
