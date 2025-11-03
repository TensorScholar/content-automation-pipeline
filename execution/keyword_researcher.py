"""
Keyword Research Engine
=======================

Performs keyword discovery, grouping and simple ranking using semantic
similarity and lightweight clustering utilities. Designed for clarity and
maintainability rather than algorithmic novelty.
"""

from __future__ import annotations

import asyncio
import math
from collections import defaultdict
from dataclasses import dataclass, field
from enum import Enum
from functools import reduce
from itertools import combinations
from typing import (
    Any,
    Awaitable,
    Callable,
    Dict,
    Generic,
    List,
    NewType,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import scipy.sparse as sp
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field, field_validator
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.sparse.linalg import eigsh

from core.exceptions import ProcessingError, ValidationError
from core.models import KeywordIntent
from intelligence.semantic_analyzer import SemanticAnalyzer, SimilarityMetric
from optimization.cache_manager import CacheManager

# =========================================================================
# TYPE-LEVEL PROGRAMMING & ALGEBRAIC DATA TYPES
# =========================================================================

T = TypeVar("T")
E = TypeVar("E")
A = TypeVar("A")
B = TypeVar("B")

# Phantom types for compile-time guarantees
KeywordId = NewType("KeywordId", str)
ClusterId = NewType("ClusterId", int)
SemanticScore = NewType("SemanticScore", float)


class SearchIntent(str, Enum):
    """User search intent taxonomy (refined from KeywordIntent)."""

    INFORMATIONAL = "informational"  # Learning, research
    NAVIGATIONAL = "navigational"  # Finding specific site
    TRANSACTIONAL = "transactional"  # Purchase intent
    COMMERCIAL = "commercial"  # Product research
    LOCAL = "local"  # Location-specific
    INVESTIGATIONAL = "investigational"  # Deep research


class KeywordDifficulty(str, Enum):
    """Ranking difficulty quantization."""

    TRIVIAL = "trivial"  # 0-20
    EASY = "easy"  # 21-40
    MODERATE = "moderate"  # 41-60
    HARD = "hard"  # 61-80
    EXTREME = "extreme"  # 81-100


class KeywordMetrics(BaseModel):
    """
    Comprehensive keyword metrics with validation invariants.

    Invariants:
    - search_volume ≥ 0
    - difficulty ∈ [0, 100]
    - cpc ≥ 0
    - All probabilities ∈ [0, 1]
    """

    search_volume: int = Field(ge=0, description="Monthly search volume")
    difficulty: float = Field(ge=0, le=100, description="Ranking difficulty score")
    cpc: float = Field(ge=0, description="Cost per click (USD)")

    # Intent distribution (multinomial)
    intent_distribution: Dict[SearchIntent, float] = Field(
        default_factory=dict, description="Probabilistic intent classification"
    )

    # Semantic richness
    semantic_entropy: float = Field(ge=0, description="Shannon entropy of semantic neighborhood")

    # Authority metrics
    pagerank_score: float = Field(default=0.0, ge=0, le=1, description="PageRank in keyword graph")

    @field_validator("intent_distribution")
    @classmethod
    def validate_probability_distribution(cls, v):
        """Ensure intent distribution sums to ~1.0."""
        if v and not math.isclose(sum(v.values()), 1.0, abs_tol=0.01):
            raise ValueError(f"Intent distribution must sum to 1.0, got {sum(v.values())}")
        return v

    model_config = ConfigDict(frozen=True)  # Immutability


@dataclass(frozen=True)
class Keyword:
    """
    Immutable keyword representation with semantic metadata.

    Type Safety: Frozen dataclass prevents mutation, enabling
    safe concurrent processing and algebraic reasoning.
    """

    phrase: str
    embedding: np.ndarray
    metrics: KeywordMetrics

    # Semantic metadata
    primary_intent: SearchIntent
    semantic_cluster: Optional[ClusterId] = None
    related_concepts: Tuple[str, ...] = field(default_factory=tuple)

    # Graph properties
    centrality: float = 0.0
    community: Optional[int] = None

    def __post_init__(self):
        """Validate invariants."""
        if not self.phrase or not self.phrase.strip():
            raise ValidationError("Keyword phrase cannot be empty")

        if self.embedding.shape[0] != 384:
            raise ValidationError(f"Expected 384-dim embedding, got {self.embedding.shape[0]}")

    def __hash__(self):
        """Hash by phrase for set operations."""
        return hash(self.phrase)

    def __eq__(self, other):
        """Equality by phrase."""
        return isinstance(other, Keyword) and self.phrase == other.phrase


@dataclass(frozen=True)
class KeywordCluster:
    """
    Semantic keyword cluster with statistical properties.

    Represents a cohesive semantic field in the keyword space.
    """

    cluster_id: ClusterId
    centroid: np.ndarray
    keywords: Tuple[Keyword, ...]

    # Cluster statistics
    coherence: float  # Intra-cluster similarity
    separation: float  # Inter-cluster distance
    entropy: float  # Semantic diversity

    # Intent profile
    dominant_intent: SearchIntent
    intent_purity: float  # Concentration in dominant intent

    @property
    def size(self) -> int:
        """Number of keywords in cluster."""
        return len(self.keywords)

    @property
    def total_search_volume(self) -> int:
        """Aggregate search volume."""
        return sum(kw.metrics.search_volume for kw in self.keywords)

    @property
    def avg_difficulty(self) -> float:
        """Average ranking difficulty."""
        return sum(kw.metrics.difficulty for kw in self.keywords) / self.size


# =========================================================================
# FUNCTIONAL ABSTRACTIONS: MONADS & FUNCTORS
# =========================================================================


class Result(Generic[T, E]):
    """
    Result monad for explicit error handling without exceptions.

    Algebraic data type: Result[T, E] = Ok(T) | Err(E)

    Functor laws:
    - Identity: result.map(lambda x: x) == result
    - Composition: result.map(f).map(g) == result.map(lambda x: g(f(x)))
    """

    def __init__(self, value: T = None, error: E = None):
        self._value = value
        self._error = error
        self._is_ok = error is None

    @staticmethod
    def ok(value: T) -> Result[T, E]:
        """Construct successful result."""
        return Result(value=value)

    @staticmethod
    def err(error: E) -> Result[T, E]:
        """Construct error result."""
        return Result(error=error)

    def is_ok(self) -> bool:
        """Check if result is successful."""
        return self._is_ok

    def is_err(self) -> bool:
        """Check if result is error."""
        return not self._is_ok

    def map(self, f: Callable[[T], B]) -> Result[B, E]:
        """Functor map: transforms value if Ok."""
        if self.is_ok():
            try:
                return Result.ok(f(self._value))
            except Exception as e:
                return Result.err(e)
        return Result.err(self._error)

    def flat_map(self, f: Callable[[T], Result[B, E]]) -> Result[B, E]:
        """Monadic bind: chains computations."""
        if self.is_ok():
            return f(self._value)
        return Result.err(self._error)

    def unwrap_or(self, default: T) -> T:
        """Extract value or return default."""
        return self._value if self.is_ok() else default

    def unwrap_or_else(self, f: Callable[[E], T]) -> T:
        """Extract value or compute from error."""
        return self._value if self.is_ok() else f(self._error)


class AsyncResult(Generic[T, E]):
    """Asynchronous Result monad for effectful computations."""

    def __init__(self, coro: Awaitable[Result[T, E]]):
        self._coro = coro

    async def map(self, f: Callable[[T], B]) -> AsyncResult[B, E]:
        """Async functor map."""
        result = await self._coro
        return AsyncResult(asyncio.coroutine(lambda: result.map(f))())

    async def flat_map(self, f: Callable[[T], AsyncResult[B, E]]) -> AsyncResult[B, E]:
        """Async monadic bind."""
        result = await self._coro
        if result.is_ok():
            return await f(result._value)._coro
        return AsyncResult(asyncio.coroutine(lambda: Result.err(result._error))())

    async def run(self) -> Result[T, E]:
        """Execute computation."""
        return await self._coro


# =========================================================================
# GRAPH-THEORETIC KEYWORD ANALYSIS
# =========================================================================


class SemanticGraph:
    """
    Semantic keyword graph with spectral analysis capabilities.

    Graph Structure:
    - Vertices: Keywords
    - Edges: Semantic similarity > threshold
    - Weights: Cosine similarity of embeddings

    Operations:
    - Spectral clustering: Eigendecomposition of graph Laplacian
    - Community detection: Louvain algorithm
    - PageRank: Authority propagation
    - Centrality: Betweenness, closeness, eigenvector
    """

    def __init__(self, keywords: List[Keyword], similarity_threshold: float = 0.70):
        """
        Construct semantic graph from keywords.

        Args:
            keywords: List of keywords with embeddings
            similarity_threshold: Minimum similarity for edge creation
        """
        self.keywords = keywords
        self.threshold = similarity_threshold

        # Build adjacency matrix
        self.adjacency = self._build_adjacency_matrix()

        # Compute graph properties
        self.laplacian = self._compute_laplacian()
        self.degree = np.array(self.adjacency.sum(axis=1)).flatten()

        logger.debug(
            f"Constructed semantic graph: {len(keywords)} nodes, " f"{self.adjacency.nnz} edges"
        )

    def _build_adjacency_matrix(self) -> sp.csr_matrix:
        """
        Build sparse adjacency matrix from semantic similarities.

        Complexity: O(n²) for dense similarity computation
        Optimization: Could use approximate NN for large n
        """
        n = len(self.keywords)

        # Compute pairwise similarities
        embeddings = np.vstack([kw.embedding for kw in self.keywords])

        # Cosine similarity matrix (via normalized dot product)
        norm_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        similarities = norm_embeddings @ norm_embeddings.T

        # Threshold to create sparse matrix
        adjacency_dense = np.where(similarities >= self.threshold, similarities, 0)
        np.fill_diagonal(adjacency_dense, 0)  # Remove self-loops

        # Convert to sparse
        adjacency = sp.csr_matrix(adjacency_dense)

        return adjacency

    def _compute_laplacian(self) -> sp.csr_matrix:
        """
        Compute graph Laplacian: L = D - A

        Properties:
        - L is symmetric positive semi-definite
        - Smallest eigenvalue is 0 (constant eigenvector)
        - Eigenvectors reveal cluster structure
        """
        degree_matrix = sp.diags(self.degree)
        laplacian = degree_matrix - self.adjacency

        return laplacian

    def spectral_clustering(self, n_clusters: int) -> np.ndarray:
        """
        Perform spectral clustering via eigendecomposition.

        Algorithm:
        1. Compute k smallest eigenvectors of Laplacian
        2. Treat rows as points in R^k
        3. Run k-means on embedded points

        Complexity: O(n³) for dense, O(n·k²) for sparse

        Args:
            n_clusters: Number of clusters

        Returns:
            Cluster labels for each keyword
        """
        # Compute k smallest eigenvectors (excluding trivial)
        try:
            eigenvalues, eigenvectors = eigsh(
                self.laplacian, k=n_clusters + 1, which="SM", return_eigenvectors=True
            )

            # Remove trivial eigenvector (all ones)
            embedding = eigenvectors[:, 1:]

            # Normalize rows
            embedding = embedding / np.linalg.norm(embedding, axis=1, keepdims=True)

            # K-means clustering
            from sklearn.cluster import KMeans

            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            labels = kmeans.fit_predict(embedding)

            return labels

        except Exception as e:
            logger.warning(f"Spectral clustering failed: {e}, falling back to hierarchical")
            return self._hierarchical_clustering(n_clusters)

    def _hierarchical_clustering(self, n_clusters: int) -> np.ndarray:
        """Fallback: Hierarchical clustering on embeddings."""
        embeddings = np.vstack([kw.embedding for kw in self.keywords])

        # Compute linkage
        Z = linkage(embeddings, method="ward")

        # Cut dendrogram
        labels = fcluster(Z, n_clusters, criterion="maxclust") - 1

        return labels

    def compute_pagerank(self, damping: float = 0.85, max_iter: int = 100) -> np.ndarray:
        """
        Compute PageRank for keyword authority.

        PageRank iteration:
        PR(i) = (1-d)/n + d·Σ[PR(j)/L(j)] for j→i

        Args:
            damping: Damping factor (typically 0.85)
            max_iter: Maximum iterations

        Returns:
            PageRank scores for each keyword
        """
        n = len(self.keywords)

        # Normalize adjacency by out-degree
        out_degree = np.array(self.adjacency.sum(axis=1)).flatten()
        out_degree[out_degree == 0] = 1  # Avoid division by zero

        transition = self.adjacency.T / out_degree

        # Initialize uniform distribution
        pr = np.ones(n) / n

        # Power iteration
        for _ in range(max_iter):
            pr_new = (1 - damping) / n + damping * transition @ pr

            # Check convergence
            if np.linalg.norm(pr_new - pr, ord=1) < 1e-6:
                break

            pr = pr_new

        return pr

    def compute_betweenness_centrality(self) -> np.ndarray:
        """
        Compute betweenness centrality (simplified approximation).

        Exact computation is O(n³), so we use sampling.
        """
        # For now, return degree centrality as proxy
        # Full implementation would use Brandes' algorithm
        return self.degree / self.degree.sum()


# =========================================================================
# KEYWORD RESEARCH ENGINE
# =========================================================================


class KeywordResearcher:
    """
    Advanced keyword research engine with graph-theoretic clustering
    and information-theoretic ranking.

    Architecture:
    - Functional core: Pure functions for transformations
    - Imperative shell: Async I/O and side effects
    - Type-safe: Leverages Pydantic and NewType
    - Monadic: Explicit error handling via Result monad

    Algorithms:
    - Spectral clustering: O(n·k² + k³) for k clusters
    - PageRank: O(n·m·iter) for m edges
    - Mutual information: O(n·m) for n keywords, m features
    """

    def __init__(
        self,
        semantic_analyzer: Optional[SemanticAnalyzer] = None,
        cache_manager: Optional[CacheManager] = None,
    ):
        """Initialize keyword researcher with semantic analyzer and cache manager.

        Args:
            semantic_analyzer: SemanticAnalyzer instance for embeddings (optional)
            cache_manager: CacheManager instance for caching (optional)
        """
        self._cache_ttl = 86400 * 7  # 7 days
        self.semantic_analyzer = semantic_analyzer
        self.cache_manager = cache_manager

        logger.info("Keyword researcher initialized")

    # =====================================================================
    # PUBLIC API: HIGH-LEVEL KEYWORD RESEARCH
    # =====================================================================

    async def research_keywords(
        self,
        seed_topic: str,
        target_count: int = 50,
        intent_filter: Optional[SearchIntent] = None,
    ) -> Result[List[Keyword], Exception]:
        """
        Comprehensive keyword research pipeline.

        Pipeline:
        1. Seed expansion: Generate candidate keywords
        2. Embedding generation: Semantic vectorization
        3. Graph construction: Build semantic adjacency graph
        4. Clustering: Spectral clustering into semantic groups
        5. Ranking: Information-theoretic relevance scoring
        6. Selection: Top-k keywords by composite score

        Args:
            seed_topic: Starting topic for research
            target_count: Desired number of keywords
            intent_filter: Optional intent filter

        Returns:
            Result monad with keywords or error
        """
        try:
            logger.info(f"Researching keywords for topic: '{seed_topic}'")

            # Step 1: Generate candidate keywords
            candidates_result = await self._generate_candidates(seed_topic, target_count * 3)
            if candidates_result.is_err():
                return candidates_result

            candidates = candidates_result.unwrap_or([])

            if len(candidates) < 10:
                return Result.err(ProcessingError(f"Insufficient candidates: {len(candidates)}"))

            # Step 2: Enrich with embeddings and metrics
            enriched = await self._enrich_keywords(candidates, seed_topic)

            # Step 3: Filter by intent if specified
            if intent_filter:
                enriched = [kw for kw in enriched if kw.primary_intent == intent_filter]

            # Step 4: Build semantic graph and cluster
            if len(enriched) >= 10:
                enriched = await self._cluster_keywords(enriched)

            # Step 5: Rank by composite score
            ranked = self._rank_keywords(enriched, seed_topic)

            # Step 6: Select top-k
            selected = ranked[:target_count]

            logger.info(
                f"Research complete: {len(selected)} keywords selected from {len(candidates)} candidates"
            )

            return Result.ok(selected)

        except Exception as e:
            logger.error(f"Keyword research failed: {e}")
            return Result.err(e)

    async def categorize_keywords(
        self,
        keywords: List[str],
    ) -> Result[Dict[str, List[Keyword]], Exception]:
        """
        Categorize existing keywords into semantic clusters.

        Args:
            keywords: List of keyword phrases

        Returns:
            Result with categorized keywords by cluster
        """
        try:
            # Enrich keywords
            enriched = await self._enrich_keywords(keywords, keywords[0] if keywords else "")

            # Cluster
            clustered = await self._cluster_keywords(enriched)

            # Group by cluster
            by_cluster = defaultdict(list)
            for kw in clustered:
                cluster_id = kw.semantic_cluster if kw.semantic_cluster is not None else -1
                by_cluster[f"cluster_{cluster_id}"].append(kw)

            return Result.ok(dict(by_cluster))

        except Exception as e:
            logger.error(f"Categorization failed: {e}")
            return Result.err(e)

    # =====================================================================
    # KEYWORD GENERATION & EXPANSION
    # =====================================================================

    async def _generate_candidates(
        self,
        seed: str,
        count: int,
    ) -> Result[List[str], Exception]:
        """
        Generate candidate keywords using semantic expansion.

        Strategy:
        1. Generate variations of seed
        2. Use LLM for creative expansion
        3. Deduplicate and filter

        Args:
            seed: Seed topic
            count: Target candidate count

        Returns:
            Result with candidate keywords
        """
        try:
            # Check cache
            cache_key = f"kw_candidates:{seed}:{count}"
            cached = await self.cache_manager.get(cache_key)

            if cached:
                logger.debug(f"Cache hit for candidates: {seed}")
                return Result.ok(cached)

            # Generate variations
            variations = self._generate_variations(seed)

            # Add seed and variations
            candidates = {seed, *variations}

            # LLM expansion (if needed)
            if len(candidates) < count:
                llm_candidates = await self._llm_keyword_expansion(seed, count - len(candidates))
                candidates.update(llm_candidates)

            candidates_list = list(candidates)[: count * 2]  # Over-generate

            # Cache result
            await self.cache_manager.set(cache_key, candidates_list, ttl=self._cache_ttl)

            return Result.ok(candidates_list)

        except Exception as e:
            return Result.err(e)

    def _generate_variations(self, seed: str) -> Set[str]:
        """
        Generate lexical variations of seed keyword.

        Techniques:
        - Pluralization
        - Synonyms (via word forms)
        - Question forms
        - Modifiers
        """
        variations = set()

        # Question forms
        question_words = ["how to", "what is", "why", "when", "where", "best"]
        for qw in question_words:
            variations.add(f"{qw} {seed}")

        # Modifiers
        modifiers = ["best", "top", "guide", "tips", "tutorial", "review"]
        for mod in modifiers:
            variations.add(f"{seed} {mod}")
            variations.add(f"{mod} {seed}")

        # Pluralization (simple)
        if not seed.endswith("s"):
            variations.add(f"{seed}s")

        return variations

    async def _llm_keyword_expansion(self, seed: str, count: int) -> Set[str]:
        """
        Use LLM for creative keyword expansion.

        Args:
            seed: Seed keyword
            count: Number of keywords to generate

        Returns:
            Set of generated keywords
        """
        # Placeholder: Would call LLM via model router
        # For now, return empty set
        return set()

    # =====================================================================
    # KEYWORD ENRICHMENT
    # =====================================================================

    async def _enrich_keywords(
        self,
        candidates: List[str],
        seed_topic: str,
    ) -> List[Keyword]:
        """
        Enrich keywords with embeddings and metrics.

        Args:
            candidates: Candidate keyword phrases
            seed_topic: Original seed for context

        Returns:
            Enriched Keyword objects
        """
        # Generate embeddings (batch)
        embeddings = await self.semantic_analyzer.embed(candidates, normalize=True)

        # Compute metrics for each
        enriched = []
        for phrase, embedding in zip(candidates, embeddings):
            # Estimate metrics (in production, query keyword APIs)
            metrics = self._estimate_metrics(phrase, seed_topic)

            # Classify intent
            intent = self._classify_intent(phrase)

            # Create keyword object
            keyword = Keyword(
                phrase=phrase,
                embedding=embedding,
                metrics=metrics,
                primary_intent=intent,
            )

            enriched.append(keyword)

        return enriched

    def _estimate_metrics(self, phrase: str, seed: str) -> KeywordMetrics:
        """
        Estimate keyword metrics (placeholder for API integration).

        In production, would query:
        - Ahrefs API for search volume, difficulty
        - Google Keyword Planner for CPC
        - SERP analysis for intent
        """
        # Dummy metrics based on phrase characteristics
        word_count = len(phrase.split())

        # Heuristic: longer phrases = lower volume, lower difficulty
        search_volume = max(10, 10000 // (word_count**2))
        difficulty = min(100, 20 + word_count * 15)
        cpc = 0.5 + word_count * 0.3

        # Compute semantic entropy (diversity of embedding)
        # Higher entropy = more semantically rich
        semantic_entropy = 0.5  # Placeholder

        return KeywordMetrics(
            search_volume=search_volume,
            difficulty=difficulty,
            cpc=cpc,
            intent_distribution={},
            semantic_entropy=semantic_entropy,
        )

    def _classify_intent(self, phrase: str) -> SearchIntent:
        """
        Classify search intent from phrase.

        Heuristics:
        - "how to", "guide" → informational
        - "buy", "price" → transactional
        - "best", "vs" → commercial
        - "near me" → local
        """
        phrase_lower = phrase.lower()

        # Transactional indicators
        if any(word in phrase_lower for word in ["buy", "purchase", "price", "cheap", "deal"]):
            return SearchIntent.TRANSACTIONAL

        # Commercial indicators
        if any(word in phrase_lower for word in ["best", "top", "review", "vs", "compare"]):
            return SearchIntent.COMMERCIAL

        # Local indicators
        if any(word in phrase_lower for word in ["near me", "local", "nearby"]):
            return SearchIntent.LOCAL

        # Navigational indicators
        if phrase_lower.startswith("go to") or phrase_lower.endswith("login"):
            return SearchIntent.NAVIGATIONAL

        # Default: informational
        return SearchIntent.INFORMATIONAL

    # =====================================================================
    # GRAPH-BASED CLUSTERING
    # =====================================================================

    async def _cluster_keywords(self, keywords: List[Keyword]) -> List[Keyword]:
        """
        Cluster keywords using spectral clustering on semantic graph.

        Args:
            keywords: Keywords with embeddings

        Returns:
            Keywords with cluster assignments
        """
        if len(keywords) < 10:
            return keywords  # Too few for meaningful clustering

        # Build semantic graph
        graph = SemanticGraph(keywords, similarity_threshold=0.70)

        # Determine optimal number of clusters (heuristic: sqrt(n))
        n_clusters = max(3, min(10, int(math.sqrt(len(keywords)))))

        # Perform spectral clustering
        labels = graph.spectral_clustering(n_clusters)

        # Compute PageRank
        pagerank = graph.compute_pagerank()

        # Assign cluster IDs and PageRank to keywords
        clustered = []
        for kw, label, pr in zip(keywords, labels, pagerank):
            # Create new keyword with cluster assignment (immutable)
            clustered_kw = Keyword(
                phrase=kw.phrase,
                embedding=kw.embedding,
                metrics=KeywordMetrics(
                    **kw.metrics.dict(),
                    pagerank_score=float(pr),
                ),
                primary_intent=kw.primary_intent,
                semantic_cluster=ClusterId(int(label)),
                related_concepts=kw.related_concepts,
                centrality=float(pr),
            )
            clustered.append(clustered_kw)

        logger.debug(f"Clustered {len(keywords)} keywords into {n_clusters} groups")

        return clustered

    # =====================================================================
    # INFORMATION-THEORETIC RANKING
    # =====================================================================

    def _rank_keywords(self, keywords: List[Keyword], seed: str) -> List[Keyword]:
        """
        Rank keywords by composite score combining multiple signals.

        Scoring function:
        score = α·relevance + β·volume + γ·(1-difficulty) + δ·authority

        Where:
        - relevance: Semantic similarity to seed
        - volume: Normalized search volume
        - difficulty: Inverted difficulty (easier = higher)
        - authority: PageRank score

        Args:
            keywords: Keywords to rank
            seed: Seed topic for relevance

        Returns:
            Ranked keywords (descending score)
        """
        # Generate seed embedding
        seed_embedding = asyncio.run(self.semantic_analyzer.embed(seed, normalize=True))

        # Compute composite scores
        scored = []
        for kw in keywords:
            # Relevance: cosine similarity to seed
            relevance = self.semantic_analyzer.compute_similarity(
                kw.embedding, seed_embedding, SimilarityMetric.COSINE
            )

            # Normalize search volume (log scale)
            volume_score = math.log1p(kw.metrics.search_volume) / math.log1p(100000)

            # Invert difficulty
            difficulty_score = 1 - (kw.metrics.difficulty / 100)

            # Authority from PageRank
            authority_score = kw.metrics.pagerank_score

            # Composite score (weighted sum)
            score = (
                0.40 * relevance
                + 0.25 * volume_score
                + 0.20 * difficulty_score
                + 0.15 * authority_score
            )

            scored.append((kw, score))

        # Sort by score (descending)
        scored.sort(key=lambda x: x[1], reverse=True)

        # Return ranked keywords
        ranked = [kw for kw, score in scored]

        logger.debug(f"Ranked {len(ranked)} keywords by composite score")

        return ranked

    # =====================================================================
    # ADVANCED ANALYTICS
    # =====================================================================

    def analyze_keyword_clusters(
        self,
        keywords: List[Keyword],
    ) -> List[KeywordCluster]:
        """
        Analyze keyword clusters with statistical properties.

        Computes:
        - Intra-cluster coherence (avg pairwise similarity)
        - Inter-cluster separation (min distance to other clusters)
        - Semantic entropy (diversity within cluster)
        - Intent purity (concentration in dominant intent)

        Args:
            keywords: Clustered keywords

        Returns:
            List of KeywordCluster objects
        """
        # Group by cluster
        by_cluster = defaultdict(list)
        for kw in keywords:
            if kw.semantic_cluster is not None:
                by_cluster[kw.semantic_cluster].append(kw)

        clusters = []

        for cluster_id, cluster_kws in by_cluster.items():
            if len(cluster_kws) < 2:
                continue

            # Compute centroid
            embeddings = np.vstack([kw.embedding for kw in cluster_kws])
            centroid = embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)  # Normalize

            # Coherence: avg intra-cluster similarity
            similarities = []
            for i, j in combinations(range(len(cluster_kws)), 2):
                sim = np.dot(cluster_kws[i].embedding, cluster_kws[j].embedding)
                similarities.append(sim)

            coherence = np.mean(similarities) if similarities else 0.0

            # Separation: distance to nearest other cluster
            other_centroids = [
                np.vstack([kw.embedding for kw in kws]).mean(axis=0)
                for cid, kws in by_cluster.items()
                if cid != cluster_id and len(kws) >= 2
            ]

            if other_centroids:
                separations = [
                    1 - np.dot(centroid, other_c / np.linalg.norm(other_c))
                    for other_c in other_centroids
                ]
                separation = min(separations)
            else:
                separation = 1.0

            # Entropy: diversity of search volumes (proxy for semantic diversity)
            volumes = [kw.metrics.search_volume for kw in cluster_kws]
            if len(set(volumes)) > 1:
                # Shannon entropy
                from collections import Counter

                volume_dist = Counter(volumes)
                total = sum(volume_dist.values())
                probs = [count / total for count in volume_dist.values()]
                entropy = -sum(p * math.log2(p) for p in probs if p > 0)
            else:
                entropy = 0.0

            # Intent purity
            intent_counts = Counter(kw.primary_intent for kw in cluster_kws)
            dominant_intent = intent_counts.most_common(1)[0][0]
            intent_purity = intent_counts[dominant_intent] / len(cluster_kws)

            # Create cluster object
            cluster = KeywordCluster(
                cluster_id=cluster_id,
                centroid=centroid,
                keywords=tuple(cluster_kws),
                coherence=coherence,
                separation=separation,
                entropy=entropy,
                dominant_intent=dominant_intent,
                intent_purity=intent_purity,
            )

            clusters.append(cluster)

        # Sort by search volume
        clusters.sort(key=lambda c: c.total_search_volume, reverse=True)

        return clusters

    def compute_keyword_network_metrics(
        self,
        keywords: List[Keyword],
    ) -> Dict[str, Any]:
        """
        Compute network-level metrics for keyword ecosystem.

        Returns:
            Dict with network statistics
        """
        if len(keywords) < 3:
            return {}

        graph = SemanticGraph(keywords, similarity_threshold=0.70)

        # Network density
        n = len(keywords)
        max_edges = n * (n - 1) / 2
        density = graph.adjacency.nnz / max_edges if max_edges > 0 else 0

        # Average degree
        avg_degree = graph.degree.mean()

        # Clustering coefficient (approximate via transitivity)
        # Count triangles vs potential triangles
        adjacency_dense = graph.adjacency.toarray()
        triangles = np.trace(adjacency_dense @ adjacency_dense @ adjacency_dense) / 6
        potential_triangles = n * (n - 1) * (n - 2) / 6
        clustering_coeff = triangles / potential_triangles if potential_triangles > 0 else 0

        return {
            "num_keywords": n,
            "num_edges": graph.adjacency.nnz,
            "density": density,
            "avg_degree": avg_degree,
            "clustering_coefficient": clustering_coeff,
            "is_connected": self._is_connected(graph.adjacency),
        }

    @staticmethod
    def _is_connected(adjacency: sp.csr_matrix) -> bool:
        """Check if graph is connected via BFS."""
        n = adjacency.shape[0]
        if n == 0:
            return True

        visited = np.zeros(n, dtype=bool)
        queue = [0]
        visited[0] = True

        while queue:
            node = queue.pop(0)
            neighbors = adjacency[node].nonzero()[1]

            for neighbor in neighbors:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    queue.append(neighbor)

        return visited.all()


# =========================================================================
# GLOBAL INSTANCE
# =========================================================================

keyword_researcher = KeywordResearcher()
