"""
Semantic Analyzer - NLP Intelligence Core
==========================================

Unified semantic intelligence layer providing:
- Embedding generation with automatic caching
- Vector similarity computations (cosine, euclidean)
- Semantic clustering (hierarchical, k-means)
- Text comparison and deduplication
- Batch processing optimizations

Design: Single abstraction layer for all semantic operations,
ensuring consistency and enabling global optimizations.
"""

import hashlib
from enum import Enum
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

from core.exceptions import ValidationError
from infrastructure.redis_client import redis_client


class SimilarityMetric(str, Enum):
    """Supported similarity metrics."""

    COSINE = "cosine"
    EUCLIDEAN = "euclidean"
    DOT_PRODUCT = "dot_product"


class ClusteringMethod(str, Enum):
    """Supported clustering algorithms."""

    KMEANS = "kmeans"
    HIERARCHICAL = "hierarchical"
    DBSCAN = "dbscan"


class SemanticAnalyzer:
    """
    Core semantic intelligence engine.

    Provides high-level semantic operations with automatic optimization:
    - Caching: Embeddings cached in Redis
    - Batch processing: Vectorized operations
    - Normalization: Automatic vector normalization
    """

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize semantic analyzer with embedding model.

        Args:
            model_name: SentenceTransformer model identifier
        """
        try:
            self.model = SentenceTransformer(model_name)
            self.embedding_dim = self.model.get_sentence_embedding_dimension()
            self.cache_enabled = True

            logger.info(f"Semantic analyzer initialized: {model_name} ({self.embedding_dim}d)")

        except Exception as e:
            logger.error(f"Failed to initialize semantic analyzer: {e}")
            raise ValidationError(f"Model initialization failed: {e}")

    # =========================================================================
    # EMBEDDING GENERATION
    # =========================================================================

    async def embed(
        self,
        text: Union[str, List[str]],
        normalize: bool = True,
        use_cache: bool = True,
    ) -> Union[np.ndarray, List[np.ndarray]]:
        """
        Generate semantic embeddings with automatic caching.

        Args:
            text: Single text or list of texts
            normalize: If True, normalize to unit vectors (recommended)
            use_cache: If True, check cache before computing

        Returns:
            Embedding vector(s) as numpy array(s)
        """
        # Handle single text
        if isinstance(text, str):
            return await self._embed_single(text, normalize, use_cache)

        # Handle batch
        return await self._embed_batch(text, normalize, use_cache)

    async def _embed_single(
        self,
        text: str,
        normalize: bool,
        use_cache: bool,
    ) -> np.ndarray:
        """Generate embedding for single text."""
        if not text or not text.strip():
            raise ValidationError("Cannot embed empty text")

        # Check cache
        if use_cache and self.cache_enabled:
            cache_key = self._compute_cache_key(text, normalize)
            cached = await redis_client.get_embedding(cache_key, shape=(self.embedding_dim,))

            if cached is not None:
                logger.debug(f"Cache hit: {cache_key[:16]}...")
                return cached

        # Generate embedding
        embedding = self.model.encode(
            text, convert_to_numpy=True, show_progress_bar=False, normalize_embeddings=normalize
        )

        # Cache for future use
        if use_cache and self.cache_enabled:
            await redis_client.store_embedding(
                key=cache_key, embedding=embedding, ttl=86400 * 90  # 90 days
            )

        return embedding

    async def _embed_batch(
        self,
        texts: List[str],
        normalize: bool,
        use_cache: bool,
    ) -> List[np.ndarray]:
        """Generate embeddings for multiple texts (optimized)."""
        if not texts:
            return []

        # Separate cached and uncached
        results = [None] * len(texts)
        uncached_indices = []
        uncached_texts = []

        if use_cache and self.cache_enabled:
            for idx, text in enumerate(texts):
                cache_key = self._compute_cache_key(text, normalize)
                cached = await redis_client.get_embedding(cache_key, shape=(self.embedding_dim,))

                if cached is not None:
                    results[idx] = cached
                else:
                    uncached_indices.append(idx)
                    uncached_texts.append(text)
        else:
            uncached_indices = list(range(len(texts)))
            uncached_texts = texts

        # Batch compute uncached embeddings
        if uncached_texts:
            logger.debug(f"Computing {len(uncached_texts)} embeddings (batch)")

            embeddings = self.model.encode(
                uncached_texts,
                convert_to_numpy=True,
                show_progress_bar=False,
                normalize_embeddings=normalize,
                batch_size=32,
            )

            # Store in cache and results
            for idx, embedding in zip(uncached_indices, embeddings):
                results[idx] = embedding

                if use_cache and self.cache_enabled:
                    cache_key = self._compute_cache_key(texts[idx], normalize)
                    await redis_client.store_embedding(cache_key, embedding, ttl=86400 * 90)

        return results

    @staticmethod
    def _compute_cache_key(text: str, normalized: bool) -> str:
        """
        Compute deterministic cache key for text.

        Args:
            text: Input text
            normalized: Whether embedding will be normalized

        Returns:
            Cache key string
        """
        # Hash text for fixed-length key
        text_hash = hashlib.md5(text.encode()).hexdigest()
        norm_suffix = "_norm" if normalized else ""
        return f"emb_minilm:{text_hash}{norm_suffix}"

    # =========================================================================
    # SIMILARITY OPERATIONS
    # =========================================================================

    def compute_similarity(
        self,
        vec1: np.ndarray,
        vec2: np.ndarray,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> float:
        """
        Compute similarity between two vectors.

        Args:
            vec1: First embedding vector
            vec2: Second embedding vector
            metric: Similarity metric to use

        Returns:
            Similarity score (interpretation depends on metric)
        """
        if vec1.shape != vec2.shape:
            raise ValidationError(f"Vector shape mismatch: {vec1.shape} vs {vec2.shape}")

        if metric == SimilarityMetric.COSINE:
            # Cosine similarity: -1 to 1 (1 = identical direction)
            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return float(dot_product / (norm1 * norm2))

        elif metric == SimilarityMetric.EUCLIDEAN:
            # Euclidean distance: 0 to inf (0 = identical)
            # Convert to similarity: 1 / (1 + distance)
            distance = np.linalg.norm(vec1 - vec2)
            return float(1 / (1 + distance))

        elif metric == SimilarityMetric.DOT_PRODUCT:
            # Dot product (assumes normalized vectors)
            return float(np.dot(vec1, vec2))

        else:
            raise ValidationError(f"Unsupported metric: {metric}")

    def compute_similarity_matrix(
        self,
        vectors: List[np.ndarray],
        metric: SimilarityMetric = SimilarityMetric.COSINE,
    ) -> np.ndarray:
        """
        Compute pairwise similarity matrix.

        Args:
            vectors: List of embedding vectors
            metric: Similarity metric

        Returns:
            N×N similarity matrix
        """
        if not vectors:
            return np.array([])

        stacked = np.vstack(vectors)

        if metric == SimilarityMetric.COSINE:
            return sklearn_cosine_similarity(stacked)

        elif metric == SimilarityMetric.DOT_PRODUCT:
            # Assumes normalized vectors
            return np.dot(stacked, stacked.T)

        elif metric == SimilarityMetric.EUCLIDEAN:
            # Compute pairwise euclidean distances
            from scipy.spatial.distance import cdist

            distances = cdist(stacked, stacked, metric="euclidean")
            # Convert to similarity
            return 1 / (1 + distances)

        else:
            raise ValidationError(f"Unsupported metric: {metric}")

    def find_most_similar(
        self,
        query_vector: np.ndarray,
        candidate_vectors: List[np.ndarray],
        top_k: int = 5,
        metric: SimilarityMetric = SimilarityMetric.COSINE,
        threshold: Optional[float] = None,
    ) -> List[Tuple[int, float]]:
        """
        Find most similar vectors to query.

        Args:
            query_vector: Query embedding
            candidate_vectors: List of candidate embeddings
            top_k: Number of results to return
            metric: Similarity metric
            threshold: Optional minimum similarity threshold

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        if not candidate_vectors:
            return []

        # Compute similarities
        similarities = [
            self.compute_similarity(query_vector, vec, metric) for vec in candidate_vectors
        ]

        # Filter by threshold
        if threshold is not None:
            similarities = [(idx, sim) for idx, sim in enumerate(similarities) if sim >= threshold]
        else:
            similarities = list(enumerate(similarities))

        # Sort by similarity (descending)
        similarities.sort(key=lambda x: x[1], reverse=True)

        return similarities[:top_k]

    # =========================================================================
    # CLUSTERING OPERATIONS
    # =========================================================================

    def cluster_embeddings(
        self,
        embeddings: List[np.ndarray],
        n_clusters: int,
        method: ClusteringMethod = ClusteringMethod.KMEANS,
    ) -> Tuple[List[int], np.ndarray]:
        """
        Cluster embeddings into semantic groups.

        Args:
            embeddings: List of embedding vectors
            n_clusters: Number of clusters
            method: Clustering algorithm

        Returns:
            Tuple of (cluster_labels, cluster_centroids)
        """
        if len(embeddings) < n_clusters:
            raise ValidationError(
                f"Cannot create {n_clusters} clusters from {len(embeddings)} vectors"
            )

        stacked = np.vstack(embeddings)

        if method == ClusteringMethod.KMEANS:
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(stacked)
            centroids = clusterer.cluster_centers_

        elif method == ClusteringMethod.HIERARCHICAL:
            clusterer = AgglomerativeClustering(
                n_clusters=n_clusters, linkage="average", metric="cosine"
            )
            labels = clusterer.fit_predict(stacked)

            # Compute centroids manually
            centroids = np.array([np.mean(stacked[labels == i], axis=0) for i in range(n_clusters)])

        else:
            raise ValidationError(f"Unsupported clustering method: {method}")

        logger.debug(f"Clustered {len(embeddings)} embeddings into {n_clusters} groups")

        return labels.tolist(), centroids

    # =========================================================================
    # SEMANTIC DEDUPLICATION
    # =========================================================================

    def deduplicate_by_similarity(
        self,
        texts: List[str],
        embeddings: List[np.ndarray],
        threshold: float = 0.95,
    ) -> List[int]:
        """
        Find duplicate texts based on semantic similarity.

        Args:
            texts: List of texts
            embeddings: Corresponding embeddings
            threshold: Similarity threshold (0-1) for considering duplicates
        Returns:
            List of indices to KEEP (duplicates removed)
        """
        if len(texts) != len(embeddings):
            raise ValidationError("Texts and embeddings must have same length")

        keep_indices = []
        seen_embeddings = []

        for idx, embedding in enumerate(embeddings):
            # Check against all previously seen embeddings
            is_duplicate = False

            for seen_emb in seen_embeddings:
                similarity = self.compute_similarity(
                    embedding, seen_emb, metric=SimilarityMetric.COSINE
                )

                if similarity >= threshold:
                    is_duplicate = True
                    logger.debug(f"Duplicate detected: text {idx} (similarity: {similarity:.3f})")
                    break

            if not is_duplicate:
                keep_indices.append(idx)
                seen_embeddings.append(embedding)

        logger.info(f"Deduplication: kept {len(keep_indices)}/{len(texts)} texts")
        return keep_indices


# =========================================================================
# SEMANTIC SEARCH
# =========================================================================


async def semantic_search(
    self,
    query: str,
    corpus: List[str],
    top_k: int = 5,
    threshold: float = 0.7,
) -> List[Tuple[str, float]]:
    """
    Search corpus for semantically similar texts.

    Args:
        query: Search query
        corpus: List of texts to search
        top_k: Maximum results to return
        threshold: Minimum similarity threshold

    Returns:
        List of (text, similarity_score) tuples
    """
    # Generate embeddings
    query_emb = await self.embed(query, normalize=True)
    corpus_embs = await self.embed(corpus, normalize=True)

    # Find most similar
    results = self.find_most_similar(
        query_vector=query_emb,
        candidate_vectors=corpus_embs,
        top_k=top_k,
        metric=SimilarityMetric.COSINE,
        threshold=threshold,
    )

    # Map back to texts
    return [(corpus[idx], score) for idx, score in results]


# =========================================================================
# UTILITY METHODS
# =========================================================================


@staticmethod
def normalize_vector(vector: np.ndarray) -> np.ndarray:
    """
    Normalize vector to unit length.

    Args:
        vector: Input vector

    Returns:
        Normalized vector (L2 norm = 1)
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        return vector
    return vector / norm


@staticmethod
def compute_centroid(vectors: List[np.ndarray], normalize: bool = True) -> np.ndarray:
    """
    Compute centroid (mean) of multiple vectors.

    Args:
        vectors: List of vectors
        normalize: If True, normalize result to unit vector

    Returns:
        Centroid vector
    """
    if not vectors:
        raise ValidationError("Cannot compute centroid of empty list")

    stacked = np.vstack(vectors)
    centroid = np.mean(stacked, axis=0)

    if normalize:
        centroid = SemanticAnalyzer.normalize_vector(centroid)

    return centroid


def compute_embedding_statistics(self, embeddings: List[np.ndarray]) -> Dict[str, float]:
    """
    Compute statistical properties of embedding set.

    Useful for understanding semantic diversity and coherence.

    Args:
        embeddings: List of embedding vectors

    Returns:
        Dict with statistics:
        {
            'mean_pairwise_similarity': float,
            'std_pairwise_similarity': float,
            'centroid_coherence': float,  # avg similarity to centroid
        }
    """
    if len(embeddings) < 2:
        return {
            "mean_pairwise_similarity": 0.0,
            "std_pairwise_similarity": 0.0,
            "centroid_coherence": 0.0,
        }

    # Compute pairwise similarities
    sim_matrix = self.compute_similarity_matrix(embeddings, metric=SimilarityMetric.COSINE)

    # Extract upper triangle (excluding diagonal)
    triu_indices = np.triu_indices_from(sim_matrix, k=1)
    pairwise_sims = sim_matrix[triu_indices]

    mean_similarity = float(np.mean(pairwise_sims))
    std_similarity = float(np.std(pairwise_sims))

    # Compute centroid coherence
    centroid = self.compute_centroid(embeddings, normalize=True)
    centroid_sims = [
        self.compute_similarity(emb, centroid, SimilarityMetric.COSINE) for emb in embeddings
    ]
    centroid_coherence = float(np.mean(centroid_sims))

    return {
        "mean_pairwise_similarity": mean_similarity,
        "std_pairwise_similarity": std_similarity,
        "centroid_coherence": centroid_coherence,
    }


def disable_cache(self):
    """Disable embedding caching (useful for testing)."""
    self.cache_enabled = False
    logger.warning("Embedding cache disabled")


def enable_cache(self):
    """Enable embedding caching."""
    self.cache_enabled = True
    logger.info("Embedding cache enabled")


# =========================================================================
# GLOBAL INSTANCE
# =========================================================================
# Singleton instance for application-wide use
semantic_analyzer = SemanticAnalyzer()
