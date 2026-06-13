"""
Simplified Keyword Researcher

This module replaces the previous over-engineered implementation with a
practical, readable keyword researcher suitable for production. It uses
standard Python types, simple heuristics or placeholders for external SEO
APIs, and conventional try/except error handling.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Dict, List, Optional, Set

from loguru import logger

from config.settings import get_settings
from intelligence.semantic_analyzer import SemanticAnalyzer
from optimization.cache_manager import CacheManager

if TYPE_CHECKING:
    from infrastructure.llm_client import AbstractLLMClient


@dataclass
class Keyword:
    phrase: str
    embedding: Optional[List[float]] = None
    metrics: Dict = field(default_factory=dict)
    primary_intent: str = "informational"
    semantic_cluster: Optional[int] = None


class KeywordResearcher:
    """Practical keyword researcher using heuristics and pluggable analyzers.

    - Returns standard Python lists/dicts
    - Uses try/except for error handling
    - Embeddings are provided by `SemanticAnalyzer`
    - LLM-powered keyword expansion when llm_client is provided
    """

    def __init__(
        self,
        semantic_analyzer: Optional[SemanticAnalyzer] = None,
        cache_manager: Optional[CacheManager] = None,
        llm_client: Optional["AbstractLLMClient"] = None,
    ) -> None:
        self.semantic_analyzer = semantic_analyzer
        self.cache_manager = cache_manager
        self.llm_client = llm_client
        self._cache_ttl = 86400 * 7

        logger.info("Keyword researcher (simplified) initialized")

    async def research_keywords(
        self,
        seed_topic: str,
        target_count: int = 50,
        intent_filter: Optional[str] = None,
    ) -> List[Keyword]:
        """Public entry: returns a list of enriched Keyword objects."""
        try:
            # Candidate expansion
            candidates = list(self._generate_variations(seed_topic))

            # Optionally call LLM-based expansion (placeholder)
            if len(candidates) < target_count:
                more = await self._llm_keyword_expansion(seed_topic, target_count - len(candidates))
                candidates.extend(list(more))

            # Deduplicate and trim
            candidates = list(dict.fromkeys(candidates))[: max(10, target_count * 2)]

            # Enrich with embeddings and metrics
            enriched = await self._enrich_keywords(candidates, seed_topic)

            # Simple clustering: bucket by first letter as a deterministic heuristic
            for idx, kw in enumerate(enriched):
                kw.semantic_cluster = ord(kw.phrase[0].lower()) % 10

            # Filter by intent if provided
            if intent_filter:
                enriched = [k for k in enriched if k.primary_intent == intent_filter]

            # Rank by simple score: search_volume (if present) then length
            enriched.sort(key=lambda k: (-(k.metrics.get("search_volume", 0)), len(k.phrase)))

            return enriched[:target_count]

        except Exception as exc:  # pragma: no cover - simple top-level handling
            logger.exception(f"Keyword research failed for '{seed_topic}': {exc}")
            return []

    async def categorize_keywords(self, keywords: List[str]) -> Dict[str, List[Keyword]]:
        try:
            enriched = await self._enrich_keywords(keywords, keywords[0] if keywords else "")
            by_cluster: Dict[str, List[Keyword]] = {}
            for kw in enriched:
                cid = kw.semantic_cluster or -1
                key = f"cluster_{cid}"
                by_cluster.setdefault(key, []).append(kw)
            return by_cluster
        except Exception as exc:
            logger.exception(f"Categorize keywords failed: {exc}")
            return {}

    def _generate_variations(self, seed: str) -> Set[str]:
        variations: Set[str] = set()
        question_words = ["how to", "what is", "why", "where", "best"]
        for qw in question_words:
            variations.add(f"{qw} {seed}")
        modifiers = ["best", "guide", "review", "tips"]
        for mod in modifiers:
            variations.add(f"{seed} {mod}")
            variations.add(f"{mod} {seed}")
        if not seed.endswith("s"):
            variations.add(f"{seed}s")
        variations.add(seed)
        return variations

    async def _llm_keyword_expansion(self, seed: str, count: int) -> Set[str]:
        """Expand seed keyword using LLM for SEO-focused variations.

        Falls back to simple modifier-based expansion if LLM unavailable.
        """
        if not self.llm_client:
            # Fallback: simple modifier-based expansion
            modifiers = ["beginner", "advanced", "2024", "complete", "ultimate",
                        "free", "professional", "easy", "step by step", "explained"]
            return {f"{seed} {mod}" for mod in modifiers[:count]}

        try:
            prompt = f"""Generate {count} SEO-optimized keyword variations for: "{seed}"

Requirements:
- Include long-tail keywords (3-5 words)
- Mix of informational, commercial, and transactional intent
- Include question-based keywords (how, what, why, best)
- Focus on search volume potential

Output format: One keyword per line, no numbering or bullets.
"""
            settings = get_settings()
            response = await self.llm_client.complete(
                prompt=prompt,
                model=settings.llm.keyword_model,  # Use centralized model config
                temperature=0.8,  # Higher creativity for diverse keywords
                max_tokens=500,
            )

            # Parse response into keyword set
            keywords = set()
            for line in response.content.strip().split("\n"):
                kw = line.strip().strip("-").strip("•").strip()
                if kw and len(kw) > 2:
                    keywords.add(kw.lower())

            logger.debug(f"LLM generated {len(keywords)} keywords for '{seed}'")
            return keywords if keywords else {f"{seed} {i}" for i in range(1, count + 1)}

        except Exception as e:
            logger.warning(f"LLM keyword expansion failed: {e}. Using fallback.")
            return {f"{seed} {i}" for i in range(1, count + 1)}

    async def _enrich_keywords(self, candidates: List[str], seed_topic: str) -> List[Keyword]:
        # Obtain embeddings if analyzer available
        embeddings = None
        if self.semantic_analyzer:
            try:
                embeddings = await self.semantic_analyzer.embed(candidates, normalize=True)
            except Exception:
                logger.exception("Embedding generation failed; continuing without embeddings")

        enriched: List[Keyword] = []
        for i, phrase in enumerate(candidates):
            emb = None
            if embeddings and i < len(embeddings):
                emb = list(embeddings[i])

            metrics = self._estimate_metrics(phrase, seed_topic)
            intent = self._classify_intent(phrase)

            enriched.append(Keyword(phrase=phrase, embedding=emb, metrics=metrics, primary_intent=intent))

        return enriched

    def _estimate_metrics(self, phrase: str, seed: str) -> Dict:
        word_count = len(phrase.split())
        search_volume = max(10, 10000 // (word_count or 1))
        difficulty = min(100, 20 + word_count * 15)
        cpc = round(0.5 + word_count * 0.3, 2)
        return {"search_volume": search_volume, "difficulty": difficulty, "cpc": cpc}

    def _classify_intent(self, phrase: str) -> str:
        p = phrase.lower()
        if any(w in p for w in ["buy", "price", "cheap"]):
            return "transactional"
        if any(w in p for w in ["best", "top", "review", "compare"]):
            return "commercial"
        if any(w in p for w in ["how to", "guide", "tutorial"]):
            return "informational"
        if any(w in p for w in ["near me", "local"]):
            return "local"
        return "informational"


def get_keyword_researcher(
    semantic_analyzer: Optional[SemanticAnalyzer] = None,
    cache_manager: Optional[CacheManager] = None,
    llm_client: Optional["AbstractLLMClient"] = None,
) -> KeywordResearcher:
    """Factory function to create a properly configured KeywordResearcher instance."""
    return KeywordResearcher(
        semantic_analyzer=semantic_analyzer,
        cache_manager=cache_manager,
        llm_client=llm_client,
    )
