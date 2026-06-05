"""
LLM-Based Context Management

Uses LLM's native capabilities for intelligent context compression and synthesis.
More reliable and semantic-aware than custom algorithms.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional

from loguru import logger

from infrastructure.llm_client import AbstractLLMClient


class CompressionLevel(str, Enum):
    """Compression aggressiveness levels."""
    LIGHT = "light"      # Keep 80-90% of content
    MEDIUM = "medium"    # Keep 50-70% of content
    HEAVY = "heavy"      # Keep 30-50% of content
    EXTREME = "extreme"  # Keep 20-30% of content


@dataclass
class CompressionResult:
    """Result of context compression operation."""
    compressed_text: str
    original_tokens: int
    compressed_tokens: int
    compression_ratio: float
    method: str = "llm_compression"


class LLMContextManager:
    """
    Manages context using LLM's native understanding.

    Benefits over custom algorithms:
    - Semantic understanding (not just word frequency)
    - Maintains coherence and readability
    - Adapts to content type automatically
    - Multi-lingual support
    - Task-aware compression
    """

    def __init__(self, llm_client: AbstractLLMClient):
        """
        Initialize context manager.

        Args:
            llm_client: LLM client for compression operations
        """
        self.llm_client = llm_client
        logger.info("LLM context manager initialized")

    async def compress_context(
        self,
        text: str,
        target_tokens: int,
        purpose: str = "content generation",
        level: CompressionLevel = CompressionLevel.MEDIUM,
    ) -> CompressionResult:
        """
        Compress context using LLM's understanding.

        This is superior to custom algorithms because:
        1. LLM understands semantic importance
        2. Maintains coherence and flow
        3. Preserves critical information for the task
        4. Produces natural, readable output

        Args:
            text: Text to compress
            target_tokens: Desired token count
            purpose: What the context will be used for
            level: Compression aggressiveness

        Returns:
            CompressionResult with compressed text and metrics
        """
        if not text or len(text.strip()) < 100:
            return CompressionResult(
                compressed_text=text,
                original_tokens=len(text.split()),
                compressed_tokens=len(text.split()),
                compression_ratio=1.0,
            )

        # Estimate original tokens (rough approximation)
        original_tokens = len(text.split()) * 1.3  # ~1.3 tokens per word average

        # Build compression prompt based on level
        compression_instructions = self._get_compression_instructions(level)

        prompt = f"""Compress the following text for {purpose}.

{compression_instructions}

Target: Approximately {target_tokens} tokens.

Important: Maintain the most critical information for {purpose}. Remove redundancy, verbose examples, and tangential details.

TEXT TO COMPRESS:
{text}

COMPRESSED VERSION:"""

        try:
            # Use smaller, faster model for compression (cost optimization)
            response = await self.llm_client.complete(
                prompt=prompt,
                max_tokens=target_tokens,
                temperature=0.3,  # Lower temperature for consistent compression
                model_preference="small",  # Use cheaper model when available
            )

            compressed_text = response.get("content", "").strip()
            compressed_tokens = response.get("usage", {}).get("total_tokens", 0)

            # If LLM doesn't provide token count, estimate
            if compressed_tokens == 0:
                compressed_tokens = len(compressed_text.split()) * 1.3

            actual_ratio = compressed_tokens / original_tokens if original_tokens > 0 else 1.0

            logger.debug(
                f"Compressed: {original_tokens:.0f} → {compressed_tokens:.0f} tokens "
                f"({actual_ratio:.1%})"
            )

            return CompressionResult(
                compressed_text=compressed_text,
                original_tokens=int(original_tokens),
                compressed_tokens=int(compressed_tokens),
                compression_ratio=actual_ratio,
            )

        except Exception as e:
            logger.warning(f"LLM compression failed: {e}. Using truncation fallback.")
            return self._fallback_truncation(text, target_tokens)

    async def synthesize_multi_source(
        self,
        sources: Dict[str, str],
        target_tokens: int,
        purpose: str = "content generation",
    ) -> str:
        """
        Synthesize multiple sources into coherent context.

        LLM is excellent at this because it can:
        - Identify and merge duplicate information
        - Resolve contradictions
        - Create a coherent narrative
        - Prioritize by relevance

        Args:
            sources: Dict of source_name -> content
            target_tokens: Target token count
            purpose: What the synthesis will be used for

        Returns:
            Synthesized context string
        """
        if not sources:
            return ""

        # Format sources
        formatted_sources = []
        for name, content in sources.items():
            formatted_sources.append(f"[SOURCE: {name}]\n{content}\n")

        combined = "\n---\n".join(formatted_sources)

        prompt = f"""Synthesize the following sources into a coherent context for {purpose}.

Instructions:
1. Merge duplicate or overlapping information
2. Resolve any contradictions (prioritize more specific sources)
3. Organize logically
4. Keep only information relevant to {purpose}
5. Target approximately {target_tokens} tokens

SOURCES:
{combined}

SYNTHESIZED CONTEXT:"""

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                max_tokens=target_tokens,
                temperature=0.3,
            )

            synthesized = response.get("content", "").strip()
            logger.debug(f"Synthesized {len(sources)} sources into {len(synthesized.split())} words")

            return synthesized

        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}. Using simple concatenation.")
            # Fallback: just concatenate and truncate
            simple = "\n\n".join(f"{name}: {content}" for name, content in sources.items())
            words = simple.split()
            target_words = int(target_tokens * 0.75)  # Conservative estimate
            return " ".join(words[:target_words]) if len(words) > target_words else simple

    async def extract_key_information(
        self,
        text: str,
        focus_areas: List[str],
        max_tokens: int = 500,
    ) -> Dict[str, str]:
        """
        Extract specific information from text using LLM.

        More accurate than regex or keyword matching because LLM
        understands context and meaning.

        Args:
            text: Source text
            focus_areas: What to extract (e.g., ["tone", "style", "audience"])
            max_tokens: Max tokens for extraction

        Returns:
            Dict mapping focus_area -> extracted information
        """
        areas_str = ", ".join(focus_areas)

        prompt = f"""Extract the following information from the text: {areas_str}

TEXT:
{text}

For each area, provide a concise summary. Format as:
AREA: information

EXTRACTIONS:"""

        try:
            response = await self.llm_client.complete(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.2,
            )

            content = response.get("content", "")

            # Parse response into dict
            extractions = {}
            for line in content.split("\n"):
                if ":" in line:
                    area, info = line.split(":", 1)
                    area = area.strip().lower()
                    if any(fa.lower() in area for fa in focus_areas):
                        extractions[area] = info.strip()

            return extractions

        except Exception as e:
            logger.error(f"Information extraction failed: {e}")
            return {area: "Could not extract" for area in focus_areas}

    @staticmethod
    def _get_compression_instructions(level: CompressionLevel) -> str:
        """Get compression instructions based on level."""
        instructions = {
            CompressionLevel.LIGHT: "Lightly edit for conciseness. Remove only obvious redundancy. Keep most details.",
            CompressionLevel.MEDIUM: "Compress significantly. Keep key points and essential details. Remove examples and elaborations.",
            CompressionLevel.HEAVY: "Aggressive compression. Keep only critical information. Be extremely concise.",
            CompressionLevel.EXTREME: "Maximum compression. Extract only the absolute essentials. Ultra-brief.",
        }
        return instructions.get(level, instructions[CompressionLevel.MEDIUM])

    def _fallback_truncation(self, text: str, target_tokens: int) -> CompressionResult:
        """
        Simple truncation fallback if LLM compression fails.

        Not ideal, but better than crashing.
        """
        words = text.split()
        target_words = int(target_tokens * 0.75)  # Conservative

        if len(words) <= target_words:
            return CompressionResult(
                compressed_text=text,
                original_tokens=len(words),
                compressed_tokens=len(words),
                compression_ratio=1.0,
            )

        truncated = " ".join(words[:target_words]) + "..."

        return CompressionResult(
            compressed_text=truncated,
            original_tokens=len(words),
            compressed_tokens=target_words,
            compression_ratio=target_words / len(words),
        )

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """
        Estimate token count from text.

        Rough approximation: ~1.3 tokens per word for English.
        Good enough for planning purposes.
        """
        if not text:
            return 0
        return int(len(text.split()) * 1.3)


# =============================================================================
# USAGE EXAMPLES
# =============================================================================

"""
Example 1: Basic Compression
-----------------------------
manager = LLMContextManager(llm_client)

result = await manager.compress_context(
    text=long_documentation,
    target_tokens=1000,
    purpose="content generation",
    level=CompressionLevel.MEDIUM,
)

print(f"Compressed: {result.original_tokens} → {result.compressed_tokens} tokens")
print(result.compressed_text)


Example 2: Multi-Source Synthesis
----------------------------------
sources = {
    "website_content": "...",
    "previous_articles": "...",
    "brand_guidelines": "...",
}

synthesized = await manager.synthesize_multi_source(
    sources=sources,
    target_tokens=2000,
    purpose="blog post generation",
)


Example 3: Targeted Extraction
-------------------------------
extractions = await manager.extract_key_information(
    text=website_content,
    focus_areas=["tone", "target_audience", "key_topics"],
    max_tokens=300,
)

print(f"Tone: {extractions.get('tone')}")
print(f"Audience: {extractions.get('target_audience')}")
"""
