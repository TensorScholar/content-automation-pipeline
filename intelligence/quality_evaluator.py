"""LLM-as-a-Judge quality evaluation module for content assessment."""

import json
from typing import Any, Dict, List, Optional

from loguru import logger

from infrastructure.llm_client import LLMClient


class QualityEvaluator:
    """Evaluates content quality using LLM-as-a-Judge pattern."""

    def __init__(self, llm_client: LLMClient) -> None:
        """Initialize evaluator with LLM client.
        
        Args:
            llm_client: Injected LLM client for API calls.
        """
        self.llm_client = llm_client

    async def evaluate_article(
        self, topic: str, content: str, audience: str
    ) -> Dict[str, Any]:
        """Evaluate article quality using LLM judgment.
        
        Args:
            topic: Article topic/title.
            content: Full article content.
            audience: Target audience description.
            
        Returns:
            Dictionary with keys: score, status, critique, metrics.
        """
        word_count = len(content.split())
        
        if word_count < 100:
            logger.warning(f"Content too short ({word_count} words), skipping LLM evaluation")
            return {
                "score": 0,
                "status": "REJECTED_TOO_SHORT",
                "critique": ["Content must be at least 100 words"],
                "metrics": {"word_count": word_count}
            }

        system_prompt = self._build_system_prompt()
        user_prompt = self._build_user_prompt(topic, content, audience)

        for attempt in range(2):
            try:
                response = await self.llm_client.complete(
                    prompt=user_prompt,
                    system_prompt=system_prompt,
                    model="gpt-4",
                    temperature=0.3,
                    max_tokens=1000
                )

                if response is None:
                    logger.error("LLM client returned None")
                    continue

                result = json.loads(response)
                
                if not self._validate_response(result):
                    logger.warning(f"Invalid response structure on attempt {attempt + 1}")
                    continue

                logger.info(f"Quality evaluation successful: score={result.get('score')}")
                return result

            except json.JSONDecodeError as e:
                logger.warning(f"JSON parsing failed on attempt {attempt + 1}: {e}")
                continue
            except Exception as e:
                logger.error(f"LLM evaluation error on attempt {attempt + 1}: {e}")
                continue

        logger.error("All evaluation attempts failed, returning fallback")
        return self._fallback_response(word_count)

    def _build_system_prompt(self) -> str:
        """Construct system prompt for LLM judge."""
        return """You are a Senior Editor evaluating content quality.

Analyze the provided article and respond with ONLY a valid JSON object containing:
- "score": integer from 0 to 100
- "critique": array of specific improvement suggestions (strings)
- "metrics": object with keys like "readability", "structure", "relevance" (each 0-10)
- "status": "APPROVED" or "NEEDS_REVISION"

Be strict but constructive. Focus on clarity, accuracy, and audience fit."""

    def _build_user_prompt(self, topic: str, content: str, audience: str) -> str:
        """Construct user prompt with article details."""
        return f"""Evaluate this article:

Topic: {topic}
Target Audience: {audience}

Content:
{content[:4000]}

Respond with JSON only."""

    def _validate_response(self, result: Any) -> bool:
        """Validate LLM response structure."""
        if not isinstance(result, dict):
            return False
        
        required_keys = {"score", "critique", "metrics", "status"}
        if not required_keys.issubset(result.keys()):
            return False
        
        if not isinstance(result["score"], (int, float)):
            return False
        
        if not isinstance(result["critique"], list):
            return False
        
        if not isinstance(result["metrics"], dict):
            return False
        
        return True

    def _fallback_response(self, word_count: int) -> Dict[str, Any]:
        """Return safe fallback when LLM evaluation fails."""
        return {
            "score": 50,
            "status": "EVALUATION_FAILED",
            "critique": ["Automated evaluation unavailable - manual review required"],
            "metrics": {
                "word_count": word_count,
                "automated_check": False
            }
        }
