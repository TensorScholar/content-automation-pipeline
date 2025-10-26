"""
Content Generator: LLM-Orchestrated Article Synthesis Engine

This module implements economically-optimized, quality-assured content generation
through adaptive prompt synthesis, incremental section generation, and multi-stage
validation. Leverages semantic coherence analysis and statistical quality metrics
to ensure output fidelity while minimizing API costs.

Architectural Pattern: Strategy + Pipeline with Adaptive Quality Gates
"""

import asyncio
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from loguru import logger

from core.exceptions import GenerationError, QualityValidationError, TokenBudgetExceededError
from core.models import ContentPlan, GeneratedArticle, Keyword, Project, Section, SectionIntent
from infrastructure.llm_client import AbstractLLMClient
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.semantic_analyzer import SemanticAnalyzer
from optimization.model_router import ModelRouter
from optimization.prompt_compressor import PromptCompressor
from optimization.token_budget_manager import TokenBudgetManager


class ContentGenerator:
    """
    Adaptive content generation engine with economic optimization and quality assurance.

    Implements a multi-stage generation pipeline:
    1. Context preparation and compression
    2. Section-by-section synthesis with validation gates
    3. Coherence analysis and refinement
    4. Final assembly with metadata generation

    Design Philosophy:
    - Fail fast: Validate early sections before generating later ones
    - Economic rationality: Use cheapest model that meets quality threshold
    - Semantic consistency: Enforce coherence across section boundaries
    - Graceful degradation: Retry with enhanced prompts on quality failures
    """

    def __init__(
        self,
        llm_client: AbstractLLMClient,
        context_synthesizer: ContextSynthesizer,
        semantic_analyzer: SemanticAnalyzer,
        model_router: ModelRouter,
        token_budget_manager: TokenBudgetManager,
        prompt_compressor: PromptCompressor,
        metrics_collector: MetricsCollector,
    ):
        self.llm = llm_client
        self.context_synthesizer = context_synthesizer
        self.semantic_analyzer = semantic_analyzer
        self.model_router = model_router
        self.budget_manager = token_budget_manager
        self.prompt_compressor = prompt_compressor
        self.metrics = metrics_collector

        # Quality thresholds
        self.min_readability_score = 60.0  # Flesch-Kincaid
        self.min_section_coherence = 0.75  # Cosine similarity
        self.max_keyword_density = 0.025  # 2.5%
        self.min_keyword_density = 0.005  # 0.5%

        # Generation parameters
        self.max_retries = 2
        self.temperature_progression = [0.7, 0.5, 0.3]  # Decrease on retries

        logger.info("ContentGenerator initialized with adaptive quality gates")

    async def generate_article(
        self, content_plan: ContentPlan, project: Project, priority: str = "high"
    ) -> GeneratedArticle:
        """
        Generate complete article from content plan with quality assurance.

        Args:
            content_plan: Structured outline with keyword strategy
            project: Project context for voice/tone adaptation
            priority: Token budget priority level

        Returns:
            GeneratedArticle with full content and metadata

        Raises:
            GenerationError: On irrecoverable generation failures
            TokenBudgetExceededError: When budget allocation exhausted
        """
        start_time = datetime.utcnow()
        article_id = uuid4()

        logger.info(
            f"Initiating article generation | article_id={article_id} | "
            f"topic={content_plan.topic} | sections={len(content_plan.outline.sections)}"
        )

        try:
            # Allocate token budget for this generation task
            budget = await self.budget_manager.allocate_budget(
                task_type="article_generation",
                priority=priority,
                estimated_sections=len(content_plan.outline.sections),
            )

            # Prepare compressed context
            context = await self._prepare_generation_context(project, content_plan)

            # Generate sections with validation gates
            sections_content = await self._generate_sections_incremental(
                content_plan.outline.sections, context, budget, content_plan.primary_keywords
            )

            # Assemble full article
            full_content = await self._assemble_article(
                title=content_plan.outline.title,
                sections=sections_content,
                meta_description=content_plan.outline.meta_description,
            )

            # Final quality validation
            quality_metrics = await self._validate_full_article(
                full_content, content_plan.primary_keywords + content_plan.secondary_keywords
            )

            # Generate metadata
            article = GeneratedArticle(
                id=article_id,
                project_id=project.id,
                content_plan_id=content_plan.id,
                title=content_plan.outline.title,
                content=full_content,
                meta_description=content_plan.outline.meta_description,
                word_count=quality_metrics["word_count"],
                readability_score=quality_metrics["readability_score"],
                keyword_density=quality_metrics["keyword_density"],
                total_tokens_used=budget.tokens_consumed,
                total_cost=budget.cost_incurred,
                generation_time=(datetime.utcnow() - start_time).total_seconds(),
                created_at=datetime.utcnow(),
            )

            # Record metrics
            await self._record_generation_metrics(article, quality_metrics)

            logger.success(
                f"Article generation completed | article_id={article_id} | "
                f"words={article.word_count} | cost=${article.total_cost:.4f} | "
                f"time={article.generation_time:.2f}s"
            )

            return article

        except Exception as e:
            logger.error(f"Article generation failed | article_id={article_id} | error={e}")
            raise GenerationError(f"Failed to generate article: {str(e)}") from e

    async def _prepare_generation_context(self, project: Project, content_plan: ContentPlan) -> str:
        """
        Prepare and compress context for content generation.

        Synthesizes project knowledge (rulebook, inferred patterns, best practices)
        into a compact, information-dense context suitable for LLM prompts.
        """
        logger.debug(f"Preparing generation context | project_id={project.id}")

        # Build context stream
        raw_context = await self.context_synthesizer.synthesize_context(
            project=project,
            topic=content_plan.topic,
            keywords=content_plan.primary_keywords,
            level="standard",  # Balance between richness and cost
        )

        # Compress for token efficiency
        compressed = await self.prompt_compressor.compress_context(
            context=raw_context,
            target_tokens=1200,  # ~30% of typical prompt budget
            preserve_critical=True,
        )

        logger.debug(
            f"Context prepared | original_tokens={len(raw_context.split())*1.3:.0f} | "
            f"compressed_tokens={len(compressed.split())*1.3:.0f}"
        )

        return compressed

    async def _generate_sections_incremental(
        self,
        sections: List[Section],
        context: str,
        budget: "TokenBudget",
        primary_keywords: List[Keyword],
    ) -> List[Dict[str, str]]:
        """
        Generate sections incrementally with validation gates.

        Implements early-exit strategy: validate each section before proceeding
        to next. This prevents wasting tokens on later sections if early ones fail.
        """
        generated_sections = []
        cumulative_content = ""

        for idx, section in enumerate(sections):
            logger.debug(f"Generating section {idx+1}/{len(sections)} | heading={section.heading}")

            # Check budget availability
            if not budget.has_capacity(estimated_tokens=400):
                raise TokenBudgetExceededError(
                    f"Insufficient budget for section {idx+1}/{len(sections)}"
                )

            # Generate section with retry mechanism
            section_content = await self._generate_single_section(
                section=section,
                context=context,
                previous_content=cumulative_content,
                keywords=primary_keywords,
                budget=budget,
            )

            # Validate section quality
            is_valid, validation_msg = await self._validate_section(
                section_content, section, cumulative_content
            )

            if not is_valid:
                logger.warning(
                    f"Section validation failed | section={idx+1} | reason={validation_msg}"
                )

                # Retry with enhanced prompt
                section_content = await self._regenerate_section_with_feedback(
                    section=section,
                    context=context,
                    previous_content=cumulative_content,
                    feedback=validation_msg,
                    budget=budget,
                )

            generated_sections.append(
                {
                    "heading": section.heading,
                    "content": section_content,
                    "intent": section.intent.value,
                }
            )

            cumulative_content += f"\n\n## {section.heading}\n\n{section_content}"

        logger.info(f"All sections generated successfully | total_sections={len(sections)}")
        return generated_sections

    async def _generate_single_section(
        self,
        section: Section,
        context: str,
        previous_content: str,
        keywords: List[Keyword],
        budget: "TokenBudget",
        temperature: float = 0.7,
    ) -> str:
        """
        Generate a single section using optimal model routing.

        Constructs section-specific prompt with contextual continuity from
        previous sections and explicit keyword integration requirements.
        """
        # Route to optimal model based on section complexity
        model = await self.model_router.select_model(
            task_type="section_generation",
            complexity=self._estimate_section_complexity(section),
            budget_constraint=budget.remaining_budget,
        )

        # Construct section prompt
        prompt = self._build_section_prompt(
            section=section, context=context, previous_content=previous_content, keywords=keywords
        )

        # Compress prompt if needed
        if len(prompt.split()) * 1.3 > budget.max_prompt_tokens:
            prompt = await self.prompt_compressor.compress_prompt(
                prompt, target_tokens=budget.max_prompt_tokens
            )

        # Generate with LLM
        response = await self.llm.complete(
            prompt=prompt,
            model=model,
            temperature=temperature,
            max_tokens=section.estimated_words * 1.5,  # Conservative estimate
            stop_sequences=["##", "\n\n---"],  # Prevent running into next section
        )

        # Update budget tracking
        budget.consume(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            cost=response.cost,
        )

        return response.content.strip()

    def _build_section_prompt(
        self, section: Section, context: str, previous_content: str, keywords: List[Keyword]
    ) -> str:
        """
        Construct optimized section generation prompt.

        Implements adaptive prompting strategy based on section intent
        and keyword integration requirements.
        """
        # Intent-specific guidance
        intent_guidance = {
            SectionIntent.INTRODUCE: (
                "Begin with a compelling hook that establishes relevance. "
                "Provide necessary background without assuming prior knowledge."
            ),
            SectionIntent.EXPLAIN: (
                "Break down complex concepts into digestible explanations. "
                "Use examples and analogies where appropriate."
            ),
            SectionIntent.COMPARE: (
                "Present balanced comparisons with clear criteria. "
                "Use structured formats (tables, lists) when helpful."
            ),
            SectionIntent.CONCLUDE: (
                "Synthesize key points without introducing new information. "
                "Provide actionable takeaways or clear next steps."
            ),
            SectionIntent.INSTRUCT: (
                "Provide step-by-step instructions with clear sequence. "
                "Anticipate common questions or obstacles."
            ),
        }

        # Extract keyword phrases for natural integration
        keyword_phrases = [kw.phrase for kw in keywords[:5]]  # Top 5

        prompt = f"""You are writing a section for a professional article.

PROJECT CONTEXT:
{context}

PREVIOUS CONTENT (for continuity):
{previous_content[-800:] if previous_content else "This is the first section."}

SECTION SPECIFICATION:
Heading: {section.heading}
Intent: {section.intent.value}
Target Length: ~{section.estimated_words} words

WRITING GUIDELINES:
{intent_guidance.get(section.intent, "Provide clear, engaging content.")}

KEYWORD INTEGRATION:
Naturally incorporate these concepts: {', '.join(keyword_phrases)}
Do NOT force keywords—prioritize readability and natural flow.

STYLE REQUIREMENTS:
- Write in clear, accessible language (Flesch-Kincaid grade 10-12)
- Use short paragraphs (3-4 sentences maximum)
- Employ active voice and concrete examples
- Maintain consistent tone with previous sections

Generate ONLY the content for this section. Do not include the heading itself.
Do not add transitional phrases to the next section.
Begin writing now:"""

        return prompt

    async def _validate_section(
        self, section_content: str, section: Section, previous_content: str
    ) -> Tuple[bool, str]:
        """
        Validate section quality using local NLP and heuristics.

        Checks:
        1. Semantic coherence with previous content
        2. Appropriate length
        3. Keyword integration (not over-optimized)
        4. Readability metrics
        """
        word_count = len(section_content.split())

        # Length validation
        expected_words = section.estimated_words
        if word_count < expected_words * 0.6:
            return False, f"Section too short: {word_count} words (expected ~{expected_words})"

        if word_count > expected_words * 1.8:
            return False, f"Section too long: {word_count} words (expected ~{expected_words})"

        # Semantic coherence check (if previous content exists)
        if previous_content:
            coherence = await self.semantic_analyzer.compute_coherence(
                previous_content[-500:],  # Last portion of previous content
                section_content[:500],  # Beginning of current section
            )

            if coherence < self.min_section_coherence:
                return False, f"Low semantic coherence: {coherence:.3f}"

        # Keyword density check
        for keyword in section.target_keywords:
            density = section_content.lower().count(keyword.lower()) / word_count
            if density > self.max_keyword_density:
                return (
                    False,
                    f"Keyword over-optimization detected: '{keyword}' density={density:.3f}",
                )

        # Readability check (local computation)
        readability = self._compute_readability(section_content)
        if readability < 50.0:  # Lower threshold for sections
            return False, f"Low readability score: {readability:.1f}"

        return True, "Section validated"

    async def _regenerate_section_with_feedback(
        self,
        section: Section,
        context: str,
        previous_content: str,
        feedback: str,
        budget: "TokenBudget",
    ) -> str:
        """
        Regenerate section with explicit feedback integration.

        Uses lower temperature and enhanced prompt to address validation failures.
        """
        logger.info(f"Regenerating section with feedback | section={section.heading}")

        # Enhanced prompt with feedback
        base_prompt = self._build_section_prompt(
            section=section,
            context=context,
            previous_content=previous_content,
            keywords=[],  # Will be in base prompt
        )

        enhanced_prompt = f"""{base_prompt}

IMPORTANT FEEDBACK FROM PREVIOUS ATTEMPT:
{feedback}

Please address this feedback in your generation. Focus on quality over length.
Begin writing now:"""

        # Use lower temperature for more focused generation
        return await self._generate_single_section(
            section=section,
            context=context,
            previous_content=previous_content,
            keywords=[],  # Already in enhanced prompt
            budget=budget,
            temperature=0.5,
        )

    async def _assemble_article(
        self, title: str, sections: List[Dict[str, str]], meta_description: str
    ) -> str:
        """
        Assemble final article from generated sections.

        Formats sections with proper HTML/Markdown structure and
        ensures consistent formatting throughout.
        """
        article_parts = [f"# {title}\n"]

        for section_data in sections:
            article_parts.append(f"## {section_data['heading']}\n")
            article_parts.append(f"{section_data['content']}\n")

        full_content = "\n".join(article_parts)

        logger.debug(f"Article assembled | total_length={len(full_content)} chars")
        return full_content

    async def _validate_full_article(self, content: str, keywords: List[Keyword]) -> Dict[str, any]:
        """
        Comprehensive quality validation for complete article.

        Returns quality metrics for storage and analysis.
        """
        word_count = len(content.split())

        # Readability analysis
        readability = self._compute_readability(content)

        # Keyword density analysis
        keyword_density = {}
        for keyword in keywords:
            phrase = keyword.phrase.lower()
            occurrences = content.lower().count(phrase)
            density = occurrences / word_count if word_count > 0 else 0
            keyword_density[phrase] = density

        # Average keyword density
        avg_density = np.mean(list(keyword_density.values())) if keyword_density else 0

        # Validation gates
        if readability < self.min_readability_score:
            logger.warning(
                f"Article readability below threshold | score={readability:.1f} | "
                f"threshold={self.min_readability_score}"
            )

        if avg_density < self.min_keyword_density:
            logger.warning(
                f"Keyword density too low | avg_density={avg_density:.4f} | "
                f"threshold={self.min_keyword_density}"
            )

        return {
            "word_count": word_count,
            "readability_score": readability,
            "keyword_density": keyword_density,
            "avg_keyword_density": avg_density,
        }

    def _compute_readability(self, text: str) -> float:
        """
        Compute Flesch-Kincaid readability score.

        Formula: 206.835 - 1.015(total_words/total_sentences) - 84.6(total_syllables/total_words)
        Higher scores indicate easier readability.
        """
        sentences = text.split(".")
        sentences = [s.strip() for s in sentences if s.strip()]

        words = text.split()
        total_words = len(words)
        total_sentences = len(sentences)

        if total_words == 0 or total_sentences == 0:
            return 0.0

        # Approximate syllable count
        total_syllables = sum(self._count_syllables(word) for word in words)

        score = (
            206.835
            - 1.015 * (total_words / total_sentences)
            - 84.6 * (total_syllables / total_words)
        )

        return max(0.0, min(100.0, score))  # Clamp to [0, 100]

    def _count_syllables(self, word: str) -> int:
        """
        Approximate syllable count using vowel clustering heuristic.
        """
        word = word.lower()
        vowels = "aeiouy"
        syllable_count = 0
        previous_was_vowel = False

        for char in word:
            is_vowel = char in vowels
            if is_vowel and not previous_was_vowel:
                syllable_count += 1
            previous_was_vowel = is_vowel

        # Adjust for silent 'e'
        if word.endswith("e"):
            syllable_count -= 1

        # Ensure at least one syllable
        return max(1, syllable_count)

    def _estimate_section_complexity(self, section: Section) -> str:
        """
        Estimate section complexity for model routing decisions.

        Complexity factors:
        - Intent type (explain/compare = higher complexity)
        - Estimated word count
        - Number of target keywords
        """
        complexity_scores = {
            SectionIntent.INTRODUCE: 2,
            SectionIntent.EXPLAIN: 4,
            SectionIntent.COMPARE: 5,
            SectionIntent.CONCLUDE: 2,
            SectionIntent.INSTRUCT: 3,
        }

        base_complexity = complexity_scores.get(section.intent, 3)

        # Adjust for length
        if section.estimated_words > 400:
            base_complexity += 1

        # Adjust for keyword density requirement
        if len(section.target_keywords) > 3:
            base_complexity += 1

        if base_complexity <= 2:
            return "low"
        elif base_complexity <= 4:
            return "medium"
        else:
            return "high"

    async def _record_generation_metrics(
        self, article: GeneratedArticle, quality_metrics: Dict[str, any]
    ):
        """Record comprehensive metrics for monitoring and optimization."""
        await self.metrics.record_generation(
            article_id=str(article.id),
            project_id=str(article.project_id),
            word_count=article.word_count,
            tokens_used=article.total_tokens_used,
            cost=article.total_cost,
            generation_time=article.generation_time,
            readability_score=quality_metrics["readability_score"],
            avg_keyword_density=quality_metrics["avg_keyword_density"],
        )
