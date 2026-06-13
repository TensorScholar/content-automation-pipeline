"""Content Generator - Strategic Article Production Engine.

Executes content plans by orchestrating:
- Parallel section generation using asyncio.gather for throughput
- Token budget verification before generation
- Style refinement via critic/editor pattern
- Fact verification to reduce hallucinations
- Quality and SEO analysis
- Article persistence via repository pattern

Architecture:
    Orchestration pattern with explicit dependency injection.
    Uses UnifiedLLMClient for all LLM calls without stochastic routing.

Example:
    >>> generator = ContentGenerator(llm_client, ...)
    >>> article = await generator.generate_article(
    ...     project_id=uuid.uuid4(),
    ...     plan=content_plan,
    ...     model="gpt-4o-mini"
    ... )
"""

import asyncio
import html
import re
import time
import uuid
from datetime import datetime, timezone
from typing import Optional
from uuid import UUID

from bs4 import BeautifulSoup
from loguru import logger

from config.settings import get_settings
from core.exceptions import WorkflowError
from core.models import ContentPlan, GeneratedArticle, Keyword, QualityMetrics
from infrastructure.llm_client import UnifiedLLMClient
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.article_repository import ArticleRepository
from optimization.prompt_compressor import prompt_compressor
from optimization.token_budget_manager import TokenBudgetManager


def _smart_truncate_html(text: str, max_chars: int = 8000) -> str:
    """Intelligently truncate HTML without breaking tag structure.

    Uses BeautifulSoup to ensure all tags are properly closed after truncation.
    Attempts to truncate at natural boundaries (paragraphs, sentences).

    Args:
        text: HTML content to truncate.
        max_chars: Maximum character count (default: 8000).

    Returns:
        Truncated HTML with all tags properly closed.
    """
    if len(text) <= max_chars:
        return text

    # Parse with BeautifulSoup first to work with valid HTML
    soup = BeautifulSoup(text, "html.parser")

    # Get text length and calculate how many elements to keep
    current_length = 0
    elements_to_keep = []

    for element in soup.children:
        element_text = str(element)
        if current_length + len(element_text) <= max_chars:
            elements_to_keep.append(element)
            current_length += len(element_text)
        else:
            # Try to include part of this element if it's text
            if element.name is None:  # Text node
                remaining = max_chars - current_length
                if remaining > 100:  # Only if we can fit meaningful text
                    # Find sentence boundary
                    text_str = str(element)[:remaining]
                    last_sentence = max(
                        text_str.rfind(". "),
                        text_str.rfind(".\n"),
                        text_str.rfind("? "),
                        text_str.rfind("! "),
                    )
                    if last_sentence > 0:
                        from bs4 import NavigableString
                        elements_to_keep.append(NavigableString(text_str[:last_sentence + 1]))
            break

    # Reconstruct soup with kept elements
    new_soup = BeautifulSoup("", "html.parser")
    for elem in elements_to_keep:
        new_soup.append(elem)

    return str(new_soup)


class ContentGenerator:
    """Strategic article generation system with quality assurance.

    Orchestrates the complete article generation workflow:
    1. Budget validation before generation
    2. Parallel section generation for speed
    3. Style refinement using critic/editor pattern
    4. Fact verification to reduce hallucinations
    5. Quality metrics calculation
    6. Article persistence

    Attributes:
        llm_client: Unified LLM client for all provider interactions.
        context_synthesizer: Aggregates contextual information.
        semantic_analyzer: Provides NLP and embedding operations.
        budget_manager: Tracks and enforces token budgets.
        article_repo: Data access layer for articles.
        metrics: Telemetry and metrics collection.
        rulebook: Optional style and tone guidelines.
    """

    def __init__(
        self,
        llm_client: UnifiedLLMClient,
        context_synthesizer: ContextSynthesizer,
        semantic_analyzer: SemanticAnalyzer,
        token_budget_manager: TokenBudgetManager,
        article_repository: ArticleRepository,
        metrics_collector: MetricsCollector,
        rulebook: dict | None = None,
    ):
        self.llm_client = llm_client
        self.context_synthesizer = context_synthesizer
        self.semantic_analyzer = semantic_analyzer
        self.budget_manager = token_budget_manager
        self.article_repo = article_repository
        self.metrics = metrics_collector
        self.rulebook = rulebook or {}

    def _get_rulebook_value(self, key: str, default: str) -> str:
        """Safely fetch a rulebook attribute from dicts or objects."""
        if not self.rulebook:
            return default
        if isinstance(self.rulebook, dict):
            return self.rulebook.get(key, default)
        return getattr(self.rulebook, key, default)

    async def generate_article(
        self,
        project_id: UUID,
        plan: ContentPlan,
        planning_model: Optional[str] = None,
        writing_model: Optional[str] = None,
        verification_model: Optional[str] = None,
        temperature: Optional[float] = None,
        **kwargs,
    ) -> GeneratedArticle:
        settings = get_settings()
        planning_model = planning_model or settings.llm.planning_model
        writing_model = writing_model or settings.llm.writing_model
        verification_model = verification_model or settings.llm.verification_model

        # Read temperature explicitly from kwargs (defaults to 0.7 if not set)
        temperature = temperature if temperature is not None else kwargs.get("temperature", 0.7)
        if temperature is None:
            temperature = 0.7

        overall_start_time = datetime.now(timezone.utc)
        """Generate complete article from content plan with quality assurance.

        Workflow:
            1. Validate budget
            2. Generate sections in parallel (asyncio.gather)
            3. Refine style using critic/editor pattern
            4. Verify facts for accuracy
            5. Calculate quality metrics
            6. Save to repository

        Args:
            project_id: Project UUID for budget tracking.
            plan: ContentPlan with outline, keywords, and structure.
            planning_model: Model used for reasoning/outline refinement steps.
            writing_model: Model used for drafting and stylistic refinement.
            verification_model: Model used for fact-checking passes.

        Returns:
            Generated article with content, metrics, and metadata.

        Raises:
            WorkflowError: If budget insufficient or all sections fail.
        """
        logger.info(f"Starting article generation for plan: {plan.outline.title}")
        logger.info(
            "Using models | planning=%s | writing=%s | verification=%s",
            planning_model,
            writing_model,
            verification_model,
        )
        start_time = time.perf_counter()

        # Check budget before starting
        estimated_tokens = len(plan.outline.sections) * 500  # Rough estimate: 500 tokens/section
        estimated_cost = len(plan.outline.sections) * 0.05  # Rough estimate
        if not await self.budget_manager.can_afford(
            tokens=estimated_tokens,
            cost=estimated_cost,
        ):
            raise WorkflowError(
                f"Insufficient budget for project {project_id}. "
                f"Estimated cost: ${estimated_cost:.2f}"
            )

        # Stage 2: Generate Content Sections
        logger.info(
            f"Starting section generation | plan_id={plan.id} | "
            f"sections={len(plan.outline.sections)} | temperature={temperature}"
        )
        section_tasks = [
            self._generate_section(project_id, plan, section, writing_model, temperature)
            for section in plan.outline.sections
        ]

        section_results = await asyncio.gather(*section_tasks, return_exceptions=True)

        # Check for failures in parallel execution
        article_content_parts = []
        total_tokens = 0
        total_cost = 0.0

        for i, result in enumerate(section_results):
            if isinstance(result, Exception):
                logger.error(f"Section {i} failed: {result}")
                # Continue with other sections rather than failing completely
                article_content_parts.append(
                    f"<h2>{plan.outline.sections[i].heading}</h2>\n"
                    f"<p><em>[Content generation failed for this section]</em></p>\n"
                )
            else:
                formatted_section, tokens, cost = result
                article_content_parts.append(formatted_section)
                total_tokens += tokens
                total_cost += cost

        # Assemble draft article
        draft_content = "\n".join(article_content_parts)
        generation_time = time.perf_counter() - start_time

        # Handle case where all sections failed
        successful_sections = sum(1 for r in section_results if not isinstance(r, Exception))
        if successful_sections == 0:
            raise WorkflowError("No content sections were generated - all sections failed")

        logger.info(
            f"✓ Generated {len(article_content_parts)} sections in {generation_time:.1f}s "
            f"({total_tokens} tokens, ${total_cost:.4f})"
        )

        # Refine (Style) - Critic/Editor pattern
        logger.info("Refining content for style compliance...")
        refined_content, refine_tokens, refine_cost = await self._refine_content(
            draft_content, plan, writing_model
        )
        total_tokens += refine_tokens
        total_cost += refine_cost

        # Verify (Facts) - LLM hallucination self-check
        logger.info("Verifying facts in generated content...")
        verified_content, verify_tokens, verify_cost = await self._verify_facts(
            refined_content, verification_model
        )
        total_tokens += verify_tokens
        total_cost += verify_cost

        # Run quality analysis
        quality_metrics = await self._run_quality_analysis(verified_content, plan.primary_keywords)

        # Validate content against original requirements
        logger.info("Validating content against requirements...")
        validation_details = self._validate_against_requirements(
            verified_content,
            plan,
            quality_metrics
        )

        # Extract keyword phrases for the article
        keywords = [kw.phrase for kw in plan.all_keywords] if hasattr(plan, 'all_keywords') else []

        # Create and save the article object
        article = GeneratedArticle(
            id=uuid.uuid4(),
            project_id=project_id,
            content_plan_id=plan.id,
            title=plan.outline.title,
            content=verified_content,
            meta_description=plan.outline.meta_description,
            keywords=keywords,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            generation_time_seconds=generation_time,
            quality_metrics=quality_metrics,
            validation_result=validation_details.validation_result,
            validation_details=validation_details,
            model_used=writing_model,
            status="completed",
            created_at=datetime.now(timezone.utc),
            updated_at=datetime.now(timezone.utc),
        )

        await self.article_repo.save_generated_article(article)

        # Export article to filesystem
        await self._export_article_to_files(article, plan)

        logger.success(
            f"✓ Article generated and saved: {article.title} "
            f"({total_tokens} tokens, ${total_cost:.4f}, {generation_time:.1f}s)"
        )
        return article

    async def _generate_section(
        self,
        project_id: UUID,
        plan: ContentPlan,
        section,
        writing_model: str,
        temperature: float = 0.7,
    ) -> tuple[str, int, float]:
        """Generate a single article section with budget tracking.

        Called concurrently for all sections via asyncio.gather.

        Args:
            project_id: Project UUID for budget recording.
            plan: Content plan providing context.
            section: Section specification with heading and intent.
            writing_model: LLM model to use for drafting.

        Returns:
            Tuple of (formatted_html, tokens_used, cost_usd).

        Raises:
            Exception: If section generation fails (caught by caller).
        """
        try:
            # Build section-specific prompt
            section_prompt = await self._build_section_prompt(plan, section)

            # Call LLM to generate section content
            response = await self.llm_client.complete(
                prompt=section_prompt,
                model=writing_model,
                temperature=temperature,
                max_tokens=2000,
            )

            # Clean LLM response before formatting
            section_content = self._clean_llm_section_output(response.content)

            # Don't wrap in <p> if content already has block-level HTML elements
            if re.search(r'<(?:p|ul|ol|h[2-4]|blockquote|div|table)', section_content, re.IGNORECASE):
                formatted_section = f"<h2>{section.heading}</h2>\n{section_content}\n"
            else:
                formatted_section = f"<h2>{section.heading}</h2>\n<p>{section_content}</p>\n"

            # No need to update budget manager here; can_afford already checked budget
            # In production, you could call report_actual_usage for post-hoc correction
            # await self.budget_manager.report_actual_usage(
            #     request_id=str(project_id),
            #     actual_tokens=response.usage.total_tokens,
            #     actual_cost=response.cost,
            # )

            logger.debug(
                f"✓ Section '{section.heading}' generated "
                f"({response.usage.total_tokens} tokens, ${response.cost:.4f})"
            )

            return formatted_section, response.usage.total_tokens, response.cost

        except Exception as e:
            logger.error(f"Failed to generate section '{section.heading}': {e}")
            raise

    def _clean_llm_section_output(self, content: str) -> str:
        """Clean LLM output by removing full HTML document wrappers and code fences."""
        # Strip markdown code fences (```html ... ``` or ``` ... ```)
        content = re.sub(r'```(?:html|HTML)?\s*\n?', '', content)

        # Remove full HTML document tags (DOCTYPE, html, head, body wrappers)
        content = re.sub(r'<!DOCTYPE[^>]*>', '', content, flags=re.IGNORECASE)
        content = re.sub(r'</?html[^>]*>', '', content, flags=re.IGNORECASE)
        # Remove entire <head>...</head> block
        content = re.sub(r'<head[^>]*>.*?</head>', '', content, flags=re.IGNORECASE | re.DOTALL)
        content = re.sub(r'</?body[^>]*>', '', content, flags=re.IGNORECASE)

        # Remove excessive whitespace left behind
        content = re.sub(r'\n{3,}', '\n\n', content)

        return content.strip()

    async def _build_section_prompt(self, plan: ContentPlan, section) -> str:
        """Build LLM prompt for writing a specific article section.

        Uses prompt compression for cost optimization.
        Supports both English and Persian (Farsi) output.

        Args:
            plan: Content plan with article context.
            section: Section specification with heading.

        Returns:
            Formatted prompt string for LLM.
        """
        target_word_count = getattr(plan, "target_word_count", None) or 500
        tone = self._get_rulebook_value("tone", "authoritative")
        style = self._get_rulebook_value("style", "concise")
        language = getattr(plan, "language", "en")

        # Language-specific instructions
        if language == "fa":
            language_instruction = """
CRITICAL: Write the entire article in Persian (Farsi/فارسی).
- Use formal Persian writing style (نوشتار رسمی فارسی)
- Write right-to-left (RTL)
- Use Persian numerals (۱, ۲, ۳) where appropriate
- Avoid unnecessary English loanwords when Persian equivalents exist
- Use proper Persian punctuation (،  ؛  ؟)
- Maintain a professional and authoritative tone in Persian
"""
        else:
            language_instruction = "Write the article in English."

        # Use prompt compression for cost optimization
        result = await prompt_compressor.compress_prompt(
            template_name="content_generation",
            variables={
                "topic": plan.outline.title,
                "context": section.heading,
                "tone": tone,
                "style": style,
                "structure": "Use <h2> for main sections, <h3> for subsections",
                "target_word_count": str(target_word_count),
                "word_count": str(target_word_count),
                "language_instruction": language_instruction,
            }
        )
        return result.compressed_prompt

    async def _refine_content(
        self, draft: str, plan: ContentPlan, writing_model: str
    ) -> tuple[str, int, float]:
        """Apply critic/editor pattern for style refinement.

        Two-step process:
        1. Critic: Review draft against style guidelines and identify issues
        2. Editor: Apply critique to improve content

        Args:
            draft: Initial generated content.
            plan: Content plan for context.
            model: LLM model to use.

        Returns:
            Tuple of (refined_content, tokens_used, cost_usd).

        Note:
            Skips editor step if critic finds no issues.
            Returns original draft if refinement fails.
        """
        total_tokens = 0
        total_cost = 0.0

        # Critic step: Ask LLM to critique the content
        rulebook_summary = ""
        if self.rulebook:
            rulebook_summary = (
                f"Tone: {self._get_rulebook_value('tone', 'N/A')}, "
                f"Style: {self._get_rulebook_value('style', 'N/A')}"
            )

        critic_prompt = f"""
        You are a quality assurance editor. Critique the following article draft against these style guidelines:
        {rulebook_summary if rulebook_summary else 'No specific style guidelines provided.'}

        Draft:
        {_smart_truncate_html(draft)}

        List specific issues (if any) related to:
        1. Tone consistency
        2. Clarity and readability
        3. Paragraph flow
        4. Any awkward phrasing

        Respond with a numbered list of issues. If no issues, say "No issues found."
        """

        try:
            critic_response = await self.llm_client.complete(
                prompt=critic_prompt, model=writing_model, temperature=0.3, max_tokens=500
            )
            total_tokens += critic_response.usage.total_tokens
            total_cost += critic_response.cost
            critique = critic_response.content
        except Exception as e:
            logger.warning(f"Critic step failed: {e}. Skipping refinement.")
            return draft, total_tokens, total_cost

        # If no issues, return draft as is
        if "no issues" in critique.lower():
            logger.debug("Critic found no issues; skipping editor step.")
            return draft, total_tokens, total_cost

        # Editor step: Ask LLM to apply the critique
        editor_prompt = f"""
        You are an expert editor. Apply the following critique to improve this article draft.

        Critique:
        {critique}

        Draft:
        {_smart_truncate_html(draft)}

        CRITICAL FORMAT REQUIREMENTS:
        - Output the improved draft in MARKDOWN format ONLY
        - Use ## for H2 headings, ### for H3 headings
        - Do NOT use HTML tags (no <h2>, <p>, <div>, etc.)
        - Do NOT include DOCTYPE, <html>, <head>, or <body> tags
        - Preserve the original structure and headings
        - Only fix the specific issues mentioned in the critique
        """

        try:
            editor_response = await self.llm_client.complete(
                prompt=editor_prompt, model=writing_model, temperature=0.4, max_tokens=4000
            )
            total_tokens += editor_response.usage.total_tokens
            total_cost += editor_response.cost
            refined = editor_response.content
        except Exception as e:
            logger.warning(f"Editor step failed: {e}. Returning original draft.")
            return draft, total_tokens, total_cost

        logger.debug("Content refinement complete.")
        return refined, total_tokens, total_cost

    async def _verify_facts(self, content: str, verification_model: str) -> tuple[str, int, float]:
        """Verify factual accuracy using LLM fact-checking.

        Process:
        1. Extract statistical claims, dates, and factual statements
        2. Rate confidence level for each claim (HIGH/MEDIUM/LOW)
        3. Add disclaimers or corrections for LOW confidence claims

        Args:
            content: Article content to verify.
            model: LLM model to use for fact-checking.

        Returns:
            Tuple of (verified_content, tokens_used, cost_usd).

        Note:
            Returns original content unchanged if verification fails.
        """
        total_tokens = 0
        total_cost = 0.0

        # Step 1: Extract and verify claims
        verify_prompt = f"""You are a fact-checker. Analyze this article for factual accuracy.

Article:
{_smart_truncate_html(content)}

For each factual claim (statistics, dates, percentages, named facts):
1. Extract the exact claim
2. Rate confidence: HIGH (verified/common knowledge), MEDIUM (plausible), LOW (suspect/unverifiable)
3. If LOW, suggest a correction or note it needs verification

Format your response as:
CLAIM: [exact claim from text]
CONFIDENCE: [HIGH/MEDIUM/LOW]
ISSUE: [only if LOW - what's wrong and suggested fix]
---

If no issues found, respond with: ALL_FACTS_VERIFIED"""

        try:
            verify_response = await self.llm_client.complete(
                prompt=verify_prompt, model=verification_model, temperature=0.1, max_tokens=1000
            )
            total_tokens += verify_response.usage.total_tokens
            total_cost += verify_response.cost
            verification_report = verify_response.content
        except Exception as e:
            logger.warning(f"Fact verification failed: {e}. Returning content as-is.")
            return content, total_tokens, total_cost

        # Log verification report for auditing
        logger.info(f"Fact verification report:\n{verification_report[:500]}...")

        # Step 2: If issues found, add disclaimers or request corrections
        if "ALL_FACTS_VERIFIED" in verification_report:
            logger.debug("All facts verified - no corrections needed")
            return content, total_tokens, total_cost

        # Parse LOW confidence claims
        low_confidence_claims = []
        for block in verification_report.split("---"):
            if "CONFIDENCE: LOW" in block:
                lines = block.strip().split("\n")
                claim = next((line.replace("CLAIM:", "").strip() for line in lines if "CLAIM:" in line), None)
                issue = next((line.replace("ISSUE:", "").strip() for line in lines if "ISSUE:" in line), None)
                if claim:
                    low_confidence_claims.append({"claim": claim, "issue": issue})

        if not low_confidence_claims:
            return content, total_tokens, total_cost

        # Step 3: Apply corrections or add disclaimers
        logger.warning(f"Found {len(low_confidence_claims)} low-confidence claims")

        correction_prompt = f"""You are an editor. The following article contains claims that need attention.

Article:
{_smart_truncate_html(content)}

Claims needing correction or disclaimer:
{chr(10).join(f"- {c['claim']}: {c['issue']}" for c in low_confidence_claims)}

Instructions:
1. For unverifiable statistics, replace with hedging language ("studies suggest", "approximately", "according to some estimates")
2. For incorrect facts, correct them if you know the right answer, otherwise remove or hedge
3. Preserve the article structure and flow
4. Output the corrected article only, no explanations

CRITICAL FORMAT REQUIREMENTS:
- Output in MARKDOWN format ONLY
- Use ## for H2 headings, ### for H3 headings
- Do NOT use HTML tags (no <h2>, <p>, <div>, etc.)
- Do NOT include DOCTYPE, <html>, <head>, or <body> tags"""

        try:
            correction_response = await self.llm_client.complete(
                prompt=correction_prompt, model=verification_model, temperature=0.3, max_tokens=4000
            )
            total_tokens += correction_response.usage.total_tokens
            total_cost += correction_response.cost
            corrected_content = correction_response.content

            logger.info(f"Applied corrections for {len(low_confidence_claims)} claims")
            return corrected_content, total_tokens, total_cost

        except Exception as e:
            logger.warning(f"Fact correction failed: {e}. Returning original with disclaimer.")
            # Add a general disclaimer if correction fails
            disclaimer = "\n\n---\n*Note: Some statistics in this article may require independent verification.*\n"
            return content + disclaimer, total_tokens, total_cost

    async def _run_quality_analysis(self, content: str, keywords: list[Keyword]) -> QualityMetrics:
        """
        Analyzes the generated content for quality metrics.
        (This is a simplified implementation. Real-world would be more complex.)
        """
        logger.debug("Running quality analysis...")
        import re as _re

        # Strip HTML tags for text analysis
        text_only = _re.sub(r'<[^>]+>', '', content)

        # Tokenize words and sentences
        words = text_only.split()
        # Split on sentence-ending punctuation followed by space or end of string
        sentences = [s.strip() for s in _re.split(r'[.!?]+', text_only) if s.strip()]
        word_count = len(words)
        sentence_count = max(len(sentences), 1)
        avg_sentence_length = word_count / sentence_count

        # Simple lexical diversity: unique words / total words
        unique_words = len(set(w.lower() for w in words if w.isalnum()))
        lexical_diversity = unique_words / max(word_count, 1)

        # Flesch-Kincaid Reading Ease calculation
        def _count_syllables(word: str) -> int:
            """Estimate syllable count using vowel-group heuristic."""
            word = word.lower().strip(".,!?;:'\"()-")
            if not word:
                return 1
            vowels = "aeiouy"
            count = 0
            prev_vowel = False
            for char in word:
                is_vowel = char in vowels
                if is_vowel and not prev_vowel:
                    count += 1
                prev_vowel = is_vowel
            # Silent 'e' at end
            if word.endswith("e") and count > 1:
                count -= 1
            return max(count, 1)

        total_syllables = sum(_count_syllables(w) for w in words if w.isalnum())
        avg_syllables_per_word = total_syllables / max(word_count, 1)

        # Flesch Reading Ease: 206.835 - 1.015*(words/sentences) - 84.6*(syllables/words)
        readability_score = max(0.0, min(100.0,
            206.835 - (1.015 * avg_sentence_length) - (84.6 * avg_syllables_per_word)
        ))

        # Calculate keyword density
        keyword_density = {}
        content_lower = content.lower()
        for kw in keywords:
            kw_lower = kw.phrase.lower()
            count = content_lower.count(kw_lower)
            density = count / max(word_count, 1)
            keyword_density[kw.phrase] = density

        return QualityMetrics(
            word_count=word_count,
            readability_score=readability_score,
            lexical_diversity=lexical_diversity,
            keyword_density=keyword_density,
            avg_sentence_length=avg_sentence_length,
            paragraph_count=len([p for p in content.split("\n\n") if p.strip()]),
        )

    def _validate_against_requirements(
        self,
        content: str,
        plan: ContentPlan,
        quality_metrics: QualityMetrics
    ) -> "ValidationDetails":
        """
        Validate generated content against original requirements from ContentPlan.

        Compares the generated article against the original constraints:
        - Word count target
        - Required keywords (primary)
        - Required sections from outline
        - Language consistency

        Args:
            content: Generated article content
            plan: Original content plan with requirements
            quality_metrics: Computed quality metrics for the article

        Returns:
            ValidationDetails with specific issues and overall validation result
        """
        from core.enums import ValidationResult
        from core.models import ValidationDetails, ValidationIssue

        issues = []

        # 1. Validate word count
        target_word_count = plan.target_word_count
        actual_word_count = quality_metrics.word_count
        word_count_deviation_pct = ((actual_word_count - target_word_count) / target_word_count) * 100

        word_count_met = True
        if abs(word_count_deviation_pct) > 30:  # More than 30% deviation is an error
            word_count_met = False
            issues.append(ValidationIssue(
                severity="error",
                category="word_count",
                message=f"Word count significantly deviates from target",
                expected=f"{target_word_count} words (±30%)",
                actual=f"{actual_word_count} words ({word_count_deviation_pct:+.1f}% deviation)",
                suggestion=f"{'Add' if actual_word_count < target_word_count else 'Remove'} approximately {abs(actual_word_count - target_word_count)} words"
            ))
        elif abs(word_count_deviation_pct) > 15:  # 15-30% deviation is a warning
            issues.append(ValidationIssue(
                severity="warning",
                category="word_count",
                message=f"Word count moderately deviates from target",
                expected=f"{target_word_count} words",
                actual=f"{actual_word_count} words ({word_count_deviation_pct:+.1f}% deviation)",
                suggestion=f"Consider {'expanding' if actual_word_count < target_word_count else 'trimming'} the content"
            ))

        # 2. Validate primary keywords presence
        content_lower = content.lower()
        missing_keywords = []
        for keyword in plan.primary_keywords:
            kw_phrase = keyword.phrase.lower()
            if kw_phrase not in content_lower:
                missing_keywords.append(keyword.phrase)

        keywords_met = len(missing_keywords) == 0
        if missing_keywords:
            severity = "error" if len(missing_keywords) > len(plan.primary_keywords) // 2 else "warning"
            issues.append(ValidationIssue(
                severity=severity,
                category="keywords",
                message=f"Missing {len(missing_keywords)} primary keyword(s)",
                expected=f"All {len(plan.primary_keywords)} primary keywords present",
                actual=f"{len(missing_keywords)} keyword(s) not found: {', '.join(missing_keywords[:3])}{'...' if len(missing_keywords) > 3 else ''}",
                suggestion=f"Add the missing keywords naturally: {', '.join(missing_keywords)}"
            ))

        # 3. Validate required sections presence
        missing_sections = []
        for section in plan.outline.sections:
            section_heading = section.heading.lower()
            # Check if heading appears in content (as Markdown heading or HTML heading)
            if section_heading not in content_lower:
                missing_sections.append(section.heading)

        structure_met = len(missing_sections) == 0
        if missing_sections:
            severity = "warning" if len(missing_sections) < 3 else "error"
            issues.append(ValidationIssue(
                severity=severity,
                category="structure",
                message=f"Missing {len(missing_sections)} required section(s) from outline",
                expected=f"All {len(plan.outline.sections)} outlined sections present",
                actual=f"{len(missing_sections)} section(s) not found: {', '.join(missing_sections[:3])}{'...' if len(missing_sections) > 3 else ''}",
                suggestion=f"Add missing sections: {', '.join(missing_sections)}"
            ))

        # 4. Determine overall validation result
        error_count = sum(1 for issue in issues if issue.severity == "error")
        warning_count = sum(1 for issue in issues if issue.severity == "warning")

        if error_count > 0:
            validation_result = ValidationResult.FAIL
        elif warning_count > 0:
            validation_result = ValidationResult.PASS_WITH_WARNINGS
        else:
            validation_result = ValidationResult.PASS

        logger.info(
            f"Validation complete: {validation_result.value} "
            f"(errors={error_count}, warnings={warning_count})"
        )

        return ValidationDetails(
            validation_result=validation_result,
            issues=issues,
            word_count_met=word_count_met,
            word_count_deviation_pct=word_count_deviation_pct,
            keywords_met=keywords_met,
            missing_keywords=missing_keywords,
            structure_met=structure_met,
            missing_sections=missing_sections,
        )

    def _sanitize_content_for_html(self, content: str) -> str:
        """
        Sanitize article content before inserting into HTML wrapper.

        This function:
        1. Detects if content already contains HTML tags
        2. Extracts body content if it's a full HTML document
        3. Converts Markdown to clean HTML
        4. Removes any duplicate DOCTYPE/html/head/body tags

        Args:
            content: Raw article content (may be Markdown or HTML)

        Returns:
            Clean HTML content ready to be wrapped
        """
        import re

        from bs4 import BeautifulSoup

        # Try to import markdown library, use fallback if not available
        try:
            import markdown
            has_markdown = True
        except ImportError:
            has_markdown = False
            logger.debug("Markdown library not available, using regex fallback")

        # Check if content contains HTML tags
        has_html_tags = bool(re.search(r'<(?:html|head|body|!DOCTYPE)', content, re.IGNORECASE))

        if has_html_tags:
            # Content already has HTML - extract body content only
            try:
                soup = BeautifulSoup(content, 'html.parser')

                # Try to find body tag first
                body = soup.find('body')
                if body:
                    # Extract everything inside body, excluding the body tag itself
                    clean_content = ''.join(str(child) for child in body.children)
                else:
                    # No body tag - just remove DOCTYPE, html, head tags
                    for tag_name in ['!DOCTYPE', 'html', 'head', 'meta', 'title', 'style', 'script']:
                        for tag in soup.find_all(tag_name):
                            tag.decompose()
                    clean_content = str(soup)

                # Remove empty lines and excessive whitespace
                clean_content = re.sub(r'\n\s*\n', '\n\n', clean_content)
                return clean_content.strip()
            except Exception as e:
                logger.warning(f"Failed to parse HTML content, using as-is: {e}")
                # Fallback - remove obvious HTML wrapper tags with regex
                content = re.sub(r'<!DOCTYPE[^>]*>', '', content, flags=re.IGNORECASE)
                content = re.sub(r'</?html[^>]*>', '', content, flags=re.IGNORECASE)
                content = re.sub(r'</?head[^>]*>', '', content, flags=re.IGNORECASE)
                content = re.sub(r'</?body[^>]*>', '', content, flags=re.IGNORECASE)
                return content.strip()
        else:
            # Content is Markdown - convert to HTML
            if has_markdown:
                try:
                    # Convert Markdown to HTML with extensions for better formatting
                    html_content = markdown.markdown(
                        content,
                        extensions=['extra', 'nl2br', 'sane_lists']
                    )
                    return html_content
                except Exception as e:
                    logger.warning(f"Failed to convert Markdown to HTML: {e}")
                    has_markdown = False  # Fall through to regex conversion

            # Fallback - basic Markdown conversion with regex
            content = re.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re.MULTILINE)
            content = re.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re.MULTILINE)
            content = re.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re.MULTILINE)
            content = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
            content = re.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
            # Convert paragraphs (double newlines to </p><p>)
            content = re.sub(r'\n\n+', '</p>\n<p>', content)
            # Wrap in paragraph tags if not already wrapped
            if not content.strip().startswith('<'):
                content = f'<p>{content}</p>'
            return content

    async def _export_article_to_files(self, article, plan) -> None:
        """
        Export generated article to filesystem in HTML and Markdown formats.

        Articles are saved to ./exports/{project_id}/{date}/{article_id}/
        with both .html and .md files for flexibility.

        Args:
            article: Generated article with content and metadata
            plan: Content plan with language setting
        """
        import os
        import re
        from pathlib import Path

        try:
            # Get language from plan
            language = getattr(plan, "language", "fa")
            is_rtl = language == "fa"

            # Create export directory structure
            date_str = article.created_at.strftime("%Y-%m-%d")
            export_base = Path("exports") / str(article.project_id) / date_str / str(article.id)
            export_base.mkdir(parents=True, exist_ok=True)

            # Sanitize title for filename (remove special chars)
            safe_title = re.sub(r'[^\w\s\u0600-\u06FF-]', '', article.title)[:50].strip()
            if not safe_title:
                safe_title = str(article.id)[:8]

            # Sanitize article content before wrapping in HTML
            # This removes duplicate DOCTYPE/HTML tags and converts Markdown to clean HTML
            clean_article_content = self._sanitize_content_for_html(article.content)

            # Generate HTML content with proper RTL support (escape user content for safety)
            safe_meta = html.escape(article.meta_description or "", quote=True)
            safe_keywords = html.escape(', '.join(article.keywords or []), quote=True)
            safe_html_title = html.escape(article.title or "", quote=True)
            html_content = f'''<!DOCTYPE html>
<html lang="{language}" dir="{"rtl" if is_rtl else "ltr"}">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <meta name="description" content="{safe_meta}">
    <meta name="keywords" content="{safe_keywords}">
    <meta name="generator" content="Smarlux Studio">
    <title>{safe_html_title}</title>
    <style>
        * {{ box-sizing: border-box; }}
        body {{
            font-family: {"'Vazirmatn', 'B Nazanin', Tahoma" if is_rtl else "'Inter', 'Segoe UI', Arial"}, sans-serif;
            line-height: 2;
            padding: 40px;
            max-width: 900px;
            margin: 0 auto;
            background: #fafafa;
            color: #333;
        }}
        h1 {{
            color: #1a1a2e;
            border-bottom: 3px solid #4f46e5;
            padding-bottom: 15px;
            margin-bottom: 30px;
        }}
        h2 {{ color: #2c3e50; margin-top: 40px; }}
        h3 {{ color: #34495e; }}
        p {{ margin: 15px 0; text-align: justify; }}
        .meta {{
            background: #f0f0f0;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 30px;
            font-size: 0.9em;
        }}
        .meta strong {{ color: #4f46e5; }}
        .keywords {{
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-top: 10px;
        }}
        .keyword {{
            background: #4f46e5;
            color: white;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.85em;
        }}
        footer {{
            margin-top: 50px;
            padding-top: 20px;
            border-top: 1px solid #ddd;
            font-size: 0.8em;
            color: #888;
        }}
    </style>
</head>
<body>
    <article>
        <h1>{article.title}</h1>

        <div class="meta">
            <p><strong>{"توضیحات متا:" if is_rtl else "Meta Description:"}</strong> {article.meta_description}</p>
            <p><strong>{"تعداد کلمات:" if is_rtl else "Word Count:"}</strong> {article.quality_metrics.word_count}</p>
            <div class="keywords">
                {"".join(f'<span class="keyword">{kw}</span>' for kw in article.keywords[:10])}
            </div>
        </div>

        {clean_article_content}
    </article>

    <footer>
        <p>{"تولید شده توسط اسمارلوکس استودیو" if is_rtl else "Generated by Smarlux Studio"} | {article.created_at.strftime("%Y-%m-%d %H:%M")}</p>
        <p>{"هزینه تولید:" if is_rtl else "Generation Cost:"} ${article.total_cost_usd:.4f} | {"زمان تولید:" if is_rtl else "Time:"} {article.generation_time_seconds:.1f}s</p>
    </footer>
</body>
</html>'''

            # Generate Markdown content
            md_content = f'''# {article.title}

> {article.meta_description}

**{"کلمات کلیدی" if is_rtl else "Keywords"}:** {", ".join(article.keywords)}

---

{article.content}

---

*{"تولید شده توسط اسمارلوکس استودیو" if is_rtl else "Generated by Smarlux Studio"}*
*{article.created_at.strftime("%Y-%m-%d %H:%M")} | {"تعداد کلمات" if is_rtl else "Words"}: {article.quality_metrics.word_count} | {"هزینه" if is_rtl else "Cost"}: ${article.total_cost_usd:.4f}*
'''

            # Write files
            html_path = export_base / f"{safe_title}.html"
            md_path = export_base / f"{safe_title}.md"

            with open(html_path, "w", encoding="utf-8") as f:
                f.write(html_content)

            with open(md_path, "w", encoding="utf-8") as f:
                f.write(md_content)

            logger.info(
                f"✓ Article exported to files | "
                f"html={html_path} | md={md_path}"
            )

        except Exception as e:
            # Don't fail the entire generation if export fails
            logger.warning(f"Failed to export article to files: {e}")
