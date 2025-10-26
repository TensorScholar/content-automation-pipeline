"""
Content Generator - Strategic Article Execution Engine
=====================================================

The ContentGenerator executes ContentPlan sections by:
- Iterating through plan sections sequentially
- Using ModelRouter for CREATIVE_GENERATION capability
- Checking budget before each section generation
- Implementing quality and SEO analysis
- Persisting final articles via ArticleRepository

Architecture: Orchestration pattern with dependency injection
"""

import time
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional
from uuid import UUID

from loguru import logger

from core.exceptions import WorkflowError
from core.models import ContentPlan, GeneratedArticle, Keyword, QualityMetrics
from infrastructure.llm_client import AbstractLLMClient
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.article_repository import ArticleRepository
from optimization.model_router import ModelCapability, ModelRouter, RoutingTask, TaskComplexity
from optimization.token_budget_manager import TokenBudgetManager


class ContentGenerator:
    """
    Strategic content generation system.

    Executes ContentPlan sections by orchestrating model routing,
    budget management, and quality analysis to produce final articles.
    """

    def __init__(
        self,
        model_router: ModelRouter,
        llm_client: AbstractLLMClient,
        context_synthesizer: ContextSynthesizer,
        semantic_analyzer: SemanticAnalyzer,
        token_budget_manager: TokenBudgetManager,
        article_repository: ArticleRepository,
        metrics_collector: MetricsCollector,
    ):
        self.model_router = model_router
        self.llm_client = llm_client
        self.context_synthesizer = context_synthesizer
        self.semantic_analyzer = semantic_analyzer
        self.budget_manager = token_budget_manager
        self.article_repo = article_repository
        self.metrics = metrics_collector

    async def generate_article(
        self,
        project_id: UUID,
        plan: ContentPlan,
    ) -> GeneratedArticle:
        """
        Generates a full article based on the provided content plan.
        """
        logger.info(f"Starting article generation for plan: {plan.outline.title}")
        start_time = time.perf_counter()

        article_content_parts = []
        total_tokens = 0
        total_cost = 0.0

        # 1. Iterate through each section in the plan
        for section in plan.outline.sections:
            logger.debug(f"Generating section: {section.heading}")

            # 2. Check budget BEFORE generation
            if not await self.budget_manager.can_afford(
                project_id=project_id, estimated_cost=0.05  # Assume a small cost per section
            ):
                logger.warning(
                    f"Token budget exceeded for project {project_id}. Stopping generation."
                )
                break  # Stop generation if budget is hit

            # 3. Build section-specific prompt
            section_prompt = self._build_section_prompt(plan, section)

            # 4. Route for *Creative Generation*
            routing_task = RoutingTask(
                task_id=f"gen_{plan.id}_{section.heading[:20]}",
                capability_required=ModelCapability.CREATIVE_GENERATION,
                complexity=TaskComplexity.MODERATE,
                estimated_input_tokens=len(section_prompt) // 3,
                estimated_output_tokens=plan.target_word_count // len(plan.outline.sections),
            )
            decision = await self.model_router.route(routing_task)

            # 5. Call LLM to generate section content
            response = await self.llm_client.complete(
                prompt=section_prompt,
                model=decision.selected_model,
                temperature=0.7,  # Generation can be more creative
                max_tokens=2000,
            )

            # 6. Format and append content (e.g., wrap in <h2> and <p>)
            formatted_section = f"<h2>{section.heading}</h2>\n<p>{response.content}</p>\n"
            article_content_parts.append(formatted_section)

            # 7. Update costs and report to budget manager
            total_tokens += response.usage.total_tokens
            total_cost += response.cost
            await self.budget_manager.record_usage(
                project_id=project_id, tokens=response.usage.total_tokens, cost=response.cost
            )

        # 8. Assemble final article
        final_content = "\n".join(article_content_parts)
        generation_time = time.perf_counter() - start_time

        # 9. Run quality analysis
        quality_metrics = await self._run_quality_analysis(final_content, plan.primary_keywords)

        # 10. Create and save the article object
        article = GeneratedArticle(
            id=uuid.uuid4(),
            project_id=project_id,
            content_plan_id=plan.id,
            title=plan.outline.title,
            content=final_content,
            meta_description=plan.outline.meta_description,
            total_tokens_used=total_tokens,
            total_cost_usd=total_cost,
            generation_time_seconds=generation_time,
            quality_metrics=quality_metrics,
            model_used=decision.selected_model,
            status="completed",
            created_at=datetime.utcnow(timezone.utc),
            updated_at=datetime.utcnow(timezone.utc),
        )

        await self.article_repo.save_generated_article(article)
        logger.success(f"Article generated and saved: {article.title}")
        return article

    def _build_section_prompt(self, plan: ContentPlan, section) -> str:
        """Constructs the prompt to instruct the LLM to *write* a specific section."""
        return f"""
        Act as an expert writer. Your task is to write a single, comprehensive section for an article.
        
        **Article Title:** {plan.outline.title}
        **Target Audience:** Professional audience
        **Tone:** Authoritative and informative
        
        **Section to Write:** {section.heading}
        
        **Instructions for this section (You MUST follow these):**
        Write comprehensive content for the "{section.heading}" section. Include relevant information, examples, and insights that would be valuable to readers.
        
        Write *only* the text for this section. Do NOT include the heading. Do NOT add any preamble or sign-off.
        """

    async def _run_quality_analysis(self, content: str, keywords: List[Keyword]) -> QualityMetrics:
        """
        Analyzes the generated content for quality metrics.
        (This is a simplified implementation. Real-world would be more complex.)
        """
        logger.debug("Running quality analysis...")
        # Use existing SemanticAnalyzer for text processing
        word_count = len(self.semantic_analyzer.tokenize(content))
        sentence_count = len(self.semantic_analyzer.sentencize(content))
        avg_sentence_length = word_count / max(sentence_count, 1)
        lexical_diversity = await self.semantic_analyzer.calculate_lexical_diversity(content)

        # Placeholder for readability
        readability_score = 70.0  # Placeholder

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
