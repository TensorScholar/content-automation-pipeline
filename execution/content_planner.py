"""
Content Planner - Strategic Content Architecture
===============================================

The ContentPlanner orchestrates the creation of strategic content plans by:
- Synthesizing context from all decision layers
- Routing to optimal reasoning models
- Generating structured JSON plans via LLM
- Validating and persisting plans

Architecture: Orchestration pattern with dependency injection
"""

import json
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

from loguru import logger

from core.exceptions import WorkflowError
from core.models import ContentPlan, Keyword, Project, ProjectContext
from infrastructure.llm_client import AbstractLLMClient
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
from knowledge.article_repository import ArticleRepository
from optimization.model_router import ModelCapability, ModelRouter, RoutingTask, TaskComplexity


class ContentPlanner:
    """
    Strategic content planning system.

    Orchestrates the creation of content plans by synthesizing context,
    routing to optimal models, and generating structured plans via LLM.
    """

    def __init__(
        self,
        decision_engine: DecisionEngine,
        context_synthesizer: ContextSynthesizer,
        model_router: ModelRouter,
        llm_client: AbstractLLMClient,  # Use AbstractLLMClient for type hinting
        article_repository: ArticleRepository,
        metrics_collector: MetricsCollector,
    ):
        self.decision_engine = decision_engine
        self.context_synthesizer = context_synthesizer
        self.model_router = model_router
        self.llm_client = llm_client
        self.article_repo = article_repository
        self.metrics = metrics_collector

    async def create_content_plan(
        self,
        project: Project,
        topic: str,
        keywords: List[Keyword],
        custom_instructions: Optional[str] = None,
    ) -> ContentPlan:
        """
        Generates a strategic content plan (outline) for a given topic.
        """
        logger.info(f"Creating content plan for topic: {topic}")

        # 1. Synthesize all available context (L1, L2, L3)
        context = await self.context_synthesizer.synthesize_context(
            project_id=project.id,
            topic=topic,
            keywords=keywords,
            custom_instructions=custom_instructions,
        )

        # 2. Build the LLM prompt for *planning*
        planning_prompt = self._build_planning_prompt(topic, keywords, context)

        # 3. Route to find the best *reasoning* model
        routing_task = RoutingTask(
            task_id=f"plan_{project.id}_{uuid.uuid4()}",
            capability_required=ModelCapability.REASONING,
            complexity=TaskComplexity.COMPLEX,
            estimated_input_tokens=len(planning_prompt) // 3,  # Rough estimate
            estimated_output_tokens=1000,
        )
        decision = await self.model_router.route(routing_task)
        logger.debug(f"Routing content plan to model: {decision.selected_model}")

        # 4. Call the LLM to get the plan (as JSON)
        try:
            response = await self.llm_client.complete(
                prompt=planning_prompt,
                model=decision.selected_model,
                temperature=0.3,  # Planning should be deterministic
                max_tokens=2000,
            )

            # 5. Parse and Validate the plan
            plan_data = self._parse_llm_json_response(response.content)

            # Create Outline from sections
            from core.models import Outline, Section

            sections = [
                Section(
                    heading=section["heading"],
                    theme_embedding=[],  # Will be populated during generation
                    target_keywords=[kw.phrase for kw in keywords[:3]],  # Use first 3 keywords
                    estimated_words=plan_data.get("target_word_count", 1500)
                    // len(plan_data["sections"]),
                    intent="explain",  # Default intent
                )
                for section in plan_data["sections"]
            ]

            outline = Outline(
                title=plan_data["title"],
                meta_description=plan_data["meta_description"],
                sections=sections,
            )

            plan = ContentPlan(
                id=uuid.uuid4(),
                project_id=project.id,
                topic=topic,
                primary_keywords=keywords[:3] if len(keywords) >= 3 else keywords,
                secondary_keywords=keywords[3:10] if len(keywords) > 3 else [],
                outline=outline,
                target_word_count=plan_data.get("target_word_count", 1500),
                readability_target="grade_10-12",
                estimated_cost_usd=0.0,
                created_at=datetime.now(timezone.utc),
            )

            # 6. Save and return
            await self.article_repo.save_content_plan(plan)
            logger.info(f"Content plan created successfully: {plan.outline.title}")
            return plan

        except Exception as e:
            logger.error(f"Failed to create content plan: {e}")
            raise WorkflowError(f"ContentPlanner failed: {e}")

    def _build_planning_prompt(
        self, topic: str, keywords: List[Keyword], context: ProjectContext
    ) -> str:
        """Constructs the prompt to instruct the LLM to generate a JSON content plan."""

        keyword_list = ", ".join([kw.phrase for kw in keywords])

        # We must instruct the LLM to return *only* JSON that matches our Pydantic model
        return f"""
        Act as a world-class SEO content strategist. Your task is to create a detailed content plan for an article.
        
        **Topic:** {topic}
        **Keywords:** {keyword_list}
        
        **Strategic Context & Rules (You MUST follow these):**
        - **Target Audience:** {context.target_audience}
        - **Tone:** {context.tone}
        - **Style Guide:** {context.style_guide}
        - **Custom Instructions:** {context.custom_instructions or 'N/A'}
        
        **Your Output MUST be a single JSON object (no markdown, no preamble) matching this exact schema:**
        {{
          "title": "A compelling, SEO-optimized title for the article.",
          "meta_description": "A 155-character meta description.",
          "target_word_count": 1500,
          "target_audience": "The specific audience persona.",
          "tone": "The primary tone (e.g., 'Authoritative', 'Technical').",
          "sections": [
            {{"heading": "Section 1 Title", "prompt": "A detailed prompt for the AI writer for this section, listing key points and keywords to include."}},
            {{"heading": "Section 2 Title", "prompt": "Detailed prompt for section 2..."}},
            {{"heading": "Conclusion", "prompt": "Detailed prompt for the conclusion, summarizing key takeaways."}}
          ]
        }}
        
        JSON ONLY:
        """

    def _parse_llm_json_response(self, response_content: str) -> dict:
        """Safely parses the LLM's JSON output."""
        try:
            # Clean the response (LLMs sometimes add markdown)
            json_str = response_content.strip().lstrip("```json").rstrip("```")
            return json.loads(json_str)
        except json.JSONDecodeError as e:
            logger.error(f"Failed to decode LLM JSON response: {e}")
            raise WorkflowError(
                f"ContentPlanner received invalid JSON from LLM: {response_content}"
            )
