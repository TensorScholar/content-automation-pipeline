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

from core.enums import SectionIntent
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
        language: str = "fa",  # Default to Persian for this platform
        target_word_count: int = 1500,
        seo_settings: Optional[Dict] = None,
        **kwargs
    ) -> ContentPlan:
        """
        Generates a strategic content plan (outline) for a given topic.

        Args:
            language: Content language - 'fa' for Persian (Farsi), 'en' for English
            target_word_count: Target word count for the article
            seo_settings: Dictionary of SEO settings (images, LSI, etc.)
        """
        logger.info(f"Creating content plan for topic: {topic} | words={target_word_count}")

        # 1. Synthesize all available context (L1, L2, L3)
        context = await self.context_synthesizer.synthesize_context(
            project_id=project.id,
            topic=topic,
            keywords=keywords,
            custom_instructions=custom_instructions,
        )

        # Override context with specific parameters if provided
        if kwargs.get("tone"):
            context.tone = kwargs.get("tone")
        if kwargs.get("target_audience"):
            context.target_audience = kwargs.get("target_audience")
        if kwargs.get("content_structure"):
            context.style_guide = kwargs.get("content_structure")

        # 2. Build the LLM prompt for *planning*
        content_memory_guidance = await self._load_content_memory_guidance(project.id, topic)
        planning_prompt = self._build_planning_prompt(
            topic,
            keywords,
            context,
            language,
            target_word_count,
            seo_settings,
            content_memory_guidance,
        )

        # 3. Route to find the best *reasoning* model
        routing_task = RoutingTask(
            task_id=f"plan_{project.id}_{uuid.uuid4()}",
            capability_required=ModelCapability.REASONING,
            complexity=TaskComplexity.COMPLEX,
            estimated_input_tokens=len(planning_prompt) // 3,  # Rough estimate
            estimated_output_tokens=1000,
        )
        model_override = kwargs.get("model_override")
        if model_override:
            selected_model = model_override
            logger.debug(f"Routing content plan to requested model override: {selected_model}")
        else:
            decision = await self.model_router.route(routing_task)
            selected_model = decision.selected_model
            logger.debug(f"Routing content plan to model: {selected_model}")

        # 4. Call the LLM to get the plan (as JSON)
        try:
            response = await self.llm_client.complete(
                prompt=planning_prompt,
                model=selected_model,
                temperature=0.3,  # Planning should be deterministic
                max_tokens=4000,  # Increased for full content plans
            )

            # 5. Parse and Validate the plan
            # Handle both LLMResponse objects and raw dict responses
            if isinstance(response, dict):
                content = response.get('content')
                planning_cost = float(response.get("cost") or 0.0)
            else:
                content = getattr(response, 'content', None)
                planning_cost = float(getattr(response, "cost", 0.0) or 0.0)
            if not isinstance(content, str) or not content.strip():
                raise WorkflowError("Content planning provider returned an empty response")
            plan_data = self._parse_llm_json_response(content)

            # Create Outline from sections
            from core.models import Outline, Section

            section_count = len(plan_data["sections"])
            base_words, remainder = divmod(target_word_count, section_count)
            sections = [
                Section(
                    heading=section["heading"],
                    theme_embedding=[],  # Will be populated during generation
                    target_keywords=[kw.phrase for kw in keywords[:3]],  # Use first 3 keywords
                    estimated_words=base_words + (1 if index < remainder else 0),
                    intent=SectionIntent.EXPLAIN,
                )
                for index, section in enumerate(plan_data["sections"])
            ]

            # Truncate meta_description if too long (max 160 chars)
            meta_description = plan_data.get("meta_description", "")[:160]
            if len(meta_description) == 160:
                # Truncate at last space to avoid cutting words
                meta_description = meta_description[:meta_description.rfind(' ')] + '...'

            outline = Outline(
                title=plan_data["title"],
                meta_description=meta_description,
                sections=sections,
            )

            # Convert dataclass keywords to Pydantic Keyword models
            from core.enums import KeywordIntent
            from core.models import Keyword as PydanticKeyword

            def convert_keyword(kw) -> PydanticKeyword:
                """Convert dataclass Keyword to Pydantic Keyword."""
                # Map intent string to enum (only 4 valid values in KeywordIntent)
                intent_map = {
                    "informational": KeywordIntent.INFORMATIONAL,
                    "commercial": KeywordIntent.COMMERCIAL,
                    "transactional": KeywordIntent.TRANSACTIONAL,
                    "navigational": KeywordIntent.NAVIGATIONAL,
                }
                raw_intent = getattr(kw, 'primary_intent', 'informational')
                # Fallback to INFORMATIONAL for unknown intents (e.g., "local")
                intent = intent_map.get(raw_intent.lower() if raw_intent else 'informational', KeywordIntent.INFORMATIONAL)

                return PydanticKeyword(
                    phrase=kw.phrase,
                    search_volume=kw.metrics.get("search_volume") if hasattr(kw, 'metrics') else None,
                    difficulty=kw.metrics.get("difficulty") if hasattr(kw, 'metrics') else None,
                    intent=intent,
                    embedding=kw.embedding if hasattr(kw, 'embedding') else None,
                )

            primary_kws = [convert_keyword(kw) for kw in (keywords[:3] if len(keywords) >= 3 else keywords)]
            secondary_kws = [convert_keyword(kw) for kw in (keywords[3:10] if len(keywords) > 3 else [])]

            plan = ContentPlan(
                id=uuid.uuid4(),
                project_id=project.id,
                topic=topic,
                primary_keywords=primary_kws,
                secondary_keywords=secondary_kws,
                outline=outline,
                target_word_count=target_word_count,
                readability_target="grade_10-12",
                language=language,  # Pass language through to generation
                estimated_cost_usd=max(0.0, planning_cost),
                created_at=datetime.now(timezone.utc).replace(
                    tzinfo=None
                ),  # PostgreSQL column stores naive UTC.
            )

            # 6. Save and return
            await self.article_repo.save_content_plan(plan)
            logger.info(f"Content plan created successfully: {plan.outline.title}")
            return plan

        except Exception as e:
            import traceback
            error_trace = traceback.format_exc()
            logger.error(f"Failed to create content plan: {e}\nTraceback:\n{error_trace}")

            # Sanitize error message for Loguru/WorkflowError (escape braces)
            safe_error = str(e).replace("{", "{{").replace("}", "}}")
            raise WorkflowError(f"ContentPlanner failed: {safe_error}")

    def _build_planning_prompt(
        self,
        topic: str,
        keywords: List[Keyword],
        context: ProjectContext,
        language: str,
        target_word_count: int,
        seo_settings: Optional[Dict],
        content_memory_guidance: Optional[str] = None,
    ) -> str:
        """Constructs the prompt to instruct the LLM to generate a JSON content plan."""

        keyword_list = ", ".join([kw.phrase for kw in keywords])
        seo_instructions = ""
        if seo_settings:
            if seo_settings.get("use_lsi_keywords"):
                seo_instructions += "- Use LSI (Latent Semantic Indexing) keywords naturally.\n"
            if seo_settings.get("seo_optimized"):
                seo_instructions += "- Ensure hierarchy (H2, H3) matches SEO best practices.\n"

        lang_instruction = "PERSIAN (FARSI)" if language == "fa" else "ENGLISH"

        # We must instruct the LLM to return *only* JSON that matches our Pydantic model
        return f"""
        Act as a world-class SEO content strategist. Your task is to create a detailed content plan for an article in **{lang_instruction}**.

        **Topic:** {topic}
        **Keywords:** {keyword_list}
        **Target Word Count:** ~{target_word_count} words

        **Strategic Context & Rules (You MUST follow these):**
        - **Target Audience:** {context.target_audience}
        - **Tone:** {context.tone}
        - **Style Guide:** {context.style_guide}
        - **Custom Instructions:** {context.custom_instructions or 'N/A'}
        - **Content Memory:** {content_memory_guidance or 'No prior generated content memory for this project.'}
        {seo_instructions}

        **Your Output MUST be a single JSON object (no markdown, no preamble) matching this exact schema:**
        {{
          "title": "A compelling, SEO-optimized title for the article.",
          "meta_description": "A 160-character meta description.",
          "target_word_count": {target_word_count},
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

    async def _load_content_memory_guidance(self, project_id, topic: str) -> Optional[str]:
        """Load compact historical guidance without making planning dependent on it."""
        try:
            from services.content_memory_service import ContentMemoryService

            return await ContentMemoryService(self.article_repo).build_planning_guidance(
                project_id, topic
            )
        except Exception as exc:
            logger.warning(f"Content memory guidance unavailable: {exc}")
            return None

    def _parse_llm_json_response(self, response_content: str) -> dict:
        """Safely parses and validates the LLM's JSON output with comprehensive error recovery."""
        import re

        try:
            # Clean the response (LLMs sometimes add markdown)
            json_str = response_content.strip()

            # Remove markdown code blocks
            json_str = re.sub(r'^```json\s*', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'^```\s*', '', json_str, flags=re.MULTILINE)
            json_str = re.sub(r'\s*```$', '', json_str, flags=re.MULTILINE)
            json_str = json_str.strip()

            # CRITICAL FIX: Normalize JSON keys to remove newlines/extra whitespace
            # This fixes KeyError: '\n  description' issues
            json_str = re.sub(r'"\s*\n\s*([^"]+)"(\s*):', r'"\1":', json_str)

            # Try to parse as-is first
            try:
                parsed = json.loads(json_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Initial JSON parse failed: {e}, attempting repair...")

                # If truncated, try to fix common issues
                open_braces = json_str.count('{')
                close_braces = json_str.count('}')
                open_brackets = json_str.count('[')
                close_brackets = json_str.count(']')

                # If clearly truncated, try to close it properly
                if open_braces > close_braces or open_brackets > close_brackets:
                    logger.warning("JSON appears truncated, attempting to repair...")

                    # Find the last complete section entry
                    sections_match = re.search(r'"sections"\s*:\s*\[(.*)$', json_str, re.DOTALL)
                    if sections_match:
                        sections_content = sections_match.group(1)
                        # Find complete section objects
                        complete_sections = re.findall(
                            r'\{\s*"heading"\s*:\s*"[^"]+"\s*,\s*"prompt"\s*:\s*"[^"]+"\s*\}',
                            sections_content
                        )

                        if complete_sections:
                            # Rebuild with complete sections only
                            prefix = json_str[:sections_match.start()] + '"sections": ['
                            repaired = prefix + ', '.join(complete_sections) + ']}'
                            try:
                                parsed = json.loads(repaired)
                            except json.JSONDecodeError:
                                pass

                # Final attempt: add missing closing brackets/braces
                if 'parsed' not in locals():
                    while open_brackets > close_brackets:
                        json_str += ']'
                        close_brackets += 1
                    while open_braces > close_braces:
                        json_str += '}'
                        close_braces += 1

                    parsed = json.loads(json_str)

            # VALIDATION: Ensure required fields exist with defaults
            if not isinstance(parsed, dict):
                raise WorkflowError(f"LLM returned non-dict JSON: {type(parsed)}")

            # Normalize all keys (remove whitespace) - prevents KeyError on malformed keys
            normalized = {}
            for key, value in parsed.items():
                clean_key = key.strip()
                normalized[clean_key] = value

            # Validate required fields
            if "title" not in normalized:
                normalized["title"] = "Generated Article"
                logger.warning("Missing 'title' in LLM response, using default")

            if "sections" not in normalized or not normalized["sections"]:
                normalized["sections"] = [
                    {"heading": "Introduction", "prompt": "Write an introduction"},
                    {"heading": "Conclusion", "prompt": "Write a conclusion"}
                ]
                logger.warning("Missing 'sections' in LLM response, using default structure")

            if "meta_description" not in normalized:
                # Generate from title
                normalized["meta_description"] = normalized["title"][:155]

            if "target_word_count" not in normalized:
                normalized["target_word_count"] = 1500

            return normalized

        except json.JSONDecodeError as e:
            error_msg = f"Failed to decode LLM JSON response after all repair attempts: {str(e)}"
            logger.error(error_msg)
            logger.error(f"Problematic JSON (first 500 chars): {response_content[:500]}")
            raise WorkflowError(f"ContentPlanner received unparseable JSON from LLM: {str(e)}")
        except Exception as e:
            # Sanitize error message to prevent Loguru formatting issues
            safe_error = str(e).replace("{", "{{").replace("}", "}}")
            logger.error(f"Unexpected error parsing LLM response: {safe_error}")
            raise WorkflowError(f"JSON parsing failed: {safe_error}")

    def _validate_plan(self, plan_data: dict) -> bool:
        """
        Validate the structure and content of the generated plan.

        Checks for:
        - Required fields (title, sections)
        - Minimum section count
        - Reasonable word counts
        """
        if not plan_data.get("title"):
            logger.warning("Plan validation failed: Missing title")
            return False

        sections = plan_data.get("sections", [])
        if not sections or len(sections) < 2:
            logger.warning("Plan validation failed: Insufficient sections")
            return False

        # Check for hallucinated word counts
        total_words = plan_data.get("target_word_count", 0)
        if total_words < 100 or total_words > 10000:
            logger.warning(f"Plan validation failed: Unreasonable word count ({total_words})")
            return False

        return True

    def _refine_plan(self, plan_data: dict, feedback: str) -> dict:
        """
        Refine the plan based on validation feedback or specific instructions.

        Applies deterministic corrections for common validation failures:
        - Missing title: derive from first section or topic
        - Insufficient sections: expand outline with intro/conclusion
        - Unreasonable word counts: clamp to sensible range

        Args:
            plan_data: The plan dictionary to refine
            feedback: Description of the issue to address

        Returns:
            Refined plan dictionary
        """
        logger.info(f"Plan refinement requested: {feedback}")
        refined = dict(plan_data)

        # Fix missing title
        if not refined.get("title"):
            sections = refined.get("sections", [])
            if sections and isinstance(sections[0], dict):
                refined["title"] = sections[0].get("heading", "Untitled Article")
            else:
                refined["title"] = "Untitled Article"
            logger.info(f"Refined plan: generated title '{refined['title']}'")

        # Fix insufficient sections - pad with intro and conclusion if needed
        sections = refined.get("sections", [])
        if not sections:
            sections = []

        if len(sections) < 2:
            has_intro = any(
                "intro" in (s.get("heading", "") or "").lower() for s in sections if isinstance(s, dict)
            )
            has_conclusion = any(
                "conclu" in (s.get("heading", "") or "").lower() for s in sections if isinstance(s, dict)
            )

            if not has_intro:
                sections.insert(0, {
                    "heading": "Introduction",
                    "description": f"Introduction to {refined.get('title', 'the topic')}",
                    "target_word_count": 200,
                })
            if not has_conclusion:
                sections.append({
                    "heading": "Conclusion",
                    "description": f"Key takeaways and summary of {refined.get('title', 'the topic')}",
                    "target_word_count": 200,
                })

            # Add a body section if still under 3 sections
            if len(sections) < 3:
                sections.insert(1, {
                    "heading": "Key Insights",
                    "description": f"Main analysis and insights about {refined.get('title', 'the topic')}",
                    "target_word_count": 600,
                })

            refined["sections"] = sections
            logger.info(f"Refined plan: expanded to {len(sections)} sections")

        # Fix unreasonable word counts
        total_words = refined.get("target_word_count", 0)
        if total_words < 100:
            refined["target_word_count"] = max(
                500,
                sum(s.get("target_word_count", 200) for s in sections if isinstance(s, dict))
            )
            logger.info(f"Refined plan: increased word count to {refined['target_word_count']}")
        elif total_words > 10000:
            refined["target_word_count"] = 3000
            logger.info("Refined plan: capped word count to 3000")

        return refined
