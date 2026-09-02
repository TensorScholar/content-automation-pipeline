"""Content Agent - Master Orchestration Controller.

Coordinates end-to-end content automation pipeline:
1. Project context loading (3-layer decision hierarchy)
2. Keyword research (semantic clustering)
3. Strategic planning (outline synthesis)
4. Article generation (parallel section creation)
5. Quality validation (multi-metric scoring)

Architecture:
    State machine with event sourcing for reproducibility and audit trails.
    Fail-safe design with graceful degradation on non-critical errors.

Example:
    >>> agent = ContentAgent(db, rulebook_mgr, ...)
    >>> result = await agent.execute_pipeline(
    ...     project_id=uuid.uuid4(),
    ...     topic="AI in Healthcare",
    ...     model="gpt-4o-mini"
    ... )
"""

import asyncio
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, cast
from uuid import UUID

from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from config.settings import settings
from core.exceptions import (
    InsufficientContextError,
    ProjectNotFoundError,
    TokenBudgetExceededError,
    WorkflowError,
    is_retryable,
)
from core.models import ContentPlan, GeneratedArticle, InferredPatterns, Keyword, Project, Rulebook
from execution.article_quality_gate import evaluate_article_quality
from execution.content_generator import ContentGenerator
from execution.content_planner import ContentPlanner
from execution.keyword_researcher import Keyword as ResearchedKeyword
from execution.keyword_researcher import KeywordResearcher
from infrastructure.database import DatabaseManager
from infrastructure.llm_usage import llm_usage_context
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
from intelligence.semantic_analyzer import SemanticAnalyzer
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager
from knowledge.website_analyzer import WebsiteAnalyzer
from optimization.token_budget_manager import TokenBudgetManager


class WorkflowState(str, Enum):
    """Workflow execution states for state machine."""

    INITIALIZED = "initialized"
    CONTEXT_LOADING = "context_loading"
    KEYWORD_RESEARCH = "keyword_research"
    CONTENT_PLANNING = "content_planning"
    CONTENT_GENERATION = "content_generation"
    QUALITY_VALIDATION = "quality_validation"
    COMPLETED = "completed"
    FAILED = "failed"


KeywordInput = Keyword | ResearchedKeyword


def _quality_diagnostics_from_release_gate(
    release_gate: Dict[str, Any], *, regeneration_attempted: bool
) -> Dict[str, Any]:
    """Map the existing release-gate result to the task-detail API contract."""
    field_map = {
        "word_count": "actual_word_count",
        "minimum_word_count": "min_word_count",
        "maximum_word_count": "max_word_count",
        "heading_count": "headings_count",
        "paragraph_count": "paragraphs_count",
        "language": "language",
        "findings": "findings",
    }
    diagnostics = {
        target: release_gate[source]
        for source, target in field_map.items()
        if source in release_gate
    }
    diagnostics["regeneration_attempted"] = regeneration_attempted
    return diagnostics


def _wrap_workflow_error(error: Exception, *, workflow_step: str) -> WorkflowError:
    """Preserve structured retry metadata when crossing the agent boundary."""
    wrapped_error = WorkflowError(
        f"Workflow execution failed at {workflow_step}: {str(error)}",
        workflow_step=workflow_step,
    )
    wrapped_error.retryable = is_retryable(error)
    if isinstance(error, WorkflowError):
        wrapped_error.context.update(error.context)
    source_error_code = getattr(error, "error_code", None)
    if source_error_code:
        wrapped_error.context["source_error_code"] = source_error_code
    return wrapped_error


class WorkflowEvent(BaseModel):
    """Event record for workflow execution tracking."""

    timestamp: datetime
    state: WorkflowState
    message: str
    metadata: Dict = Field(default_factory=dict)


class ContentAgentConfig(BaseModel):
    """Configuration for content-generation behavior."""

    model_config = ConfigDict(extra="forbid")

    max_generation_retries: int = 2
    enable_pattern_inference: bool = True
    default_priority: str = "high"


class ContentAgent:
    """Master orchestrator for end-to-end content automation workflows.

    Pipeline Stages:
        1. Context Resolution: Load project rules, patterns, and best practices
        2. Keyword Research: Semantic analysis and clustering
        3. Strategic Planning: Outline and structure synthesis
        4. Generation: Parallel section creation with quality checks
        5. Validation: Quality scoring and fact verification

    Design Principles:
        - Event sourcing: Full execution trail for debugging
        - Adaptive intelligence: 3-layer decision hierarchy with fallbacks
        - Budget awareness: Cost tracking at every stage
        - Fault tolerance: Graceful degradation on failures
    """

    def __init__(
        self,
        database_manager: DatabaseManager,
        semantic_analyzer: SemanticAnalyzer,
        keyword_researcher: KeywordResearcher,
        content_planner: ContentPlanner,
        content_generator: ContentGenerator,
        project_repository: ProjectRepository,
        website_analyzer: WebsiteAnalyzer,
        redis_client: Any,
        metrics_collector: MetricsCollector,
        token_budget_manager: Optional[TokenBudgetManager] = None,
        config: Optional[ContentAgentConfig] = None,
    ):
        self.database_manager = database_manager
        self.semantic_analyzer = semantic_analyzer
        self.keyword_researcher = keyword_researcher
        self.content_planner = content_planner
        self.content_generator = content_generator
        self.project_repo = project_repository
        self.website_analyzer = website_analyzer
        self.redis_client = redis_client
        self.metrics = metrics_collector
        self.budget_manager = token_budget_manager
        self.config = config or ContentAgentConfig()

        # WARNING: current_workflow and workflow_events are per-invocation state.
        # ContentAgent must NOT be used as a singleton / cached instance.
        # Each call to create_content() resets these fields — concurrent calls on
        # the same instance will corrupt each other's state.
        # Always instantiate ContentAgent fresh per task (see api/dependencies.py).
        self.current_workflow: Optional[Dict] = None
        self.workflow_events: List[WorkflowEvent] = []

        logger.info("ContentAgent initialized")

    async def create_content(
        self,
        project_id: UUID,
        topic: str,
        priority: Optional[str] = None,
        custom_instructions: Optional[str] = None,
        project_context: Optional[Dict] = None,
        language: str = "fa",  # Default to Persian
        **kwargs: Any,  # Extended parameters (word_count, tone, seo_settings, etc.)
    ) -> GeneratedArticle:
        """
        Execute complete content creation workflow from topic to a validated generated article.
        Args:
            project_id: Target project identifier
            topic: High-level content topic/theme
            priority: Token budget priority ("low", "medium", "high", "critical")
            custom_instructions: Optional override instructions for this generation
            project_context: Optional pre-loaded context (e.g. from sync repository)
            language: Content language - 'fa' for Persian (default), 'en' for English
            **kwargs: Additional parameters like word_count_range, tone, seo_settings
        """
        defer_finalization = bool(kwargs.pop("_defer_finalization", False))
        workflow_id = f"workflow_{datetime.now(timezone.utc).timestamp()}"
        start_time = datetime.now(timezone.utc)
        usage_context = llm_usage_context(
            project_id=project_id,
            user_id=kwargs.pop("_llm_user_id", None),
            task_id=kwargs.pop("_llm_task_id", None),
            operation_type="content_generation",
        )

        priority = priority or self.config.default_priority

        logger.info(
            f"Content creation initiated | workflow_id={workflow_id} | "
            f"project_id={project_id} | topic={topic} | priority={priority} | "
            f"language={language} | extra_params={list(kwargs.keys())}"
        )

        # Initialize workflow state
        self.current_workflow = {
            "id": workflow_id,
            "project_id": project_id,
            "topic": topic,
            "priority": priority,
            "state": WorkflowState.INITIALIZED,
            "start_time": start_time,
            "settings": kwargs.get("seo_settings", {}),  # Store SEO settings for validation
        }
        self.workflow_events = []

        usage_context.__enter__()
        try:
            # Stage 1: Load and resolve project context
            await self._transition_state(WorkflowState.CONTEXT_LOADING)

            # Use injected context if available (avoiding async DB access issues in Celery)
            if project_context:
                logger.debug(f"Using injected project context | project_id={project_id}")
            else:
                project_context = await self._load_project_context(project_id)

            # Stage 2: Conduct keyword research
            await self._transition_state(WorkflowState.KEYWORD_RESEARCH)
            keywords = await self._conduct_keyword_research(
                project=project_context["project"], topic=topic
            )

            # Merge user-provided keywords into researched categories
            if kwargs.get("primary_keyword") or kwargs.get("secondary_keywords"):
                from core.enums import KeywordIntent
                from core.models import Keyword

                existing_phrases = set()
                for kw_list in keywords.values():
                    for kw in kw_list:
                        existing_phrases.add(kw.phrase.lower())

                if kwargs.get("primary_keyword"):
                    pk = kwargs["primary_keyword"]
                    if pk.lower() not in existing_phrases:
                        keywords["primary"].insert(
                            0, Keyword(phrase=pk, intent=KeywordIntent.INFORMATIONAL)
                        )
                        existing_phrases.add(pk.lower())

                for sk in kwargs.get("secondary_keywords") or []:
                    if sk.lower() not in existing_phrases:
                        keywords["secondary"].insert(
                            0, Keyword(phrase=sk, intent=KeywordIntent.INFORMATIONAL)
                        )
                        existing_phrases.add(sk.lower())

            # Stage 3: Plan content structure
            await self._transition_state(WorkflowState.CONTENT_PLANNING)
            content_plan = await self._plan_content(
                project=project_context["project"],
                topic=topic,
                keywords=keywords,
                custom_instructions=custom_instructions,
                language=language,
                **kwargs,  # Propagate extended params
            )

            # Stage 4: Generate article
            await self._transition_state(WorkflowState.CONTENT_GENERATION)
            article = await self._generate_article(
                project=project_context["project"],
                content_plan=content_plan,
                priority=priority,
                topic=topic,
                **kwargs,  # Propagate extended params
            )
            article.total_cost_usd += max(
                0.0, float(getattr(content_plan, "estimated_cost_usd", 0.0) or 0.0)
            )

            # Stage 5: Quality validation
            await self._transition_state(WorkflowState.QUALITY_VALIDATION)
            validation_result = await self._validate_article_quality(
                article,
                language=language,
                target_word_count=content_plan.target_word_count,
                require_faq=bool(kwargs.get("include_faq")),
            )

            if not validation_result["passed"]:
                logger.warning(
                    f"Article failed quality validation | workflow_id={workflow_id} | "
                    f"issues={validation_result['issues']}"
                )

                # Attempt regeneration if retries available
                if self.current_workflow.get("retry_count", 0) < min(
                    self.config.max_generation_retries, 1
                ):
                    logger.info("Attempting article regeneration with enhanced parameters")
                    self.current_workflow["retry_count"] = (
                        self.current_workflow.get("retry_count", 0) + 1
                    )

                    try:
                        consumed_tokens = article.total_tokens_used
                        consumed_cost = article.total_cost_usd
                        consumed_time = article.generation_time_seconds
                        # Regenerate with stricter parameters
                        article = await self._regenerate_with_feedback(
                            content_plan=content_plan,
                            project=project_context["project"],
                            feedback=validation_result["issues"],
                            release_gate=validation_result["release_gate"],
                            model_override=kwargs.get("model_override"),
                        )
                        article.total_tokens_used += consumed_tokens
                        article.total_cost_usd += consumed_cost
                        article.generation_time_seconds += consumed_time
                        validation_result = await self._validate_article_quality(
                            article,
                            language=language,
                            target_word_count=content_plan.target_word_count,
                            require_faq=bool(kwargs.get("include_faq")),
                        )
                    except Exception as regen_err:
                        diagnostics = validation_result["release_gate"]
                        error = WorkflowError(
                            f"Article regeneration failed: {regen_err}",
                            workflow_step=WorkflowState.QUALITY_VALIDATION.value,
                        )
                        error.context["release_gate"] = diagnostics
                        error.context["regeneration_attempted"] = True
                        error.context["quality_diagnostics"] = (
                            _quality_diagnostics_from_release_gate(
                                diagnostics,
                                regeneration_attempted=True,
                            )
                        )
                        error.retryable = is_retryable(regen_err)
                        raise error from regen_err

            if not validation_result["passed"]:
                findings = "; ".join(validation_result["issues"])
                diagnostics = validation_result["release_gate"]
                regeneration_attempted = self.current_workflow.get("retry_count", 0) > 0
                error = WorkflowError(
                    f"Article failed release gate: {findings} | "
                    f"regeneration_attempted={regeneration_attempted}",
                    workflow_step=WorkflowState.QUALITY_VALIDATION.value,
                )
                error.context["release_gate"] = diagnostics
                error.context["regeneration_attempted"] = regeneration_attempted
                error.context["quality_diagnostics"] = _quality_diagnostics_from_release_gate(
                    diagnostics,
                    regeneration_attempted=regeneration_attempted,
                )
                error.context["terminal_reason"] = "release_quality"
                error.retryable = False
                raise error

            if defer_finalization:
                # Celery commits accepted content together with its durable task
                # result and export request. Keeping this method side-effect-free
                # on that path prevents duplicate persistence on redelivery.
                return article

            await self.content_generator.finalize_article(article, content_plan)

            # Workflow completion
            await self._transition_state(WorkflowState.COMPLETED)

            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            logger.success(
                f"Content creation completed | workflow_id={workflow_id} | "
                f"article_id={article.id} | execution_time={execution_time:.2f}s | "
                f"total_cost=${article.total_cost_usd:.4f}"
            )

            # Record workflow metrics
            await self._record_workflow_metrics(article, execution_time)

            return article

        except TokenBudgetExceededError:
            await self._transition_state(WorkflowState.FAILED)
            logger.warning(
                "Content generation blocked by persistent LLM budget | "
                f"workflow_id={workflow_id} | project_id={project_id}"
            )
            raise
        except Exception as e:
            current_state = self.current_workflow["state"]
            failed_step = (
                current_state.value
                if isinstance(current_state, WorkflowState)
                else str(current_state)
            )
            await self._transition_state(WorkflowState.FAILED)
            execution_time = (datetime.now(timezone.utc) - start_time).total_seconds()

            # Record failure metrics
            self.metrics.record_workflow_completion(
                project_id=str(project_id),
                workflow_type="content_generation",
                duration_seconds=execution_time,
                cost=0.0,  # No cost on failure
                success=False,
                error_type=type(e).__name__,
            )

            logger.error(
                f"Content creation failed | workflow_id={workflow_id} | "
                f"state={self.current_workflow['state']} | error={e}"
            )
            wrapped_error = _wrap_workflow_error(e, workflow_step=failed_step)
            raise wrapped_error from e
        finally:
            usage_context.__exit__(None, None, None)

    async def _load_project_context(self, project_id: UUID) -> Dict:
        """
        Load complete project context with 3-layer decision hierarchy.

        OPTIMIZED: Uses Redis caching and batch queries to reduce latency.

        Layer 1: Explicit rulebook (if exists)
        Layer 2: Inferred patterns (if analyzed)
        Layer 3: Best practices (always available)

        Returns enriched project context dictionary.
        """
        logger.debug(f"Loading project context | project_id={project_id}")
        # NOTE: Project context is NOT cached in Redis because the context dict
        # contains live domain objects (Project, Rulebook, InferredPatterns) that
        # cannot be round-tripped through JSON without losing type information.
        # The expensive sub-queries (rulebook, patterns) are fast enough for
        # small-scale use. Re-introduce caching only if profiling shows a hot path.

        # Initialize database manager if not already initialized
        if (
            not hasattr(self.database_manager, "_initialized")
            or not self.database_manager._initialized
        ):
            await self.database_manager.initialize()

        # OPTIMIZED: Load all data in parallel using asyncio.gather
        async with self.database_manager.session() as session:
            project_repo = ProjectRepository(self.database_manager)

            # Parallel data loading
            gathered = await asyncio.gather(
                project_repo.get_by_id(project_id),
                self.project_repo.get_inferred_patterns(project_id),
                return_exceptions=True,
            )
            project_result = cast(Project | None | Exception, gathered[0])
            patterns_result = cast(InferredPatterns | None | Exception, gathered[1])

            # Handle exceptions from parallel execution
            if isinstance(project_result, Exception):
                raise ProjectNotFoundError(project_id)
            if project_result is None:
                raise ProjectNotFoundError(project_id)
            project = project_result

            if isinstance(patterns_result, Exception):
                logger.warning(f"Failed to load patterns: {patterns_result}")
                patterns = None
            else:
                patterns = patterns_result

            context: Dict[str, Any] = {
                "project": project,
                "rulebook": None,
                "inferred_patterns": None,
                "decision_strategy": "best_practices",  # Default
            }

            # Layer 1: Load rulebook if exists
            try:
                rulebook_mgr = RulebookManager(session, self.semantic_analyzer)
                rulebook = await rulebook_mgr.get_latest_rulebook(project_id)
                if rulebook:
                    context["rulebook"] = rulebook
                    context["decision_strategy"] = "explicit_rules"
                    logger.info(
                        f"Rulebook loaded | project_id={project_id} | rules={len(rulebook.rules)}"
                    )
            except AttributeError:
                logger.warning(f"RulebookManager doesn't support get_latest_rulebook method")
                context["rulebook"] = None
            except Exception as e:
                logger.warning(f"Failed to load rulebook: {e}")
                context["rulebook"] = None

            # Layer 2: Load inferred patterns if available
            if patterns and patterns.confidence > 0.65:  # Minimum confidence threshold
                context["inferred_patterns"] = patterns
                if not context["rulebook"]:
                    context["decision_strategy"] = "inferred_patterns"
                logger.info(
                    f"Inferred patterns loaded | project_id={project_id} | "
                    f"confidence={patterns.confidence:.3f}"
                )

            # If no rulebook and patterns are stale/missing, trigger analysis
            if not context["rulebook"] and (not patterns or patterns.confidence < 0.65):
                if self.config.enable_pattern_inference and project.domain:
                    logger.info(f"Triggering website analysis | domain={project.domain}")
                    try:
                        patterns = await self.website_analyzer.analyze_website(
                            project_id, project.domain
                        )
                        context["inferred_patterns"] = patterns
                        context["decision_strategy"] = "inferred_patterns"
                    except Exception as e:
                        logger.warning(f"Website analysis failed: {e}")

            # Layer 3: Best practices always loaded in decision engine

            self._record_event(
                WorkflowState.CONTEXT_LOADING,
                f"Context loaded | strategy={context['decision_strategy']}",
                {"strategy": context["decision_strategy"]},
            )

            return context

    async def _conduct_keyword_research(
        self, project: Project, topic: str
    ) -> Dict[str, List[KeywordInput]]:
        """
        Execute keyword research with semantic clustering.

        Returns categorized keywords (primary, secondary, long-tail).
        """
        logger.debug(f"Conducting keyword research | topic={topic}")

        keywords = await self.keyword_researcher.research_keywords(
            seed_topic=topic, target_count=25
        )

        # Handle both Result objects and direct lists
        if hasattr(keywords, "unwrap_or"):
            keywords = keywords.unwrap_or([])

        # Categorize keywords by search intent and volume
        categorized: Dict[str, List[KeywordInput]] = {
            "primary": [keyword for keyword in keywords[:5]],  # Top performers
            "secondary": [keyword for keyword in keywords[5:15]],  # Supporting keywords
            "long_tail": [keyword for keyword in keywords[15:]],  # Niche, specific queries
        }

        self._record_event(
            WorkflowState.KEYWORD_RESEARCH,
            f"Keyword research completed | total={len(keywords)}",
            {
                "primary_count": len(categorized["primary"]),
                "secondary_count": len(categorized["secondary"]),
                "long_tail_count": len(categorized["long_tail"]),
            },
        )

        logger.info(
            f"Keywords researched | primary={len(categorized['primary'])} | "
            f"secondary={len(categorized['secondary'])} | "
            f"long_tail={len(categorized['long_tail'])}"
        )

        return categorized

    async def _plan_content(
        self,
        project: Project,
        topic: str,
        keywords: Dict[str, List[KeywordInput]],
        custom_instructions: Optional[str],
        language: str = "fa",
        **kwargs: Any,
    ) -> ContentPlan:
        """
        Generate strategic content plan with outline and metadata.

        Uses decision engine to determine optimal structure and approach.
        """
        logger.debug(f"Planning content structure | topic={topic}")

        # Synthesize context for planning
        from core.models import ProjectContext

        # Extract context from kwargs if available
        context = ProjectContext(
            target_audience=kwargs.get("target_audience") or "technical professionals",
            tone=kwargs.get("tone") or "professional",
            style_guide=kwargs.get("content_structure") or "standard",
            custom_instructions=custom_instructions,
        )

        # Determine word count target
        word_count_range = kwargs.get("word_count_range")
        target_word_count = kwargs.get("target_word_count")

        if word_count_range:
            # Parse "1000-1500" -> 1250
            try:
                if "-" in word_count_range:
                    parts = word_count_range.split("-")
                    target_word_count = (int(parts[0]) + int(parts[1])) // 2
                elif "+" in word_count_range:
                    target_word_count = int(word_count_range.replace("+", "")) + 500
            except ValueError:
                logger.warning(f"Failed to parse word count range: {word_count_range}")
                target_word_count = 1500  # Default fallback

        if not target_word_count:
            target_word_count = 1500  # Default

        # Generate content plan
        # Remove duplicate parameters from kwargs to avoid conflicts
        filtered_kwargs = {
            k: v
            for k, v in kwargs.items()
            if k not in ["target_word_count", "seo_settings", "custom_instructions", "language"]
        }

        content_plan = await self.content_planner.create_content_plan(
            project=project,
            topic=topic,
            keywords=cast(List[Keyword], keywords["primary"]),
            custom_instructions=custom_instructions,
            language=language,
            target_word_count=target_word_count,
            seo_settings=kwargs.get("seo_settings"),
            **filtered_kwargs,  # Pass only non-conflicting params
        )

        self._record_event(
            WorkflowState.CONTENT_PLANNING,
            f"Content plan created | sections={len(content_plan.outline.sections)}",
            {
                "target_words": content_plan.target_word_count,
                "estimated_cost": content_plan.estimated_cost_usd,
                "sections": len(content_plan.outline.sections),
            },
        )

        logger.info(
            f"Content planned | sections={len(content_plan.outline.sections)} | "
            f"target_words={content_plan.target_word_count} | "
            f"estimated_cost=${content_plan.estimated_cost_usd:.4f}"
        )

        return content_plan

    async def _generate_article(
        self,
        project: Project,
        content_plan: ContentPlan,
        priority: str,
        topic: str,
        **kwargs: Any,
    ) -> GeneratedArticle:
        """
        Generate complete article using content generator.

        Implements budget-aware generation with quality gates.
        """
        logger.debug(f"Generating article | plan_id={content_plan.id}")

        # Select models from centralized configuration, with an optional
        # per-request override validated by the API/worker before execution.
        model_override = kwargs.get("model_override")
        planning_model = model_override or settings.llm.planning_model
        writing_model = model_override or settings.llm.writing_model
        verification_model = model_override or settings.llm.verification_model

        # Generate article using ContentGenerator
        article = await self.content_generator.generate_article(
            project_id=project.id,
            plan=content_plan,
            planning_model=planning_model,
            writing_model=writing_model,
            verification_model=verification_model,
            persist=False,
            **kwargs,
        )

        self._record_event(
            WorkflowState.CONTENT_GENERATION,
            f"Article generated | id={article.id}",
            {
                "word_count": article.word_count,
                "tokens_used": article.total_tokens_used,
                "cost": article.total_cost_usd,
                "generation_time": article.generation_time_seconds,
            },
        )

        logger.info(
            f"Article generated | id={article.id} | words={article.word_count} | "
            f"cost=${article.total_cost_usd:.4f} | time={article.generation_time_seconds:.2f}s"
        )

        return article

    async def _validate_article_quality(
        self,
        article: GeneratedArticle,
        *,
        language: str,
        target_word_count: int,
        require_faq: bool = False,
    ) -> Dict:
        """
        Comprehensive quality validation using multi-metric analysis.

        Returns validation result with pass/fail and detailed feedback.
        """
        logger.debug(f"Validating article quality | article_id={article.id}")

        release_gate = evaluate_article_quality(
            article.content or "",
            language=language,
            target_word_count=target_word_count,
            require_faq=require_faq,
        )
        article.quality_metrics.word_count = release_gate.word_count
        issues = [finding.message for finding in release_gate.findings]
        title_present = bool(article.title and article.title.strip())
        release_gate_data = release_gate.to_dict()
        if not title_present:
            issues.append("Article title is required.")
            release_gate_data["passed"] = False
            release_gate_data["findings"].append(
                {
                    "code": "missing_title",
                    "severity": "error",
                    "message": "Article title is required.",
                    "expected": "a non-empty article title",
                    "actual": "empty title",
                }
            )
        advisories = []

        # Readability check (null-safe)
        readability = 0.0
        if article.quality_metrics and hasattr(article.quality_metrics, "readability_score"):
            readability = article.quality_metrics.readability_score or 0.0
        else:
            advisories.append("Quality metrics unavailable - manual review required")

        if language.lower().startswith("en") and readability < 60.0:
            advisories.append(f"Low readability: {readability:.1f} (target: 60+)")

        word_count = release_gate.word_count

        # Keyword density validation
        avg_density = self._calculate_keyword_density(article.content, article.keywords)

        if avg_density < 0.005:
            advisories.append(f"Keyword density too low: {avg_density:.4f} (minimum: 0.005)")
        elif avg_density > 0.025:
            advisories.append(f"Keyword over-optimization: {avg_density:.4f} (maximum: 0.025)")

        # Plagiarism / Uniqueness Check (Requested Feature)
        workflow = cast(Dict[str, Any], self.current_workflow)
        workflow_settings = workflow.get("settings", {})
        if workflow_settings.get("check_plagiarism", False):
            # Internal Uniqueness Check (Repetition Detection)
            # Detects if AI is looping or repeating phrases (internal plagiarism)
            grams = [article.content[i : i + 50] for i in range(0, len(article.content) - 50, 50)]
            unique_grams = set(grams)
            repetition_ratio = 1.0 - (len(unique_grams) / len(grams)) if grams else 0

            if repetition_ratio > 0.2:  # More than 20% repetitive chunks
                advisories.append(
                    f"High repetition detected (Pseudo-Plagiarism): {repetition_ratio:.1%} content overlap"
                )
            else:
                logger.info(
                    f"Plagiarism/Uniqueness check passed | article_id={article.id} | score={1-repetition_ratio:.2f}"
                )

        passed = release_gate.passed and title_present

        validation_result = {
            "passed": passed,
            "issues": issues,
            "advisories": advisories,
            "release_gate": release_gate_data,
            "metrics": {
                "readability": readability,
                "word_count": word_count,
                "avg_keyword_density": avg_density,
            },
        }

        self._record_event(
            WorkflowState.QUALITY_VALIDATION,
            f"Validation {'passed' if passed else 'failed'}",
            validation_result,
        )

        if passed:
            logger.success(f"Article passed quality validation | article_id={article.id}")
        else:
            logger.warning(
                f"Article failed quality validation | article_id={article.id} | "
                f"issues={len(issues)}"
            )

        return validation_result

    async def _regenerate_with_feedback(
        self,
        content_plan: ContentPlan,
        project: Project,
        feedback: List[str],
        release_gate: Optional[Dict[str, Any]] = None,
        model_override: Optional[str] = None,
    ) -> GeneratedArticle:
        """
        Regenerate article with enhanced parameters based on validation feedback.

        Adjusts generation strategy to address specific quality issues.
        """
        logger.info(f"Regenerating article with quality feedback: {feedback}")

        if content_plan is None:
            logger.warning("Content plan is None, cannot regenerate with feedback")
            # Create a basic article as fallback
            return await self._generate_article(
                project=project, content_plan=None, priority="medium", topic="Regenerated content"
            )

        # Preserve the request target; regeneration corrects output rather than
        # silently changing the user's contract.
        enhanced_plan = content_plan.model_copy(deep=True)

        gate = release_gate or {}
        generation_feedback = (
            "The previous candidate failed the release gate. "
            f"Actual words: {gate.get('word_count', 'unknown')}; "
            f"allowed range: {gate.get('minimum_word_count', 'unknown')}-"
            f"{gate.get('maximum_word_count', 'unknown')}; "
            f"requested target: {enhanced_plan.target_word_count}. "
            f"Correct these findings without changing the requested target: {'; '.join(feedback)}"
        )

        # Regenerate with adjusted plan
        article = await self.content_generator.generate_article(
            project_id=project.id,
            plan=enhanced_plan,
            planning_model=model_override or settings.llm.planning_model,
            writing_model=model_override or settings.llm.writing_model,
            verification_model=model_override or settings.llm.verification_model,
            generation_feedback=generation_feedback,
            persist=False,
        )

        return article

    async def _transition_state(self, new_state: WorkflowState) -> None:
        """Transition workflow to new state with event recording."""
        workflow = cast(Dict[str, Any], self.current_workflow)
        old_state = workflow.get("state")
        workflow["state"] = new_state

        logger.debug(f"Workflow state transition | {old_state} → {new_state}")

        # Record the state transition as an event for audit trail
        self._record_event(
            state=new_state,
            message=f"Transitioned from {old_state} to {new_state}",
            metadata={"previous_state": str(old_state) if old_state else None},
        )

    def _record_event(
        self,
        state: WorkflowState,
        message: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Record workflow event for audit trail."""
        event = WorkflowEvent(
            timestamp=datetime.now(timezone.utc),
            state=state,
            message=message,
            metadata=metadata or {},
        )
        self.workflow_events.append(event)

    def _calculate_keyword_density(self, content: str, keywords: List[str]) -> float:
        """
        Calculate average keyword density across all target keywords.

        Keyword density = (keyword occurrences / total words) * 100

        Args:
            content: Article content (HTML or plain text)
            keywords: List of target keywords

        Returns:
            Average keyword density as a decimal (e.g., 0.015 = 1.5%)
        """
        import re

        if not content or not keywords:
            return 0.0

        # Strip HTML tags for accurate word count
        text = re.sub(r"<[^>]+>", "", content).lower()
        # Use Unicode-aware word matching that includes Persian/Arabic characters
        words = re.findall(r"[\w\u0600-\u06FF\u0750-\u077F\uFB50-\uFDFF\uFE70-\uFEFF]+", text)
        total_words = len(words)

        if total_words == 0:
            return 0.0

        total_density = 0.0
        for keyword in keywords:
            keyword_lower = keyword.lower()
            keyword_words = keyword_lower.split()

            if len(keyword_words) == 1:
                # Single word - count occurrences
                count = text.count(keyword_lower)
            else:
                # Multi-word phrase - count phrase occurrences
                # Use lookaround with non-word chars (supports Persian/Arabic)
                count = len(
                    re.findall(r"(?<![^\s])" + re.escape(keyword_lower) + r"(?![^\s])", text)
                )

            # Density for this keyword
            density = count / total_words
            total_density += density

        # Average density across all keywords
        avg_density = total_density / len(keywords) if keywords else 0.0
        return avg_density

    async def _record_workflow_metrics(
        self, article: GeneratedArticle, execution_time: float
    ) -> None:
        """Record comprehensive workflow metrics for monitoring."""
        self.metrics.record_workflow_completion(
            project_id=str(article.project_id),
            workflow_type="content_generation",
            duration_seconds=execution_time,
            cost=article.total_cost_usd,
            success=True,  # Assuming this method is called only on success
        )

    async def get_workflow_status(self) -> Dict:
        """
        Get current workflow execution status.

        Returns workflow state, events, and progress metrics.
        """
        if not self.current_workflow:
            return {"status": "no_active_workflow"}

        return {
            "workflow_id": self.current_workflow["id"],
            "state": self.current_workflow["state"],
            "project_id": str(self.current_workflow["project_id"]),
            "topic": self.current_workflow["topic"],
            "start_time": self.current_workflow["start_time"].isoformat(),
            "events": [
                {
                    "timestamp": event.timestamp.isoformat(),
                    "state": event.state,
                    "message": event.message,
                }
                for event in self.workflow_events
            ],
        }
