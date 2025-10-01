"""
Content Agent: Master Orchestration Controller

Central coordinator for the entire content automation pipeline. Implements stateful
workflow management with adaptive decision-making, economic optimization, and
comprehensive error handling. Serves as the primary interface between user directives
and system execution.

Architectural Pattern: Orchestrator + State Machine with Event Sourcing
"""

import asyncio
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional
from uuid import UUID

from loguru import logger
from pydantic import BaseModel

from core.exceptions import InsufficientContextError, ProjectNotFoundError, WorkflowError
from core.models import ContentPlan, GeneratedArticle, InferredPatterns, Keyword, Project, Rulebook
from execution.content_generator import ContentGenerator
from execution.content_planner import ContentPlanner
from execution.distributer import Distributor
from execution.keyword_researcher import KeywordResearcher
from infrastructure.monitoring import MetricsCollector
from intelligence.context_synthesizer import ContextSynthesizer
from intelligence.decision_engine import DecisionEngine
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
    DISTRIBUTION = "distribution"
    COMPLETED = "completed"
    FAILED = "failed"


class WorkflowEvent(BaseModel):
    """Event record for workflow execution tracking."""

    timestamp: datetime
    state: WorkflowState
    message: str
    metadata: Dict = {}


class ContentAgentConfig(BaseModel):
    """Configuration for content agent behavior."""

    enable_auto_distribution: bool = True
    require_manual_approval: bool = False
    max_generation_retries: int = 2
    enable_pattern_inference: bool = True
    default_priority: str = "high"


class ContentAgent:
    """
    Master orchestration controller for end-to-end content automation.

    Workflow Pipeline:
    1. Project Context Resolution (3-layer decision hierarchy)
    2. Keyword Research (semantic clustering)
    3. Content Strategy Planning (outline synthesis)
    4. Article Generation (section-by-section with validation)
    5. Quality Assurance (multi-metric evaluation)
    6. Distribution (multi-channel delivery)

    Design Principles:
    - Stateful execution: Full event sourcing for reproducibility
    - Adaptive intelligence: Decision hierarchy with fallbacks
    - Economic rationality: Budget-aware at every stage
    - Fail-safe: Graceful degradation on non-critical failures
    """

    def __init__(
        self,
        project_repository: ProjectRepository,
        rulebook_manager: RulebookManager,
        website_analyzer: WebsiteAnalyzer,
        decision_engine: DecisionEngine,
        context_synthesizer: ContextSynthesizer,
        keyword_researcher: KeywordResearcher,
        content_planner: ContentPlanner,
        content_generator: ContentGenerator,
        distributor: Distributor,
        budget_manager: TokenBudgetManager,
        metrics_collector: MetricsCollector,
        config: Optional[ContentAgentConfig] = None,
    ):
        self.projects = project_repository
        self.rulebook_mgr = rulebook_manager
        self.website_analyzer = website_analyzer
        self.decision_engine = decision_engine
        self.context_synthesizer = context_synthesizer
        self.keyword_researcher = keyword_researcher
        self.content_planner = content_planner
        self.content_generator = content_generator
        self.distributor = distributor
        self.budget_manager = budget_manager
        self.metrics = metrics_collector

        self.config = config or ContentAgentConfig()

        # Workflow state tracking
        self.current_workflow: Optional[Dict] = None
        self.workflow_events: List[WorkflowEvent] = []

        logger.info(
            "ContentAgent initialized | "
            f"auto_distribution={self.config.enable_auto_distribution} | "
            f"manual_approval={self.config.require_manual_approval}"
        )

    async def create_content(
        self,
        project_id: UUID,
        topic: str,
        priority: Optional[str] = None,
        custom_instructions: Optional[str] = None,
    ) -> GeneratedArticle:
        """
        Execute complete content creation workflow from topic to published article.
        Args:
            project_id: Target project identifier
            topic: High-level content topic/theme
            priority: Token budget priority ("low", "medium", "high", "critical")
            custom_instructions: Optional override instructions for this generation

        Returns:
            GeneratedArticle with full metadata and distribution status

        Raises:
            ProjectNotFoundError: Invalid project_id
            WorkflowError: Critical workflow failures
            InsufficientContextError: Unable to resolve project context
        """
        workflow_id = f"workflow_{datetime.utcnow().timestamp()}"
        start_time = datetime.utcnow()

        priority = priority or self.config.default_priority

        logger.info(
            f"Content creation initiated | workflow_id={workflow_id} | "
            f"project_id={project_id} | topic={topic} | priority={priority}"
        )

        # Initialize workflow state
        self.current_workflow = {
            "id": workflow_id,
            "project_id": project_id,
            "topic": topic,
            "priority": priority,
            "state": WorkflowState.INITIALIZED,
            "start_time": start_time,
        }
        self.workflow_events = []

        try:
            # Stage 1: Load and resolve project context
            await self._transition_state(WorkflowState.CONTEXT_LOADING)
            project_context = await self._load_project_context(project_id)

            # Stage 2: Conduct keyword research
            await self._transition_state(WorkflowState.KEYWORD_RESEARCH)
            keywords = await self._conduct_keyword_research(
                project=project_context["project"], topic=topic
            )

            # Stage 3: Plan content structure
            await self._transition_state(WorkflowState.CONTENT_PLANNING)
            content_plan = await self._plan_content(
                project=project_context["project"],
                topic=topic,
                keywords=keywords,
                custom_instructions=custom_instructions,
            )

            # Stage 4: Generate article
            await self._transition_state(WorkflowState.CONTENT_GENERATION)
            article = await self._generate_article(
                project=project_context["project"], content_plan=content_plan, priority=priority
            )

            # Stage 5: Quality validation
            await self._transition_state(WorkflowState.QUALITY_VALIDATION)
            validation_result = await self._validate_article_quality(article)

            if not validation_result["passed"] and not self.config.require_manual_approval:
                logger.warning(
                    f"Article failed quality validation | workflow_id={workflow_id} | "
                    f"issues={validation_result['issues']}"
                )

                # Attempt regeneration if retries available
                if self.current_workflow.get("retry_count", 0) < self.config.max_generation_retries:
                    logger.info("Attempting article regeneration with enhanced parameters")
                    self.current_workflow["retry_count"] = (
                        self.current_workflow.get("retry_count", 0) + 1
                    )

                    # Regenerate with stricter parameters
                    article = await self._regenerate_with_feedback(
                        content_plan=content_plan,
                        project=project_context["project"],
                        feedback=validation_result["issues"],
                    )

            # Stage 6: Distribution (if enabled and approved)
            if self.config.enable_auto_distribution and not self.config.require_manual_approval:
                await self._transition_state(WorkflowState.DISTRIBUTION)
                distribution_result = await self._distribute_article(
                    article=article, project=project_context["project"]
                )
                article.distributed_at = datetime.utcnow()
                article.distribution_channels = distribution_result["channels"]

            # Workflow completion
            await self._transition_state(WorkflowState.COMPLETED)

            execution_time = (datetime.utcnow() - start_time).total_seconds()

            logger.success(
                f"Content creation completed | workflow_id={workflow_id} | "
                f"article_id={article.id} | execution_time={execution_time:.2f}s | "
                f"total_cost=${article.total_cost:.4f}"
            )

            # Record workflow metrics
            await self._record_workflow_metrics(article, execution_time)

            return article

        except Exception as e:
            await self._transition_state(WorkflowState.FAILED)
            logger.error(
                f"Content creation failed | workflow_id={workflow_id} | "
                f"state={self.current_workflow['state']} | error={e}"
            )
            raise WorkflowError(
                f"Workflow execution failed at {self.current_workflow['state']}: {str(e)}"
            ) from e


async def _load_project_context(self, project_id: UUID) -> Dict:
    """
    Load complete project context with 3-layer decision hierarchy.

    Layer 1: Explicit rulebook (if exists)
    Layer 2: Inferred patterns (if analyzed)
    Layer 3: Best practices (always available)

    Returns enriched project context dictionary.
    """
    logger.debug(f"Loading project context | project_id={project_id}")

    # Load project metadata
    project = await self.projects.get_by_id(project_id)
    if not project:
        raise ProjectNotFoundError(f"Project not found: {project_id}")

    context = {
        "project": project,
        "rulebook": None,
        "inferred_patterns": None,
        "decision_strategy": "best_practices",  # Default
    }

    # Layer 1: Load rulebook if exists
    rulebook = await self.rulebook_mgr.get_rulebook(project_id)
    if rulebook:
        context["rulebook"] = rulebook
        context["decision_strategy"] = "explicit_rules"
        logger.info(f"Rulebook loaded | project_id={project_id} | rules={len(rulebook.rules)}")

    # Layer 2: Load inferred patterns if available
    patterns = await self.projects.get_inferred_patterns(project_id)
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
            patterns = await self.website_analyzer.analyze_website(project.domain, project_id)
            context["inferred_patterns"] = patterns
            context["decision_strategy"] = "inferred_patterns"

    # Layer 3: Best practices always loaded in decision engine

    self._record_event(
        WorkflowState.CONTEXT_LOADING,
        f"Context loaded | strategy={context['decision_strategy']}",
        {"strategy": context["decision_strategy"]},
    )

    return context


async def _conduct_keyword_research(self, project: Project, topic: str) -> Dict[str, List[Keyword]]:
    """
    Execute keyword research with semantic clustering.

    Returns categorized keywords (primary, secondary, long-tail).
    """
    logger.debug(f"Conducting keyword research | topic={topic}")

    keywords = await self.keyword_researcher.research_keywords(
        topic=topic, project=project, max_keywords=25
    )

    # Categorize keywords by search intent and volume
    categorized = {
        "primary": keywords[:5],  # Top performers
        "secondary": keywords[5:15],  # Supporting keywords
        "long_tail": keywords[15:],  # Niche, specific queries
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
    keywords: Dict[str, List[Keyword]],
    custom_instructions: Optional[str],
) -> ContentPlan:
    """
    Generate strategic content plan with outline and metadata.

    Uses decision engine to determine optimal structure and approach.
    """
    logger.debug(f"Planning content structure | topic={topic}")

    # Synthesize context for planning
    context = await self.context_synthesizer.synthesize_context(
        project=project, topic=topic, keywords=keywords["primary"], level="standard"
    )

    # Generate content plan
    content_plan = await self.content_planner.create_plan(
        project=project,
        topic=topic,
        primary_keywords=keywords["primary"],
        secondary_keywords=keywords["secondary"],
        context=context,
        custom_instructions=custom_instructions,
    )

    self._record_event(
        WorkflowState.CONTENT_PLANNING,
        f"Content plan created | sections={len(content_plan.outline.sections)}",
        {
            "target_words": content_plan.target_word_count,
            "estimated_cost": content_plan.estimated_cost,
            "sections": len(content_plan.outline.sections),
        },
    )

    logger.info(
        f"Content planned | sections={len(content_plan.outline.sections)} | "
        f"target_words={content_plan.target_word_count} | "
        f"estimated_cost=${content_plan.estimated_cost:.4f}"
    )

    return content_plan


async def _generate_article(
    self, project: Project, content_plan: ContentPlan, priority: str
) -> GeneratedArticle:
    """
    Generate complete article using content generator.

    Implements budget-aware generation with quality gates.
    """
    logger.debug(f"Generating article | plan_id={content_plan.id}")

    article = await self.content_generator.generate_article(
        content_plan=content_plan, project=project, priority=priority
    )

    self._record_event(
        WorkflowState.CONTENT_GENERATION,
        f"Article generated | id={article.id}",
        {
            "word_count": article.word_count,
            "tokens_used": article.total_tokens_used,
            "cost": article.total_cost,
            "generation_time": article.generation_time,
        },
    )

    logger.info(
        f"Article generated | id={article.id} | words={article.word_count} | "
        f"cost=${article.total_cost:.4f} | time={article.generation_time:.2f}s"
    )

    return article


async def _validate_article_quality(self, article: GeneratedArticle) -> Dict:
    """
    Comprehensive quality validation using multi-metric analysis.

    Returns validation result with pass/fail and detailed feedback.
    """
    logger.debug(f"Validating article quality | article_id={article.id}")

    issues = []

    # Readability check
    if article.readability_score < 60.0:
        issues.append(f"Low readability: {article.readability_score:.1f} (target: 60+)")

    # Length validation
    if article.word_count < 800:
        issues.append(f"Article too short: {article.word_count} words (minimum: 800)")
    elif article.word_count > 3500:
        issues.append(f"Article too long: {article.word_count} words (maximum: 3500)")

    # Keyword density validation
    avg_density = (
        sum(article.keyword_density.values()) / len(article.keyword_density)
        if article.keyword_density
        else 0
    )

    if avg_density < 0.005:
        issues.append(f"Keyword density too low: {avg_density:.4f} (minimum: 0.005)")
    elif avg_density > 0.025:
        issues.append(f"Keyword over-optimization: {avg_density:.4f} (maximum: 0.025)")

    passed = len(issues) == 0

    validation_result = {
        "passed": passed,
        "issues": issues,
        "metrics": {
            "readability": article.readability_score,
            "word_count": article.word_count,
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
    self, content_plan: ContentPlan, project: Project, feedback: List[str]
) -> GeneratedArticle:
    """
    Regenerate article with enhanced parameters based on validation feedback.

    Adjusts generation strategy to address specific quality issues.
    """
    logger.info("Regenerating article with quality feedback")

    # Enhance content plan based on feedback
    enhanced_plan = content_plan.copy(deep=True)

    # Adjust parameters based on feedback
    for issue in feedback:
        if "too short" in issue.lower():
            enhanced_plan.target_word_count = int(enhanced_plan.target_word_count * 1.3)
        elif "too long" in issue.lower():
            enhanced_plan.target_word_count = int(enhanced_plan.target_word_count * 0.8)
        elif "readability" in issue.lower():
            # Will be handled by generator's temperature adjustment
            pass

    # Regenerate with adjusted plan
    article = await self.content_generator.generate_article(
        content_plan=enhanced_plan,
        project=project,
        priority="critical",  # Use higher priority for regeneration
    )

    return article


async def _distribute_article(self, article: GeneratedArticle, project: Project) -> Dict:
    """
    Distribute article through configured channels.

    Currently supports Telegram; extensible for additional channels.
    """
    logger.debug(f"Distributing article | article_id={article.id}")

    channels_used = []

    # Telegram distribution
    if project.telegram_channel:
        try:
            await self.distributor.distribute_to_telegram(
                article=article, channel=project.telegram_channel
            )
            channels_used.append("telegram")
            logger.info(f"Article distributed to Telegram | channel={project.telegram_channel}")
        except Exception as e:
            logger.error(f"Telegram distribution failed | error={e}")

    # Future: WordPress, email, etc.

    self._record_event(
        WorkflowState.DISTRIBUTION,
        f"Article distributed | channels={channels_used}",
        {"channels": channels_used},
    )

    return {"channels": channels_used, "distributed_at": datetime.utcnow()}


async def _transition_state(self, new_state: WorkflowState):
    """Transition workflow to new state with event recording."""
    old_state = self.current_workflow.get("state")
    self.current_workflow["state"] = new_state

    logger.debug(f"Workflow state transition | {old_state} → {new_state}")


def _record_event(self, state: WorkflowState, message: str, metadata: Dict = None):
    """Record workflow event for audit trail."""
    event = WorkflowEvent(
        timestamp=datetime.utcnow(), state=state, message=message, metadata=metadata or {}
    )
    self.workflow_events.append(event)


async def _record_workflow_metrics(self, article: GeneratedArticle, execution_time: float):
    """Record comprehensive workflow metrics for monitoring."""
    await self.metrics.record_workflow(
        workflow_id=self.current_workflow["id"],
        project_id=str(article.project_id),
        article_id=str(article.id),
        execution_time=execution_time,
        total_cost=article.total_cost,
        total_tokens=article.total_tokens_used,
        word_count=article.word_count,
        quality_score=article.readability_score,
        events=len(self.workflow_events),
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
