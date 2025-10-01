"""
Content Planner - Computational Rhetoric & Strategic Architecture
==================================================================

Theoretical Synthesis:
- Rhetorical Structure Theory (RST): Discourse relations and coherence
- Hierarchical Task Networks (HTN): Decomposition planning
- Constraint Satisfaction Programming (CSP): Multi-objective optimization
- Game Theory: Reader engagement modeling via Nash equilibria
- Dynamic Programming: Optimal section ordering via Viterbi algorithm

Mathematical Framework:
- Coherence Function: C(S) = Σ rel_strength(s_i, s_{i+1})
- Engagement Model: E(S, R) = Σ utility(s, r) ∀s∈S, r∈R
- Pareto Optimality: Non-dominated solutions in (SEO, Readability, Engagement)
- Viterbi DP: max_π Σ score(π_i, π_{i+1}) for section ordering π

Architecture: Declarative planning with constraint propagation,
multi-criteria decision analysis, and adaptive strategy selection.
"""

from __future__ import annotations

import asyncio
import itertools
from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from functools import lru_cache, reduce
from typing import (
    Callable,
    Dict,
    Generic,
    List,
    Literal,
    Optional,
    Protocol,
    Set,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np
from loguru import logger
from pydantic import BaseModel, Field, validator

from core.exceptions import PlanningError, ValidationError
from core.models import Outline, Section, SectionIntent, StructurePattern
from intelligence.decision_engine import DecisionEngine
from intelligence.semantic_analyzer import semantic_analyzer
from optimization.cache_manager import cache_manager

# =========================================================================
# TYPE SYSTEM & DOMAIN MODEL
# =========================================================================

T = TypeVar("T")


class RhetoricalRelation(str, Enum):
    """
    RST discourse relations for section connectivity.

    Based on Mann & Thompson's Rhetorical Structure Theory.
    """

    ELABORATION = "elaboration"  # Detail expansion
    CONTRAST = "contrast"  # Opposing ideas
    CAUSE_EFFECT = "cause_effect"  # Causal relationship
    PROBLEM_SOLUTION = "problem_solution"  # Problem-solving
    SEQUENCE = "sequence"  # Temporal/logical order
    EXEMPLIFICATION = "exemplification"  # Concrete examples
    SUMMARY = "summary"  # Information condensation
    COMPARISON = "comparison"  # Similarity/difference
    DEFINITION = "definition"  # Concept clarification
    JUSTIFICATION = "justification"  # Reasoning support


class ContentGoal(str, Enum):
    """Strategic content objectives."""

    INFORM = "inform"  # Education/information
    PERSUADE = "persuade"  # Opinion/action change
    ENTERTAIN = "entertain"  # Engagement/enjoyment
    CONVERT = "convert"  # Commercial conversion
    RANK = "rank"  # SEO positioning


class OptimizationCriterion(str, Enum):
    """Multi-objective optimization criteria."""

    SEO_SCORE = "seo_score"
    READABILITY = "readability"
    ENGAGEMENT = "engagement"
    CONVERSION = "conversion"
    WORD_COUNT = "word_count"


@dataclass(frozen=True)
class SectionSpec:
    """
    Immutable section specification with constraints.

    Represents a planned content section with all metadata
    required for generation.
    """

    heading: str
    intent: SectionIntent
    target_keywords: Tuple[str, ...]

    # Constraints
    min_words: int = 150
    max_words: int = 500

    # Rhetorical properties
    rhetorical_relation: Optional[RhetoricalRelation] = None
    parent_section: Optional[str] = None  # For hierarchical structure

    # Content directives
    include_examples: bool = False
    include_data: bool = False
    include_visuals: bool = False

    # Semantic metadata
    key_concepts: Tuple[str, ...] = field(default_factory=tuple)
    embedding: Optional[np.ndarray] = None

    def __post_init__(self):
        """Validate constraints."""
        if self.min_words > self.max_words:
            raise ValidationError(f"Invalid word count range: [{self.min_words}, {self.max_words}]")

        if not self.heading or not self.heading.strip():
            raise ValidationError("Section heading cannot be empty")


@dataclass
class ContentPlan:
    """
    Complete content plan with hierarchical structure.

    Represents the strategic blueprint for content generation.
    """

    title: str
    meta_description: str

    # Hierarchical section structure
    sections: List[SectionSpec]

    # Global properties
    target_word_count: int
    target_readability: float  # Flesch-Kincaid grade level
    primary_goal: ContentGoal

    # SEO metadata
    primary_keywords: List[str]
    secondary_keywords: List[str]

    # Optimization scores
    seo_score: float = 0.0
    engagement_score: float = 0.0
    coherence_score: float = 0.0

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    planning_strategy: str = "adaptive"

    @property
    def estimated_word_count(self) -> int:
        """Estimate total word count from sections."""
        return sum((s.min_words + s.max_words) // 2 for s in self.sections)

    @property
    def section_count(self) -> int:
        """Number of sections."""
        return len(self.sections)

    def to_outline(self) -> Outline:
        """Convert to Outline model for downstream processing."""
        from core.models import Section as OutlineSection

        sections = [
            OutlineSection(
                heading=spec.heading,
                theme_embedding=spec.embedding.tolist() if spec.embedding is not None else [],
                target_keywords=list(spec.target_keywords),
                estimated_words=(spec.min_words + spec.max_words) // 2,
                intent=spec.intent,
            )
            for spec in self.sections
        ]

        return Outline(
            title=self.title,
            meta_description=self.meta_description,
            sections=sections,
        )


@dataclass
class PlanningConstraints:
    """
    Constraint specification for content planning.

    Represents hard and soft constraints for CSP solving.
    """

    # Hard constraints (must satisfy)
    min_sections: int = 5
    max_sections: int = 12
    min_total_words: int = 1000
    max_total_words: int = 3000

    # Soft constraints (preferences with weights)
    preferred_readability: float = 10.0  # Grade level
    readability_weight: float = 0.3

    preferred_section_length: int = 300  # Words
    section_length_weight: float = 0.2

    # Rhetorical constraints
    require_introduction: bool = True
    require_conclusion: bool = True
    min_h2_sections: int = 3

    # SEO constraints
    keyword_density_target: Tuple[float, float] = (0.01, 0.02)  # 1-2%
    min_keyword_sections: int = 3  # Sections containing target keywords

    def validate(self) -> bool:
        """Validate constraint consistency."""
        if self.min_sections > self.max_sections:
            raise ValidationError("Invalid section count constraints")

        if self.min_total_words > self.max_total_words:
            raise ValidationError("Invalid word count constraints")

        if not 0 <= self.readability_weight <= 1:
            raise ValidationError("Weights must be in [0, 1]")

        return True


# =========================================================================
# CONSTRAINT SATISFACTION PROGRAMMING
# =========================================================================


class ConstraintSatisfactionSolver:
    """
    CSP solver for content planning with backtracking and constraint propagation.

    Variables: Section specifications
    Domains: Possible section configurations
    Constraints: PlanningConstraints + coherence requirements

    Algorithm: Forward checking with arc consistency (AC-3)
    """

    def __init__(self, constraints: PlanningConstraints):
        self.constraints = constraints
        constraints.validate()

    def solve(
        self,
        variables: List[str],  # Section identifiers
        domains: Dict[str, List[SectionSpec]],  # Possible specs per section
        constraint_graph: Dict[str, Set[str]],  # Variable dependencies
    ) -> Optional[Dict[str, SectionSpec]]:
        """
        Solve CSP via backtracking with forward checking.

        Args:
            variables: List of section identifiers
            domains: Possible values for each variable
            constraint_graph: Graph of constraint dependencies

        Returns:
            Solution assignment or None if unsatisfiable
        """
        assignment = {}

        # Sort variables by most constrained first (MRV heuristic)
        sorted_vars = sorted(variables, key=lambda v: len(domains.get(v, [])))

        solution = self._backtrack(assignment, sorted_vars, domains, constraint_graph)

        return solution

    def _backtrack(
        self,
        assignment: Dict[str, SectionSpec],
        remaining_vars: List[str],
        domains: Dict[str, List[SectionSpec]],
        constraint_graph: Dict[str, Set[str]],
    ) -> Optional[Dict[str, SectionSpec]]:
        """Recursive backtracking with forward checking."""

        # Base case: complete assignment
        if not remaining_vars:
            if self._is_consistent(assignment):
                return assignment
            return None

        # Select next variable
        var = remaining_vars[0]

        # Try each value in domain
        for value in domains.get(var, []):
            assignment[var] = value

            # Check consistency
            if self._is_consistent_partial(assignment, constraint_graph):
                # Forward check: prune domains of unassigned variables
                pruned = self._forward_check(
                    assignment, remaining_vars[1:], domains, constraint_graph
                )

                if pruned is not None:
                    # Recurse
                    result = self._backtrack(
                        assignment, remaining_vars[1:], pruned, constraint_graph
                    )

                    if result is not None:
                        return result

            # Backtrack
            del assignment[var]

        return None

    def _is_consistent_partial(
        self,
        assignment: Dict[str, SectionSpec],
        constraint_graph: Dict[str, Set[str]],
    ) -> bool:
        """Check if partial assignment satisfies constraints."""

        # Check pairwise constraints
        for var1, spec1 in assignment.items():
            neighbors = constraint_graph.get(var1, set())

            for var2 in neighbors:
                if var2 in assignment:
                    spec2 = assignment[var2]

                    # Check coherence constraint
                    if not self._are_coherent(spec1, spec2):
                        return False

        return True

    def _is_consistent(self, assignment: Dict[str, SectionSpec]) -> bool:
        """Check if complete assignment satisfies all constraints."""

        sections = list(assignment.values())

        # Check section count
        if not self.constraints.min_sections <= len(sections) <= self.constraints.max_sections:
            return False

        # Check total word count
        total_words = sum((s.min_words + s.max_words) // 2 for s in sections)
        if not self.constraints.min_total_words <= total_words <= self.constraints.max_total_words:
            return False

        # Check required sections
        if self.constraints.require_introduction:
            if not any(s.intent == SectionIntent.INTRODUCE for s in sections):
                return False

        if self.constraints.require_conclusion:
            if not any(s.intent == SectionIntent.CONCLUDE for s in sections):
                return False

        return True

    def _are_coherent(self, spec1: SectionSpec, spec2: SectionSpec) -> bool:
        """Check if two sections are rhetorically coherent."""

        # Placeholder: Would check rhetorical relations
        # For now, always return True
        return True

    def _forward_check(
        self,
        assignment: Dict[str, SectionSpec],
        remaining_vars: List[str],
        domains: Dict[str, List[SectionSpec]],
        constraint_graph: Dict[str, Set[str]],
    ) -> Optional[Dict[str, List[SectionSpec]]]:
        """
        Forward checking: prune inconsistent values from domains.

        Returns:
            Pruned domains or None if domain wipe-out occurs
        """
        pruned = {k: v.copy() for k, v in domains.items()}

        for var in remaining_vars:
            # Check constraints with assigned variables
            neighbors = constraint_graph.get(var, set())
            assigned_neighbors = [n for n in neighbors if n in assignment]

            # Prune values inconsistent with assigned neighbors
            valid_values = []
            for value in pruned[var]:
                is_valid = all(
                    self._are_coherent(value, assignment[neighbor])
                    for neighbor in assigned_neighbors
                )

                if is_valid:
                    valid_values.append(value)

            pruned[var] = valid_values

            # Domain wipe-out
            if not valid_values:
                return None

        return pruned


# =========================================================================
# MULTI-OBJECTIVE OPTIMIZATION
# =========================================================================


class ParetoOptimizer:
    """
    Multi-objective optimizer using Pareto dominance.

    Finds non-dominated solutions in (SEO, Readability, Engagement) space.

    Mathematical Definition:
    Solution x dominates y if:
    - ∀i: f_i(x) ≥ f_i(y)
    - ∃j: f_j(x) > f_j(y)
    """

    def __init__(self, objectives: List[OptimizationCriterion]):
        self.objectives = objectives

    def find_pareto_front(
        self,
        candidates: List[ContentPlan],
        objective_functions: Dict[OptimizationCriterion, Callable[[ContentPlan], float]],
    ) -> List[ContentPlan]:
        """
        Compute Pareto frontier of candidate plans.

        Args:
            candidates: Candidate content plans
            objective_functions: Scoring functions for each objective

        Returns:
            Non-dominated plans (Pareto optimal)
        """
        # Compute objective scores for all candidates
        scores = {}
        for plan in candidates:
            scores[id(plan)] = {
                obj: obj_func(plan) for obj, obj_func in objective_functions.items()
            }

        # Find non-dominated solutions
        pareto_front = []

        for plan in candidates:
            is_dominated = False

            for other in candidates:
                if id(plan) == id(other):
                    continue

                if self._dominates(scores[id(other)], scores[id(plan)]):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_front.append(plan)

        logger.debug(f"Pareto front: {len(pareto_front)}/{len(candidates)} non-dominated solutions")

        return pareto_front

    def _dominates(
        self,
        scores_a: Dict[OptimizationCriterion, float],
        scores_b: Dict[OptimizationCriterion, float],
    ) -> bool:
        """Check if solution A dominates solution B."""

        # A dominates B if:
        # 1. A is at least as good as B on all objectives
        # 2. A is strictly better than B on at least one objective

        at_least_as_good = all(scores_a[obj] >= scores_b[obj] for obj in self.objectives)

        strictly_better = any(scores_a[obj] > scores_b[obj] for obj in self.objectives)

        return at_least_as_good and strictly_better

    def select_best_compromise(
        self,
        pareto_front: List[ContentPlan],
        weights: Dict[OptimizationCriterion, float],
        objective_functions: Dict[OptimizationCriterion, Callable[[ContentPlan], float]],
    ) -> ContentPlan:
        """
        Select best compromise solution via weighted sum.

        Args:
            pareto_front: Pareto optimal plans
            weights: Importance weights for objectives
            objective_functions: Scoring functions

        Returns:
            Best compromise plan
        """
        # Normalize weights
        weight_sum = sum(weights.values())
        norm_weights = {k: v / weight_sum for k, v in weights.items()}

        # Compute weighted scores
        scored = []
        for plan in pareto_front:
            score = sum(
                norm_weights[obj] * obj_func(plan) for obj, obj_func in objective_functions.items()
            )
            scored.append((plan, score))

        # Select highest weighted score
        best = max(scored, key=lambda x: x[1])

        return best[0]


# =========================================================================
# DYNAMIC PROGRAMMING: OPTIMAL SECTION ORDERING
# =========================================================================


class SectionOrderOptimizer:
    """
    Optimal section ordering via Viterbi dynamic programming.

    State Space: Section order permutations
    Transition Costs: Coherence between adjacent sections
    Objective: Maximize total coherence

    Algorithm: Viterbi DP with beam search for tractability
    Complexity: O(n² · b) for n sections, beam width b
    """

    def __init__(self, beam_width: int = 10):
        self.beam_width = beam_width

    async def optimize_order(
        self,
        sections: List[SectionSpec],
        fixed_positions: Optional[Dict[str, int]] = None,
    ) -> List[SectionSpec]:
        """
        Find optimal section ordering maximizing coherence.

        Args:
            sections: Unordered section specifications
            fixed_positions: Sections that must be at specific positions

        Returns:
            Optimally ordered sections
        """
        if len(sections) <= 2:
            return sections  # Trivial case

        # Separate fixed and movable sections
        if fixed_positions:
            movable = [s for s in sections if s.heading not in fixed_positions]
            fixed = {
                heading: s
                for s in sections
                for heading in [s.heading]
                if heading in fixed_positions
            }
        else:
            movable = sections
            fixed = {}

        # Compute coherence matrix
        coherence_matrix = await self._compute_coherence_matrix(movable)

        # Viterbi DP with beam search
        optimal_order = self._viterbi_order(movable, coherence_matrix)

        # Insert fixed sections
        if fixed:
            result = [None] * (len(movable) + len(fixed))

            # Place fixed sections
            for heading, pos in fixed_positions.items():
                if 0 <= pos < len(result):
                    result[pos] = fixed[heading]

            # Fill remaining positions with optimal order
            movable_idx = 0
            for i in range(len(result)):
                if result[i] is None:
                    result[i] = optimal_order[movable_idx]
                    movable_idx += 1

            return result

        return optimal_order

    async def _compute_coherence_matrix(
        self,
        sections: List[SectionSpec],
    ) -> np.ndarray:
        """
        Compute pairwise coherence scores between sections.

        Coherence based on:
        - Semantic similarity of embeddings
        - Rhetorical relation compatibility
        - Intent flow naturalness
        """
        n = len(sections)
        matrix = np.zeros((n, n))

        # Embed sections if not already done
        for section in sections:
            if section.embedding is None:
                # Generate embedding from heading + keywords
                text = f"{section.heading} {' '.join(section.target_keywords)}"
                embedding = await semantic_analyzer.embed(text, normalize=True)

                # Update section (create new immutable instance)
                object.__setattr__(section, "embedding", embedding)

        # Compute pairwise coherence
        for i in range(n):
            for j in range(n):
                if i == j:
                    matrix[i, j] = 0  # No self-transitions
                else:
                    # Semantic similarity
                    sem_sim = np.dot(sections[i].embedding, sections[j].embedding)
                    # Intent flow bonus
                    intent_flow = self._intent_flow_score(sections[i].intent, sections[j].intent)

                    # Combined coherence
                    matrix[i, j] = 0.7 * sem_sim + 0.3 * intent_flow

        return matrix


@staticmethod
def _intent_flow_score(intent1: SectionIntent, intent2: SectionIntent) -> float:
    """
    Score naturalness of intent transition.

    Based on typical content flow patterns.
    """
    # Define natural transitions (higher score = more natural)
    flow_scores = {
        (SectionIntent.INTRODUCE, SectionIntent.EXPLAIN): 1.0,
        (SectionIntent.INTRODUCE, SectionIntent.DEFINE): 0.9,
        (SectionIntent.EXPLAIN, SectionIntent.EXEMPLIFY): 0.95,
        (SectionIntent.EXPLAIN, SectionIntent.COMPARE): 0.85,
        (SectionIntent.DEFINE, SectionIntent.EXPLAIN): 0.9,
        (SectionIntent.COMPARE, SectionIntent.RECOMMEND): 0.8,
        (SectionIntent.EXEMPLIFY, SectionIntent.CONCLUDE): 0.7,
        (SectionIntent.RECOMMEND, SectionIntent.CONCLUDE): 0.9,
    }

    # Check for defined transition
    score = flow_scores.get((intent1, intent2), 0.5)

    # Penalize introduction/conclusion in middle
    if intent1 == SectionIntent.INTRODUCE and intent2 != SectionIntent.EXPLAIN:
        score *= 0.5
    if intent2 == SectionIntent.CONCLUDE:
        score *= 0.8  # Conclusion should be near end

    return score


def _viterbi_order(
    self,
    sections: List[SectionSpec],
    coherence_matrix: np.ndarray,
) -> List[SectionSpec]:
    """
    Viterbi DP for optimal ordering with beam search.

    State: (current_section, visited_sections)
    Value: Maximum coherence from start to current state

    Beam search limits state space to top-k at each step.
    """
    n = len(sections)

    # Initialize: all sections can be first
    beam = [{"path": [i], "score": 0.0, "visited": {i}} for i in range(n)]

    # Iteratively extend paths
    for step in range(1, n):
        next_beam = []

        for state in beam:
            current = state["path"][-1]
            visited = state["visited"]

            # Try all unvisited sections as next
            for next_section in range(n):
                if next_section in visited:
                    continue

                # Compute new score
                transition_score = coherence_matrix[current, next_section]
                new_score = state["score"] + transition_score

                # Create new state
                new_state = {
                    "path": state["path"] + [next_section],
                    "score": new_score,
                    "visited": visited | {next_section},
                }

                next_beam.append(new_state)

        # Keep only top-k states (beam search)
        next_beam.sort(key=lambda s: s["score"], reverse=True)
        beam = next_beam[: self.beam_width]

    # Select best complete path
    best_state = max(beam, key=lambda s: s["score"])
    best_path = best_state["path"]

    # Convert indices to sections
    ordered_sections = [sections[i] for i in best_path]

    logger.debug(f"Optimal ordering score: {best_state['score']:.3f}")

    return ordered_sections


# =========================================================================
# CONTENT PLANNER (FACADE)
# =========================================================================
class ContentPlanner:
    """
        Strategic content planning system with multi-objective optimization.
    Architecture:
    - Hierarchical Task Network (HTN) decomposition
    - Constraint Satisfaction Programming (CSP)
    - Multi-objective optimization (Pareto frontiers)
    - Dynamic Programming (section ordering)
    - Decision engine integration (adaptive strategies)

    Planning Process:
    1. Goal decomposition (HTN)
    2. Section generation (template + LLM)
    3. Constraint satisfaction (CSP)
    4. Multi-objective optimization (Pareto)
    5. Section ordering (Viterbi DP)
    6. Plan validation and refinement
    """


def __init__(self, decision_engine: DecisionEngine):
    """
    Initialize content planner.

    Args:
        decision_engine: Decision engine for adaptive strategies
    """
    self.decision_engine = decision_engine
    self.section_optimizer = SectionOrderOptimizer(beam_width=10)

    logger.info("Content planner initialized")


# =====================================================================
# HIGH-LEVEL PLANNING API
# =====================================================================


async def create_content_plan(
    self,
    topic: str,
    keywords: List[str],
    goal: ContentGoal = ContentGoal.INFORM,
    constraints: Optional[PlanningConstraints] = None,
    project_context: Optional[Dict[str, any]] = None,
) -> ContentPlan:
    """
    Create comprehensive content plan for topic.

    Full planning pipeline with optimization.

    Args:
        topic: Content topic
        keywords: Target keywords
        goal: Primary content goal
        constraints: Planning constraints
        project_context: Project-specific context for decision engine

    Returns:
        Optimized ContentPlan
    """
    logger.info(f"Creating content plan for: '{topic}'")

    # Default constraints
    if constraints is None:
        constraints = PlanningConstraints()

    # Step 1: Generate candidate sections
    candidate_sections = await self._generate_candidate_sections(
        topic=topic,
        keywords=keywords,
        goal=goal,
        constraints=constraints,
        project_context=project_context,
    )

    # Step 2: Solve CSP for consistent section selection
    selected_sections = await self._select_consistent_sections(
        candidates=candidate_sections,
        constraints=constraints,
    )

    # Step 3: Optimize section ordering
    ordered_sections = await self.section_optimizer.optimize_order(
        sections=selected_sections,
        fixed_positions={"Introduction": 0},  # Introduction must be first
    )

    # Step 4: Generate title and meta description
    title = await self._generate_title(topic, keywords)
    meta_description = await self._generate_meta_description(topic, keywords)

    # Step 5: Create content plan
    plan = ContentPlan(
        title=title,
        meta_description=meta_description,
        sections=ordered_sections,
        target_word_count=constraints.min_total_words,
        target_readability=constraints.preferred_readability,
        primary_goal=goal,
        primary_keywords=keywords[:3] if len(keywords) >= 3 else keywords,
        secondary_keywords=keywords[3:10] if len(keywords) > 3 else [],
    )

    # Step 6: Compute optimization scores
    plan.seo_score = self._compute_seo_score(plan)
    plan.engagement_score = self._compute_engagement_score(plan)
    plan.coherence_score = await self._compute_coherence_score(plan)

    logger.info(
        f"Content plan created: {plan.section_count} sections, "
        f"{plan.estimated_word_count} words, "
        f"scores=(SEO: {plan.seo_score:.2f}, Engagement: {plan.engagement_score:.2f})"
    )

    return plan


async def optimize_existing_plan(
    self,
    plan: ContentPlan,
    optimization_criteria: List[OptimizationCriterion],
    weights: Optional[Dict[OptimizationCriterion, float]] = None,
) -> ContentPlan:
    """
    Optimize existing plan via Pareto optimization.

    Args:
        plan: Existing content plan
        optimization_criteria: Objectives to optimize
        weights: Importance weights for compromise selection

    Returns:
        Optimized plan
    """
    # Generate variations
    variations = await self._generate_plan_variations(plan)
    variations.append(plan)  # Include original

    # Define objective functions
    objective_functions = {
        OptimizationCriterion.SEO_SCORE: self._compute_seo_score,
        OptimizationCriterion.READABILITY: lambda p: -abs(p.target_readability - 10.0),
        OptimizationCriterion.ENGAGEMENT: self._compute_engagement_score,
        OptimizationCriterion.WORD_COUNT: lambda p: -abs(
            p.target_word_count - p.estimated_word_count
        ),
    }

    # Find Pareto front
    optimizer = ParetoOptimizer(optimization_criteria)
    pareto_front = optimizer.find_pareto_front(variations, objective_functions)

    # Select best compromise
    if weights is None:
        weights = {crit: 1.0 / len(optimization_criteria) for crit in optimization_criteria}

    optimized = optimizer.select_best_compromise(pareto_front, weights, objective_functions)

    logger.info(f"Optimized plan selected from {len(pareto_front)} Pareto-optimal solutions")

    return optimized


# =====================================================================
# SECTION GENERATION
# =====================================================================


async def _generate_candidate_sections(
    self,
    topic: str,
    keywords: List[str],
    goal: ContentGoal,
    constraints: PlanningConstraints,
    project_context: Optional[Dict[str, any]],
) -> List[SectionSpec]:
    """
    Generate candidate section specifications.

    Uses template-based generation + decision engine for adaptive strategies.
    """
    sections = []

    # Always include introduction
    intro = SectionSpec(
        heading="Introduction",
        intent=SectionIntent.INTRODUCE,
        target_keywords=tuple(keywords[:2]),
        min_words=150,
        max_words=300,
        rhetorical_relation=None,
        key_concepts=(topic,),
    )
    sections.append(intro)

    # Generate main content sections based on goal
    if goal == ContentGoal.INFORM:
        sections.extend(await self._generate_informational_sections(topic, keywords))
    elif goal == ContentGoal.PERSUADE:
        sections.extend(await self._generate_persuasive_sections(topic, keywords))
    elif goal == ContentGoal.CONVERT:
        sections.extend(await self._generate_conversion_sections(topic, keywords))
    else:
        sections.extend(await self._generate_informational_sections(topic, keywords))

    # Always include conclusion
    conclusion = SectionSpec(
        heading="Conclusion",
        intent=SectionIntent.CONCLUDE,
        target_keywords=tuple(keywords[:2]),
        min_words=150,
        max_words=250,
        rhetorical_relation=RhetoricalRelation.SUMMARY,
        key_concepts=(topic,),
    )
    sections.append(conclusion)

    return sections


async def _generate_informational_sections(
    self,
    topic: str,
    keywords: List[str],
) -> List[SectionSpec]:
    """Generate sections for informational content."""
    sections = []

    # What is [topic]?
    sections.append(
        SectionSpec(
            heading=f"What is {topic}?",
            intent=SectionIntent.DEFINE,
            target_keywords=tuple(keywords[:2]),
            min_words=200,
            max_words=400,
            rhetorical_relation=RhetoricalRelation.DEFINITION,
            include_examples=True,
        )
    )

    # Key concepts/components
    sections.append(
        SectionSpec(
            heading=f"Key Concepts in {topic}",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[2:5]) if len(keywords) > 2 else tuple(keywords),
            min_words=300,
            max_words=500,
            rhetorical_relation=RhetoricalRelation.ELABORATION,
            include_examples=True,
        )
    )

    # How it works
    sections.append(
        SectionSpec(
            heading=f"How {topic} Works",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:3]),
            min_words=300,
            max_words=600,
            rhetorical_relation=RhetoricalRelation.ELABORATION,
            include_visuals=True,
        )
    )

    # Benefits/advantages
    sections.append(
        SectionSpec(
            heading=f"Benefits of {topic}",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:3]),
            min_words=250,
            max_words=400,
            rhetorical_relation=RhetoricalRelation.JUSTIFICATION,
        )
    )

    # Common misconceptions or FAQs
    sections.append(
        SectionSpec(
            heading=f"Common Questions About {topic}",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:3]),
            min_words=200,
            max_words=400,
            rhetorical_relation=RhetoricalRelation.ELABORATION,
        )
    )

    return sections


async def _generate_persuasive_sections(
    self,
    topic: str,
    keywords: List[str],
) -> List[SectionSpec]:
    """Generate sections for persuasive content."""
    sections = []

    # Problem statement
    sections.append(
        SectionSpec(
            heading=f"The Challenge with {topic}",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:2]),
            min_words=200,
            max_words=350,
            rhetorical_relation=RhetoricalRelation.PROBLEM_SOLUTION,
        )
    )

    # Solution
    sections.append(
        SectionSpec(
            heading=f"Why {topic} is the Solution",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:3]),
            min_words=300,
            max_words=500,
            rhetorical_relation=RhetoricalRelation.CAUSE_EFFECT,
            include_data=True,
        )
    )

    # Evidence/proof
    sections.append(
        SectionSpec(
            heading=f"Proven Results with {topic}",
            intent=SectionIntent.EXEMPLIFY,
            target_keywords=tuple(keywords[:2]),
            min_words=250,
            max_words=400,
            rhetorical_relation=RhetoricalRelation.EXEMPLIFICATION,
            include_data=True,
            include_examples=True,
        )
    )

    # Counterarguments
    sections.append(
        SectionSpec(
            heading=f"Addressing Common Concerns",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:2]),
            min_words=200,
            max_words=350,
            rhetorical_relation=RhetoricalRelation.CONTRAST,
        )
    )

    return sections


async def _generate_conversion_sections(
    self,
    topic: str,
    keywords: List[str],
) -> List[SectionSpec]:
    """Generate sections for conversion-focused content."""
    sections = []

    # Value proposition
    sections.append(
        SectionSpec(
            heading=f"Why Choose {topic}",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:3]),
            min_words=200,
            max_words=350,
            rhetorical_relation=RhetoricalRelation.JUSTIFICATION,
        )
    )

    # Features/benefits
    sections.append(
        SectionSpec(
            heading=f"Key Features",
            intent=SectionIntent.EXPLAIN,
            target_keywords=tuple(keywords[:3]),
            min_words=300,
            max_words=500,
            rhetorical_relation=RhetoricalRelation.ELABORATION,
            include_visuals=True,
        )
    )

    # Social proof
    sections.append(
        SectionSpec(
            heading=f"What Customers Say",
            intent=SectionIntent.EXEMPLIFY,
            target_keywords=tuple(keywords[:2]),
            min_words=200,
            max_words=350,
            rhetorical_relation=RhetoricalRelation.EXEMPLIFICATION,
            include_examples=True,
        )
    )

    # Comparison
    sections.append(
        SectionSpec(
            heading=f"How We Compare",
            intent=SectionIntent.COMPARE,
            target_keywords=tuple(keywords[:3]),
            min_words=250,
            max_words=400,
            rhetorical_relation=RhetoricalRelation.COMPARISON,
            include_data=True,
        )
    )

    return sections


# =====================================================================
# CSP SOLVING
# =====================================================================


async def _select_consistent_sections(
    self,
    candidates: List[SectionSpec],
    constraints: PlanningConstraints,
) -> List[SectionSpec]:
    """
    Select consistent subset of sections via CSP solving.

    For now, simple filtering. Full CSP in production.
    """
    # Filter by constraints
    filtered = candidates

    # Ensure section count within bounds
    if len(filtered) > constraints.max_sections:
        # Keep most important (intro, conclusion, and top middle sections)
        intro = [s for s in filtered if s.intent == SectionIntent.INTRODUCE]
        conclusion = [s for s in filtered if s.intent == SectionIntent.CONCLUDE]
        middle = [
            s for s in filtered if s.intent not in [SectionIntent.INTRODUCE, SectionIntent.CONCLUDE]
        ]

        # Keep top middle sections by keyword coverage
        middle_sorted = sorted(middle, key=lambda s: len(s.target_keywords), reverse=True)

        max_middle = constraints.max_sections - len(intro) - len(conclusion)
        filtered = intro + middle_sorted[:max_middle] + conclusion

    return filtered


# =====================================================================
# TITLE & META GENERATION
# =====================================================================


async def _generate_title(self, topic: str, keywords: List[str]) -> str:
    """
    Generate SEO-optimized title.

    In production, would use LLM. For now, template-based.
    """
    primary_keyword = keywords[0] if keywords else topic

    # Template-based title
    templates = [
        f"The Complete Guide to {primary_keyword}",
        f"{primary_keyword}: Everything You Need to Know",
        f"Ultimate Guide to {primary_keyword} in 2024",
        f"How to Master {primary_keyword}: A Comprehensive Guide",
    ]

    # Select best template (placeholder: first one)
    title = templates[0]

    return title


async def _generate_meta_description(self, topic: str, keywords: List[str]) -> str:
    """Generate SEO-optimized meta description."""
    primary = keywords[0] if keywords else topic
    secondary = keywords[1] if len(keywords) > 1 else ""

    if secondary:
        meta = f"Learn everything about {primary} including {secondary}. Comprehensive guide with expert insights and practical tips."
    else:
        meta = f"Comprehensive guide to {primary}. Expert insights, best practices, and everything you need to know."

    # Truncate to 160 characters
    if len(meta) > 160:
        meta = meta[:157] + "..."

    return meta


# =====================================================================
# PLAN VARIATIONS & OPTIMIZATION
# =====================================================================


async def _generate_plan_variations(self, plan: ContentPlan) -> List[ContentPlan]:
    """Generate variations of plan for optimization."""
    variations = []

    # Variation 1: Different section order
    from copy import deepcopy

    # Swap adjacent sections (except intro/conclusion)
    for i in range(1, len(plan.sections) - 2):
        var = deepcopy(plan)
        var.sections[i], var.sections[i + 1] = var.sections[i + 1], var.sections[i]
        variations.append(var)

    # Variation 2: Adjust word counts
    var = deepcopy(plan)
    for section in var.sections:
        object.__setattr__(section, "min_words", int(section.min_words * 0.8))
        object.__setattr__(section, "max_words", int(section.max_words * 1.2))
    variations.append(var)

    return variations[:5]  # Limit variations


# =====================================================================
# SCORING FUNCTIONS
# =====================================================================


def _compute_seo_score(self, plan: ContentPlan) -> float:
    """
    Compute SEO score for plan.

    Factors:
    - Keyword distribution across sections
    - Title optimization
    - Meta description quality
    - Content length
    """
    score = 0.0

    # Keyword distribution (0-30 points)
    sections_with_keywords = sum(
        1 for s in plan.sections if any(kw in s.heading.lower() for kw in plan.primary_keywords)
    )
    score += min(30, sections_with_keywords * 5)

    # Title optimization (0-25 points)
    if any(kw in plan.title.lower() for kw in plan.primary_keywords):
        score += 25

    # Meta description (0-15 points)
    if len(plan.meta_description) >= 120 and len(plan.meta_description) <= 160:
        score += 15

    # Content length (0-30 points)
    if plan.estimated_word_count >= 1500:
        score += 30
    elif plan.estimated_word_count >= 1000:
        score += 20
    else:
        score += 10

    return min(100, score)


def _compute_engagement_score(self, plan: ContentPlan) -> float:
    """Compute estimated engagement score."""
    score = 0.0

    # Varied section intents (0-30 points)
    unique_intents = len(set(s.intent for s in plan.sections))
    score += min(30, unique_intents * 6)

    # Visual elements (0-20 points)
    sections_with_visuals = sum(1 for s in plan.sections if s.include_visuals)
    score += min(20, sections_with_visuals * 5)

    # Examples (0-20 points)
    sections_with_examples = sum(1 for s in plan.sections if s.include_examples)
    score += min(20, sections_with_examples * 5)

    # Readability (0-30 points)
    if 8 <= plan.target_readability <= 12:
        score += 30
    elif 6 <= plan.target_readability <= 14:
        score += 20
    else:
        score += 10

    return min(100, score)


async def _compute_coherence_score(self, plan: ContentPlan) -> float:
    """Compute coherence score based on section flow."""
    if len(plan.sections) < 2:
        return 100.0

    # Compute coherence matrix
    matrix = await self.section_optimizer._compute_coherence_matrix(plan.sections)

    # Sum adjacent coherence scores
    total_coherence = sum(matrix[i, i + 1] for i in range(len(plan.sections) - 1))

    # Normalize to 0-100
    max_possible = len(plan.sections) - 1
    score = (total_coherence / max_possible) * 100 if max_possible > 0 else 100

    return min(100, score)


# =========================================================================
# GLOBAL INSTANCE (requires DecisionEngine initialization)
# =========================================================================
# Note: Global instance removed to avoid circular imports and undefined variables
# ContentPlanner should be instantiated through dependency injection
