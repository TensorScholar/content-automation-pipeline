"""
Implements a 3-layer decision hierarchy (Rulebook, Inferred Patterns, Best Practices) to guide content generation.
"""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from uuid import UUID

import numpy as np
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import DecisionError, ValidationError
from core.models import InferredPatterns, RuleType
from intelligence.best_practices_kb import BestPracticesKB
from intelligence.semantic_analyzer import SimilarityMetric, semantic_analyzer
from knowledge.project_repository import ProjectRepository
from knowledge.rulebook_manager import RulebookManager


class DecisionLayer(str, Enum):
    """Source layer for decision."""

    EXPLICIT_RULE = "explicit_rule"
    INFERRED_PATTERN = "inferred_pattern"
    BEST_PRACTICE = "best_practice"
    COMPOSITE = "composite"  # Multiple sources combined


class DecisionConfidence(str, Enum):
    """Confidence levels for decisions."""

    VERY_HIGH = "very_high"  # 0.90+
    HIGH = "high"  # 0.75-0.90
    MEDIUM = "medium"  # 0.60-0.75
    LOW = "low"  # 0.45-0.60
    VERY_LOW = "very_low"  # <0.45


@dataclass
class Evidence:
    """
    Single piece of evidence contributing to a decision.

    Represents information from any layer with associated metadata.
    """

    source_layer: DecisionLayer
    content: str
    embedding: np.ndarray
    confidence: float  # 0-1
    authority: float  # Layer-specific weight (1=highest, 0.7=medium, 0.5=low)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate evidence parameters."""
        if not 0 <= self.confidence <= 1:
            raise ValidationError(f"Confidence must be in [0,1], got {self.confidence}")
        if not 0 <= self.authority <= 1:
            raise ValidationError(f"Authority must be in [0,1], got {self.authority}")


@dataclass
class Decision:
    """
    Final decision with full provenance and confidence scoring.

    Includes all evidence that contributed to the decision for auditability.
    """

    query: str
    choice: str
    confidence_score: float
    confidence_level: DecisionConfidence
    primary_layer: DecisionLayer
    evidence_chain: List[Evidence]
    reasoning: str
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize decision for logging/storage."""
        return {
            "query": self.query,
            "choice": self.choice,
            "confidence_score": self.confidence_score,
            "confidence_level": self.confidence_level.value,
            "primary_layer": self.primary_layer.value,
            "evidence_count": len(self.evidence_chain),
            "reasoning": self.reasoning,
            "timestamp": self.timestamp.isoformat(),
        }


@dataclass
class DecisionContext:
    """Context for decision-making process."""

    project_id: UUID
    decision_type: str
    query: str
    additional_context: Optional[Dict[str, Any]] = None


class DecisionManifold:
    """
    High-dimensional decision space where choices emerge geometrically.

    Mathematical Foundation:
    - Each evidence source creates a "force field" in decision space
    - Forces have direction (recommendation embedding) and magnitude (confidence × authority)
    - Final decision is the equilibrium point (resultant vector)
    - No conditional branching - pure vector algebra
    """

    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim

    def resolve(
        self,
        query_embedding: np.ndarray,
        evidence_list: List[Evidence],
    ) -> Tuple[np.ndarray, float]:
        """
        Resolve decision through vector space equilibrium.

        Process:
        1. Project query into decision manifold
        2. Compute influence vectors from each evidence
        3. Weight by (similarity × confidence × authority)
        4. Find resultant vector (equilibrium point)
        5. Return decision vector and confidence

        Args:
            query_embedding: Query representation in semantic space
            evidence_list: All available evidence

        Returns:
            Tuple of (decision_vector, aggregate_confidence)
        """
        if not evidence_list:
            raise DecisionError("Cannot resolve decision without evidence")

        # Compute influence vectors
        influences = []
        total_weight = 0.0

        for evidence in evidence_list:
            # Semantic similarity (how relevant is this evidence?)
            similarity = semantic_analyzer.compute_similarity(
                query_embedding, evidence.embedding, metric=SimilarityMetric.COSINE
            )

            # Combined weight: similarity × confidence × authority
            weight = similarity * evidence.confidence * evidence.authority

            # Influence vector: weighted evidence embedding
            influence = evidence.embedding * weight

            influences.append(influence)
            total_weight += weight

        # Resultant vector (equilibrium point)
        if total_weight > 0:
            decision_vector = np.sum(influences, axis=0) / total_weight
        else:
            # Fallback: unweighted average
            decision_vector = np.mean([e.embedding for e in evidence_list], axis=0)
            total_weight = 1.0

        # Normalize to unit vector
        decision_vector = semantic_analyzer.normalize_vector(decision_vector)

        # Aggregate confidence (weighted average)
        aggregate_confidence = min(1.0, total_weight / len(evidence_list))

        return decision_vector, aggregate_confidence

    def find_best_match(
        self,
        decision_vector: np.ndarray,
        candidate_embeddings: Dict[str, np.ndarray],
    ) -> Tuple[str, float]:
        """
        Find closest candidate to decision vector.

        Args:
            decision_vector: Resolved decision in semantic space
            candidate_embeddings: Dict mapping choices to embeddings

        Returns:
            Tuple of (best_choice, similarity_score)
        """
        best_choice = None
        best_similarity = -1.0

        for choice, embedding in candidate_embeddings.items():
            similarity = semantic_analyzer.compute_similarity(
                decision_vector, embedding, metric=SimilarityMetric.COSINE
            )

            if similarity > best_similarity:
                best_similarity = similarity
                best_choice = choice

        return best_choice, best_similarity


class DecisionEngine:
    """
    Core intelligence orchestrator implementing 3-layer decision hierarchy.

    Architecture:
    - Pure functional composition of evidence from multiple sources
    - Bayesian-inspired evidence accumulation
    - Geometric decision resolution (no conditional logic)
    - Full provenance tracking for auditability
    """

    def __init__(
        self,
        session: AsyncSession,
        rulebook_manager: RulebookManager,
        best_practices: BestPracticesKB,
    ):
        self.session = session
        self.rulebook_manager = rulebook_manager
        self.best_practices = best_practices
        self.manifold = DecisionManifold()

        # Authority weights for each layer
        self.layer_authorities = {
            DecisionLayer.EXPLICIT_RULE: 1.0,  # Highest authority
            DecisionLayer.INFERRED_PATTERN: 0.7,  # Medium authority
            DecisionLayer.BEST_PRACTICE: 0.5,  # Baseline authority
        }

    # =========================================================================
    # MAIN DECISION INTERFACE
    # =========================================================================

    async def make_decision(
        self,
        project_id: UUID,
        query: str,
        decision_context: Optional[Dict[str, Any]] = None,
        rule_type: Optional[RuleType] = None,
    ) -> Decision:
        """
        Make intelligent decision using 3-layer hierarchy.

        Process Flow:
        1. Generate query embedding
        2. Gather evidence from all layers (parallel)
        3. Resolve decision via geometric manifold
        4. Generate human-readable reasoning

        Args:
            project_id: Project context
            query: Natural language decision query
            decision_context: Optional context (e.g., target audience, content type)
            rule_type: Optional filter for specific rule types

        Returns:
            Decision object with full provenance
        """
        try:
            logger.info(f"Decision query: '{query}' for project {project_id}")

            # Generate query embedding
            query_embedding = await semantic_analyzer.embed(query, normalize=True)

            # Gather evidence from all layers
            evidence = await self._gather_evidence(
                project_id=project_id,
                query=query,
                query_embedding=query_embedding,
                rule_type=rule_type,
                context=decision_context or {},
            )

            if not evidence:
                raise DecisionError(f"No evidence found for query: {query}")

            # Resolve decision through manifold
            decision_vector, confidence = self.manifold.resolve(
                query_embedding=query_embedding,
                evidence_list=evidence,
            )

            # Find best matching choice
            # For open-ended decisions, use evidence content directly
            choice = await self._synthesize_choice(decision_vector, evidence)

            # Determine primary layer and confidence level
            primary_layer = self._determine_primary_layer(evidence)
            confidence_level = self._classify_confidence(confidence)

            # Generate reasoning
            reasoning = self._generate_reasoning(evidence, choice, confidence)

            decision = Decision(
                query=query,
                choice=choice,
                confidence_score=confidence,
                confidence_level=confidence_level,
                primary_layer=primary_layer,
                evidence_chain=evidence,
                reasoning=reasoning,
            )

            logger.info(
                f"Decision made: {choice} (confidence: {confidence:.2f}, layer: {primary_layer.value})"
            )
            return decision

        except Exception as e:
            logger.error(f"Decision failed: {e}")
            raise DecisionError(f"Decision process failed: {e}")

    # =========================================================================
    # EVIDENCE GATHERING (Layer 1, 2, 3)
    # =========================================================================

    async def _gather_evidence(
        self,
        project_id: UUID,
        query: str,
        query_embedding: np.ndarray,
        rule_type: Optional[RuleType],
        context: Dict[str, Any],
    ) -> List[Evidence]:
        """
        Gather evidence from all available layers in parallel.

        Args:
            project_id: Project identifier
            query: Decision query
            query_embedding: Query vector
            rule_type: Optional rule type filter
            context: Decision context

        Returns:
            List of Evidence objects from all layers
        """
        evidence = []

        # Layer 1: Query explicit rules
        explicit_evidence = await self._query_explicit_rules(
            project_id=project_id,
            query=query,
            query_embedding=query_embedding,
            rule_type=rule_type,
        )
        evidence.extend(explicit_evidence)

        # Layer 2: Query inferred patterns
        pattern_evidence = await self._query_inferred_patterns(
            project_id=project_id,
            query=query,
            query_embedding=query_embedding,
            context=context,
        )
        evidence.extend(pattern_evidence)

        # Layer 3: Query best practices
        best_practice_evidence = await self._query_best_practices(
            query=query,
            query_embedding=query_embedding,
            rule_type=rule_type,
        )
        evidence.extend(best_practice_evidence)

        logger.debug(f"Gathered {len(evidence)} evidence items across all layers")
        return evidence

    async def _query_explicit_rules(
        self,
        project_id: UUID,
        query: str,
        query_embedding: np.ndarray,
        rule_type: Optional[RuleType],
    ) -> List[Evidence]:
        """
        Layer 1: Query explicit rulebook via vector similarity.

        High threshold (0.80+) ensures only relevant rules contribute.
        """
        try:
            rules = await self.rulebook_manager.query_rules(
                project_id=project_id,
                query=query,
                rule_type=rule_type,
                top_k=3,
                similarity_threshold=0.80,
            )

            evidence = []
            for rule, similarity in rules:
                # Generate embedding for rule content if not cached
                rule_embedding = await semantic_analyzer.embed(rule.content, normalize=True)

                evidence.append(
                    Evidence(
                        source_layer=DecisionLayer.EXPLICIT_RULE,
                        content=rule.content,
                        embedding=rule_embedding,
                        confidence=similarity,  # Similarity = confidence
                        authority=self.layer_authorities[DecisionLayer.EXPLICIT_RULE],
                        metadata={
                            "rule_id": str(rule.id),
                            "rule_type": rule.rule_type.value,
                            "priority": rule.priority,
                            "context": rule.context,
                        },
                    )
                )

            if evidence:
                logger.debug(f"Layer 1: Found {len(evidence)} explicit rules")

            return evidence

        except Exception as e:
            logger.warning(f"Layer 1 query failed: {e}")
            return []

    async def _query_inferred_patterns(
        self,
        project_id: UUID,
        query: str,
        query_embedding: np.ndarray,
        context: Dict[str, Any],
    ) -> List[Evidence]:
        """
        Layer 2: Query inferred patterns with statistical confidence.

        Translates linguistic patterns into decision recommendations.
        """
        try:
            # Retrieve inferred patterns
            patterns = await self._get_inferred_patterns(project_id)

            if not patterns:
                return []

            evidence = []

            # Extract relevant pattern-based recommendations
            # Example: tone query → tone embedding similarity
            if "tone" in query.lower() or "voice" in query.lower():
                recommendation = self._pattern_to_tone_recommendation(patterns)

                evidence.append(
                    Evidence(
                        source_layer=DecisionLayer.INFERRED_PATTERN,
                        content=recommendation,
                        embedding=patterns.tone_embedding,
                        confidence=patterns.confidence,
                        authority=self.layer_authorities[DecisionLayer.INFERRED_PATTERN],
                        metadata={
                            "pattern_id": str(patterns.id),
                            "sample_size": patterns.sample_size,
                            "analyzed_at": patterns.analyzed_at.isoformat(),
                        },
                    )
                )

            # Structure patterns
            if "structure" in query.lower() or "format" in query.lower():
                for structure in patterns.structure_patterns[:2]:  # Top 2
                    recommendation = self._pattern_to_structure_recommendation(structure)
                    rec_embedding = await semantic_analyzer.embed(recommendation, normalize=True)

                    evidence.append(
                        Evidence(
                            source_layer=DecisionLayer.INFERRED_PATTERN,
                            content=recommendation,
                            embedding=rec_embedding,
                            confidence=structure.frequency * patterns.confidence,
                            authority=self.layer_authorities[DecisionLayer.INFERRED_PATTERN],
                            metadata={
                                "pattern_type": structure.pattern_type,
                                "frequency": structure.frequency,
                            },
                        )
                    )

            # Readability recommendations
            if "readability" in query.lower() or "complexity" in query.lower():
                recommendation = self._pattern_to_readability_recommendation(patterns)
                rec_embedding = await semantic_analyzer.embed(recommendation, normalize=True)

                evidence.append(
                    Evidence(
                        source_layer=DecisionLayer.INFERRED_PATTERN,
                        content=recommendation,
                        embedding=rec_embedding,
                        confidence=patterns.confidence,
                        authority=self.layer_authorities[DecisionLayer.INFERRED_PATTERN],
                        metadata={
                            "readability_score": patterns.readability_score,
                            "avg_sentence_length": patterns.avg_sentence_length,
                        },
                    )
                )

            if evidence:
                logger.debug(f"Layer 2: Found {len(evidence)} inferred patterns")

            return evidence

        except Exception as e:
            logger.warning(f"Layer 2 query failed: {e}")
            return []

    async def _query_best_practices(
        self,
        query: str,
        query_embedding: np.ndarray,
        rule_type: Optional[RuleType],
    ) -> List[Evidence]:
        """
        Layer 3: Query universal best practices knowledge base.

        Provides baseline recommendations when project-specific knowledge is unavailable.
        """
        try:
            practices = await self.best_practices.query(
                query=query,
                query_embedding=query_embedding,
                rule_type=rule_type,
                top_k=2,
            )

            evidence = []
            for practice, similarity in practices:
                evidence.append(
                    Evidence(
                        source_layer=DecisionLayer.BEST_PRACTICE,
                        content=practice.content,
                        embedding=practice.embedding,
                        confidence=similarity * 0.9,  # Slightly reduce for baseline
                        authority=self.layer_authorities[DecisionLayer.BEST_PRACTICE],
                        metadata={
                            "practice_id": practice.id,
                            "category": practice.category,
                        },
                    )
                )

            if evidence:
                logger.debug(f"Layer 3: Found {len(evidence)} best practices")

            return evidence

        except Exception as e:
            logger.warning(f"Layer 3 query failed: {e}")
            return []

    # =========================================================================
    # PATTERN TRANSLATION
    # =========================================================================

    @staticmethod
    def _pattern_to_tone_recommendation(patterns: InferredPatterns) -> str:
        """Translate linguistic patterns to tone recommendation."""
        # Analyze readability and sentence structure
        if patterns.readability_score < 10:
            formality = "conversational and accessible"
        elif patterns.readability_score < 13:
            formality = "professional yet approachable"
        else:
            formality = "formal and authoritative"

        if patterns.lexical_diversity > 0.7:
            vocabulary = "rich and varied vocabulary"
        else:
            vocabulary = "clear and straightforward language"

        return (
            f"Adopt a {formality} tone with {vocabulary}. "
            f"Maintain average sentence length around {patterns.avg_sentence_length:.0f} words."
        )

    @staticmethod
    def _pattern_to_structure_recommendation(structure) -> str:
        """Translate structure pattern to recommendation."""
        return (
            f"Consider using a {structure.pattern_type} format "
            f"(observed in {structure.frequency * 100:.0f}% of existing content). "
            f"Typical structure: {', '.join(structure.typical_sections)}. "
            f"Target approximately {structure.avg_word_count} words."
        )

    @staticmethod
    def _pattern_to_readability_recommendation(patterns: InferredPatterns) -> str:
        """Translate readability patterns to recommendation."""
        return (
            f"Target Flesch-Kincaid grade level {patterns.readability_score:.1f}. "
            f"Use average sentence length of {patterns.avg_sentence_length:.0f} words "
            f"(±{patterns.sentence_length_std:.0f} std dev). "
            f"Maintain lexical diversity around {patterns.lexical_diversity:.2f}."
        )

    # =========================================================================
    # DECISION SYNTHESIS
    # =========================================================================

    async def _synthesize_choice(
        self,
        decision_vector: np.ndarray,
        evidence: List[Evidence],
    ) -> str:
        """
        Synthesize final choice from decision vector and evidence.

        For open-ended decisions, combines evidence content weighted by influence.
        """
        # Weight evidence by (confidence × authority)
        weights = [e.confidence * e.authority for e in evidence]
        total_weight = sum(weights)

        if total_weight == 0:
            # Fallback: concatenate all evidence
            return " ".join(e.content for e in evidence)

        # Weighted combination (prioritize high-weight evidence)
        # Take top 3 most influential pieces
        indexed_evidence = list(zip(evidence, weights))
        indexed_evidence.sort(key=lambda x: x[1], reverse=True)

        top_evidence = indexed_evidence[:3]

        # Synthesize from top evidence
        synthesized = " ".join(e.content for e, w in top_evidence)

        return synthesized

    @staticmethod
    def _determine_primary_layer(evidence: List[Evidence]) -> DecisionLayer:
        """Determine which layer contributed most to decision."""
        if not evidence:
            return DecisionLayer.BEST_PRACTICE

        # Weight by (confidence × authority)
        layer_weights = {}
        for e in evidence:
            weight = e.confidence * e.authority
            layer_weights[e.source_layer] = layer_weights.get(e.source_layer, 0) + weight

        # Return layer with highest total weight
        return max(layer_weights, key=layer_weights.get)

    @staticmethod
    def _classify_confidence(confidence: float) -> DecisionConfidence:
        """Map numeric confidence to categorical level."""
        if confidence >= 0.90:
            return DecisionConfidence.VERY_HIGH
        elif confidence >= 0.75:
            return DecisionConfidence.HIGH
        elif confidence >= 0.60:
            return DecisionConfidence.MEDIUM
        elif confidence >= 0.45:
            return DecisionConfidence.LOW
        else:
            return DecisionConfidence.VERY_LOW

    def _generate_reasoning(
        self,
        evidence: List[Evidence],
        choice: str,
        confidence: float,
    ) -> str:
        """
        Generate human-readable reasoning for decision.

        Explains which evidence contributed and why.
        """
        reasoning_parts = [f"Decision confidence: {confidence:.2%}"]

        # Group evidence by layer
        by_layer = {}
        for e in evidence:
            by_layer.setdefault(e.source_layer, []).append(e)

        # Explain each layer's contribution
        for layer in [
            DecisionLayer.EXPLICIT_RULE,
            DecisionLayer.INFERRED_PATTERN,
            DecisionLayer.BEST_PRACTICE,
        ]:
            if layer in by_layer:
                count = len(by_layer[layer])
                avg_conf = sum(e.confidence for e in by_layer[layer]) / count
                reasoning_parts.append(
                    f"{layer.value}: {count} source(s) with avg confidence {avg_conf:.2%}"
                )

        return ". ".join(reasoning_parts)

    # =========================================================================
    # UTILITY METHODS
    # =========================================================================

    async def _get_inferred_patterns(self, project_id: UUID) -> Optional[InferredPatterns]:
        """Retrieve inferred patterns for project."""
        try:
            query = """
                SELECT id, project_id, avg_sentence_length, sentence_length_std,
                       lexical_diversity, readability_score, tone_embedding,
                       structure_patterns, confidence, sample_size, analyzed_at
                FROM inferred_patterns
                WHERE project_id = :project_id
                ORDER BY analyzed_at DESC
                LIMIT 1;
            """

            result = await self.session.execute(query, {"project_id": project_id})
            row = result.fetchone()

            if not row:
                return None

            # Load tone embedding from database
            tone_embedding_query = """
                SELECT tone_embedding FROM inferred_patterns WHERE id = :id;
            """
            emb_result = await self.session.execute(tone_embedding_query, {"id": row[0]})
            emb_row = emb_result.fetchone()

            # Parse tone embedding (stored as PostgreSQL vector)
            tone_embedding = np.array(emb_row[0]) if emb_row and emb_row[0] else np.zeros(384)

            # Parse structure patterns (stored as JSONB)
            import json

            structure_patterns = json.loads(row[7]) if row[7] else []

            from core.models import StructurePattern

            structure_objs = [StructurePattern(**p) for p in structure_patterns]

            return InferredPatterns(
                id=row[0],
                project_id=row[1],
                avg_sentence_length=row[2],
                sentence_length_std=row[3],
                lexical_diversity=row[4],
                readability_score=row[5],
                tone_embedding=tone_embedding,
                structure_patterns=structure_objs,
                confidence=row[8],
                sample_size=row[9],
                analyzed_at=row[10],
            )

        except Exception as e:
            logger.error(f"Failed to load inferred patterns: {e}")
            return None
