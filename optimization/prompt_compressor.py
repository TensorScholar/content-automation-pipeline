"""
Prompt Compressor - Optimization Through Architecture
======================================================

Reduces prompt token consumption through intelligent composition:

1. Template Components: Modular, reusable prompt building blocks
2. Variable Substitution: Parameterized templates reduce repetition
3. Structural Optimization: Remove redundant formatting/instructions
4. Semantic Compression: Distill to essential information
5. Few-Shot Optimization: Dynamic example selection

Target: 40-60% token reduction vs naive prompting.

Design: Functional composition of prompt components with automatic
optimization at assembly time.
"""

import re
from dataclasses import dataclass, field
from enum import Enum
from string import Template
from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from core.exceptions import ValidationError


class PromptRole(str, Enum):
    """Prompt component roles."""

    SYSTEM = "system"
    INSTRUCTION = "instruction"
    CONTEXT = "context"
    EXAMPLES = "examples"
    OUTPUT_FORMAT = "output_format"
    CONSTRAINTS = "constraints"


@dataclass
class PromptComponent:
    """
    Modular prompt component with metadata.

    Components are building blocks that can be composed into full prompts.
    """

    id: str
    role: PromptRole
    template: str
    priority: int  # 1-10, higher = more important
    required: bool = True
    estimated_tokens: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def render(self, variables: Dict[str, Any]) -> str:
        """
        Render component with variable substitution.

        Args:
            variables: Dict of template variables

        Returns:
            Rendered component text
        """
        try:
            template = Template(self.template)
            rendered = template.safe_substitute(variables)
            return rendered.strip()
        except Exception as e:
            logger.error(f"Failed to render component {self.id}: {e}")
            return self.template


@dataclass
class CompressionResult:
    """Result of prompt compression."""

    original_prompt: str
    compressed_prompt: str
    original_tokens: int
    compressed_tokens: int
    components_used: List[str]
    components_omitted: List[str]

    @property
    def tokens_saved(self) -> int:
        """Tokens saved through compression."""
        return self.original_tokens - self.compressed_tokens

    @property
    def compression_ratio(self) -> float:
        """Compression ratio (0-1)."""
        return self.compressed_tokens / self.original_tokens if self.original_tokens > 0 else 1.0

    @property
    def reduction_percent(self) -> float:
        """Percentage reduction."""
        return (1 - self.compression_ratio) * 100


class PromptTemplate:
    """
    Composable prompt template with optimization.

    Templates are assembled from components based on:
    - Available token budget
    - Component priority
    - Required vs optional components
    """

    def __init__(self, name: str):
        self.name = name
        self.components: Dict[str, PromptComponent] = {}
        self._assembly_order: List[str] = []

    def add_component(self, component: PromptComponent) -> None:
        """Add component to template."""
        self.components[component.id] = component

        # Maintain priority-sorted assembly order
        self._assembly_order.append(component.id)
        self._assembly_order.sort(key=lambda cid: self.components[cid].priority, reverse=True)

    def render(
        self,
        variables: Dict[str, Any],
        token_budget: Optional[int] = None,
    ) -> Tuple[str, List[str]]:
        """
        Render template with budget-aware component selection.

        Args:
            variables: Template variables
            token_budget: Maximum tokens (None = no limit)

        Returns:
            Tuple of (rendered_prompt, included_component_ids)
        """
        sections = []
        included = []
        omitted = []
        tokens_used = 0

        # Render components in priority order
        for component_id in self._assembly_order:
            component = self.components[component_id]

            # Render component
            rendered = component.render(variables)

            # Estimate tokens
            component_tokens = self._estimate_tokens(rendered)

            # Check if we have budget
            if token_budget and tokens_used + component_tokens > token_budget:
                if component.required:
                    logger.warning(f"Required component {component_id} exceeds budget")
                    # Include anyway, but log
                    sections.append(rendered)
                    included.append(component_id)
                    tokens_used += component_tokens
                else:
                    # Skip optional component
                    omitted.append(component_id)
                    logger.debug(f"Omitted optional component {component_id} due to budget")
                    continue
            else:
                sections.append(rendered)
                included.append(component_id)
                tokens_used += component_tokens

        # Assemble prompt
        prompt = "\n\n".join(sections)

        return prompt, included

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count."""
        return len(text) // 4


class PromptCompressor:
    """
    Intelligent prompt compression and optimization engine.

    Strategies:
    1. Component composition (modular architecture)
    2. Redundancy elimination
    3. Format optimization
    4. Few-shot example selection
    """

    def __init__(self):
        """Initialize compressor with template library."""
        self.templates: Dict[str, PromptTemplate] = {}
        self._initialize_standard_templates()

        logger.info("Prompt compressor initialized")

    # =========================================================================
    # TEMPLATE LIBRARY
    # =========================================================================

    def _initialize_standard_templates(self) -> None:
        """Initialize library of standard prompt templates."""

        # Content Generation Template
        content_template = PromptTemplate("content_generation")

        content_template.add_component(
            PromptComponent(
                id="instruction",
                role=PromptRole.INSTRUCTION,
                template="Generate a high-quality article on the topic: $topic",
                priority=10,
                required=True,
            )
        )

        content_template.add_component(
            PromptComponent(
                id="context",
                role=PromptRole.CONTEXT,
                template="Context: $context",
                priority=8,
                required=False,
            )
        )

        content_template.add_component(
            PromptComponent(
                id="tone_guidelines",
                role=PromptRole.CONSTRAINTS,
                template="Tone: $tone",
                priority=7,
                required=False,
            )
        )

        content_template.add_component(
            PromptComponent(
                id="structure_guidelines",
                role=PromptRole.CONSTRAINTS,
                template="Structure: $structure",
                priority=6,
                required=False,
            )
        )

        content_template.add_component(
            PromptComponent(
                id="output_format",
                role=PromptRole.OUTPUT_FORMAT,
                template="Output as markdown with H2/H3 headings. Target $word_count words.",
                priority=5,
                required=True,
            )
        )

        self.templates["content_generation"] = content_template

        # Keyword Research Template
        keyword_template = PromptTemplate("keyword_research")

        keyword_template.add_component(
            PromptComponent(
                id="instruction",
                role=PromptRole.INSTRUCTION,
                template="Analyze and categorize the following keywords for topic: $topic",
                priority=10,
                required=True,
            )
        )

        keyword_template.add_component(
            PromptComponent(
                id="keywords",
                role=PromptRole.CONTEXT,
                template="Keywords: $keywords",
                priority=9,
                required=True,
            )
        )

        keyword_template.add_component(
            PromptComponent(
                id="output_format",
                role=PromptRole.OUTPUT_FORMAT,
                template='Return JSON with structure: {"primary": [], "secondary": [], "long_tail": []}',
                priority=8,
                required=True,
            )
        )

        self.templates["keyword_research"] = keyword_template

        # Outline Generation Template
        outline_template = PromptTemplate("outline_generation")

        outline_template.add_component(
            PromptComponent(
                id="instruction",
                role=PromptRole.INSTRUCTION,
                template="Create detailed article outline for: $topic",
                priority=10,
                required=True,
            )
        )

        outline_template.add_component(
            PromptComponent(
                id="keywords",
                role=PromptRole.CONTEXT,
                template="Target keywords: $keywords",
                priority=8,
                required=True,
            )
        )

        outline_template.add_component(
            PromptComponent(
                id="guidelines",
                role=PromptRole.CONSTRAINTS,
                template="$guidelines",
                priority=7,
                required=False,
            )
        )

        outline_template.add_component(
            PromptComponent(
                id="output_format",
                role=PromptRole.OUTPUT_FORMAT,
                template="Format: H1 title, H2 sections, H3 subsections with brief descriptions.",
                priority=6,
                required=True,
            )
        )

        self.templates["outline_generation"] = outline_template

    # =========================================================================
    # COMPRESSION OPERATIONS
    # =========================================================================

    async def compress_prompt(
        self,
        template_name: str,
        variables: Dict[str, Any],
        token_budget: Optional[int] = None,
    ) -> CompressionResult:
        """
        Compress prompt using template-based composition.

        Args:
            template_name: Name of template to use
            variables: Template variables
            token_budget: Optional token budget

        Returns:
            CompressionResult with metrics
        """
        if template_name not in self.templates:
            raise ValidationError(f"Unknown template: {template_name}")

        template = self.templates[template_name]

        # Render with budget
        compressed_prompt, included = template.render(variables, token_budget)

        # For comparison, render without budget
        original_prompt, _ = template.render(variables, token_budget=None)

        # Calculate metrics
        original_tokens = self._estimate_tokens(original_prompt)
        compressed_tokens = self._estimate_tokens(compressed_prompt)

        omitted = [cid for cid in template.components.keys() if cid not in included]

        result = CompressionResult(
            original_prompt=original_prompt,
            compressed_prompt=compressed_prompt,
            original_tokens=original_tokens,
            compressed_tokens=compressed_tokens,
            components_used=included,
            components_omitted=omitted,
        )

        logger.info(
            f"Compressed prompt: {result.reduction_percent:.1f}% reduction "
            f"({result.tokens_saved} tokens saved)"
        )

        return result

    def optimize_formatting(self, prompt: str) -> str:
        """
        Optimize prompt formatting to reduce tokens.

        Techniques:
        - Remove excess whitespace
        - Consolidate instructions
        - Remove redundant markers

        Args:
            prompt: Input prompt

        Returns:
            Optimized prompt
        """
        # Remove multiple newlines
        optimized = re.sub(r"\n{3,}", "\n\n", prompt)

        # Remove excess spaces
        optimized = re.sub(r" {2,}", " ", optimized)

        # Remove trailing whitespace
        lines = [line.rstrip() for line in optimized.split("\n")]
        optimized = "\n".join(lines)

        # Remove redundant section markers
        optimized = re.sub(r"^\s*[-=*]{3,}\s*$", "", optimized, flags=re.MULTILINE)

        return optimized.strip()

    def select_few_shot_examples(
        self,
        examples: List[Dict[str, str]],
        query: str,
        max_examples: int = 2,
    ) -> List[Dict[str, str]]:
        """
        Select most relevant few-shot examples.

        Uses simple keyword matching (can be enhanced with embeddings).

        Args:
            examples: List of example dicts
            query: Query to match against
            max_examples: Maximum examples to select

        Returns:
            Selected examples
        """
        if len(examples) <= max_examples:
            return examples

        # Score examples by keyword overlap
        query_words = set(query.lower().split())

        scored = []
        for example in examples:
            # Combine all text in example
            example_text = " ".join(str(v) for v in example.values()).lower()
            example_words = set(example_text.split())

            # Jaccard similarity
            overlap = len(query_words & example_words)
            union = len(query_words | example_words)
            score = overlap / union if union > 0 else 0

            scored.append((example, score))

        # Sort by score and take top K
        scored.sort(key=lambda x: x[1], reverse=True)
        selected = [ex for ex, score in scored[:max_examples]]

        logger.debug(f"Selected {len(selected)}/{len(examples)} most relevant examples")
        return selected

    # =========================================================================
    # UTILITIES
    # =========================================================================

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """Estimate token count for text."""
        return len(text) // 4

    def get_template(self, name: str) -> Optional[PromptTemplate]:
        """Get template by name."""
        return self.templates.get(name)

    def register_template(self, template: PromptTemplate) -> None:
        """Register custom template."""
        self.templates[template.name] = template
        logger.info(f"Registered template: {template.name}")

    def list_templates(self) -> List[str]:
        """List available template names."""
        return list(self.templates.keys())


# =========================================================================
# GLOBAL INSTANCE
# =========================================================================

prompt_compressor = PromptCompressor()
