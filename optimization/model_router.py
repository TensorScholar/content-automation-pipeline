"""
Model Router - Intelligent LLM Selection
========================================

Routes tasks to the most appropriate LLM based on capability, complexity, and cost.
"""

from enum import Enum
from typing import Optional

from pydantic import BaseModel

from config.settings import settings


class ModelCapability(str, Enum):
    REASONING = "reasoning"
    CREATIVE = "creative"
    CODING = "coding"
    ANALYSIS = "analysis"

class TaskComplexity(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"

class RoutingTask(BaseModel):
    task_id: str
    capability_required: ModelCapability
    complexity: TaskComplexity
    estimated_input_tokens: int
    estimated_output_tokens: int

class RoutingDecision(BaseModel):
    selected_model: str
    reasoning: str

class ModelRouter:
    """
    Simple router that selects models based on configuration.
    """

    async def route(self, task: RoutingTask) -> RoutingDecision:
        """
        Route the task to the configured primary or secondary model.
        """
        # For now, we just use the primary model for everything to keep it simple
        # and ensure we use the user's configured Gemini model.
        model = settings.llm.primary_model

        return RoutingDecision(
            selected_model=model,
            reasoning="Default routing to primary model"
        )
