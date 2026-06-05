"""
API Schemas: Request/Response Models

Centralized Pydantic models for API request/response validation.
Implements Domain Transfer Objects (DTOs) pattern for clean API contracts.
"""

from datetime import datetime
from typing import List, Optional

import bleach
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CreateProjectRequest(BaseModel):
    """Command: Create new project with validation."""

    name: str = Field(..., min_length=1, max_length=255, description="Project name")
    domain: Optional[str] = Field(None, description="Target website domain (e.g., 'example.com')")
    telegram_channel: Optional[str] = Field(None, description="Telegram channel ID")
    wordpress_url: Optional[str] = Field(
        None, description="WordPress site URL (e.g., 'https://example.com')"
    )
    wordpress_username: Optional[str] = Field(None, description="WordPress username")
    wordpress_app_password: Optional[str] = Field(
        None, description="WordPress Application Password"
    )
    rulebook_content: Optional[str] = Field(None, description="Initial rulebook content")
    vertical: Optional[str] = Field(None, max_length=100, description="Business vertical/category")
    description: Optional[str] = Field(None, max_length=500, description="Project description")

    @field_validator("domain")
    @classmethod
    def validate_domain(cls, v):
        """Strip protocol and trailing slash from domain."""
        if v:
            v = v.replace("https://", "").replace("http://", "").rstrip("/")
        return v

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "name": "TechBlog AI",
                "domain": "https://techblog.ai",
                "telegram_channel": "@techblogai",
                "rulebook_content": "Write in conversational tone...",
            }
        }
    )


class ProjectResponse(BaseModel):
    """Query result: Project representation."""

    id: str
    name: str
    domain: Optional[str]
    telegram_channel: Optional[str]
    wordpress_url: Optional[str]
    wordpress_username: Optional[str]
    # DO NOT expose the password in the response
    created_at: datetime
    total_articles_generated: int
    has_rulebook: bool
    has_inferred_patterns: bool

    model_config = ConfigDict(from_attributes=True)


class GenerateContentRequest(BaseModel):
    """Command: Generate content with strategic parameters."""

    topic: str = Field(..., min_length=10, max_length=500, description="Content topic")
    priority: str = Field("high", pattern="^(low|medium|high|critical)$")
    custom_instructions: Optional[str] = Field(None, max_length=2000)
    async_execution: bool = Field(False, description="Execute as background task")
    keywords: Optional[List[str]] = Field(default=None, max_length=20, description="Target keywords")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0, description="LLM Creativity")

    @field_validator("topic", "custom_instructions")
    @classmethod
    def sanitize_text_input(cls, v: Optional[str], info) -> Optional[str]:
        """Sanitize user input to prevent XSS and injection attacks."""
        if v is None:
            return v

        # Remove HTML tags and scripts
        cleaned = bleach.clean(v, tags=[], strip=True)

        # Validate against injection patterns
        dangerous_patterns = ['<script', 'javascript:', 'onerror=', 'onclick=', 'onload=', 'eval(', 'exec(']
        cleaned_lower = cleaned.lower()
        for pattern in dangerous_patterns:
            if pattern in cleaned_lower:
                raise ValueError(f"Invalid characters detected in input: '{pattern}'")

        final_value = cleaned.strip()

        # Enforce min_length for 'topic' AFTER sanitization
        if info.field_name == "topic" and len(final_value) < 10:
            raise ValueError("Topic must be at least 10 characters long (after removing HTML/whitespace).")

        return final_value

    @field_validator("keywords")
    @classmethod
    def sanitize_keywords(cls, v: Optional[List[str]]) -> Optional[List[str]]:
        """Sanitize keyword list."""
        if v is None:
            return v

        sanitized = []
        for keyword in v:
            # Remove HTML and validate
            cleaned = bleach.clean(keyword, tags=[], strip=True)
            if cleaned and len(cleaned) <= 100:  # Max keyword length
                sanitized.append(cleaned.strip())

        return sanitized[:20]  # Max 20 keywords

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "topic": "Advanced NLP Techniques for Content Generation",
                "priority": "high",
                "custom_instructions": "Focus on practical implementations",
                "async_execution": False,
            }
        }
    )


class ArticleResponse(BaseModel):
    """Query result: Generated article metadata."""

    article_id: str
    project_id: str
    title: str
    word_count: int
    cost: float
    generation_time: float
    readability_score: float
    distributed: bool
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class TaskStatusResponse(BaseModel):
    """Query result: Async task execution status."""

    task_id: str
    state: str
    ready: bool
    successful: Optional[bool]
    failed: Optional[bool]
    result: Optional[dict]
    error: Optional[str]
    progress: Optional[dict]


class WorkflowStatusResponse(BaseModel):
    """Query result: Real-time workflow state."""

    workflow_id: str
    state: str
    project_id: str
    topic: str
    start_time: datetime
    events: list


class HealthCheckResponse(BaseModel):
    """System health status."""

    status: str
    timestamp: datetime
    version: str
    dependencies: dict


class ErrorResponse(BaseModel):
    """Standardized error response."""

    error: str
    detail: str
    timestamp: datetime
    request_id: Optional[str]
