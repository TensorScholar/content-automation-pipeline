from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from api.routes.content import ContentQualityMetrics, get_quality_metrics
from core.models import User


@pytest.mark.asyncio
async def test_quality_endpoint_preserves_structured_service_metrics():
    article_id = uuid4()
    result = {
        "article_id": str(article_id),
        "readability_score": 82.0,
        "readability_grade": "Easy",
        "keyword_density": 1.4,
        "keyword_analysis": {
            "overall_density": 1.4,
            "keywords": {"quality": {"count": 7, "density": 1.4}},
            "issues": [],
        },
        "semantic_coherence": {"score": 0.86, "details": {}},
        "structure_score": {"score": 0.84, "details": {"h2_count": 5}},
        "seo_score": {"score": 0.72, "recommendations": []},
        "overall_quality": {"score": 0.81, "grade": "Good"},
    }
    content_service = AsyncMock()
    content_service.get_quality_metrics.return_value = result
    user = User(
        id=uuid4(),
        email="quality@example.com",
        role="user",
        full_name="Quality User",
        hashed_password="",
        is_active=True,
    )

    response = await get_quality_metrics(
        article_id=article_id,
        content_service=content_service,
        user=user,
    )

    assert response == ContentQualityMetrics.model_validate(result)
    assert response.overall_quality["score"] == 0.81
    assert response.structure_score["score"] == 0.84
    assert response.keyword_density == 1.4
