import pytest
from pydantic import ValidationError

from orchestration.content_agent import ContentAgent, ContentAgentConfig, WorkflowState


def test_content_agent_configuration_is_generation_only_and_fail_fast():
    assert set(ContentAgentConfig.model_fields) == {
        "max_generation_retries",
        "enable_pattern_inference",
        "default_priority",
    }

    with pytest.raises(ValidationError):
        ContentAgentConfig(timeout_seconds=300)


def test_content_agent_does_not_own_publication_or_distribution_state():
    assert "DISTRIBUTION" not in WorkflowState.__members__
    assert not hasattr(ContentAgent, "_distribute_article")
