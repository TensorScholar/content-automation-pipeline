import pytest
from pydantic import ValidationError

from api.routes.content import GenerateContentRequest
from orchestration.tasks import GenerateContentInput


def _request(**overrides):
    return GenerateContentRequest(
        project_id="00000000-0000-0000-0000-000000000001",
        topic="Valid article topic",
        **overrides,
    )


@pytest.mark.parametrize("word_count_range", ["799-1000", "800-3501"])
def test_generation_request_rejects_out_of_contract_word_count_ranges(word_count_range):
    with pytest.raises(ValidationError, match="between 800 and 3500"):
        _request(word_count_range=word_count_range)


def test_generation_request_accepts_default_product_word_count_range():
    request = _request(word_count_range="800-1000")

    assert request.word_count_range == "800-1000"


def test_generation_request_rejects_maximum_below_minimum():
    with pytest.raises(ValidationError, match="greater than or equal to"):
        _request(word_count_range="1000-800")


def test_generation_request_rejects_legacy_target_outside_product_contract():
    with pytest.raises(ValidationError, match="between 800 and 3500"):
        _request(target_word_count=799)


@pytest.mark.parametrize("word_count_range", ["799-1000", "800-3501", "1000-800"])
def test_worker_rejects_invalid_word_count_range_before_generation(word_count_range):
    with pytest.raises(ValidationError):
        GenerateContentInput(
            project_id="00000000-0000-0000-0000-000000000001",
            topic="Valid article topic",
            word_count_range=word_count_range,
        )


def test_worker_normalizes_valid_word_count_contract_values():
    task_input = GenerateContentInput(
        project_id="00000000-0000-0000-0000-000000000001",
        topic="Valid article topic",
        word_count_range=" 800 - 1000 ",
        target_word_count=900,
    )

    assert task_input.word_count_range == "800-1000"
    assert task_input.target_word_count == 900


def test_worker_rejects_legacy_target_outside_product_contract():
    with pytest.raises(ValidationError, match="between 800 and 3500"):
        GenerateContentInput(
            project_id="00000000-0000-0000-0000-000000000001",
            topic="Valid article topic",
            target_word_count=3501,
        )


@pytest.mark.parametrize("temperature", [-0.01, 2.01])
def test_worker_rejects_invalid_temperature_before_generation(temperature):
    with pytest.raises(ValidationError):
        GenerateContentInput(
            project_id="00000000-0000-0000-0000-000000000001",
            topic="Valid article topic",
            temperature=temperature,
        )


# ── Structured attribute contract tests ──────────────────────────────


@pytest.mark.parametrize(
    "field,value",
    [
        ("tone", "professional"),
        ("tone", "friendly"),
        ("tone", "persuasive"),
        ("target_audience", "general"),
        ("target_audience", "technical"),
        ("target_audience", "business"),
        ("content_structure", "standard"),
        ("content_structure", "listicle"),
        ("content_structure", "howto"),
        ("content_structure", "pillar"),
    ],
)
def test_request_accepts_machine_value_for_structured_attribute(field, value):
    """UI select options use stable machine keys (not localized labels)."""
    request = _request(**{field: value})
    assert getattr(request, field) == value


def test_request_accepts_custom_tone_string():
    """When user picks 'Custom', the raw free-text value is transmitted."""
    request = _request(tone="conversational and witty")
    assert request.tone == "conversational and witty"


def test_request_accepts_custom_target_audience_string():
    request = _request(target_audience="CTOs at B2B SaaS startups")
    assert request.target_audience == "CTOs at B2B SaaS startups"


def test_request_accepts_custom_content_structure_string():
    request = _request(content_structure="interview-format")
    assert request.content_structure == "interview-format"


def test_request_defaults_structured_attributes_to_none():
    """When not supplied, the fields default to None (backend uses project rules)."""
    request = _request()
    assert request.tone is None
    assert request.target_audience is None
    assert request.content_structure is None


def test_worker_sanitizes_structured_attributes():
    """Verify that GenerateContentInput sanitizes tone, audience, and structure."""
    task_input = GenerateContentInput(
        project_id="00000000-0000-0000-0000-000000000001",
        topic="Valid article topic",
        tone="  professional\x00 ",
        target_audience="\n general\t ",
        content_structure="  standard \n",
    )

    assert task_input.tone == "professional"
    assert task_input.target_audience == "general"
    assert task_input.content_structure == "standard"
