import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

import pytest

from execution.content_generator import (
    ContentGenerator,
    _completion_token_limit,
    _section_word_target,
)
from execution.content_planner import ContentPlanner
from orchestration.content_agent import ContentAgent


def _plan(*estimated_words: int, target_word_count: int = 900, language: str = "fa"):
    sections = [
        SimpleNamespace(heading=f"Section {index}", estimated_words=words)
        for index, words in enumerate(estimated_words, start=1)
    ]
    return SimpleNamespace(
        target_word_count=target_word_count,
        language=language,
        outline=SimpleNamespace(title="Article title", sections=sections),
    )


def test_section_targets_are_normalized_to_the_article_contract() -> None:
    plan = _plan(200, 300, 400)

    targets = [_section_word_target(plan, section) for section in plan.outline.sections]

    assert targets == [200, 300, 400]
    assert sum(targets) == plan.target_word_count


def test_completion_budget_scales_with_locale_and_requested_words() -> None:
    assert _completion_token_limit(200, "fa") == 856
    assert _completion_token_limit(200, "en") == 656
    assert _completion_token_limit(1, "fa") == 259


@pytest.mark.asyncio
async def test_section_writer_uses_section_target_in_prompt_and_token_limit() -> None:
    plan = _plan(300, 300, 300)
    section = plan.outline.sections[0]
    generator = object.__new__(ContentGenerator)
    generator.llm_client = SimpleNamespace(
        complete=AsyncMock(
            return_value=SimpleNamespace(
                content="متن بخش",
                usage=SimpleNamespace(total_tokens=12),
                cost=0.001,
                model="google/gemini-2.5-flash-lite",
                provider=SimpleNamespace(value="openai_compatible"),
            )
        )
    )
    generator._build_section_prompt = AsyncMock(return_value="section prompt")

    result = await generator._generate_section(
        uuid4(),
        plan,
        section,
        "model",
        generation_feedback="previous output was too long",
    )

    generator._build_section_prompt.assert_awaited_once_with(
        plan,
        section,
        300,
        "previous output was too long",
    )
    assert generator.llm_client.complete.await_args.kwargs["max_tokens"] == 1156
    assert result[3] == "openai_compatible/google/gemini-2.5-flash-lite"


@pytest.mark.asyncio
async def test_section_prompt_forbids_writing_the_full_article() -> None:
    plan = _plan(300, 300, 300)
    section = plan.outline.sections[0]
    generator = object.__new__(ContentGenerator)
    generator.rulebook = {}
    compressed = SimpleNamespace(compressed_prompt="compressed base")

    with patch(
        "execution.content_generator.prompt_compressor.compress_prompt",
        AsyncMock(return_value=compressed),
    ) as compress:
        prompt = await generator._build_section_prompt(
            plan,
            section,
            300,
            "actual words 6000; allowed maximum 1800",
        )

    assert compress.await_args.kwargs["variables"]["word_count"] == "300"
    assert "complete article target is 900 words" in prompt
    assert "This section must contain 255-345 words" in prompt
    assert "Never expand this section into a complete article" in prompt
    assert "allowed maximum 1800" in prompt


@pytest.mark.asyncio
async def test_regeneration_preserves_target_and_passes_structured_length_feedback() -> None:
    agent = object.__new__(ContentAgent)
    enhanced_plan = SimpleNamespace(target_word_count=900)
    content_plan = SimpleNamespace(
        target_word_count=900,
        model_copy=lambda **_kwargs: enhanced_plan,
    )
    generated = object()
    agent.content_generator = SimpleNamespace(
        generate_article=AsyncMock(return_value=generated)
    )

    result = await agent._regenerate_with_feedback(
        content_plan,
        SimpleNamespace(id=uuid4()),
        ["Article exceeds the allowed length."],
        release_gate={
            "word_count": 6361,
            "minimum_word_count": 800,
            "maximum_word_count": 1800,
        },
        model_override="provider/model",
    )

    assert result is generated
    assert enhanced_plan.target_word_count == 900
    call = agent.content_generator.generate_article.await_args.kwargs
    assert call["persist"] is False
    assert "Actual words: 6361" in call["generation_feedback"]
    assert "allowed range: 800-1800" in call["generation_feedback"]
    assert "requested target: 900" in call["generation_feedback"]


@pytest.mark.asyncio
async def test_planner_keeps_request_target_authoritative() -> None:
    planner = object.__new__(ContentPlanner)
    planner.context_synthesizer = SimpleNamespace(
        synthesize_context=AsyncMock(
            return_value=SimpleNamespace(
                target_audience="general",
                tone="professional",
                style_guide="standard",
                custom_instructions=None,
            )
        )
    )
    planner._load_content_memory_guidance = AsyncMock(return_value=None)
    planner.llm_client = SimpleNamespace(
        complete=AsyncMock(
            return_value=SimpleNamespace(
                content=json.dumps(
                    {
                        "title": "A valid article title",
                        "meta_description": "A sufficiently long meta description for plan validation and storage.",
                        "target_word_count": 9999,
                        "sections": [
                            {"heading": "Introduction"},
                            {"heading": "Analysis"},
                            {"heading": "Conclusion"},
                        ],
                    }
                ),
                cost=0.004,
            )
        )
    )
    planner.article_repo = SimpleNamespace(save_content_plan=AsyncMock())

    plan = await planner.create_content_plan(
        project=SimpleNamespace(id=uuid4()),
        topic="Topic",
        keywords=[],
        target_word_count=900,
        model_override="provider/model",
        content_structure="pillar",
    )

    assert plan.target_word_count == 900
    assert [section.estimated_words for section in plan.outline.sections] == [300, 300, 300]
    assert plan.estimated_cost_usd == 0.004
    assert "Style Guide:** pillar" in planner.llm_client.complete.await_args.kwargs["prompt"]
