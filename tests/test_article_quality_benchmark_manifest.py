import json
from collections import Counter
from pathlib import Path

BENCHMARK_PATH = Path(__file__).parent / "fixtures" / "article_quality_benchmark_v1.json"


def test_phase_two_benchmark_manifest_is_complete_and_well_formed() -> None:
    benchmark = json.loads(BENCHMARK_PATH.read_text(encoding="utf-8"))
    cases = benchmark["cases"]
    rubric = benchmark["rubric"]

    assert len(cases) == benchmark["case_count"] == 40
    assert Counter(case["language"] for case in cases) == benchmark["language_distribution"]
    assert len({case["id"] for case in cases}) == len(cases)

    for case in cases:
        assert case["intent"] in {"informational", "tutorial", "commercial", "comparison"}
        assert case["risk"] in {"low", "high"}
        assert case["source_mode"] in {"none", "provided"}
        assert 800 <= case["target_words"] <= 3500
        assert len(case["required_sections"]) >= 3
        assert case["required_keywords"]

    dimensions = rubric["dimensions"]
    assert rubric["version"] == "article-quality-rubric-v1"
    assert rubric["score_scale"] == {"minimum": 0, "maximum": 100}
    assert sum(dimension["weight"] for dimension in dimensions) == 100
    assert len({dimension["id"] for dimension in dimensions}) == len(dimensions) == 9
    assert len(rubric["hard_blockers"]) == len(set(rubric["hard_blockers"]))
    assert rubric["hard_blockers"]

    human_review = rubric["human_review"]
    assert set(human_review["required_fields"]) == {
        "case_id",
        "reviewer_id",
        "review_round",
        "dimension_scores",
        "hard_blockers",
        "accepted",
        "notes",
    }
    assert human_review["allowed_decisions"] == [
        "accept",
        "reject",
        "needs_adjudication",
    ]
