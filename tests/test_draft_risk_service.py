from services.draft_risk_service import DraftRiskService


def test_missing_content_blocks_publish():
    result = DraftRiskService().assess({"title": "Example article", "content": ""})

    assert result["risk_level"] == "blocked"
    assert any(issue["id"] == "missing_content" for issue in result["blocking_issues"])


def test_complete_article_is_low_risk():
    content = """
    <h2>Overview</h2>
    This article explains a practical workflow for reliable content operations.
    It includes setup steps, review steps, and publishing checks for managers.
    The process keeps output consistent and avoids accidental publishing mistakes.
    <h2>Implementation</h2>
    Teams should configure rules, validate integrations, review drafts, and keep
    monitoring active before publishing content to production websites.
    <h2>FAQ</h2>
    Frequently asked questions help readers understand next steps and reduce
    repeated support questions after publication.
    """ * 8

    result = DraftRiskService().assess(
        {
            "title": "Reliable content operations for production publishing",
            "content": content,
            "meta_description": "A practical guide to reliable content operations and publishing checks.",
            "keywords": ["content operations", "publishing"],
        }
    )

    assert result["risk_level"] == "low"
    assert result["overall_score"] >= 80


def test_absolute_claims_are_review_warnings():
    content = """
    <h2>Overview</h2>
    This workflow is guaranteed to improve content operations and always works.
    <h2>FAQ</h2>
    Frequently asked questions are included.
    """ * 20

    result = DraftRiskService().assess(
        {
            "title": "Production content operations with review checkpoints",
            "content": content,
            "meta_description": "Review checkpoints for production content operations.",
            "keywords": ["content operations"],
        }
    )

    assert any(issue["id"] == "absolute_claims" for issue in result["warnings"])
