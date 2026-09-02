from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_only_wordpress_is_exposed_as_direct_publication_surface():
    routes = _read("api/routes/content.py")

    assert '"/{article_id}/publish/wordpress"' in routes
    assert '"/{article_id}/publish/wordpress/validate"' in routes
    assert '"/{article_id}/publish/status"' in routes
    assert '"/{article_id}/distribute"' not in routes
    assert '"/{article_id}/distribution"' not in routes
    assert "DistributionStatusResponse" not in routes


def test_fake_multi_channel_success_implementations_are_absent():
    distributor = _read("execution/distributer.py")
    service = _read("services/content_service.py")
    repository = _read("knowledge/article_repository.py")

    assert "async def distribute_to_wordpress" in distributor
    assert "distribute_to_rss" not in distributor
    assert "distribute_to_social_media" not in distributor
    assert "async def distribute_article" not in service
    assert "async def get_distribution_status" not in service
    assert "async def get_distribution_status" not in repository
    assert "smtplib" not in service


def test_readme_describes_the_real_direct_publisher():
    readme = _read("README.md")

    assert "distributer.py        # WordPress publishing adapter" in readme
    assert "distributer.py        # Multi-channel publishing" not in readme
