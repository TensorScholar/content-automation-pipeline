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


def test_generation_domain_has_no_generic_distribution_contract():
    enums = _read("core/enums.py")
    models = _read("core/models.py")
    service = _read("services/content_service.py")
    repository = _read("knowledge/article_repository.py")
    tasks = _read("orchestration/tasks.py")
    api_schemas = _read("api/schemas.py")
    api_main = _read("api/main.py")

    assert "class DistributionChannel" not in enums
    assert "ContentGenerationRequest" not in models
    assert "auto_distribute" not in models
    assert "distribution_channels" not in models
    assert "def is_distributed" not in models
    assert "def mark_distributed" not in models

    assert "generate_content_workflow" not in service
    assert "_distribute_to_wordpress" not in service
    assert "update_distribution_status" not in repository
    assert "update_article_distribution" not in repository

    assert '"distributed": len(article.distribution_channels)' not in tasks
    assert '"distributed": article.distributed_at is not None' not in tasks
    assert '"distributed_at": (' not in tasks

    assert "class ArticleResponse" not in api_schemas
    assert "ArticleResponse," not in api_main


def test_wordpress_publication_and_compatibility_mirror_remain_explicit():
    publishing_service = _read("services/publishing_service.py")
    publishing_repository = _read("knowledge/publishing_repository.py")
    schema = _read("infrastructure/schema.py")

    assert "publish_to_wordpress" in publishing_service
    assert "distribute_to_wordpress" in publishing_service
    assert 'distribution_channels=["wordpress"]' in publishing_repository
    assert 'Column("distributed_at", DateTime)' in schema
    assert 'Column("distribution_channels", JSON)' in schema
