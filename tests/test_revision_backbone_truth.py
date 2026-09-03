from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_revision_backbone_is_explicit_in_schema_and_migration():
    schema = _read("infrastructure/schema.py")
    migration = _read("alembic/versions/20260903_revision_backbone.py")

    assert 'Column("current_revision_id"' in schema
    assert 'Column("revision_number", Integer, nullable=False)' in schema
    assert 'Column("meta_description", String(500))' in schema
    assert 'Column("keywords", JSONB)' in schema
    assert '"uq_article_revisions_article_number"' in schema

    assert 'revision = "20260903_001"' in migration
    assert 'down_revision = "20260801_001"' in migration
    assert "migration_current_backfill" in migration
    assert "assign_article_revision_identity" in migration
    assert "capture_generated_article_revision" in migration
    assert "trg_generated_articles_capture_revision" in migration
    assert "current_revision_id = captured_revision_id" in migration
    assert "legacy_partial" in migration
    assert (
        "snapshot_completeness = 'complete'" not in migration
    )  # never fabricates historical completeness


def test_application_no_longer_writes_redundant_pre_edit_snapshots():
    repository = _read("knowledge/article_repository.py")
    service = _read("services/content_service.py")

    update_method = repository.split("async def update_content_with_revision", 1)[1].split(
        "async def get_review_state", 1
    )[0]
    assert "insert(article_revisions_table)" not in update_method
    assert "DB triggers append the new immutable revision" in update_method
    assert "async def create_revision" not in repository
    assert "create_revision(" not in service


def test_history_exposes_current_revision_identity_without_duplicating_it():
    repository = _read("knowledge/article_repository.py")

    assert '"revision_id"' in repository
    assert "generated_articles_table.c.current_revision_id == current_revision.c.id" in repository
    assert 'article_revisions_table.c.id != current["current_revision_id"]' in repository
    assert "article_revisions_table.c.revision_number.desc()" in repository
