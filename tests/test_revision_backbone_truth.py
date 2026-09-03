from pathlib import Path

from sqlalchemy import CheckConstraint

from infrastructure.schema import article_revisions_table, generated_articles_table

ROOT = Path(__file__).resolve().parents[1]


def _read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_revision_backbone_is_explicit_in_schema_and_migration():
    migration = _read("alembic/versions/20260903_revision_backbone.py")

    current_revision = generated_articles_table.c.current_revision_id
    assert current_revision.nullable is True
    assert {foreign_key.target_fullname for foreign_key in current_revision.foreign_keys} == {
        "article_revisions.id"
    }

    assert article_revisions_table.c.revision_number.nullable is False
    assert "meta_description" in article_revisions_table.c
    assert "keywords" in article_revisions_table.c
    assert "revision_source" in article_revisions_table.c
    assert "snapshot_completeness" in article_revisions_table.c
    assert "generation_task_id" in article_revisions_table.c

    index_names = {index.name for index in article_revisions_table.indexes}
    assert "uq_article_revisions_article_number" in index_names

    check_names = {
        constraint.name
        for constraint in article_revisions_table.constraints
        if isinstance(constraint, CheckConstraint)
    }
    assert "ck_article_revisions_revision_number_positive" in check_names
    assert "ck_article_revisions_snapshot_completeness" in check_names

    assert 'revision = "20260903_001"' in migration
    assert 'down_revision = "20260801_001"' in migration
    assert "revision_source = 'legacy_snapshot'" in migration
    assert "snapshot_completeness = 'legacy_partial'" in migration
    assert "migration_current_backfill" in migration
    assert "assign_article_revision_identity" in migration
    assert "capture_generated_article_revision" in migration
    assert "trg_generated_articles_capture_revision" in migration
    assert "current_setting('app.revision_note', true)" in migration
    assert "validate_generated_article_current_revision" in migration
    assert "prevent_article_revision_update" in migration
    assert "current_revision_id = captured_revision_id" in migration


def test_application_no_longer_writes_redundant_pre_edit_snapshots():
    repository = _read("knowledge/article_repository.py")
    service = _read("services/content_service.py")

    update_method = repository.split("async def update_content_with_revision", 1)[1].split(
        "async def get_review_state", 1
    )[0]
    assert "insert(article_revisions_table)" not in update_method
    assert "DB triggers append the new immutable revision" in update_method
    assert 'func.set_config("app.revision_note", revision_note or "", True)' in update_method
    assert "async def create_revision" not in repository
    assert "create_revision(" not in service


def test_history_exposes_current_revision_identity_without_duplicating_it():
    repository = _read("knowledge/article_repository.py")

    assert '"revision_id"' in repository
    assert "generated_articles_table.c.current_revision_id == current_revision.c.id" in repository
    assert 'article_revisions_table.c.id != current["current_revision_id"]' in repository
    assert "article_revisions_table.c.revision_number.desc()" in repository
