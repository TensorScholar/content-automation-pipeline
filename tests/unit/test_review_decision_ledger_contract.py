from pathlib import Path

from infrastructure.schema import article_review_decisions_table, generated_articles_table

ROOT = Path(__file__).resolve().parents[2]
MIGRATION = ROOT / "alembic" / "versions" / "20260904_review_decision_ledger.py"
GRAPH_VALIDATOR = ROOT / "scripts" / "maintenance" / "validate_migration_graph.py"


def test_review_decision_table_matches_phase4b1_contract() -> None:
    columns = article_review_decisions_table.c

    assert columns.article_id.foreign_keys
    assert next(iter(columns.article_id.foreign_keys)).target_fullname == "generated_articles.id"
    assert next(iter(columns.article_id.foreign_keys)).ondelete == "CASCADE"

    assert columns.article_revision_id.foreign_keys
    assert next(iter(columns.article_revision_id.foreign_keys)).target_fullname == "article_revisions.id"
    assert next(iter(columns.article_revision_id.foreign_keys)).ondelete == "CASCADE"

    # Reviewer identity is immutable historical evidence, not a live FK whose
    # ON DELETE behavior could mutate an append-only decision.
    assert not columns.reviewer_id_snapshot.foreign_keys

    check_names = {
        constraint.name
        for constraint in article_review_decisions_table.constraints
        if constraint.name
    }
    assert "ck_article_review_decisions_decision" in check_names
    assert "ck_article_review_decisions_number_positive" in check_names

    index_names = {index.name for index in article_review_decisions_table.indexes}
    assert "uq_article_review_decisions_article_number" in index_names
    assert "idx_article_review_decisions_revision" in index_names
    assert "idx_article_review_decisions_article_recorded" in index_names


def test_article_projection_points_to_review_ledger_and_has_bounded_status() -> None:
    pointer = generated_articles_table.c.current_review_decision_id
    assert pointer.foreign_keys
    foreign_key = next(iter(pointer.foreign_keys))
    assert foreign_key.target_fullname == "article_review_decisions.id"
    assert foreign_key.ondelete == "SET NULL"

    check_names = {
        constraint.name
        for constraint in generated_articles_table.constraints
        if constraint.name
    }
    assert "ck_generated_articles_review_status" in check_names


def test_migration_is_revision_bound_fail_closed_and_compatibility_aware() -> None:
    migration = MIGRATION.read_text(encoding="utf-8")
    graph_validator = GRAPH_VALIDATOR.read_text(encoding="utf-8")

    assert 'revision = "20260904_001"' in migration
    assert 'down_revision = "20260903_001"' in migration
    assert 'EXPECTED_HEAD = "20260904_001"' in graph_validator

    assert "article_review_decisions" in migration
    assert "article_revision_id" in migration
    assert "current_review_decision_id" in migration
    assert "reviewer_id_snapshot" in migration
    assert "app.expected_review_revision_id" in migration
    assert "app.review_decision_source" in migration
    assert "Stale review revision" in migration
    assert "prevent_article_review_decision_update" in migration
    assert "trg_generated_articles_invalidate_review_on_revision" in migration
    assert "trg_generated_articles_review_projection_consistency" in migration
    assert "DEFERRABLE INITIALLY DEFERRED" in migration

    # User deletion only nulls the legacy live reviewed_by column. That FK
    # maintenance must not be recorded as a new editorial decision event.
    decision_change_guard = migration.split(
        "CREATE OR REPLACE FUNCTION capture_article_review_decision_from_projection()",
        1,
    )[1].split("IF NEW.review_status = 'pending_review'", 1)[0]
    assert "OLD.review_status IS DISTINCT FROM NEW.review_status" in decision_change_guard
    assert "OLD.review_note IS DISTINCT FROM NEW.review_note" in decision_change_guard
    assert "OLD.reviewed_at IS DISTINCT FROM NEW.reviewed_at" in decision_change_guard
    assert "OLD.reviewed_by IS DISTINCT FROM NEW.reviewed_by" not in decision_change_guard
