from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent


def replace_once(path: str, old: str, new: str) -> None:
    target = ROOT / path
    content = target.read_text(encoding="utf-8")
    count = content.count(old)
    if count != 1:
        raise RuntimeError(f"{path}: expected one match, found {count}")
    target.write_text(content.replace(old, new, 1), encoding="utf-8")


# Preserve manual-edit provenance without mutating an immutable revision after insert.
replace_once(
    "knowledge/article_repository.py",
    '''        """Atomically apply a manual edit; DB triggers append the new immutable revision."""
        del revision_note  # Revision identity is payload-derived; edit events are tracked separately.
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            current_result = await session.execute(
''',
    '''        """Atomically apply a manual edit; DB triggers append the new immutable revision."""
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        async with self.db.transaction() as session:
            current_result = await session.execute(
''',
)
replace_once(
    "knowledge/article_repository.py",
    '''            if current is None:
                return None

            updated_result = await session.execute(
''',
    '''            if current is None:
                return None

            # Transaction-local context lets the DB trigger capture human edit
            # provenance at revision construction time. The setting disappears
            # automatically at transaction end, so pooled connections cannot leak it.
            await session.execute(
                select(func.set_config("app.revision_note", revision_note or "", True))
            )

            updated_result = await session.execute(
''',
)

# Capture the transaction-local note on the immutable revision itself.
replace_once(
    "alembic/versions/20260903_revision_backbone.py",
    '''                captured_revision_id uuid;
                captured_source text;
                captured_at timestamp;
''',
    '''                captured_revision_id uuid;
                captured_source text;
                captured_note text;
                captured_at timestamp;
''',
)
replace_once(
    "alembic/versions/20260903_revision_backbone.py",
    '''                ELSE
                    captured_source := 'article_payload_update';
                    captured_at := COALESCE(NEW.updated_at, NOW());
                END IF;
''',
    '''                ELSE
                    captured_source := 'article_payload_update';
                    captured_note := NULLIF(current_setting('app.revision_note', true), '');
                    captured_at := COALESCE(NEW.updated_at, NOW());
                END IF;
''',
)
replace_once(
    "alembic/versions/20260903_revision_backbone.py",
    '''                    NEW.meta_description,
                    NEW.keywords,
                    NULL,
                    NEW.word_count,
''',
    '''                    NEW.meta_description,
                    NEW.keywords,
                    captured_note,
                    NEW.word_count,
''',
)

# Add ownership and immutability guards after the capture trigger is installed.
needle = '''    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_generated_articles_capture_revision
            AFTER INSERT OR UPDATE OF title, content, meta_description, keywords, word_count
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION capture_generated_article_revision()
            """
        )
    )


def downgrade() -> None:
'''
replacement = '''    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_generated_articles_capture_revision
            AFTER INSERT OR UPDATE OF title, content, meta_description, keywords, word_count
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION capture_generated_article_revision()
            """
        )
    )

    # The simple FK proves that current_revision_id exists. This trigger also
    # proves ownership: the pointed revision must belong to the same article.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION validate_generated_article_current_revision()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                IF NEW.current_revision_id IS NULL THEN
                    RETURN NEW;
                END IF;

                PERFORM 1
                FROM article_revisions
                WHERE id = NEW.current_revision_id
                  AND article_id = NEW.id;

                IF NOT FOUND THEN
                    RAISE EXCEPTION
                        'Current revision % does not belong to article %',
                        NEW.current_revision_id,
                        NEW.id
                        USING ERRCODE = '23514';
                END IF;
                RETURN NEW;
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_generated_articles_validate_current_revision
            BEFORE INSERT OR UPDATE OF current_revision_id
            ON generated_articles
            FOR EACH ROW
            EXECUTE FUNCTION validate_generated_article_current_revision()
            """
        )
    )

    # Revisions are append-only. Deletion remains available only because an
    # article delete must cascade its historical revisions safely.
    op.execute(
        sa.text(
            """
            CREATE OR REPLACE FUNCTION prevent_article_revision_update()
            RETURNS trigger
            LANGUAGE plpgsql
            AS $$
            BEGIN
                RAISE EXCEPTION 'Article revisions are immutable once inserted'
                    USING ERRCODE = '55000';
            END;
            $$
            """
        )
    )
    op.execute(
        sa.text(
            """
            CREATE TRIGGER trg_article_revisions_prevent_update
            BEFORE UPDATE ON article_revisions
            FOR EACH ROW
            EXECUTE FUNCTION prevent_article_revision_update()
            """
        )
    )


def downgrade() -> None:
'''
replace_once("alembic/versions/20260903_revision_backbone.py", needle, replacement)
replace_once(
    "alembic/versions/20260903_revision_backbone.py",
    '''def downgrade() -> None:
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_capture_revision ON generated_articles"
    )
''',
    '''def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS trg_article_revisions_prevent_update ON article_revisions")
    op.execute("DROP FUNCTION IF EXISTS prevent_article_revision_update()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_validate_current_revision ON generated_articles"
    )
    op.execute("DROP FUNCTION IF EXISTS validate_generated_article_current_revision()")
    op.execute(
        "DROP TRIGGER IF EXISTS trg_generated_articles_capture_revision ON generated_articles"
    )
''',
)

# Strengthen the static contract.
replace_once(
    "tests/test_revision_backbone_truth.py",
    '''    assert "capture_generated_article_revision" in migration
    assert "trg_generated_articles_capture_revision" in migration
    assert "current_revision_id = captured_revision_id" in migration
''',
    '''    assert "capture_generated_article_revision" in migration
    assert "trg_generated_articles_capture_revision" in migration
    assert "current_setting('app.revision_note', true)" in migration
    assert "validate_generated_article_current_revision" in migration
    assert "prevent_article_revision_update" in migration
    assert "current_revision_id = captured_revision_id" in migration
''',
)
replace_once(
    "tests/test_revision_backbone_truth.py",
    '''    assert "insert(article_revisions_table)" not in update_method
    assert "DB triggers append the new immutable revision" in update_method
    assert "async def create_revision" not in repository
''',
    '''    assert "insert(article_revisions_table)" not in update_method
    assert "DB triggers append the new immutable revision" in update_method
    assert 'func.set_config("app.revision_note", revision_note or "", True)' in update_method
    assert "async def create_revision" not in repository
''',
)

# Exercise provenance, ownership, and append-only behavior in the real PostgreSQL migration test.
replace_once(
    "tests/integration/test_revision_backbone_migration_postgres.py",
    '''from sqlalchemy import create_engine, text
''',
    '''from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError
''',
)
replace_once(
    "tests/integration/test_revision_backbone_migration_postgres.py",
    '''        with target_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET content = 'New binary payload', word_count = 3, updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": article_id},
            )
''',
    '''        with target_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.revision_note', :note, true)"),
                {"note": "Manual edit provenance note"},
            )
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET content = 'New binary payload', word_count = 3, updated_at = NOW()
                    WHERE id = :id
                    """
                ),
                {"id": article_id},
            )
''',
)
replace_once(
    "tests/integration/test_revision_backbone_migration_postgres.py",
    '''                    SELECT article.current_revision_id, revision.revision_number,
                           revision.content, revision.revision_source,
                           revision.snapshot_completeness
''',
    '''                    SELECT article.current_revision_id, revision.revision_number,
                           revision.content, revision.revision_note, revision.revision_source,
                           revision.snapshot_completeness
''',
)
replace_once(
    "tests/integration/test_revision_backbone_migration_postgres.py",
    '''        assert current["content"] == "New binary payload"
        assert current["revision_source"] == "article_payload_update"
''',
    '''        assert current["content"] == "New binary payload"
        assert current["revision_note"] == "Manual edit provenance note"
        assert current["revision_source"] == "article_payload_update"
''',
)

ownership_anchor = '''        assert second == (1, "article_initial", "complete")

        # Concurrent payload updates serialize on the article row and must never
'''
ownership_replacement = '''        assert second == (1, "article_initial", "complete")

        # A current pointer may never cross article ownership boundaries.
        with target_engine.connect() as connection:
            foreign_revision_id = connection.execute(
                text(
                    "SELECT current_revision_id FROM generated_articles WHERE id = :id"
                ),
                {"id": second_article_id},
            ).scalar_one()
        with pytest.raises(DBAPIError):
            with target_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE generated_articles
                        SET current_revision_id = :revision_id
                        WHERE id = :article_id
                        """
                    ),
                    {"revision_id": foreign_revision_id, "article_id": article_id},
                )

        # Immutable means append-only: a persisted revision cannot be rewritten.
        with pytest.raises(DBAPIError):
            with target_engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE article_revisions SET content = 'tampered' WHERE id = :id"
                    ),
                    {"id": current_revision_id},
                )

        # Concurrent payload updates serialize on the article row and must never
'''
replace_once(
    "tests/integration/test_revision_backbone_migration_postgres.py",
    ownership_anchor,
    ownership_replacement,
)

print("Phase 4A integrity hardening staged")
