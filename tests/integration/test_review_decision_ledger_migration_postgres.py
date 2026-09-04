"""PostgreSQL evidence for the Phase 4B1 review-decision ledger.

The test upgrades a disposable database from the Phase 4A head, seeds a legacy
article-level review, then exercises immutable revision-bound review decisions,
stale-review rejection, revision invalidation, ownership guards, reviewer
snapshots, and cascade deletion using the real Alembic migration.
"""

from __future__ import annotations

import os
import subprocess
from datetime import datetime
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import DBAPIError

BASE_DATABASE_URL = os.getenv("TEST_DATABASE_URL") or (
    os.getenv("DATABASE_URL") if os.getenv("CI") == "true" else None
)
pytestmark = pytest.mark.integration


def _sync_url(url: str) -> str:
    return url.replace("postgresql+asyncpg://", "postgresql+psycopg2://", 1)


def _database_url(base_url: str, database_name: str) -> str:
    parsed = urlsplit(_sync_url(base_url))
    return urlunsplit(parsed._replace(path=f"/{database_name}"))


def _run_alembic(database_url: str, target: str) -> None:
    env = os.environ.copy()
    env["DATABASE_URL"] = database_url
    result = subprocess.run(
        ["alembic", "upgrade", target],
        cwd=os.getcwd(),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, f"{result.stdout}\n{result.stderr}"


@pytest.mark.skipif(not BASE_DATABASE_URL, reason="requires CI DATABASE_URL or TEST_DATABASE_URL")
def test_review_decisions_are_revision_bound_immutable_and_stale_safe():
    database_name = f"review_decision_ledger_{uuid4().hex}"
    admin_engine = create_engine(_sync_url(BASE_DATABASE_URL), isolation_level="AUTOCOMMIT")
    database_url = _database_url(BASE_DATABASE_URL, database_name)
    target_engine = None

    try:
        with admin_engine.connect() as connection:
            connection.execute(text(f'CREATE DATABASE "{database_name}"'))

        _run_alembic(database_url, "20260903_001")
        target_engine = create_engine(database_url)

        project_id = uuid4()
        reviewer_id = uuid4()
        article_id = uuid4()
        created_at = datetime(2026, 9, 4, 8, 0, 0)

        with target_engine.begin() as connection:
            connection.execute(
                text("INSERT INTO projects (id, name) VALUES (:id, 'Review Ledger Test')"),
                {"id": project_id},
            )
            connection.execute(
                text(
                    """
                    INSERT INTO users (id, email, hashed_password, full_name)
                    VALUES (:id, :email, 'test-password-hash', :name)
                    """
                ),
                {
                    "id": reviewer_id,
                    "email": "reviewer@example.test",
                    "name": "Review Manager",
                },
            )
            connection.execute(
                text(
                    """
                    INSERT INTO generated_articles (
                        id, project_id, title, content, meta_description,
                        keywords, word_count, created_at, updated_at
                    ) VALUES (
                        :id, :project_id, 'Review-bound article', :content,
                        'Review ledger metadata', CAST(:keywords AS jsonb),
                        20, :created_at, :created_at
                    )
                    """
                ),
                {
                    "id": article_id,
                    "project_id": project_id,
                    "content": "A sufficiently complete article payload for review ledger testing. " * 8,
                    "keywords": '["review", "integrity"]',
                    "created_at": created_at,
                },
            )

            phase4a_revision_id = connection.execute(
                text("SELECT current_revision_id FROM generated_articles WHERE id = :id"),
                {"id": article_id},
            ).scalar_one()
            assert phase4a_revision_id is not None

            # This is the pre-Phase-4B mutable review representation that must
            # be preserved as an immutable decision during migration.
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET review_status = 'approved',
                        review_note = 'Legacy approval before ledger migration',
                        reviewed_by = :reviewer_id,
                        reviewed_at = :reviewed_at,
                        review_updated_at = :reviewed_at
                    WHERE id = :article_id
                    """
                ),
                {
                    "article_id": article_id,
                    "reviewer_id": reviewer_id,
                    "reviewed_at": created_at,
                },
            )

        _run_alembic(database_url, "head")

        with target_engine.connect() as connection:
            article = (
                connection.execute(
                    text(
                        """
                        SELECT current_revision_id, current_review_decision_id,
                               review_status, review_note, reviewed_by
                        FROM generated_articles
                        WHERE id = :id
                        """
                    ),
                    {"id": article_id},
                )
                .mappings()
                .one()
            )
            decisions = (
                connection.execute(
                    text(
                        """
                        SELECT id, article_revision_id, decision_number, decision,
                               note, reviewer_id, reviewer_name_snapshot,
                               reviewer_email_snapshot, decision_source
                        FROM article_review_decisions
                        WHERE article_id = :article_id
                        ORDER BY decision_number
                        """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .all()
            )

        assert article["current_revision_id"] == phase4a_revision_id
        assert article["review_status"] == "approved"
        assert len(decisions) == 1
        legacy_decision = decisions[0]
        assert article["current_review_decision_id"] == legacy_decision["id"]
        assert legacy_decision["article_revision_id"] == phase4a_revision_id
        assert legacy_decision["decision_number"] == 1
        assert legacy_decision["decision"] == "approved"
        assert legacy_decision["note"] == "Legacy approval before ledger migration"
        assert legacy_decision["reviewer_id"] == reviewer_id
        assert legacy_decision["reviewer_name_snapshot"] == "Review Manager"
        assert legacy_decision["reviewer_email_snapshot"] == "reviewer@example.test"
        assert legacy_decision["decision_source"] == "legacy_review_backfill"

        # Native Phase-4B caller: bind the review write to the exact revision
        # that was evaluated and label the source for auditability.
        with target_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.expected_review_revision_id', :revision, true)"),
                {"revision": str(phase4a_revision_id)},
            )
            connection.execute(
                text("SELECT set_config('app.review_decision_source', 'manager_api', true)")
            )
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET review_status = 'changes_requested',
                        review_note = 'Strengthen the supporting evidence.',
                        reviewed_by = :reviewer_id,
                        reviewed_at = NOW(),
                        review_updated_at = NOW()
                    WHERE id = :article_id
                    """
                ),
                {"article_id": article_id, "reviewer_id": reviewer_id},
            )

        with target_engine.connect() as connection:
            native_decision = (
                connection.execute(
                    text(
                        """
                        SELECT id, article_revision_id, decision_number, decision,
                               decision_source
                        FROM article_review_decisions
                        WHERE article_id = :article_id
                        ORDER BY decision_number DESC
                        LIMIT 1
                        """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .one()
            )
            current_pointer = connection.execute(
                text("SELECT current_review_decision_id FROM generated_articles WHERE id = :id"),
                {"id": article_id},
            ).scalar_one()

        assert native_decision["decision_number"] == 2
        assert native_decision["article_revision_id"] == phase4a_revision_id
        assert native_decision["decision"] == "changes_requested"
        assert native_decision["decision_source"] == "manager_api"
        assert current_pointer == native_decision["id"]

        # Editing the payload creates a new immutable revision through Phase 4A
        # and must atomically invalidate the decision projection from the older
        # revision rather than silently inheriting it.
        with target_engine.begin() as connection:
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET content = content || ' New revision content.',
                        word_count = word_count + 3,
                        updated_at = NOW()
                    WHERE id = :article_id
                    """
                ),
                {"article_id": article_id},
            )

        with target_engine.connect() as connection:
            after_edit = (
                connection.execute(
                    text(
                        """
                        SELECT current_revision_id, current_review_decision_id,
                               review_status, review_note, reviewed_by, reviewed_at
                        FROM generated_articles
                        WHERE id = :article_id
                        """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .one()
            )

        new_revision_id = after_edit["current_revision_id"]
        assert new_revision_id != phase4a_revision_id
        assert after_edit["current_review_decision_id"] is None
        assert after_edit["review_status"] == "pending_review"
        assert after_edit["review_note"] is None
        assert after_edit["reviewed_by"] is None
        assert after_edit["reviewed_at"] is None

        # A manager decision evaluated against the superseded revision is a
        # concurrency conflict. The DB rejects it even if application code were
        # to miss the race after its initial readiness check.
        with pytest.raises(DBAPIError):
            with target_engine.begin() as connection:
                connection.execute(
                    text("SELECT set_config('app.expected_review_revision_id', :revision, true)"),
                    {"revision": str(phase4a_revision_id)},
                )
                connection.execute(
                    text(
                        """
                        UPDATE generated_articles
                        SET review_status = 'approved',
                            review_note = NULL,
                            reviewed_by = :reviewer_id,
                            reviewed_at = NOW(),
                            review_updated_at = NOW()
                        WHERE id = :article_id
                        """
                    ),
                    {"article_id": article_id, "reviewer_id": reviewer_id},
                )

        with target_engine.connect() as connection:
            stale_state = (
                connection.execute(
                    text(
                        """
                        SELECT review_status, current_review_decision_id
                        FROM generated_articles
                        WHERE id = :article_id
                        """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .one()
            )
            decision_count = connection.execute(
                text(
                    "SELECT COUNT(*) FROM article_review_decisions WHERE article_id = :article_id"
                ),
                {"article_id": article_id},
            ).scalar_one()

        assert stale_state["review_status"] == "pending_review"
        assert stale_state["current_review_decision_id"] is None
        assert decision_count == 2

        # The same operation succeeds when the exact current revision identity
        # is supplied.
        with target_engine.begin() as connection:
            connection.execute(
                text("SELECT set_config('app.expected_review_revision_id', :revision, true)"),
                {"revision": str(new_revision_id)},
            )
            connection.execute(
                text("SELECT set_config('app.review_decision_source', 'manager_api', true)")
            )
            connection.execute(
                text(
                    """
                    UPDATE generated_articles
                    SET review_status = 'approved',
                        review_note = NULL,
                        reviewed_by = :reviewer_id,
                        reviewed_at = NOW(),
                        review_updated_at = NOW()
                    WHERE id = :article_id
                    """
                ),
                {"article_id": article_id, "reviewer_id": reviewer_id},
            )

        with target_engine.connect() as connection:
            approved = (
                connection.execute(
                    text(
                        """
                        SELECT article.current_revision_id,
                               article.current_review_decision_id,
                               article.review_status,
                               decision.article_revision_id,
                               decision.decision_number,
                               decision.decision
                        FROM generated_articles AS article
                        JOIN article_review_decisions AS decision
                          ON decision.id = article.current_review_decision_id
                        WHERE article.id = :article_id
                        """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .one()
            )

        approved_decision_id = approved["current_review_decision_id"]
        assert approved["review_status"] == "approved"
        assert approved["article_revision_id"] == approved["current_revision_id"]
        assert approved["article_revision_id"] == new_revision_id
        assert approved["decision_number"] == 3
        assert approved["decision"] == "approved"

        # Directly switching the current revision is also review-invalidating;
        # approval may never float across immutable revision identities.
        with target_engine.begin() as connection:
            connection.execute(
                text(
                    "UPDATE generated_articles SET current_revision_id = :revision WHERE id = :article_id"
                ),
                {"article_id": article_id, "revision": phase4a_revision_id},
            )

        with target_engine.connect() as connection:
            rewound = (
                connection.execute(
                    text(
                        """
                        SELECT current_revision_id, current_review_decision_id, review_status
                        FROM generated_articles
                        WHERE id = :article_id
                        """
                    ),
                    {"article_id": article_id},
                )
                .mappings()
                .one()
            )

        assert rewound["current_revision_id"] == phase4a_revision_id
        assert rewound["current_review_decision_id"] is None
        assert rewound["review_status"] == "pending_review"

        # A decision from another revision cannot be installed as the current
        # approval pointer even when it belongs to the same article.
        with pytest.raises(DBAPIError):
            with target_engine.begin() as connection:
                connection.execute(
                    text(
                        """
                        UPDATE generated_articles
                        SET current_review_decision_id = :decision_id
                        WHERE id = :article_id
                        """
                    ),
                    {"article_id": article_id, "decision_id": approved_decision_id},
                )

        # Review decisions are append-only audit events.
        with pytest.raises(DBAPIError):
            with target_engine.begin() as connection:
                connection.execute(
                    text(
                        "UPDATE article_review_decisions SET note = 'tampered' WHERE id = :id"
                    ),
                    {"id": approved_decision_id},
                )

        # User deletion may null the live FK but must not erase historical
        # reviewer identity captured at decision time.
        with target_engine.begin() as connection:
            connection.execute(text("DELETE FROM users WHERE id = :id"), {"id": reviewer_id})

        with target_engine.connect() as connection:
            reviewer_history = (
                connection.execute(
                    text(
                        """
                        SELECT reviewer_id, reviewer_name_snapshot, reviewer_email_snapshot
                        FROM article_review_decisions
                        WHERE id = :id
                        """
                    ),
                    {"id": approved_decision_id},
                )
                .mappings()
                .one()
            )

        assert reviewer_history["reviewer_id"] is None
        assert reviewer_history["reviewer_name_snapshot"] == "Review Manager"
        assert reviewer_history["reviewer_email_snapshot"] == "reviewer@example.test"

        # Article deletion still removes its immutable review ledger through
        # the parent cascade; the current decision FK must not create a cycle
        # that prevents normal aggregate deletion.
        with target_engine.begin() as connection:
            connection.execute(
                text("DELETE FROM generated_articles WHERE id = :article_id"),
                {"article_id": article_id},
            )

        with target_engine.connect() as connection:
            remaining = connection.execute(
                text(
                    "SELECT COUNT(*) FROM article_review_decisions WHERE article_id = :article_id"
                ),
                {"article_id": article_id},
            ).scalar_one()
        assert remaining == 0
    finally:
        if target_engine is not None:
            target_engine.dispose()
        with admin_engine.connect() as connection:
            connection.execute(
                text(
                    "SELECT pg_terminate_backend(pid) FROM pg_stat_activity "
                    "WHERE datname = :database_name AND pid <> pg_backend_pid()"
                ),
                {"database_name": database_name},
            )
            connection.execute(text(f'DROP DATABASE IF EXISTS "{database_name}"'))
        admin_engine.dispose()
