"""add project relational integrity and precise cost accounting

Revision ID: 20260607_001
Revises: 20250225_001
Create Date: 2026-06-07
"""

import sqlalchemy as sa

from alembic import op

revision = "20260607_001"
down_revision = "20250225_001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Remove inaccessible legacy rows before enforcing the relationships.
    op.execute(
        """
        DELETE FROM rules
        WHERE NOT EXISTS (
            SELECT 1 FROM rulebooks WHERE rulebooks.id = rules.rulebook_id
        )
        OR rulebook_id IN (
            SELECT rulebooks.id
            FROM rulebooks
            LEFT JOIN projects ON projects.id = rulebooks.project_id
            WHERE projects.id IS NULL
        )
        """
    )
    op.execute(
        """
        DELETE FROM article_revisions
        WHERE NOT EXISTS (
            SELECT 1
            FROM generated_articles
            WHERE generated_articles.id = article_revisions.article_id
        )
        """
    )
    for table_name in (
        "generated_articles",
        "content_plans",
        "inferred_patterns",
        "rulebooks",
    ):
        op.execute(
            f"""
            DELETE FROM {table_name}
            WHERE NOT EXISTS (
                SELECT 1 FROM projects WHERE projects.id = {table_name}.project_id
            )
            """
        )

    op.execute(
        """
        UPDATE generated_articles
        SET content_plan_id = NULL
        WHERE content_plan_id IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM content_plans
              WHERE content_plans.id = generated_articles.content_plan_id
          )
        """
    )

    op.alter_column(
        "projects",
        "total_cost_usd",
        existing_type=sa.Numeric(10, 2),
        type_=sa.Numeric(14, 6),
        existing_nullable=True,
    )

    op.create_foreign_key(
        "fk_generated_articles_project",
        "generated_articles",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_generated_articles_content_plan",
        "generated_articles",
        "content_plans",
        ["content_plan_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_foreign_key(
        "fk_article_revisions_article",
        "article_revisions",
        "generated_articles",
        ["article_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_content_plans_project",
        "content_plans",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_rulebooks_project",
        "rulebooks",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_inferred_patterns_project",
        "inferred_patterns",
        "projects",
        ["project_id"],
        ["id"],
        ondelete="CASCADE",
    )
    op.create_foreign_key(
        "fk_rules_rulebook",
        "rules",
        "rulebooks",
        ["rulebook_id"],
        ["id"],
        ondelete="CASCADE",
    )


def downgrade() -> None:
    op.drop_constraint("fk_rules_rulebook", "rules", type_="foreignkey")
    op.drop_constraint("fk_inferred_patterns_project", "inferred_patterns", type_="foreignkey")
    op.drop_constraint("fk_rulebooks_project", "rulebooks", type_="foreignkey")
    op.drop_constraint("fk_content_plans_project", "content_plans", type_="foreignkey")
    op.drop_constraint("fk_article_revisions_article", "article_revisions", type_="foreignkey")
    op.drop_constraint(
        "fk_generated_articles_content_plan",
        "generated_articles",
        type_="foreignkey",
    )
    op.drop_constraint("fk_generated_articles_project", "generated_articles", type_="foreignkey")
    op.alter_column(
        "projects",
        "total_cost_usd",
        existing_type=sa.Numeric(14, 6),
        type_=sa.Numeric(10, 2),
        existing_nullable=True,
    )
