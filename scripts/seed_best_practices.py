"""
Best Practices Seeding Script

Populates the system's best practices knowledge base with curated SEO, content
marketing, and writing guidelines. These serve as Layer 3 fallback knowledge in
the decision hierarchy when projects lack explicit rules or inferred patterns.

Usage:
    python scripts/seed_best_practices.py [--reset]

Options:
    --reset    Clear existing best practices before seeding
"""

import argparse
import asyncio
from datetime import datetime
from typing import Dict, List
from uuid import uuid4

from loguru import logger
from sentence_transformers import SentenceTransformer

from infrastructure.database import DatabaseManager
from intelligence.semantic_analyzer import SemanticAnalyzer

# Curated best practices knowledge base
BEST_PRACTICES = [
    {
        "category": "tone",
        "subcategory": "general",
        "guideline": (
            "Use clear, accessible language appropriate for a general audience. "
            "Avoid jargon unless writing for a technical audience. Aim for "
            "Flesch-Kincaid grade level 10-12 for broad readability."
        ),
        "priority": 8,
        "context": "Default tone for all content types",
    },
    {
        "category": "tone",
        "subcategory": "b2b",
        "guideline": (
            "For B2B content, adopt an authoritative yet accessible tone. "
            "Demonstrate expertise without condescension. Use industry terminology "
            "appropriately but explain complex concepts clearly."
        ),
        "priority": 9,
        "context": "Business-to-business content, enterprise software, professional services",
    },
    {
        "category": "tone",
        "subcategory": "b2c",
        "guideline": (
            "For B2C content, use conversational, engaging language that builds "
            "connection with readers. Be friendly and approachable. Use second person "
            "('you') to create direct engagement."
        ),
        "priority": 9,
        "context": "Consumer-focused content, retail, lifestyle, entertainment",
    },
    {
        "category": "structure",
        "subcategory": "introduction",
        "guideline": (
            "Begin articles with a compelling hook that establishes relevance. "
            "State the problem or question being addressed in the first paragraph. "
            "Keep introductions concise—3-4 sentences maximum."
        ),
        "priority": 10,
        "context": "Opening sections of all articles",
    },
    {
        "category": "structure",
        "subcategory": "body",
        "guideline": (
            "Organize body content with clear, descriptive headings (H2, H3). "
            "Use short paragraphs (3-4 sentences). Break up text with bullet points, "
            "numbered lists, and visual elements where appropriate."
        ),
        "priority": 9,
        "context": "Main content sections",
    },
    {
        "category": "structure",
        "subcategory": "conclusion",
        "guideline": (
            "Conclude with a clear summary of key points. Provide actionable "
            "takeaways or next steps. Avoid introducing new information in conclusions. "
            "End with a call-to-action when appropriate."
        ),
        "priority": 9,
        "context": "Closing sections of articles",
    },
    {
        "category": "seo",
        "subcategory": "keywords",
        "guideline": (
            "Integrate primary keywords naturally in: title, first paragraph, "
            "at least one H2 heading, and conclusion. Avoid keyword stuffing—maintain "
            "readability over keyword density. Target 0.5-2% keyword density."
        ),
        "priority": 10,
        "context": "Keyword integration strategy",
    },
    {
        "category": "seo",
        "subcategory": "headings",
        "guideline": (
            "Use hierarchical heading structure (H1 → H2 → H3). Include keywords "
            "in headings where natural. Make headings descriptive and scannable. "
            "Limit to one H1 per page (the title)."
        ),
        "priority": 9,
        "context": "Heading optimization",
    },
    {
        "category": "seo",
        "subcategory": "meta",
        "guideline": (
            "Craft meta descriptions that summarize content in 150-160 characters. "
            "Include primary keyword naturally. Write compelling copy that encourages "
            "clicks. Avoid duplicate meta descriptions across pages."
        ),
        "priority": 10,
        "context": "Meta description creation",
    },
    {
        "category": "seo",
        "subcategory": "length",
        "guideline": (
            "Target 1,500-2,500 words for comprehensive guides and pillar content. "
            "800-1,500 words for standard blog posts. 300-800 words for news articles "
            "or updates. Prioritize value over arbitrary word counts."
        ),
        "priority": 7,
        "context": "Content length guidelines",
    },
    {
        "category": "writing_style",
        "subcategory": "voice",
        "guideline": (
            "Use active voice predominantly (target 80%+ of sentences). Passive voice "
            "is acceptable when the actor is unknown or less important than the action. "
            "Active voice improves clarity and engagement."
        ),
        "priority": 8,
        "context": "Sentence construction",
    },
    {
        "category": "writing_style",
        "subcategory": "clarity",
        "guideline": (
            "Prefer concrete, specific language over vague generalizations. "
            "Use examples to illustrate abstract concepts. Define technical terms "
            "on first use. Avoid ambiguous pronouns."
        ),
        "priority": 9,
        "context": "Writing clarity and precision",
    },
    {
        "category": "writing_style",
        "subcategory": "engagement",
        "guideline": (
            "Ask rhetorical questions to engage readers. Use storytelling techniques "
            "where appropriate. Include relevant statistics and data to support claims. "
            "Vary sentence structure to maintain rhythm."
        ),
        "priority": 7,
        "context": "Reader engagement techniques",
    },
    {
        "category": "content_types",
        "subcategory": "how_to",
        "guideline": (
            "Structure how-to guides with: clear problem statement, required materials/prerequisites, "
            "numbered step-by-step instructions, visual aids or examples, troubleshooting tips, "
            "and conclusion with next steps. Make instructions actionable and specific."
        ),
        "priority": 10,
        "context": "Instructional content, tutorials, guides",
    },
    {
        "category": "content_types",
        "subcategory": "listicle",
        "guideline": (
            "For list-based articles: use odd numbers (5, 7, 9 perform well), make items "
            "parallel in structure, include brief explanations for each item, order by "
            "importance or logical sequence, and summarize key takeaways at the end."
        ),
        "priority": 8,
        "context": "List-based articles, roundups, rankings",
    },
    {
        "category": "content_types",
        "subcategory": "comparison",
        "guideline": (
            "For comparison content: establish clear evaluation criteria upfront, "
            "present options objectively, use tables or side-by-side formats, "
            "highlight key differences, and provide a recommendation based on use cases."
        ),
        "priority": 9,
        "context": "Comparison articles, versus posts, product reviews",
    },
    {
        "category": "content_types",
        "subcategory": "thought_leadership",
        "guideline": (
            "For thought leadership content: present unique insights or perspectives, "
            "support opinions with data and examples, acknowledge counterarguments, "
            "demonstrate deep industry knowledge, and conclude with forward-looking implications."
        ),
        "priority": 8,
        "context": "Opinion pieces, industry analysis, trend predictions",
    },
    {
        "category": "formatting",
        "subcategory": "readability",
        "guideline": (
            "Break up long text blocks with subheadings every 300-400 words. Use bullet "
            "points for lists of 3+ items. Bold key concepts sparingly (1-2 per section). "
            "Keep sentences under 25 words on average."
        ),
        "priority": 9,
        "context": "Visual formatting and text structure",
    },
    {
        "category": "formatting",
        "subcategory": "emphasis",
        "guideline": (
            "Use bold for important concepts and terms (not entire sentences). "
            "Use italics for emphasis, titles, or foreign words. Avoid ALL CAPS except "
            "for acronyms. Use formatting sparingly to maintain impact."
        ),
        "priority": 7,
        "context": "Text emphasis and styling",
    },
    {
        "category": "credibility",
        "subcategory": "sources",
        "guideline": (
            "Cite authoritative sources for statistics and factual claims. Link to "
            "primary sources when possible. Include publication dates for time-sensitive "
            "information. Use reputable, well-known sources to build trust."
        ),
        "priority": 10,
        "context": "Source citation and credibility",
    },
    {
        "category": "credibility",
        "subcategory": "accuracy",
        "guideline": (
            "Verify all facts and statistics before publication. Avoid absolute statements "
            "unless certain. Use qualifiers appropriately (often, typically, generally). "
            "Update content periodically to maintain accuracy."
        ),
        "priority": 10,
        "context": "Factual accuracy and verification",
    },
    {
        "category": "user_intent",
        "subcategory": "informational",
        "guideline": (
            "For informational queries: provide comprehensive, well-researched answers. "
            "Use clear explanations and examples. Structure content for easy scanning. "
            "Answer common related questions."
        ),
        "priority": 9,
        "context": "Content targeting informational search intent",
    },
    {
        "category": "user_intent",
        "subcategory": "commercial",
        "guideline": (
            "For commercial intent: compare options objectively, highlight key features "
            "and benefits, address common objections, include pricing information where "
            "relevant, and provide clear next steps (CTAs)."
        ),
        "priority": 9,
        "context": "Content targeting commercial/transactional intent",
    },
    {
        "category": "user_intent",
        "subcategory": "navigational",
        "guideline": (
            "For navigational intent: provide direct answers quickly, include relevant "
            "internal links, ensure brand/product information is prominent, and optimize "
            "for featured snippets."
        ),
        "priority": 8,
        "context": "Content targeting navigational intent",
    },
    {
        "category": "accessibility",
        "subcategory": "general",
        "guideline": (
            "Write descriptive link text (avoid 'click here'). Use semantic HTML structure. "
            "Ensure sufficient color contrast. Provide alt text for images. Structure content "
            "logically for screen readers."
        ),
        "priority": 8,
        "context": "Content accessibility for all users",
    },
    {
        "category": "technical_seo",
        "subcategory": "internal_linking",
        "guideline": (
            "Include 2-5 relevant internal links per 1000 words. Use descriptive anchor "
            "text with target keywords. Link to both newer and authoritative older content. "
            "Ensure links add value for readers."
        ),
        "priority": 8,
        "context": "Internal linking strategy",
    },
    {
        "category": "technical_seo",
        "subcategory": "external_linking",
        "guideline": (
            "Link to 1-3 high-authority external sources per article. Open external links "
            "in new tabs (controversial but common practice). Use nofollow for sponsored "
            "or untrusted links. Verify link destinations before publishing."
        ),
        "priority": 7,
        "context": "External linking practices",
    },
    {
        "category": "technical_seo",
        "subcategory": "freshness",
        "guideline": (
            "Update evergreen content annually or when information changes. Add publication "
            "and last-updated dates. Refresh statistics and examples periodically. "
            "Archive or redirect truly outdated content."
        ),
        "priority": 7,
        "context": "Content freshness and maintenance",
    },
    {
        "category": "audience_adaptation",
        "subcategory": "expertise_level",
        "guideline": (
            "For beginner audiences: define terms, avoid jargon, use analogies, provide "
            "step-by-step guidance. For advanced audiences: assume foundational knowledge, "
            "use technical terminology, focus on nuance and edge cases."
        ),
        "priority": 9,
        "context": "Adapting content to audience expertise",
    },
    {
        "category": "audience_adaptation",
        "subcategory": "cultural",
        "guideline": (
            "Consider cultural context and sensitivities. Avoid idioms that don't translate "
            "well. Use inclusive language. Be mindful of examples that may not resonate "
            "globally. Localize dates, measurements, and currency where appropriate."
        ),
        "priority": 7,
        "context": "Cultural adaptation and inclusivity",
    },
]


class BestPracticesSeeder:
    """
        Seeds best practices knowledge base with semantic indexing.
    Processes curated guidelines through embedding generation for
    intelligent semantic retrieval during content generation.
    """

    def __init__(self, db: DatabaseManager, semantic_analyzer: SemanticAnalyzer):
        self.db = db
        self.semantic_analyzer = semantic_analyzer

        logger.info("BestPracticesSeeder initialized")

    async def seed(self, reset: bool = False):
        """
        Seed best practices into database with semantic embeddings.

        Args:
            reset: If True, clear existing best practices before seeding
        """
        logger.info(f"Starting best practices seeding | reset={reset}")

        if reset:
            await self._clear_existing()

        # Check if already seeded
        existing_count = await self._count_existing()
        if existing_count > 0 and not reset:
            logger.warning(
                f"Best practices already seeded ({existing_count} records). "
                f"Use --reset to re-seed."
            )
            return

        # Seed practices
        seeded_count = 0
        failed_count = 0

        for practice in BEST_PRACTICES:
            try:
                await self._insert_practice(practice)
                seeded_count += 1

                if seeded_count % 5 == 0:
                    logger.info(f"Seeded {seeded_count}/{len(BEST_PRACTICES)} practices")

            except Exception as e:
                logger.error(
                    f"Failed to seed practice | category={practice['category']} | error={e}"
                )
                failed_count += 1

        logger.success(
            f"Best practices seeding completed | " f"seeded={seeded_count} | failed={failed_count}"
        )

        # Create indexes for efficient querying
        await self._create_indexes()

    async def _clear_existing(self):
        """Remove all existing best practices."""
        logger.info("Clearing existing best practices")

        await self.db.execute("DELETE FROM best_practices")

        logger.info("Existing best practices cleared")

    async def _count_existing(self) -> int:
        """Count existing best practices records."""
        result = await self.db.fetch_one("SELECT COUNT(*) as count FROM best_practices")
        return result["count"] if result else 0

    async def _insert_practice(self, practice: Dict):
        """
        Insert single best practice with semantic embedding.

        Generates embedding for the guideline text to enable
        semantic similarity search during decision-making.
        """
        # Generate embedding for semantic search
        embedding = await self.semantic_analyzer.generate_embedding(practice["guideline"])

        # Insert into database
        practice_id = uuid4()

        await self.db.execute(
            """
            INSERT INTO best_practices (
                id, category, subcategory, guideline,
                embedding, priority, context, created_at
            )
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            practice_id,
            practice["category"],
            practice["subcategory"],
            practice["guideline"],
            embedding.tolist(),  # Convert numpy array to list for pgvector
            practice["priority"],
            practice["context"],
            datetime.utcnow(),
        )

    async def _create_indexes(self):
        """Create database indexes for efficient querying."""
        logger.info("Creating indexes for best practices")

        # Create vector index for similarity search
        await self.db.execute(
            """
            CREATE INDEX IF NOT EXISTS best_practices_embedding_idx
            ON best_practices
            USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 50)
            """
        )

        # Create indexes for filtering
        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS best_practices_category_idx ON best_practices (category)"
        )

        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS best_practices_subcategory_idx ON best_practices (subcategory)"
        )

        await self.db.execute(
            "CREATE INDEX IF NOT EXISTS best_practices_priority_idx ON best_practices (priority DESC)"
        )

        logger.success("Indexes created successfully")

    async def verify_seeding(self) -> Dict:
        """
        Verify seeding completeness and quality.

        Returns statistics about seeded practices.
        """
        logger.info("Verifying best practices seeding")

        # Count total records
        total = await self._count_existing()

        # Count by category
        category_counts = await self.db.fetch_all(
            """
            SELECT category, COUNT(*) as count
            FROM best_practices
            GROUP BY category
            ORDER BY count DESC
            """
        )

        # Test semantic search functionality
        test_query = "What tone should I use for writing?"
        test_embedding = await self.semantic_analyzer.generate_embedding(test_query)

        similar_practices = await self.db.fetch_all(
            """
            SELECT guideline, category, subcategory,
                   1 - (embedding <=> $1::vector) as similarity
            FROM best_practices
            ORDER BY embedding <=> $1::vector
            LIMIT 3
            """,
            test_embedding.tolist(),
        )

        verification_result = {
            "total_practices": total,
            "expected_practices": len(BEST_PRACTICES),
            "seeding_complete": total == len(BEST_PRACTICES),
            "categories": {row["category"]: row["count"] for row in category_counts},
            "semantic_search_test": {
                "query": test_query,
                "top_results": [
                    {
                        "guideline": row["guideline"][:100] + "...",
                        "category": row["category"],
                        "subcategory": row["subcategory"],
                        "similarity": float(row["similarity"]),
                    }
                    for row in similar_practices
                ],
            },
        }

        logger.info(
            f"Verification complete | seeding_complete={verification_result['seeding_complete']}"
        )

        return verification_result


async def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(description="Seed best practices knowledge base")
    parser.add_argument(
        "--reset", action="store_true", help="Clear existing best practices before seeding"
    )
    parser.add_argument(
        "--verify-only", action="store_true", help="Only verify existing seeding without re-seeding"
    )
    args = parser.parse_args()

    logger.info("Best practices seeding script started")

    # Initialize dependencies
    db = DatabaseManager()
    await db.initialize()

    semantic_analyzer = SemanticAnalyzer()

    seeder = BestPracticesSeeder(db, semantic_analyzer)

    try:
        if args.verify_only:
            # Only verify
            verification = await seeder.verify_seeding()

            print("\n" + "=" * 60)
            print("VERIFICATION RESULTS")
            print("=" * 60)
            print(
                f"Total Practices: {verification['total_practices']}/{verification['expected_practices']}"
            )
            print(f"Seeding Complete: {verification['seeding_complete']}")
            print("\nCategories:")
            for category, count in verification["categories"].items():
                print(f"  - {category}: {count}")
            print("\nSemantic Search Test:")
            print(f"  Query: {verification['semantic_search_test']['query']}")
            print("  Top Results:")
            for result in verification["semantic_search_test"]["top_results"]:
                print(
                    f"    - [{result['category']}/{result['subcategory']}] "
                    f"Similarity: {result['similarity']:.3f}"
                )
                print(f"      {result['guideline']}")
            print("=" * 60 + "\n")
        else:
            # Seed practices
            await seeder.seed(reset=args.reset)

            # Verify after seeding
            verification = await seeder.verify_seeding()

            if verification["seeding_complete"]:
                logger.success("✓ Best practices seeding verified successfully")
            else:
                logger.error(
                    f"✗ Seeding incomplete | "
                    f"expected={verification['expected_practices']} | "
                    f"actual={verification['total_practices']}"
                )

    finally:
        await db.disconnect()
        logger.info("Database connection closed")


if __name__ == "__main__":
    asyncio.run(main())
