"""
Rulebook Manager - Semantic Rule Processing
============================================

Transforms natural language rulebooks into searchable vector databases:
- Intelligent text chunking (semantic boundaries)
- Rule classification and metadata extraction
- Vector embedding generation and storage
- Semantic similarity search for rule retrieval

Design Philosophy: Rules are semantic objects, not syntax patterns.
Query by intent, not by keyword matching.
"""

import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from uuid import UUID, uuid4

import numpy as np
from loguru import logger
from sqlalchemy import delete, func, insert, select, text, update
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import DatabaseError, ValidationError
from core.models import Rule, Rulebook, RuleType
from infrastructure.redis_client import redis_client
from infrastructure.schema import rulebooks_table, rules_table


class RulebookManager:
    """
    Manages rulebook lifecycle: parsing, indexing, querying.

    Implements semantic chunking and vector indexing for NLP-native
    rule retrieval.
    """

    def __init__(self, session: AsyncSession, embedding_model):
        """
        Initialize manager with database session and embedding model.

        Args:
            session: SQLAlchemy async session
            embedding_model: Sentence-BERT model for embeddings
        """
        self.session = session
        self.embedding_model = embedding_model
        self.chunk_size = 500  # characters per semantic chunk
        self.overlap = 50  # character overlap between chunks

    # =========================================================================
    # RULEBOOK CREATION & INDEXING
    # =========================================================================

    async def create_rulebook(
        self,
        project_id: UUID,
        raw_content: str,
    ) -> Rulebook:
        """
        Create and index a new rulebook.

        Process:
        1. Chunk raw content into semantic units
        2. Classify each chunk by rule type
        3. Generate embeddings
        4. Store in database and cache

        Args:
            project_id: UUID of owning project
            raw_content: Full rulebook text

        Returns:
            Rulebook model with indexed rules
        """
        try:
            logger.info(f"Creating rulebook for project {project_id}")

            # Validate content
            if not raw_content or len(raw_content) < 50:
                raise ValidationError("Rulebook content too short (minimum 50 characters)")

            # Get next version number
            version = await self._get_next_version(project_id)

            # Create rulebook record
            rulebook_id = uuid4()
            query = (
                insert(rulebooks_table)
                .values(
                    id=rulebook_id,
                    project_id=project_id,
                    content=raw_content,
                    version=version,
                    created_at=func.now(),
                    updated_at=func.now(),
                )
                .returning(rulebooks_table)
            )

            result = await self.session.execute(query)
            row = result.fetchone()

            # Parse and index rules
            rules = await self._parse_and_index_rules(rulebook_id, raw_content)

            await self.session.commit()

            rulebook = Rulebook(
                id=row.id,
                project_id=row.project_id,
                raw_content=row.content,
                version=row.version,
                created_at=row.created_at,
                updated_at=row.updated_at,
                rules=rules,
            )

            logger.info(f"Created rulebook with {len(rules)} rules (version {version})")
            return rulebook

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to create rulebook: {e}")
            raise DatabaseError(f"Rulebook creation failed: {e}")

    async def _parse_and_index_rules(
        self,
        rulebook_id: UUID,
        raw_content: str,
    ) -> List[Rule]:
        """
        Parse rulebook into semantic chunks and index as rules.

        Strategy:
        1. Split into paragraphs (natural semantic boundaries)
        2. Further chunk long paragraphs
        3. Classify each chunk by rule type
        4. Generate embeddings
        5. Store with metadata

        Args:
            rulebook_id: UUID of parent rulebook
            raw_content: Full rulebook text

        Returns:
            List of indexed Rule objects
        """
        # Step 1: Chunk content
        chunks = self._chunk_content(raw_content)
        logger.debug(f"Chunked rulebook into {len(chunks)} semantic units")

        # Step 2: Process each chunk
        rules = []
        for idx, chunk in enumerate(chunks):
            # Classify rule type
            rule_type = self._classify_rule_type(chunk)

            # Extract priority from content (e.g., "CRITICAL:", "Important:")
            priority = self._extract_priority(chunk)

            # Extract contextual information
            context = self._extract_context(chunk)

            # Generate embedding
            embedding = self.embedding_model.encode(chunk)

            # Store in database
            rule_id = uuid4()
            query = (
                insert(rules_table)
                .values(
                    id=rule_id,
                    rulebook_id=rulebook_id,
                    rule_type=rule_type.value,
                    content=chunk,
                    embedding=embedding,
                    priority=priority,
                    context=context,
                    created_at=func.now(),
                )
                .returning(rules_table)
            )

            result = await self.session.execute(query)
            row = result.fetchone()

            rule = Rule(
                id=row.id,
                rulebook_id=row.rulebook_id,
                rule_type=RuleType(row.rule_type),
                content=row.content,
                embedding=embedding,
                priority=row.priority,
                context=row.context,
                created_at=row.created_at,
            )

            rules.append(rule)

            # Cache embedding in Redis for fast access
            await redis_client.store_embedding(
                key=f"rule_emb:{rule_id}",
                embedding=embedding.tolist(),
                ttl=86400 * 90,  # 90 days
            )

        logger.info(f"Indexed {len(rules)} rules with embeddings")
        return rules

    def _chunk_content(self, content: str) -> List[str]:
        """
        Chunk content into semantic units.

        Strategy:
        - Split on paragraph boundaries (double newline)
        - Further split long paragraphs at sentence boundaries
        - Maintain minimum chunk size for context

        Args:
            content: Full rulebook text

        Returns:
            List of text chunks
        """
        # Normalize whitespace
        content = re.sub(r"\n{3,}", "\n\n", content)

        # Split into paragraphs
        paragraphs = content.split("\n\n")

        chunks = []
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            # If paragraph is short enough, use as-is
            if len(para) <= self.chunk_size:
                chunks.append(para)
            else:
                # Split long paragraph at sentence boundaries
                sentences = re.split(r"(?<=[.!?])\s+", para)

                current_chunk = ""
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) <= self.chunk_size:
                        current_chunk += sentence + " "
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = sentence + " "

                if current_chunk:
                    chunks.append(current_chunk.strip())

        return chunks

    def _classify_rule_type(self, chunk: str) -> RuleType:
        """
        Classify chunk into rule type using keyword matching.

        Types:
        - TONE: Voice, style, personality
        - STRUCTURE: Format, organization, layout
        - TOPIC: Subject matter, themes
        - STYLE: Writing conventions, grammar
        - SEO: Keywords, optimization
        - GENERAL: Catch-all

        Args:
            chunk: Text chunk

        Returns:
            Classified RuleType
        """
        chunk_lower = chunk.lower()

        # Tone indicators
        tone_keywords = [
            "tone",
            "voice",
            "personality",
            "feel",
            "emotion",
            "attitude",
            "formal",
            "casual",
        ]
        if any(kw in chunk_lower for kw in tone_keywords):
            return RuleType.TONE

        # Structure indicators
        structure_keywords = [
            "format",
            "structure",
            "organize",
            "layout",
            "section",
            "heading",
            "outline",
        ]
        if any(kw in chunk_lower for kw in structure_keywords):
            return RuleType.STRUCTURE

        # Topic indicators
        topic_keywords = ["topic", "subject", "about", "focus on", "cover", "discuss", "theme"]
        if any(kw in chunk_lower for kw in topic_keywords):
            return RuleType.TOPIC

        # Style indicators
        style_keywords = ["style", "grammar", "punctuation", "capitalize", "write", "spelling"]
        if any(kw in chunk_lower for kw in style_keywords):
            return RuleType.STYLE

        # SEO indicators
        seo_keywords = ["seo", "keyword", "optimize", "search", "rank", "meta"]
        if any(kw in chunk_lower for kw in seo_keywords):
            return RuleType.SEO

        return RuleType.GENERAL

    def _extract_priority(self, chunk: str) -> int:
        """
        Extract priority level from chunk content.

        Looks for explicit markers:
        - CRITICAL, MANDATORY, MUST: priority 10
        - IMPORTANT, SHOULD: priority 7
        - PREFER, RECOMMEND: priority 5
        - DEFAULT: priority 3

        Args:
            chunk: Text chunk

        Returns:
            Priority level (1-10)
        """
        chunk_upper = chunk.upper()

        if any(marker in chunk_upper for marker in ["CRITICAL", "MANDATORY", "MUST"]):
            return 10
        elif any(marker in chunk_upper for marker in ["IMPORTANT", "SHOULD"]):
            return 7
        elif any(marker in chunk_upper for marker in ["PREFER", "RECOMMEND"]):
            return 5
        else:
            return 3

    def _extract_context(self, chunk: str) -> Optional[str]:
        """
        Extract contextual information (when does rule apply?).

        Looks for patterns like:
        - "When writing about X..."
        - "For articles on Y..."
        - "In technical content..."

        Args:
            chunk: Text chunk

        Returns:
            Context string or None
        """
        # Pattern: "when ... " or "for ... "
        context_pattern = r"(?:when|for|in)\s+([^,.!?]+)"

        matches = re.findall(context_pattern, chunk.lower())
        if matches:
            return matches[0].strip()

        return None

    async def _get_next_version(self, project_id: UUID) -> int:
        """Get next version number for project's rulebook using SQLAlchemy Core."""
        query = select(func.coalesce(func.max(rulebooks_table.c.version), 0) + 1).where(
            rulebooks_table.c.project_id == project_id
        )

        result = await self.session.execute(query)
        return result.scalar()

    # =========================================================================
    # SEMANTIC QUERY & RETRIEVAL
    # =========================================================================

    async def query_rules(
        self,
        project_id: UUID,
        query: str,
        rule_type: Optional[RuleType] = None,
        top_k: int = 5,
        similarity_threshold: float = 0.75,
    ) -> List[Tuple[Rule, float]]:
        """
        Semantic search for relevant rules.

        Process:
        1. Generate query embedding
        2. Vector similarity search in database
        3. Filter by type and threshold
        4. Return ranked results

        Args:
            project_id: UUID of project
            query: Natural language query
            rule_type: Optional filter by rule type
            top_k: Maximum results to return
            similarity_threshold: Minimum cosine similarity (0-1)

        Returns:
            List of (Rule, similarity_score) tuples, ordered by relevance
        """
        try:
            # Generate query embedding
            query_embedding = self.embedding_model.encode(query)

            # Build SQL query with vector similarity
            sql = """
                SELECT 
                    r.id, r.rulebook_id, r.rule_type, r.content, 
                    r.priority, r.context, r.created_at,
                    1 - (r.embedding <=> :query_embedding::vector) AS similarity
                FROM rules r
                JOIN rulebooks rb ON r.rulebook_id = rb.id
                WHERE rb.project_id = :project_id
            """

            params = {
                "project_id": project_id,
                "query_embedding": query_embedding,
            }

            if rule_type:
                sql += " AND r.rule_type = :rule_type"
                params["rule_type"] = rule_type.value

            sql += """
                AND (1 - (r.embedding <=> :query_embedding::vector)) >= :threshold
                ORDER BY similarity DESC
                LIMIT :top_k;
            """

            params["threshold"] = similarity_threshold
            params["top_k"] = top_k

            result = await self.session.execute(text(sql), params)
            rows = result.fetchall()

            # Convert to Rule objects with scores
            results = []
            for row in rows:
                rule = Rule(
                    id=row[0],
                    rulebook_id=row[1],
                    rule_type=RuleType(row[2]),
                    content=row[3],
                    embedding=[],  # Don't load full embedding
                    priority=row[4],
                    context=row[5],
                    created_at=row[6],
                )
                similarity = float(row[7])
                results.append((rule, similarity))

            logger.debug(f"Found {len(results)} rules for query: '{query[:50]}...'")
            return results

        except Exception as e:
            logger.error(f"Rule query failed: {e}")
            return []

    async def get_rules_by_type(
        self,
        project_id: UUID,
        rule_type: RuleType,
    ) -> List[Rule]:
        """
        Retrieve all rules of specific type for a project using SQLAlchemy Core.

        Args:
            project_id: UUID of project
            rule_type: Type of rules to retrieve

        Returns:
            List of Rule objects
        """
        try:
            query = (
                select(
                    rules_table.c.id,
                    rules_table.c.rulebook_id,
                    rules_table.c.rule_type,
                    rules_table.c.content,
                    rules_table.c.priority,
                    rules_table.c.context,
                    rules_table.c.created_at,
                )
                .select_from(
                    rules_table.join(
                        rulebooks_table, rules_table.c.rulebook_id == rulebooks_table.c.id
                    )
                )
                .where(
                    (rulebooks_table.c.project_id == project_id)
                    & (rules_table.c.rule_type == rule_type.value)
                )
                .order_by(rules_table.c.priority.desc(), rules_table.c.created_at.asc())
            )

            result = await self.session.execute(query)
            rows = result.fetchall()

            return [
                Rule(
                    id=row.id,
                    rulebook_id=row.rulebook_id,
                    rule_type=RuleType(row.rule_type),
                    content=row.content,
                    embedding=[],
                    priority=row.priority,
                    context=row.context,
                    created_at=row.created_at,
                )
                for row in rows
            ]

        except Exception as e:
            logger.error(f"Failed to retrieve rules by type: {e}")
            return []

    # =========================================================================
    # RULEBOOK MANAGEMENT
    # =========================================================================

    async def get_latest_rulebook(self, project_id: UUID) -> Optional[Rulebook]:
        """Retrieve the latest version of project's rulebook using SQLAlchemy Core."""
        try:
            query = (
                select(rulebooks_table)
                .where(rulebooks_table.c.project_id == project_id)
                .order_by(rulebooks_table.c.version.desc())
                .limit(1)
            )

            result = await self.session.execute(query)
            row = result.fetchone()

            if not row:
                return None

            # Load associated rules
            rules = await self._load_rules(row.id)

            return Rulebook(
                id=row.id,
                project_id=row.project_id,
                raw_content=row.content,
                version=row.version,
                created_at=row.created_at,
                updated_at=row.updated_at,
                rules=rules,
            )

        except Exception as e:
            logger.error(f"Failed to retrieve latest rulebook: {e}")
            return None

    async def _load_rules(self, rulebook_id: UUID) -> List[Rule]:
        """Load all rules for a rulebook using SQLAlchemy Core."""
        query = (
            select(
                rules_table.c.id,
                rules_table.c.rulebook_id,
                rules_table.c.rule_type,
                rules_table.c.content,
                rules_table.c.embedding,
                rules_table.c.priority,
                rules_table.c.context,
                rules_table.c.created_at,
            )
            .where(rules_table.c.rulebook_id == rulebook_id)
            .order_by(rules_table.c.priority.desc(), rules_table.c.created_at.asc())
        )

        result = await self.session.execute(query)
        rows = result.fetchall()

        return [
            Rule(
                id=row.id,
                rulebook_id=row.rulebook_id,
                rule_type=RuleType(row.rule_type),
                content=row.content,
                embedding=row.embedding,
                priority=row.priority,
                context=row.context,
                created_at=row.created_at,
            )
            for row in rows
        ]

    async def delete_rulebook(self, rulebook_id: UUID) -> bool:
        """
        Delete rulebook and all associated rules using SQLAlchemy Core.

        Cascades via foreign key constraints.

        Returns:
            True if deleted successfully
        """
        try:
            query = delete(rulebooks_table).where(rulebooks_table.c.id == rulebook_id)
            result = await self.session.execute(query)
            await self.session.commit()

            if result.rowcount > 0:
                logger.info(f"Deleted rulebook: {rulebook_id}")
                return True

            return False

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to delete rulebook: {e}")
            raise DatabaseError(f"Rulebook deletion failed: {e}")
