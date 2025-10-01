"""
Website Analyzer - Pattern Inference Engine
============================================

Scrapes and analyzes target websites to infer implicit content rules:
- Intelligent content extraction (article detection)
- Linguistic feature analysis (spaCy)
- Stylistic pattern detection
- Statistical confidence calculation

Implements "Layer 2" of decision hierarchy: learning from examples
when explicit rules are unavailable.
"""

import asyncio
import re
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse
from uuid import UUID, uuid4

from bs4 import BeautifulSoup
from loguru import logger
from playwright.async_api import Browser, Page, async_playwright
from sqlalchemy.ext.asyncio import AsyncSession

from core.exceptions import ScrapingError, ValidationError
from core.models import InferredPatterns, StructurePattern
from knowledge.pattern_extractor import PatternExtractor


class WebsiteAnalyzer:
    """
    Analyzes websites to infer content patterns and style guidelines.

    Strategy:
    1. Discover article URLs (sitemap/navigation parsing)
    2. Scrape representative sample
    3. Extract main content (removing boilerplate)
    4. Linguistic analysis via PatternExtractor
    5. Statistical aggregation
    6. Store inferred patterns
    """

    def __init__(self, session: AsyncSession, pattern_extractor: PatternExtractor):
        """
        Initialize analyzer with dependencies.

        Args:
            session: Database session
            pattern_extractor: Linguistic pattern extraction engine
        """
        self.session = session
        self.pattern_extractor = pattern_extractor
        self.browser: Optional[Browser] = None
        self.sample_size = 15  # Number of articles to analyze
        self.timeout_ms = 30000  # 30 seconds per page

    # =========================================================================
    # ANALYSIS ORCHESTRATION
    # =========================================================================

    async def analyze_website(
        self,
        project_id: UUID,
        domain: str,
        force_refresh: bool = False,
    ) -> Optional[InferredPatterns]:
        """
        Complete website analysis workflow.

        Args:
            project_id: UUID of project
            domain: Target domain (e.g., "example.com")
            force_refresh: If True, re-analyze even if patterns exist

        Returns:
            InferredPatterns model or None if analysis fails
        """
        try:
            logger.info(f"Starting website analysis for {domain}")

            # Check if recent analysis exists
            if not force_refresh:
                existing = await self._get_existing_patterns(project_id)
                if existing:
                    logger.info(f"Using existing patterns (analyzed {existing.analyzed_at})")
                    return existing

            # Initialize browser
            await self._init_browser()

            # Step 1: Discover article URLs
            article_urls = await self._discover_articles(domain)
            if len(article_urls) < 5:
                raise ScrapingError(f"Insufficient articles found: {len(article_urls)} (minimum 5)")

            # Step 2: Scrape representative sample
            sample_urls = article_urls[: self.sample_size]
            articles = await self._scrape_articles(sample_urls)

            if len(articles) < 5:
                raise ScrapingError(f"Failed to scrape minimum articles: {len(articles)}/5")

            # Step 3: Extract patterns
            patterns = await self._extract_patterns(articles)

            # Step 4: Store in database
            inferred = await self._store_patterns(project_id, patterns)

            logger.info(f"Analysis complete: {len(articles)} articles analyzed")
            return inferred

        except Exception as e:
            logger.error(f"Website analysis failed for {domain}: {e}")
            raise ScrapingError(f"Analysis failed: {e}")

        finally:
            await self._close_browser()

    # =========================================================================
    # CONTENT DISCOVERY
    # =========================================================================

    async def _discover_articles(self, domain: str) -> List[str]:
        """
        Discover article URLs from website.

        Strategy:
        1. Try sitemap.xml first (most reliable)
        2. Fallback to crawling blog/articles section
        3. Filter for content pages (heuristics)

        Args:
            domain: Target domain

        Returns:
            List of article URLs
        """
        base_url = f"https://{domain}" if not domain.startswith("http") else domain

        # Try sitemap first
        sitemap_urls = await self._parse_sitemap(base_url)
        if len(sitemap_urls) >= 10:
            logger.info(f"Discovered {len(sitemap_urls)} URLs from sitemap")
            return sitemap_urls

        # Fallback: crawl common article sections
        article_urls = await self._crawl_article_section(base_url)

        logger.info(f"Discovered {len(article_urls)} article URLs")
        return article_urls

    async def _parse_sitemap(self, base_url: str) -> List[str]:
        """
        Parse sitemap.xml to extract article URLs.

        Args:
            base_url: Website base URL

        Returns:
            List of URLs from sitemap
        """
        sitemap_urls = [
            f"{base_url}/sitemap.xml",
            f"{base_url}/sitemap_index.xml",
            f"{base_url}/post-sitemap.xml",
        ]

        all_urls = []

        for sitemap_url in sitemap_urls:
            try:
                page = await self.browser.new_page()
                await page.goto(sitemap_url, timeout=self.timeout_ms)

                content = await page.content()
                await page.close()

                # Parse XML
                soup = BeautifulSoup(content, "xml")
                loc_tags = soup.find_all("loc")

                urls = [loc.text.strip() for loc in loc_tags]

                # Filter for likely article URLs
                filtered = self._filter_article_urls(urls)
                all_urls.extend(filtered)

                if len(all_urls) >= 50:
                    break

            except Exception as e:
                logger.debug(f"Sitemap {sitemap_url} not accessible: {e}")
                continue

        return list(set(all_urls))[:100]  # Deduplicate and limit

    async def _crawl_article_section(self, base_url: str) -> List[str]:
        """
        Crawl blog/articles section to discover content.

        Args:
            base_url: Website base URL

        Returns:
            List of discovered article URLs
        """
        candidate_paths = ["/blog", "/articles", "/news", "/posts", "/resources"]

        all_urls = []

        for path in candidate_paths:
            try:
                url = urljoin(base_url, path)
                page = await self.browser.new_page()
                await page.goto(url, timeout=self.timeout_ms)

                # Extract all links
                links = await page.eval_on_selector_all(
                    "a[href]", "(elements) => elements.map(e => e.href)"
                )

                await page.close()

                # Filter for article URLs
                filtered = self._filter_article_urls(links)
                all_urls.extend(filtered)

                if len(all_urls) >= 50:
                    break

            except Exception as e:
                logger.debug(f"Failed to crawl {path}: {e}")
                continue

        return list(set(all_urls))[:100]

    @staticmethod
    def _filter_article_urls(urls: List[str]) -> List[str]:
        """
        Filter URLs to keep only likely article pages.

        Heuristics:
        - Has readable slug (no IDs or query params)
        - Path depth 2-4 levels
        - Excludes common non-content pages

        Args:
            urls: List of URLs to filter

        Returns:
            Filtered list of article URLs
        """
        filtered = []

        exclude_patterns = [
            r"/(tag|category|author|archive|page|wp-admin|wp-content)/",
            r"\.(jpg|jpeg|png|gif|pdf|zip|css|js)$",
            r"[?&]",  # Query parameters
        ]

        for url in urls:
            parsed = urlparse(url)
            path = parsed.path

            # Check exclusions
            if any(re.search(pattern, path, re.I) for pattern in exclude_patterns):
                continue

            # Check path depth (2-4 levels)
            depth = len([p for p in path.split("/") if p])
            if depth < 1 or depth > 4:
                continue

            # Check for readable slug
            if re.search(r"-|_", path):  # Has word separators
                filtered.append(url)

        return filtered

    # =========================================================================
    # CONTENT SCRAPING
    # =========================================================================

    async def _scrape_articles(self, urls: List[str]) -> List[Dict[str, str]]:
        """
        Scrape content from article URLs.

        Args:
            urls: List of article URLs

        Returns:
            List of dicts with 'url', 'title', 'content' keys
        """
        articles = []

        for url in urls:
            try:
                article = await self._scrape_single_article(url)
                if article:
                    articles.append(article)

                # Rate limiting
                await asyncio.sleep(1)

            except Exception as e:
                logger.warning(f"Failed to scrape {url}: {e}")
                continue

        return articles

    async def _scrape_single_article(self, url: str) -> Optional[Dict[str, str]]:
        """
        Scrape single article with content extraction.

        Args:
            url: Article URL

        Returns:
            Dict with article data or None if extraction fails
        """
        page = await self.browser.new_page()

        try:
            await page.goto(url, timeout=self.timeout_ms, wait_until="networkidle")

            # Extract title
            title = await page.title()

            # Extract main content (multiple strategies)
            content = await self._extract_main_content(page)

            if not content or len(content) < 200:
                logger.debug(f"Insufficient content extracted from {url}")
                return None

            return {
                "url": url,
                "title": title,
                "content": content,
            }

        finally:
            await page.close()

    async def _extract_main_content(self, page: Page) -> str:
        """
        Extract main article content, removing boilerplate.

        Strategies (in order):
        1. Look for semantic HTML5 tags (article, main)
        2. Find largest text block
        3. Use common CMS selectors

        Args:
            page: Playwright page object

        Returns:
            Extracted text content
        """
        # Strategy 1: Semantic HTML5
        selectors = [
            "article",
            "main",
            '[role="main"]',
            ".post-content",
            ".entry-content",
            ".article-content",
            "#content",
        ]

        for selector in selectors:
            try:
                element = await page.query_selector(selector)
                if element:
                    text = await element.inner_text()
                    if len(text) > 300:
                        return self._clean_text(text)
            except:
                continue

        # Strategy 2: Largest text block
        try:
            all_text_blocks = await page.eval_on_selector_all(
                "p",
                """(elements) => elements.map(e => ({
                    text: e.innerText,
                    length: e.innerText.length
                }))""",
            )

            # Find parent with most paragraph content
            body_text = await page.eval_on_selector(
                "body",
                """(body) => {
                    const paragraphs = Array.from(body.querySelectorAll('p'));
                    const containers = new Map();
                    
                    paragraphs.forEach(p => {
                        let parent = p.parentElement;
                        while (parent && parent !== body) {
                            const key = parent.tagName + (parent.className || '');
                            const current = containers.get(key) || { element: parent, length: 0 };
                            current.length += p.innerText.length;
                            containers.set(key, current);
                            parent = parent.parentElement;
                        }
                    });
                    
                    const best = Array.from(containers.values())
                        .sort((a, b) => b.length - a.length)[0];
                    
                    return best ? best.element.innerText : body.innerText;
                }""",
            )

            if body_text and len(body_text) > 300:
                return self._clean_text(body_text)

        except Exception as e:
            logger.debug(f"Fallback extraction failed: {e}")

        # Last resort: body text
        body_text = await page.eval_on_selector("body", "(body) => body.innerText")
        return self._clean_text(body_text)

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted text (remove excess whitespace, etc.).

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # Remove multiple spaces/newlines
        text = re.sub(r"\s+", " ", text)

        # Remove common boilerplate patterns
        text = re.sub(r"(Cookie Policy|Privacy Policy|Terms of Service).*$", "", text, flags=re.I)
        text = re.sub(r"^.*?(Share on|Follow us|Subscribe)", "", text, flags=re.I)

        return text.strip()

    # =========================================================================
    # PATTERN EXTRACTION & AGGREGATION
    # =========================================================================

    async def _extract_patterns(self, articles: List[Dict[str, str]]) -> Dict:
        """
        Extract linguistic patterns from article corpus.

        Aggregates patterns across all articles to compute:
        - Average sentence length (with std dev)
        - Lexical diversity
        - Readability scores
        - Tone embedding (semantic centroid)
        - Common structural patterns

        Args:
            articles: List of scraped articles

        Returns:
            Dict with aggregated pattern data
        """
        logger.info(f"Extracting patterns from {len(articles)} articles")

        # Extract features from each article
        all_features = []
        for article in articles:
            features = self.pattern_extractor.extract_features(article["content"])
            all_features.append(features)

        # Aggregate statistics
        sentence_lengths = [f["avg_sentence_length"] for f in all_features]
        avg_sentence_length = sum(sentence_lengths) / len(sentence_lengths)
        sentence_length_std = (
            sum((x - avg_sentence_length) ** 2 for x in sentence_lengths) / len(sentence_lengths)
        ) ** 0.5

        lexical_diversities = [f["lexical_diversity"] for f in all_features]
        avg_lexical_diversity = sum(lexical_diversities) / len(lexical_diversities)

        readability_scores = [f["readability_score"] for f in all_features]
        avg_readability = sum(readability_scores) / len(readability_scores)

        # Aggregate tone embedding (centroid)
        tone_embeddings = [f["tone_embedding"] for f in all_features]
        tone_centroid = self.pattern_extractor.compute_centroid(tone_embeddings)

        # Aggregate structural patterns
        structure_patterns = self._aggregate_structures(all_features)

        # Calculate confidence based on sample consistency
        confidence = self._calculate_confidence(all_features)

        return {
            "avg_sentence_length": avg_sentence_length,
            "sentence_length_std": sentence_length_std,
            "lexical_diversity": avg_lexical_diversity,
            "readability_score": avg_readability,
            "tone_embedding": tone_centroid,
            "structure_patterns": structure_patterns,
            "confidence": confidence,
            "sample_size": len(articles),
        }

    def _aggregate_structures(self, features_list: List[Dict]) -> List[StructurePattern]:
        """
        Aggregate structural patterns across articles.

        Identifies common patterns:
        - Listicles (numbered/bulleted lists)
        - How-to guides (instructional)
        - Problem-solution
        - Narrative/story

        Args:
            features_list: List of feature dicts

        Returns:
            List of StructurePattern objects with frequencies
        """
        pattern_counts = {}
        total = len(features_list)

        for features in features_list:
            detected_patterns = features.get("structure_patterns", [])
            for pattern_name in detected_patterns:
                pattern_counts[pattern_name] = pattern_counts.get(pattern_name, 0) + 1

        # Convert to StructurePattern objects
        structure_patterns = []
        for pattern_name, count in pattern_counts.items():
            frequency = count / total

            # Infer typical characteristics
            typical_sections, avg_word_count = self._infer_pattern_characteristics(
                pattern_name, features_list
            )

            structure_patterns.append(
                StructurePattern(
                    pattern_type=pattern_name,
                    frequency=frequency,
                    typical_sections=typical_sections,
                    avg_word_count=avg_word_count,
                )
            )

        return sorted(structure_patterns, key=lambda p: p.frequency, reverse=True)

    @staticmethod
    def _infer_pattern_characteristics(
        pattern_name: str, features_list: List[Dict]
    ) -> Tuple[List[str], int]:
        """
        Infer typical characteristics of a structure pattern.

        Args:
            pattern_name: Name of pattern
            features_list: List of all feature dicts

        Returns:
            Tuple of (typical_sections, avg_word_count)
        """
        # Default characteristics by pattern type
        defaults = {
            "listicle": (["Introduction", "List Items", "Conclusion"], 1200),
            "how-to": (
                ["Introduction", "Materials/Prerequisites", "Steps", "Tips", "Conclusion"],
                1500,
            ),
            "problem-solution": (["Problem Statement", "Analysis", "Solution", "Results"], 1800),
            "narrative": (["Setup", "Development", "Climax", "Resolution"], 1600),
            "comparison": (
                ["Introduction", "Option A", "Option B", "Comparison", "Recommendation"],
                2000,
            ),
        }

        return defaults.get(pattern_name, (["Introduction", "Body", "Conclusion"], 1500))

    @staticmethod
    def _calculate_confidence(features_list: List[Dict]) -> float:
        """
        Calculate confidence score based on sample consistency.

        High confidence = low variance across samples

        Args:
            features_list: List of feature dicts

        Returns:
            Confidence score (0-1)
        """
        if len(features_list) < 5:
            return 0.5  # Low confidence with small sample

        # Calculate coefficient of variation for key metrics
        metrics = ["avg_sentence_length", "lexical_diversity", "readability_score"]

        cv_scores = []
        for metric in metrics:
            values = [f[metric] for f in features_list]
            mean = sum(values) / len(values)
            std = (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
            cv = std / mean if mean > 0 else 1.0  # Coefficient of variation
            cv_scores.append(cv)

        avg_cv = sum(cv_scores) / len(cv_scores)

        # Convert to confidence (lower CV = higher confidence)
        # CV of 0.2 or less = high confidence
        confidence = max(0.5, min(1.0, 1.0 - (avg_cv / 0.4)))

        # Adjust for sample size
        sample_factor = min(1.0, len(features_list) / 15)

        return confidence * sample_factor

    # =========================================================================
    # PERSISTENCE
    # =========================================================================

    async def _store_patterns(
        self,
        project_id: UUID,
        patterns: Dict,
    ) -> InferredPatterns:
        """
        Store inferred patterns in database.

        Args:
            project_id: UUID of project
            patterns: Aggregated pattern data

        Returns:
            InferredPatterns model
        """
        try:
            # Delete existing patterns for project
            await self.session.execute(
                "DELETE FROM inferred_patterns WHERE project_id = :project_id",
                {"project_id": project_id},
            )

            # Insert new patterns
            query = """
                INSERT INTO inferred_patterns (
                    id, project_id, avg_sentence_length, sentence_length_std,
                    lexical_diversity, readability_score, tone_embedding,
                    structure_patterns, confidence, sample_size, analyzed_at
                ) VALUES (
                    :id, :project_id, :avg_sentence_length, :sentence_length_std,
                    :lexical_diversity, :readability_score, :tone_embedding,
                    :structure_patterns, :confidence, :sample_size, NOW()
                )
                RETURNING id, project_id, avg_sentence_length, sentence_length_std,
                          lexical_diversity, readability_score, confidence, 
                          sample_size, analyzed_at;
            """

            pattern_id = uuid4()

            # Convert structure patterns to JSON
            structure_json = [
                {
                    "pattern_type": p.pattern_type,
                    "frequency": p.frequency,
                    "typical_sections": p.typical_sections,
                    "avg_word_count": p.avg_word_count,
                }
                for p in patterns["structure_patterns"]
            ]

            result = await self.session.execute(
                query,
                {
                    "id": pattern_id,
                    "project_id": project_id,
                    "avg_sentence_length": patterns["avg_sentence_length"],
                    "sentence_length_std": patterns["sentence_length_std"],
                    "lexical_diversity": patterns["lexical_diversity"],
                    "readability_score": patterns["readability_score"],
                    "tone_embedding": f"[{','.join(map(str, patterns['tone_embedding']))}]",
                    "structure_patterns": str(structure_json).replace("'", '"'),
                    "confidence": patterns["confidence"],
                    "sample_size": patterns["sample_size"],
                },
            )

            await self.session.commit()

            row = result.fetchone()

            inferred = InferredPatterns(
                id=row[0],
                project_id=row[1],
                avg_sentence_length=row[2],
                sentence_length_std=row[3],
                lexical_diversity=row[4],
                readability_score=row[5],
                tone_embedding=patterns["tone_embedding"],
                structure_patterns=patterns["structure_patterns"],
                confidence=row[6],
                sample_size=row[7],
                analyzed_at=row[8],
            )

            logger.info(f"Stored inferred patterns (confidence: {patterns['confidence']:.2f})")
            return inferred

        except Exception as e:
            await self.session.rollback()
            logger.error(f"Failed to store patterns: {e}")
            raise DatabaseError(f"Pattern storage failed: {e}")

    async def _get_existing_patterns(self, project_id: UUID) -> Optional[InferredPatterns]:
        """Retrieve existing patterns if recent (< 30 days)."""
        query = """
            SELECT id, project_id, avg_sentence_length, sentence_length_std,
                   lexical_diversity, readability_score, confidence, 
                   sample_size, analyzed_at
            FROM inferred_patterns
            WHERE project_id = :project_id
            AND analyzed_at > NOW() - INTERVAL '30 days'
            ORDER BY analyzed_at DESC
            LIMIT 1;
        """

        result = await self.session.execute(query, {"project_id": project_id})
        row = result.fetchone()

        if not row:
            return None

        # Note: Not loading full tone_embedding or structure_patterns for efficiency
        return InferredPatterns(
            id=row[0],
            project_id=row[1],
            avg_sentence_length=row[2],
            sentence_length_std=row[3],
            lexical_diversity=row[4],
            readability_score=row[5],
            tone_embedding=[],  # Load separately if needed
            structure_patterns=[],
            confidence=row[6],
            sample_size=row[7],
            analyzed_at=row[8],
        )

    # =========================================================================
    # BROWSER MANAGEMENT
    # =========================================================================

    async def _init_browser(self) -> None:
        """Initialize Playwright browser instance."""
        if self.browser is None:
            playwright = await async_playwright().start()
            self.browser = await playwright.chromium.launch(headless=True)
            logger.debug("Browser initialized")

    async def _close_browser(self) -> None:
        """Close browser instance."""
        if self.browser:
            await self.browser.close()
            self.browser = None
        logger.debug("Browser closed")


# =========================================================================
# UTILITY FUNCTIONS
# =========================================================================
def normalize_domain(domain: str) -> str:
    """
    Normalize domain to consistent format.
    Args:
        domain: Raw domain input

    Returns:
        Normalized domain (lowercase, no protocol)
    """
    domain = domain.lower().strip()
    domain = re.sub(r"^https?://", "", domain)
    domain = re.sub(r"/.*$", "", domain)
    return domain
