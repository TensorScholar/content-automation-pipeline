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

import httpx
from bs4 import BeautifulSoup
from loguru import logger
from sqlalchemy.ext.asyncio import AsyncSession

from config.settings import ScrapingSettings
from core.exceptions import ScrapingError, ValidationError
from core.models import InferredPatterns, StructurePattern
from knowledge.pattern_extractor import PatternExtractor
from knowledge.project_repository import ProjectRepository


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

    def __init__(
        self,
        pattern_extractor: PatternExtractor,
        project_repository: ProjectRepository,
        scraping_settings: ScrapingSettings,
    ):
        """
        Initialize analyzer with dependencies.

        Args:
            pattern_extractor: Linguistic pattern extraction engine
            project_repository: Repository for saving inferred patterns
            scraping_settings: Configuration for scraping behavior
        """
        self.pattern_extractor = pattern_extractor
        self.project_repository = project_repository
        self.scraping_settings = scraping_settings

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
                existing = await self.project_repository.get_inferred_patterns(project_id)
                if existing:
                    logger.info(f"Using existing patterns (analyzed {existing.analyzed_at})")
                    return existing

            # Step 1: Discover article URLs
            article_urls = await self._discover_articles(domain)
            if len(article_urls) < 5:
                raise ScrapingError(f"Insufficient articles found: {len(article_urls)} (minimum 5)")

            # Step 2: Scrape representative sample
            sample_urls = article_urls[: self.scraping_settings.max_article_sample_size]
            articles = await self._scrape_articles(sample_urls)

            if len(articles) < 5:
                raise ScrapingError(f"Failed to scrape minimum articles: {len(articles)}/5")

            # Step 3: Extract patterns
            patterns = await self._extract_patterns(articles)

            # Step 4: Store in database
            inferred = await self.project_repository.save_inferred_patterns(project_id, patterns)

            logger.info(f"Analysis complete: {len(articles)} articles analyzed")
            return inferred

        except Exception as e:
            logger.error(f"Website analysis failed for {domain}: {e}")
            raise ScrapingError(f"Analysis failed: {e}")

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

        async with httpx.AsyncClient(
            timeout=self.scraping_settings.request_timeout,
            headers={"User-Agent": self.scraping_settings.user_agent},
        ) as client:
            for sitemap_url in sitemap_urls:
                try:
                    response = await client.get(sitemap_url)
                    response.raise_for_status()

                    # Parse XML
                    soup = BeautifulSoup(response.text, "xml")
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

        async with httpx.AsyncClient(
            timeout=self.scraping_settings.request_timeout,
            headers={"User-Agent": self.scraping_settings.user_agent},
        ) as client:
            for path in candidate_paths:
                try:
                    url = urljoin(base_url, path)
                    response = await client.get(url)
                    response.raise_for_status()

                    # Parse HTML and extract links
                    soup = BeautifulSoup(response.text, "html.parser")
                    links = [a.get("href") for a in soup.find_all("a", href=True)]

                    # Convert relative URLs to absolute
                    absolute_links = [urljoin(url, link) for link in links]

                    # Filter for article URLs
                    filtered = self._filter_article_urls(absolute_links)
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

        async with httpx.AsyncClient(
            timeout=self.scraping_settings.request_timeout,
            headers={"User-Agent": self.scraping_settings.user_agent},
        ) as client:
            for url in urls:
                try:
                    article = await self._scrape_single_article(client, url)
                    if article:
                        articles.append(article)

                    # Rate limiting
                    await asyncio.sleep(self.scraping_settings.min_delay_between_requests)

                except Exception as e:
                    logger.warning(f"Failed to scrape {url}: {e}")
                    continue

        return articles

    async def _scrape_single_article(
        self, client: httpx.AsyncClient, url: str
    ) -> Optional[Dict[str, str]]:
        """
        Scrape single article with content extraction.

        Args:
            client: httpx client instance
            url: Article URL

        Returns:
            Dict with article data or None if extraction fails
        """
        try:
            response = await client.get(url)
            response.raise_for_status()

            # Parse HTML
            soup = BeautifulSoup(response.text, "html.parser")

            # Extract title
            title_tag = soup.find("title")
            title = title_tag.get_text().strip() if title_tag else "Untitled"

            # Extract main content
            content = self._extract_main_content(soup)

            if not content or len(content) < self.scraping_settings.min_article_word_count:
                logger.debug(f"Insufficient content extracted from {url}")
                return None

            return {
                "url": url,
                "title": title,
                "content": content,
            }

        except Exception as e:
            logger.debug(f"Failed to scrape {url}: {e}")
            return None

    def _extract_main_content(self, soup: BeautifulSoup) -> str:
        """
        Extract main article content, removing boilerplate.

        Strategies (in order):
        1. Look for semantic HTML5 tags (article, main)
        2. Find largest text block
        3. Use common CMS selectors

        Args:
            soup: BeautifulSoup parsed HTML

        Returns:
            Extracted text content
        """
        # Remove common boilerplate sections before extraction
        for tag in soup(
            [
                "nav",
                "footer",
                "header",
                "aside",
                "script",
                "style",
                ".sidebar",
                ".menu",
                ".related-posts",
                ".comments",
            ]
        ):
            try:
                tag.decompose()
            except Exception:
                pass  # Ignore if tag not found

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
                element = soup.select_one(selector)
                if element:
                    text = element.get_text()
                    if len(text) > 300:
                        return self._clean_text(text)
            except Exception:  # nosec B112 - Continue on any selector failure
                continue

        # Strategy 2: Largest text block (find div with most paragraphs)
        paragraphs = soup.find_all("p")
        if paragraphs:
            # Find parent with most paragraph content
            parent_counts = {}
            for p in paragraphs:
                parent = p.parent
                if parent:
                    parent_key = f"{parent.name}_{parent.get('class', [])}"
                    parent_counts[parent_key] = parent_counts.get(parent_key, 0) + 1

            if parent_counts:
                best_parent_key = max(parent_counts, key=parent_counts.get)
                # Find the actual parent element
                for p in paragraphs:
                    parent = p.parent
                    if parent and f"{parent.name}_{parent.get('class', [])}" == best_parent_key:
                        text = parent.get_text()
                        if len(text) > 300:
                            return self._clean_text(text)

        return ""

    @staticmethod
    def _clean_text(text: str) -> str:
        """
        Clean extracted text (remove excess whitespace, etc.).

        Args:
            text: Raw extracted text

        Returns:
            Cleaned text
        """
        # 1. Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        # 2. Remove common boilerplate patterns (case-insensitive, multiline)
        # This removes "Cookie Policy..." and "Copyright..." etc.
        boilerplate_patterns = [
            r"cookie policy",
            r"privacy policy",
            r"terms of service",
            r"all rights reserved",
            r"copyright ©",
            r"share on",
            r"follow us",
            r"subscribe to our newsletter",
            r"related posts",
            r"leave a comment",
            r"posted on",
            r"by author",
        ]
        # Build a single regex to find and remove these phrases and nearby text
        pattern = r"(?i)(" + "|".join(boilerplate_patterns) + r").*?($|\n)"
        text = re.sub(pattern, "", text, flags=re.MULTILINE)

        # 3. Remove excess newlines (which became spaces) again
        text = re.sub(r"\s+", " ", text).strip()

        return text

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
            features = self.pattern_extractor.extract_patterns(article["content"])
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
    # UTILITY FUNCTIONS
    # =========================================================================

    @staticmethod
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
