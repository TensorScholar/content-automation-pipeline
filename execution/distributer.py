"""
Content Distribution Module
===========================

Handles distribution of generated content to various channels like Telegram, WordPress, etc.
"""

from datetime import datetime as _dt
from typing import Any, Dict, Optional

import httpx
from loguru import logger

from core.exceptions import DistributionError
from core.models import GeneratedArticle, Project
from infrastructure.credential_encryption import decrypt_credential


class Distributor:
    """
    Handles distribution of generated content to various channels.

    Currently supports:
    - WordPress (with Gutenberg blocks and schema.org)
    - RSS feeds (planned)
    - Social media (planned)
    """

    def __init__(self, max_retries: int = 5, initial_retry_delay: float = 2.0):
        """Initialize the distributor with configurable retry parameters.

        Args:
            max_retries: Maximum number of retry attempts (default: 5)
            initial_retry_delay: Initial delay in seconds before retry (default: 2.0)
        """
        self.max_retries = max_retries
        self.retry_delay = initial_retry_delay
        logger.info(
            f"Distributor initialized | max_retries={max_retries} | retry_delay={initial_retry_delay}s"
        )

    @staticmethod
    def _wordpress_password(project: Project) -> str:
        """Decrypt the stored credential only at the outbound WordPress boundary."""
        from config.settings import get_settings

        password = decrypt_credential(
            project.wordpress_app_password,
            get_settings().credential_encryption_key,
        )
        if not password:
            raise DistributionError("WordPress credentials not configured")
        return password

    def convert_to_gutenberg_blocks(self, html_content: str) -> str:
        """
        Convert HTML content to WordPress Gutenberg block format.

        M-5 fix: Replaced DOTALL regex (which corrupts nested tags like <h2><strong>Title</strong></h2>)
        with Python's html.parser for correct, safe tag-aware transformation.
        """
        from html.parser import HTMLParser

        class GutenbergParser(HTMLParser):
            """SAX-style HTML parser that emits Gutenberg block wrappers."""
            BLOCK_MAP = {
                "h2": ('<!-- wp:heading {"level":2} -->', '<!-- /wp:heading -->'),
                "h3": ('<!-- wp:heading {"level":3} -->', '<!-- /wp:heading -->'),
                "h4": ('<!-- wp:heading {"level":4} -->', '<!-- /wp:heading -->'),
                "p":  ('<!-- wp:paragraph -->', '<!-- /wp:paragraph -->'),
                "ul": ('<!-- wp:list -->', '<!-- /wp:list -->'),
                "ol": ('<!-- wp:list {"ordered":true} -->', '<!-- /wp:list -->'),
                "blockquote": ('<!-- wp:quote -->', '<!-- /wp:quote -->'),
            }

            def __init__(self):
                super().__init__()
                self._out: list[str] = []
                self._stack: list[str] = []

            def handle_starttag(self, tag, attrs):
                attr_str = "".join(
                    f' {k}="{v}"' if v else f" {k}" for k, v in attrs
                )
                raw = f"<{tag}{attr_str}>"
                if tag in self.BLOCK_MAP:
                    self._out.append(self.BLOCK_MAP[tag][0] + "\n")
                    self._stack.append(tag)
                self._out.append(raw)

            def handle_endtag(self, tag):
                self._out.append(f"</{tag}>")
                if self._stack and self._stack[-1] == tag and tag in self.BLOCK_MAP:
                    self._stack.pop()
                    self._out.append("\n" + self.BLOCK_MAP[tag][1] + "\n")

            def handle_data(self, data):
                self._out.append(data)

            def handle_entityref(self, name):
                self._out.append(f"&{name};")

            def handle_charref(self, name):
                self._out.append(f"&#{name};")

            @property
            def result(self) -> str:
                return "".join(self._out)

        parser = GutenbergParser()
        parser.feed(html_content)
        result = parser.result
        logger.debug("Converted HTML to Gutenberg blocks via html.parser")
        return result

    def generate_schema_markup(self, article: "GeneratedArticle", article_url: str = "") -> str:
        """
        Generate schema.org JSON-LD structured data for SEO.

        Creates Article schema and FAQPage schema if FAQ section detected.

        Args:
            article: Generated article with metadata
            article_url: Published URL (if available)

        Returns:
            JSON-LD script tag for insertion in <head> or article
        """
        import json

        # Base Article schema
        schema = {
            "@context": "https://schema.org",
            "@type": "Article",
            "headline": article.title,
            "description": article.meta_description,
            "datePublished": article.created_at.isoformat(),
            "dateModified": article.updated_at.isoformat()
            if article.updated_at
            else article.created_at.isoformat(),
            "author": {"@type": "Organization", "name": "Smarlux Studio"},
            "publisher": {"@type": "Organization", "name": "Smarlux Studio"},
            "keywords": ", ".join(article.keywords) if article.keywords else "",
            "wordCount": article.quality_metrics.word_count if article.quality_metrics else 0,
        }

        if article_url:
            schema["url"] = article_url
            schema["mainEntityOfPage"] = {"@type": "WebPage", "@id": article_url}

        # Check for FAQ section and add FAQPage schema
        content_lower = article.content.lower() if article.content else ""
        if (
            "سوالات متداول" in content_lower
            or "faq" in content_lower
            or "frequently asked" in content_lower
        ):
            # Add FAQ indicator - actual FAQ extraction would require parsing
            schema["@type"] = ["Article", "FAQPage"]

        json_ld = f'<script type="application/ld+json">\n{json.dumps(schema, ensure_ascii=False, indent=2)}\n</script>'

        logger.debug(f"Generated schema.org markup: {schema['@type']}")
        return json_ld

    async def distribute_to_rss(self, article: GeneratedArticle, feed_url: str) -> Dict[str, Any]:
        """
        Distribute article to RSS feed.

        Args:
            article: The generated article to distribute
            feed_url: RSS feed URL

        Returns:
            Distribution result metadata
        """
        logger.info(f"Distributing article {article.id} to RSS feed {feed_url}")

        # For now, just log the distribution
        result = {
            "feed_url": feed_url,
            "article_id": str(article.id),
            "title": article.title,
            "distributed_at": article.created_at.isoformat(),
            "status": "success",
        }

        logger.info(f"Article distributed to RSS successfully: {result}")
        return result

    async def distribute_to_social_media(
        self, article: GeneratedArticle, platforms: list[str]
    ) -> Dict[str, Any]:
        """
        Distribute article to social media platforms.

        Args:
            article: The generated article to distribute
            platforms: List of social media platforms

        Returns:
            Distribution result metadata
        """
        logger.info(f"Distributing article {article.id} to social media platforms: {platforms}")

        # For now, just log the distribution
        result = {
            "platforms": platforms,
            "article_id": str(article.id),
            "title": article.title,
            "distributed_at": article.created_at.isoformat(),
            "status": "success",
        }

        logger.info(f"Article distributed to social media successfully: {result}")
        return result

    async def validate_wordpress_connection(self, project: Project) -> tuple[bool, str]:
        """
        Validate WordPress credentials before attempting distribution.

        Args:
            project: Project with WordPress configuration

        Returns:
            Tuple of (is_valid, error_message)
        """
        if (
            not project.wordpress_url
            or not project.wordpress_username
            or not project.wordpress_app_password
        ):
            return False, "WordPress credentials not configured"

        try:
            auth = httpx.BasicAuth(
                project.wordpress_username,
                self._wordpress_password(project),
            )
            # Test connection with a simple GET request to posts endpoint
            test_url = f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/posts?per_page=1"

            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(test_url, auth=auth)

            if response.status_code == 200:
                return True, ""
            elif response.status_code == 401:
                return False, "Invalid WordPress credentials"
            elif response.status_code == 404:
                return False, "WordPress REST API not found - check URL"
            else:
                return False, f"WordPress connection failed: HTTP {response.status_code}"

        except httpx.TimeoutException:
            logger.warning(f"WordPress connection timeout for {project.wordpress_url}")
            return False, "WordPress connection timeout"
        except httpx.ConnectError as e:
            logger.error(f"WordPress connection failed: Cannot connect to {project.wordpress_url}")
            return False, f"Cannot connect to WordPress: {str(e)}"
        except Exception as e:
            logger.error(f"WordPress validation error: {str(e)}")
            return False, f"WordPress connection error: {str(e)}"

    async def _resolve_tag_ids(
        self, tag_names: list[str], project: Project, auth: httpx.BasicAuth
    ) -> list[int]:
        """Resolve tag names to WordPress tag IDs, creating missing tags as needed."""
        if not tag_names:
            return []

        tag_ids = []
        tags_url = f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/tags"

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                # Fetch existing tags
                resp = await client.get(f"{tags_url}?per_page=100", auth=auth)
                existing = (
                    {t["name"].lower(): t["id"] for t in resp.json()}
                    if resp.status_code == 200
                    else {}
                )

                for name in tag_names:
                    tid = existing.get(name.lower())
                    if tid:
                        tag_ids.append(tid)
                    else:
                        # Create missing tag
                        create_resp = await client.post(tags_url, json={"name": name}, auth=auth)
                        if create_resp.status_code == 201:
                            tag_ids.append(create_resp.json()["id"])
                        else:
                            logger.warning(
                                f"Failed to create tag '{name}': {create_resp.status_code}"
                            )
        except Exception as e:
            logger.warning(f"Tag resolution failed, skipping tags: {e}")

        return tag_ids

    async def distribute_to_wordpress(
        self, article: GeneratedArticle, project: Project, post_status: str = "draft"
    ) -> dict[str, Any]:
        """
        Publishes a generated article to a WordPress site using the REST API.

        Features:
        - Pre-flight validation of WordPress credentials
        - Automatic retry on transient failures (up to 5 attempts)
        - Converts HTML to Gutenberg block format for modern themes
        - Injects schema.org JSON-LD for SEO rich results
        - Sends meta description to Yoast SEO plugin
        - post_status controls whether to upload as 'draft' or 'publish' live

        Args:
            article: The generated article object
            project: The project with WordPress credentials
            post_status: WordPress post status — 'draft' (default) or 'publish'

        Returns:
            dict with status="published" on success or status="error" on failure
        """
        logger.info(
            f"Starting WordPress upload | article_id={article.id} | "
            f"project_id={project.id} | wp_url={project.wordpress_url} | "
            f"post_status={post_status}"
        )

        # Pre-flight validation
        is_valid, error_msg = await self.validate_wordpress_connection(project)
        if not is_valid:
            logger.error(
                f"WordPress validation failed | article_id={article.id} | "
                f"project_id={project.id} | error={error_msg}"
            )
            return {"status": "error", "reason": error_msg}

        api_url = f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/posts"

        auth = httpx.BasicAuth(project.wordpress_username, self._wordpress_password(project))

        # Convert content to Gutenberg blocks for modern WordPress
        gutenberg_content = self.convert_to_gutenberg_blocks(article.content)

        # Resolve keyword strings to WordPress tag IDs
        tag_ids = await self._resolve_tag_ids(
            article.keywords if article.keywords else [], project, auth
        )

        post_data = {
            "title": article.title,
            "content": gutenberg_content,
            "status": post_status,  # AP-1: controlled by caller, defaults to 'draft'
            "meta": {
                "_yoast_wpseo_metadesc": article.meta_description or "",
            },
            "tags": tag_ids,
        }

        # Retry logic with exponential backoff and jitter
        import asyncio
        import random

        last_error = None

        for attempt in range(self.max_retries):
            try:
                # Use connection pooling with longer timeout for large content
                timeout_config = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)
                async with httpx.AsyncClient(
                    timeout=timeout_config,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    follow_redirects=True,
                ) as client:
                    response = await client.post(api_url, json=post_data, auth=auth)

                response.raise_for_status()

                response_data = response.json()
                post_url = response_data.get("link", "")
                post_id = response_data.get("id")

                # Generate and save schema markup
                schema_markup = self.generate_schema_markup(article, post_url)

                try:
                    update_url = (
                        f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/posts/{post_id}"
                    )
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        await client.post(
                            update_url, json={"meta": {"_schema_json_ld": schema_markup}}, auth=auth
                        )
                    logger.debug(f"Schema markup saved to post {post_id}")
                except Exception as e:
                    logger.warning(f"Could not save schema markup: {e}")

                logger.success(
                    f"WordPress upload SUCCESS | article_id={article.id} | "
                    f"project_id={project.id} | post_id={post_id} | "
                    f"url={post_url} | attempts={attempt + 1} | status={post_status}"
                )

                # H-6 fix: Structured audit log for every WP publish event
                # In production, write this dict to a distribution_log DB table.
                distribution_audit = {
                    "event": "wordpress_publish",
                    "article_id": str(article.id),
                    "project_id": str(project.id),
                    "wp_post_id": post_id,
                    "wp_url": post_url,
                    "post_status": post_status,
                    "attempts": attempt + 1,
                    "timestamp": _dt.utcnow().isoformat() + "Z",
                }
                logger.info(f"DISTRIBUTION_AUDIT | {distribution_audit}")

                # L-4 fix: Structured metric for live vs draft ratio telemetry.
                # Replace with prometheus_client.Counter increment in production:
                #   wordpress_publish_total.labels(status=post_status).inc()
                logger.info(
                    f"METRIC | wordpress_publish_total | status={post_status} | "
                    f"project_id={project.id}"
                )

                return {
                    "status": "published",
                    "url": post_url,
                    "post_id": post_id,
                    "gutenberg_formatted": True,
                    "schema_generated": True,
                    "attempts": attempt + 1,
                    "post_status": post_status,
                }

            except httpx.HTTPStatusError as e:
                last_error = e
                # Don't retry on 4xx client errors (except 429 rate limit)
                if 400 <= e.response.status_code < 500 and e.response.status_code != 429:
                    logger.error(
                        f"WordPress API client error (NO RETRY) | article_id={article.id} | "
                        f"status_code={e.response.status_code} | error={e.response.text}"
                    )
                    raise DistributionError(f"WordPress API error: {e.response.text}")

                logger.warning(
                    f"WordPress HTTP error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | "
                    f"status_code={e.response.status_code} | error={str(e)}"
                )

                if attempt < self.max_retries - 1:
                    # Exponential backoff with jitter to prevent thundering herd
                    # Cap at 10 seconds to prevent unbounded growth
                    base_wait = min(self.retry_delay * (2**attempt), 10.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    logger.info(
                        f"Retry scheduled | article_id={article.id} | "
                        f"wait={wait_time:.2f}s | next_attempt={attempt + 2}/{self.max_retries}"
                    )
                    await asyncio.sleep(wait_time)

            except httpx.NetworkError as e:
                last_error = e
                logger.warning(
                    f"WordPress network error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | "
                    f"error_type={type(e).__name__} | error={str(e)}"
                )

                if attempt < self.max_retries - 1:
                    # Cap backoff at 10 seconds
                    base_wait = min(self.retry_delay * (2**attempt), 10.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    logger.info(
                        f"Retry after network error | article_id={article.id} | "
                        f"wait={wait_time:.2f}s | next_attempt={attempt + 2}/{self.max_retries}"
                    )
                    await asyncio.sleep(wait_time)

            except Exception as e:
                last_error = e
                logger.error(
                    f"Unexpected WordPress error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | "
                    f"error_type={type(e).__name__} | error={str(e)}",
                    exc_info=True,
                )

                if attempt < self.max_retries - 1:
                    # Cap backoff at 10 seconds
                    base_wait = min(self.retry_delay * (2**attempt), 10.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    wait_time = base_wait + jitter
                    logger.info(
                        f"Retry after unexpected error | article_id={article.id} | "
                        f"wait={wait_time:.2f}s | next_attempt={attempt + 2}/{self.max_retries}"
                    )
                    await asyncio.sleep(wait_time)

        # All retries exhausted
        logger.error(
            f"WordPress upload FAILED - retries exhausted | article_id={article.id} | "
            f"project_id={project.id} | max_retries={self.max_retries} | "
            f"last_error_type={type(last_error).__name__} | last_error={str(last_error)}"
        )
        raise DistributionError(
            f"WordPress distribution failed after {self.max_retries} attempts: {str(last_error)}"
        )
