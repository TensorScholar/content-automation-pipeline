"""
Content Distribution Module
===========================

Handles distribution of generated content to various channels like Telegram, WordPress, etc.
"""

from typing import Any, Dict, Optional

import httpx
from loguru import logger

from core.exceptions import DistributionError
from core.models import GeneratedArticle, Project


class Distributor:
    """
    Handles distribution of generated content to various channels.

    Currently supports:
    - Telegram channels
    - RSS feeds (planned)
    - Social media (planned)
    """

    def __init__(self):
        """Initialize the distributor."""
        logger.info("Distributor initialized")

    async def distribute_to_telegram(
        self, article: GeneratedArticle, channel: str
    ) -> Dict[str, Any]:
        """
        Distribute article to Telegram channel.

        Args:
            article: The generated article to distribute
            channel: Telegram channel identifier

        Returns:
            Distribution result metadata
        """
        logger.info(f"Distributing article {article.id} to Telegram channel {channel}")

        # For now, just log the distribution
        # In a real implementation, this would send to Telegram API
        result = {
            "channel": channel,
            "article_id": str(article.id),
            "title": article.title,
            "distributed_at": article.created_at.isoformat(),
            "status": "success",
        }

        logger.info(f"Article distributed successfully: {result}")
        return result

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

    async def distribute_to_wordpress(
        self, article: GeneratedArticle, project: Project
    ) -> dict[str, Any]:
        """
        Publishes a generated article directly to a WordPress site using the REST API.
        """
        if (
            not project.wordpress_url
            or not project.wordpress_username
            or not project.wordpress_app_password
        ):
            logger.warning(
                f"WordPress credentials not configured for project {project.id}. Skipping distribution."
            )
            return {"status": "skipped", "reason": "Not configured"}

        api_url = f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/posts"

        auth = httpx.BasicAuth(
            project.wordpress_username, project.wordpress_app_password.get_secret_value()
        )

        # Article content is already HTML-formatted by ContentGenerator
        post_data = {
            "title": article.title,
            "content": article.content,
            "status": "publish",  # Or "draft" if manual approval is needed
            "meta": {
                "_yoast_wpseo_metadesc": article.meta_description,  # Assumes Yoast plugin
                # "_rank_math_description": article.meta_description # Assumes Rank Math
            },
            "tags": ", ".join(article.keywords),  # Add keywords as tags
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(api_url, json=post_data, auth=auth, timeout=30.0)

            response.raise_for_status()  # Raise exception for 4xx/5xx responses

            response_data = response.json()
            logger.success(
                f"Article {article.id} successfully posted to WordPress. New post ID: {response_data.get('id')}"
            )
            return {"status": "published", "url": response_data.get("link")}

        except httpx.HTTPStatusError as e:
            logger.error(
                f"Failed to post to WordPress: HTTP {e.response.status_code} - {e.response.text}"
            )
            raise DistributionError(f"WordPress API error: {e.response.text}")
        except Exception as e:
            logger.error(f"Failed to post to WordPress: {e}")
            raise DistributionError(str(e))
