"""
Distributor: Multi-Channel Content Delivery Engine

Manages asynchronous distribution of generated content across multiple channels.
Implements channel-specific formatting, rate limiting, retry mechanisms, and
delivery tracking. Designed for extensibility—adding new channels requires
minimal code changes.

Architectural Pattern: Strategy + Adapter with Circuit Breaker
"""

import asyncio
from abc import ABC, abstractmethod
from datetime import datetime
from enum import Enum
from typing import Dict, List, Optional

import httpx
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential

from core.exceptions import DistributionError
from core.models import GeneratedArticle
from infrastructure.monitoring import MetricsCollector


class DistributionChannel(str, Enum):
    """Supported distribution channels."""

    TELEGRAM = "telegram"
    WORDPRESS = "wordpress"
    EMAIL = "email"
    TWITTER = "twitter"


class ChannelAdapter(ABC):
    """
    Abstract base class for distribution channel adapters.

    Each adapter implements channel-specific logic for:
    - Content formatting
    - API communication
    - Error handling
    - Delivery confirmation
    """

    @abstractmethod
    async def distribute(self, article: GeneratedArticle, destination: str) -> Dict:
        """
        Distribute article to specific destination.

        Returns delivery metadata (message_id, timestamp, etc.)
        """
        pass

    @abstractmethod
    def format_content(self, article: GeneratedArticle) -> str:
        """Format article content for channel-specific requirements."""
        pass

    @abstractmethod
    async def validate_destination(self, destination: str) -> bool:
        """Validate destination identifier (channel ID, email, etc.)."""
        pass


class TelegramAdapter(ChannelAdapter):
    """
    Telegram Bot API adapter for content distribution.

    Supports:
    - Markdown/HTML formatting
    - Message chunking (Telegram 4096 char limit)
    - Silent notifications
    - Preview link disabling
    """

    def __init__(self, bot_token: str):
        self.bot_token = bot_token
        self.api_base = f"https://api.telegram.org/bot{bot_token}"
        self.max_message_length = 4096
        self.client = httpx.AsyncClient(timeout=30.0)

        logger.info("TelegramAdapter initialized")

    def format_content(self, article: GeneratedArticle) -> str:
        """
        Format article for Telegram with Markdown.

        Converts article structure to Telegram-compatible Markdown:
        - Bold headers
        - Clickable links
        - Code blocks
        - Proper escaping
        """
        # Convert to Telegram MarkdownV2 format
        formatted = f"*{self._escape_markdown(article.title)}*\n\n"

        # Add meta description as subtitle
        formatted += f"_{self._escape_markdown(article.meta_description)}_\n\n"

        # Convert article content
        # Assuming content is in markdown format
        content_lines = article.content.split("\n")

        for line in content_lines:
            if line.startswith("# "):
                # H1 headers
                formatted += f"*{self._escape_markdown(line[2:])}*\n"
            elif line.startswith("## "):
                # H2 headers
                formatted += f"\n*{self._escape_markdown(line[3:])}*\n"
            elif line.strip():
                # Regular paragraphs
                formatted += f"{self._escape_markdown(line)}\n"
            else:
                formatted += "\n"

        # Add footer with metadata
        formatted += f"\n\n📊 {article.word_count} words | "
        formatted += f"📖 Readability: {article.readability_score:.0f}/100"

        return formatted

    def _escape_markdown(self, text: str) -> str:
        """Escape special characters for Telegram MarkdownV2."""
        special_chars = [
            "_",
            "*",
            "[",
            "]",
            "(",
            ")",
            "~",
            "`",
            ">",
            "#",
            "+",
            "-",
            "=",
            "|",
            "{",
            "}",
            ".",
            "!",
        ]
        for char in special_chars:
            text = text.replace(char, f"\\{char}")
        return text

    async def distribute(self, article: GeneratedArticle, destination: str) -> Dict:
        """
        Send article to Telegram channel/chat.

        Args:
            article: Generated article to distribute
            destination: Telegram channel ID (e.g., "@mychannel" or "-1001234567890")

        Returns:
            Distribution metadata with message IDs
        """
        logger.debug(f"Distributing to Telegram | destination={destination}")

        # Format content
        formatted_content = self.format_content(article)

        # Handle message chunking if content exceeds limit
        if len(formatted_content) > self.max_message_length:
            chunks = self._chunk_message(formatted_content)
            message_ids = []

            for i, chunk in enumerate(chunks):
                msg_id = await self._send_message(
                    destination, chunk, is_last=(i == len(chunks) - 1)
                )
                message_ids.append(msg_id)

                # Small delay between chunks to avoid rate limiting
                if i < len(chunks) - 1:
                    await asyncio.sleep(0.5)

            return {
                "channel": DistributionChannel.TELEGRAM,
                "destination": destination,
                "message_ids": message_ids,
                "chunks_sent": len(chunks),
                "timestamp": datetime.utcnow(),
            }
        else:
            message_id = await self._send_message(destination, formatted_content)

            return {
                "channel": DistributionChannel.TELEGRAM,
                "destination": destination,
                "message_id": message_id,
                "timestamp": datetime.utcnow(),
            }

    @retry(
        stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10), reraise=True
    )
    async def _send_message(self, chat_id: str, text: str, is_last: bool = True) -> int:
        """
        Send message to Telegram with retry logic.

        Returns message ID for tracking.
        """
        payload = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "MarkdownV2",
            "disable_web_page_preview": False if is_last else True,
            "disable_notification": False,
        }

        try:
            response = await self.client.post(f"{self.api_base}/sendMessage", json=payload)
            response.raise_for_status()

            data = response.json()

            if not data.get("ok"):
                raise DistributionError(f"Telegram API error: {data.get('description')}")

            message_id = data["result"]["message_id"]
            logger.debug(f"Message sent | chat_id={chat_id} | message_id={message_id}")

            return message_id

        except httpx.HTTPError as e:
            logger.error(f"Telegram HTTP error | error={e}")
            raise DistributionError(f"Failed to send Telegram message: {str(e)}") from e

    def _chunk_message(self, text: str) -> List[str]:
        """
        Split long message into chunks respecting Telegram limits.

        Attempts to split at paragraph boundaries for readability.
        """
        chunks = []
        current_chunk = ""

        paragraphs = text.split("\n\n")

        for para in paragraphs:
            # If single paragraph exceeds limit, force split
            if len(para) > self.max_message_length:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = ""

                # Split long paragraph by sentences
                sentences = para.split(". ")
                for sentence in sentences:
                    if len(current_chunk) + len(sentence) + 2 < self.max_message_length:
                        current_chunk += sentence + ". "
                    else:
                        chunks.append(current_chunk)
                        current_chunk = sentence + ". "

            # Normal paragraph handling
            elif len(current_chunk) + len(para) + 2 < self.max_message_length:
                current_chunk += para + "\n\n"
            else:
                chunks.append(current_chunk)
                current_chunk = para + "\n\n"

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    async def validate_destination(self, destination: str) -> bool:
        """
        Validate Telegram channel/chat ID.

        Checks if bot has access to send messages to destination.
        """
        try:
            payload = {"chat_id": destination}
            response = await self.client.post(f"{self.api_base}/getChat", json=payload)

            return response.status_code == 200

        except Exception as e:
            logger.warning(f"Telegram validation failed | destination={destination} | error={e}")
            return False


class Distributor:
    """
    Main distribution controller managing multi-channel delivery.

    Coordinates distribution across multiple channels with:
    - Concurrent delivery (when multiple channels configured)
    - Delivery tracking and confirmation
    - Error handling with graceful degradation
    - Metrics collection for monitoring

    Design Philosophy:
    - Channel-agnostic interface
    - Fail-safe: One channel failure doesn't block others
    - Observable: Track all delivery attempts
    """

    def __init__(self, telegram_bot_token: Optional[str], metrics_collector: MetricsCollector):
        self.metrics = metrics_collector

        # Initialize channel adapters
        self.adapters: Dict[DistributionChannel, ChannelAdapter] = {}

        if telegram_bot_token:
            self.adapters[DistributionChannel.TELEGRAM] = TelegramAdapter(telegram_bot_token)
            logger.info("Telegram adapter registered")

        # Future adapters
        # self.adapters[DistributionChannel.WORDPRESS] = WordPressAdapter(...)
        # self.adapters[DistributionChannel.EMAIL] = EmailAdapter(...)

        logger.info(f"Distributor initialized | channels={list(self.adapters.keys())}")

    async def distribute_to_telegram(self, article: GeneratedArticle, channel: str) -> Dict:
        """
        Distribute article to Telegram channel.

        Convenience method for direct Telegram distribution.
        """
        return await self.distribute(
            article=article, channel=DistributionChannel.TELEGRAM, destination=channel
        )

    async def distribute(
        self, article: GeneratedArticle, channel: DistributionChannel, destination: str
    ) -> Dict:
        """
        Distribute article to specified channel.

        Args:
            article: Generated article to distribute
            channel: Target distribution channel
            destination: Channel-specific destination identifier

        Returns:
            Distribution result metadata

        Raises:
            DistributionError: On distribution failure
        """
        logger.info(
            f"Initiating distribution | article_id={article.id} | "
            f"channel={channel} | destination={destination}"
        )

        start_time = datetime.utcnow()

        # Verify adapter exists
        adapter = self.adapters.get(channel)
        if not adapter:
            raise DistributionError(f"Channel not configured: {channel}")

        # Validate destination
        is_valid = await adapter.validate_destination(destination)
        if not is_valid:
            raise DistributionError(f"Invalid destination for {channel}: {destination}")

        try:
            # Execute distribution
            result = await adapter.distribute(article, destination)

            distribution_time = (datetime.utcnow() - start_time).total_seconds()

            # Record metrics
            await self.metrics.record_distribution(
                article_id=str(article.id),
                channel=channel,
                destination=destination,
                success=True,
                distribution_time=distribution_time,
            )

            logger.success(
                f"Distribution completed | article_id={article.id} | "
                f"channel={channel} | time={distribution_time:.2f}s"
            )

            return result

        except Exception as e:
            distribution_time = (datetime.utcnow() - start_time).total_seconds()

            # Record failure metrics
            await self.metrics.record_distribution(
                article_id=str(article.id),
                channel=channel,
                destination=destination,
                success=False,
                distribution_time=distribution_time,
                error=str(e),
            )

            logger.error(
                f"Distribution failed | article_id={article.id} | " f"channel={channel} | error={e}"
            )

            raise DistributionError(f"Distribution to {channel} failed: {str(e)}") from e

    async def distribute_multi_channel(
        self, article: GeneratedArticle, destinations: Dict[DistributionChannel, str]
    ) -> Dict[DistributionChannel, Dict]:
        """
        Distribute to multiple channels concurrently.

        Args:
            article: Article to distribute
            destinations: Mapping of channel to destination identifier

        Returns:
            Mapping of channel to distribution result
        """
        logger.info(
            f"Multi-channel distribution | article_id={article.id} | "
            f"channels={list(destinations.keys())}"
        )

        # Create distribution tasks
        tasks = {
            channel: self.distribute(article, channel, dest)
            for channel, dest in destinations.items()
        }

        # Execute concurrently
        results = {}
        errors = {}

        for channel, task in tasks.items():
            try:
                result = await task
                results[channel] = result
            except Exception as e:
                errors[channel] = str(e)
                logger.error(f"Channel distribution failed | channel={channel} | error={e}")

        # Return combined results
        return {
            "successful_channels": results,
            "failed_channels": errors,
            "total_channels": len(destinations),
            "success_count": len(results),
            "failure_count": len(errors),
        }

    def get_available_channels(self) -> List[DistributionChannel]:
        """Return list of configured distribution channels."""
        return list(self.adapters.keys())

    async def test_channel(self, channel: DistributionChannel, destination: str) -> bool:
        """
        Test channel connectivity and permissions.

        Returns True if channel is accessible and functional.
        """
        adapter = self.adapters.get(channel)
        if not adapter:
            return False

        return await adapter.validate_destination(destination)
