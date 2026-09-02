"""
WordPress Publishing Adapter
============================

Handles the repository's single supported direct publishing integration: WordPress.
"""

import asyncio
import ipaddress
import socket
from datetime import datetime as _dt
from datetime import timezone
from typing import Any, Dict, Optional
from urllib.parse import quote, urlparse

import httpx
from loguru import logger

from core.exceptions import DistributionError
from core.models import GeneratedArticle, Project
from execution.export_safety import json_for_html_script, sanitize_html_fragment
from infrastructure.credential_encryption import CredentialEncryptionError, decrypt_credential
from infrastructure.redaction import redact_text


class WordPressPublishError(DistributionError):
    """Classified WordPress publishing error safe for API/log surfaces."""

    def __init__(
        self,
        safe_message: str,
        *,
        category: str = "unknown_error",
        retryable: bool = False,
        status_code: int | None = None,
        retry_count: int = 0,
    ):
        super().__init__(safe_message)
        self.safe_message = redact_text(safe_message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code
        self.retry_count = retry_count


class Distributor:
    """Publish generated content to WordPress with bounded retry and verification."""

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

    @staticmethod
    async def _validate_wordpress_network_target(wordpress_url: str) -> None:
        """Reject production WordPress hosts that resolve to non-public networks.

        Literal private addresses are already rejected by the publishing service.
        This second outbound-boundary check also resolves hostnames immediately
        before HTTP use, preventing ordinary DNS aliases to loopback, link-local,
        private, multicast, reserved, or otherwise non-global addresses.
        """
        from config.settings import get_settings

        if get_settings().environment != "production":
            return
        parsed = urlparse(wordpress_url)
        hostname = (parsed.hostname or "").rstrip(".")
        if not hostname:
            raise WordPressPublishError(
                "WordPress target hostname is missing",
                category="unsafe_target",
                retryable=False,
            )
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            literal = ipaddress.ip_address(hostname)
            addresses = {literal}
        except ValueError:
            try:
                loop = asyncio.get_running_loop()
                resolved = await loop.getaddrinfo(
                    hostname,
                    port,
                    family=socket.AF_UNSPEC,
                    type=socket.SOCK_STREAM,
                )
            except OSError as exc:
                raise WordPressPublishError(
                    "WordPress hostname could not be resolved",
                    category="network_error",
                    retryable=True,
                ) from exc
            addresses = set()
            for item in resolved:
                raw_address = str(item[4][0]).split("%", 1)[0]
                try:
                    addresses.add(ipaddress.ip_address(raw_address))
                except ValueError:
                    continue

        if not addresses:
            raise WordPressPublishError(
                "WordPress hostname did not resolve to an IP address",
                category="network_error",
                retryable=True,
            )
        if any(not address.is_global for address in addresses):
            raise WordPressPublishError(
                "Production WordPress target resolves to a non-public network",
                category="unsafe_target",
                retryable=False,
            )

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
                "h2": ('<!-- wp:heading {"level":2} -->', "<!-- /wp:heading -->"),
                "h3": ('<!-- wp:heading {"level":3} -->', "<!-- /wp:heading -->"),
                "h4": ('<!-- wp:heading {"level":4} -->', "<!-- /wp:heading -->"),
                "p": ("<!-- wp:paragraph -->", "<!-- /wp:paragraph -->"),
                "ul": ("<!-- wp:list -->", "<!-- /wp:list -->"),
                "ol": ('<!-- wp:list {"ordered":true} -->', "<!-- /wp:list -->"),
                "blockquote": ("<!-- wp:quote -->", "<!-- /wp:quote -->"),
            }

            def __init__(self):
                super().__init__()
                self._out: list[str] = []
                self._stack: list[str] = []

            def handle_starttag(self, tag, attrs):
                if tag == "blockquote":
                    attrs = list(attrs)
                    for index, (name, value) in enumerate(attrs):
                        if name == "class":
                            classes = (value or "").split()
                            if "wp-block-quote" not in classes:
                                classes.append("wp-block-quote")
                            attrs[index] = (name, " ".join(classes))
                            break
                    else:
                        attrs.append(("class", "wp-block-quote"))

                attr_str = "".join(f' {k}="{v}"' if v else f" {k}" for k, v in attrs)
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
        parser.feed(sanitize_html_fragment(html_content))
        result = parser.result
        logger.debug("Converted HTML to Gutenberg blocks via html.parser")
        return result

    def generate_schema_markup(self, article: "GeneratedArticle", article_url: str = "") -> str:
        """
        Generate schema.org JSON-LD structured data for SEO.

        Creates an Article schema that reflects only persisted article metadata.

        Args:
            article: Generated article with metadata
            article_url: Published URL (if available)

        Returns:
            JSON-LD script tag for insertion in <head> or article
        """
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

        json_ld = (
            '<script type="application/ld+json">\n'
            f"{json_for_html_script(schema, indent=2)}\n"
            "</script>"
        )

        logger.debug(f"Generated schema.org markup: {schema['@type']}")
        return json_ld

    async def validate_wordpress_connection(
        self,
        project: Project,
        *,
        required_status: str = "draft",
    ) -> tuple[bool, str]:
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
            await self._validate_wordpress_network_target(project.wordpress_url)
            auth = httpx.BasicAuth(
                project.wordpress_username,
                self._wordpress_password(project),
            )
            # An authenticated users/me request verifies both REST availability and
            # that the Application Password belongs to a real user. context=edit
            # also proves the credential has an authenticated editing context.
            test_url = f"{project.wordpress_url.rstrip('/')}/wp-json/wp/v2/users/me?context=edit"

            async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                response = await client.get(test_url, auth=auth)

            if response.status_code == 200:
                try:
                    payload = response.json()
                except Exception:
                    return False, "WordPress returned an invalid authenticated-user response"
                if not isinstance(payload, dict) or not payload.get("id"):
                    return False, "WordPress returned an invalid authenticated-user response"
                capabilities = payload.get("capabilities")
                if not isinstance(capabilities, dict):
                    return False, "WordPress did not return authenticated capability data"
                if not bool(capabilities.get("edit_posts")):
                    return False, "WordPress account does not have permission to edit posts"
                if required_status in {"future", "publish"} and not bool(
                    capabilities.get("publish_posts")
                ):
                    return False, "WordPress account does not have permission to publish posts"
                return True, ""
            if response.status_code == 401:
                return False, "Invalid WordPress credentials"
            if response.status_code == 403:
                return False, "WordPress account does not have the required editing permission"
            if response.status_code == 404:
                return False, "WordPress REST API not found - check URL"
            if 300 <= response.status_code < 400:
                return False, (
                    f"WordPress URL redirected (HTTP {response.status_code}); "
                    "configure the canonical HTTPS URL"
                )
            return False, f"WordPress connection failed: HTTP {response.status_code}"

        except WordPressPublishError as exc:
            logger.error("WordPress target validation failed: {}", exc.safe_message)
            return False, exc.safe_message
        except CredentialEncryptionError:
            logger.error("Stored WordPress credential cannot be decrypted")
            return False, "Stored WordPress credential cannot be decrypted"
        except DistributionError:
            return False, "WordPress credentials not configured"
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
        self,
        tag_names: list[str],
        wordpress_url: str,
        auth: httpx.BasicAuth,
    ) -> tuple[list[int], list[str], list[dict[str, str]]]:
        """Resolve existing tags before the core post write; do not create remotely yet."""
        normalized_names = list(
            dict.fromkeys(str(name or "").strip() for name in tag_names if str(name or "").strip())
        )[:20]
        if not normalized_names:
            return [], [], []

        tag_ids: list[int] = []
        missing_names: list[str] = []
        warnings: list[dict[str, str]] = []
        tags_url = f"{wordpress_url.rstrip('/')}/wp-json/wp/v2/tags"

        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                response = await client.get(f"{tags_url}?per_page=100", auth=auth)
            if response.status_code != 200:
                warnings.append(
                    {
                        "category": "tag_list_failed",
                        "message": f"WordPress tags could not be loaded (HTTP {response.status_code})",
                    }
                )
                return tag_ids, normalized_names, warnings
            try:
                payload = response.json()
            except Exception:
                warnings.append(
                    {
                        "category": "tag_list_invalid_response",
                        "message": "WordPress returned an invalid tag-list response",
                    }
                )
                return tag_ids, normalized_names, warnings
            if not isinstance(payload, list):
                warnings.append(
                    {
                        "category": "tag_list_invalid_response",
                        "message": "WordPress returned an invalid tag-list response",
                    }
                )
                return tag_ids, normalized_names, warnings
            existing = {
                str(item.get("name") or "").lower(): item.get("id")
                for item in payload
                if isinstance(item, dict) and item.get("name") and item.get("id")
            }
            for name in normalized_names:
                tag_id = existing.get(name.lower())
                if tag_id:
                    tag_ids.append(int(tag_id))
                else:
                    missing_names.append(name)
        except (httpx.TimeoutException, httpx.NetworkError):
            warnings.append(
                {
                    "category": "tag_resolution_unavailable",
                    "message": "WordPress tags could not be resolved because the taxonomy endpoint was unavailable",
                }
            )
            missing_names = normalized_names
        except Exception:
            warnings.append(
                {
                    "category": "tag_resolution_failed",
                    "message": "WordPress tags could not be resolved",
                }
            )
            missing_names = normalized_names

        return tag_ids, missing_names, warnings

    async def _create_missing_tag_ids(
        self,
        tag_names: list[str],
        wordpress_url: str,
        auth: httpx.BasicAuth,
    ) -> tuple[list[int], list[dict[str, str]]]:
        """Create optional missing tags only after the core post is verified."""
        if not tag_names:
            return [], []
        tags_url = f"{wordpress_url.rstrip('/')}/wp-json/wp/v2/tags"
        tag_ids: list[int] = []
        warnings: list[dict[str, str]] = []
        try:
            async with httpx.AsyncClient(timeout=15.0, follow_redirects=False) as client:
                for name in tag_names[:20]:
                    response = await client.post(tags_url, json={"name": name}, auth=auth)
                    if response.status_code == 201:
                        try:
                            payload = response.json()
                            tag_id = payload.get("id") if isinstance(payload, dict) else None
                        except Exception:
                            tag_id = None
                        if tag_id:
                            tag_ids.append(int(tag_id))
                        else:
                            warnings.append(
                                {
                                    "category": "tag_create_invalid_response",
                                    "message": "WordPress created a tag but returned no tag ID",
                                }
                            )
                    else:
                        warnings.append(
                            {
                                "category": "tag_create_failed",
                                "message": f"A WordPress tag could not be created (HTTP {response.status_code})",
                            }
                        )
        except (httpx.TimeoutException, httpx.NetworkError):
            warnings.append(
                {
                    "category": "tag_create_unavailable",
                    "message": "Missing WordPress tags could not be created because the taxonomy endpoint was unavailable",
                }
            )
        except Exception:
            warnings.append(
                {
                    "category": "tag_create_failed",
                    "message": "Missing WordPress tags could not be created",
                }
            )
        return tag_ids, warnings

    @staticmethod
    def _wordpress_slug(article: GeneratedArticle) -> str:
        return f"smarlux-{str(article.id).replace('-', '')[:32]}"

    @staticmethod
    def _classify_http_status(status_code: int) -> tuple[str, bool]:
        if status_code in (401,):
            return "auth_error", False
        if status_code in (403,):
            return "permission_error", False
        if status_code == 404:
            return "not_found", False
        if status_code == 429:
            return "rate_limited", True
        if 400 <= status_code < 500:
            return "validation_error", False
        if 500 <= status_code < 600:
            return "wordpress_5xx", True
        return "unknown_error", False

    @staticmethod
    def _classify_exception(exc: Exception) -> tuple[str, bool]:
        if isinstance(exc, httpx.TimeoutException):
            return "timeout", True
        if isinstance(exc, httpx.NetworkError):
            return "network_error", True
        return "unknown_error", True

    @staticmethod
    def _safe_response_text(response: httpx.Response) -> str:
        value = getattr(response, "text", "")
        return value if isinstance(value, str) else str(value)

    async def _find_existing_wordpress_post(
        self,
        *,
        client: httpx.AsyncClient,
        api_url: str,
        auth: httpx.BasicAuth,
        slug: str,
    ) -> dict[str, Any] | None:
        """Find a previously created post by deterministic slug before creating."""
        lookup_url = f"{api_url}?slug={quote(slug)}&status=any&per_page=1"
        response = await client.get(lookup_url, auth=auth)
        if response.status_code != 200:
            category, retryable = self._classify_http_status(response.status_code)
            raise WordPressPublishError(
                f"WordPress duplicate lookup failed (HTTP {response.status_code})",
                category=category,
                retryable=retryable,
                status_code=response.status_code,
            )
        try:
            posts = response.json()
        except Exception as exc:
            raise WordPressPublishError(
                "WordPress duplicate lookup returned invalid JSON",
                category="invalid_response",
                retryable=True,
            ) from exc
        if isinstance(posts, list) and posts:
            first = posts[0]
            return first if isinstance(first, dict) else None
        return None

    async def _verify_wordpress_post(
        self,
        *,
        wordpress_url: str,
        auth: httpx.BasicAuth,
        post_id: str | int | None,
        expected_slug: str,
        expected_status: str,
    ) -> dict[str, Any]:
        """Read after write and reject ambiguous or mismatched remote state."""
        if not post_id:
            raise WordPressPublishError(
                "WordPress response did not contain a post ID",
                category="invalid_response",
                retryable=True,
            )
        url = f"{wordpress_url.rstrip('/')}/wp-json/wp/v2/posts/{post_id}?context=edit"
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as client:
            response = await client.get(url, auth=auth)
        if response.status_code >= 400:
            category, retryable = self._classify_http_status(response.status_code)
            raise WordPressPublishError(
                f"WordPress post verification failed (HTTP {response.status_code})",
                category=category,
                retryable=retryable,
                status_code=response.status_code,
            )
        try:
            payload = response.json()
        except Exception as exc:
            raise WordPressPublishError(
                "WordPress post verification returned invalid JSON",
                category="invalid_response",
                retryable=True,
            ) from exc
        if not isinstance(payload, dict):
            raise WordPressPublishError(
                "WordPress post verification returned an invalid response",
                category="invalid_response",
                retryable=True,
            )
        actual_id = payload.get("id")
        actual_slug = str(payload.get("slug") or "")
        actual_status = str(payload.get("status") or "")
        post_url = str(payload.get("link") or "")
        if str(actual_id) != str(post_id):
            raise WordPressPublishError(
                "WordPress post verification returned a different post ID",
                category="remote_state_mismatch",
                retryable=False,
            )
        if actual_slug != expected_slug:
            raise WordPressPublishError(
                "WordPress post verification returned a different slug",
                category="remote_state_mismatch",
                retryable=False,
            )
        if actual_status != expected_status:
            raise WordPressPublishError(
                f"WordPress post status verification failed: expected {expected_status}, received {actual_status or 'unknown'}",
                category="remote_state_mismatch",
                retryable=False,
            )
        if not post_url:
            raise WordPressPublishError(
                "WordPress post verification did not return a post URL",
                category="invalid_response",
                retryable=True,
            )
        return payload

    async def distribute_to_wordpress(
        self,
        article: GeneratedArticle,
        project: Project,
        post_status: str = "draft",
        wordpress_post_id: str | int | None = None,
        idempotency_key: str | None = None,
        scheduled_at: _dt | None = None,
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
            post_status: WordPress post status — 'draft' (default), 'future', or 'publish'
            wordpress_post_id: Existing remote post ID; when present, update instead of create.
            idempotency_key: Deterministic publish key used for traceability and duplicate lookup.
            scheduled_at: Required by caller for 'future' scheduled posts.

        Returns:
            dict with status="published" on success or status="error" on failure
        """
        logger.info(
            f"Starting WordPress upload | article_id={article.id} | "
            f"project_id={project.id} | wp_url={project.wordpress_url} | "
            f"post_status={post_status}"
        )

        # Pre-flight validation
        is_valid, error_msg = await self.validate_wordpress_connection(
            project,
            required_status=post_status,
        )
        if not is_valid:
            logger.error(
                f"WordPress validation failed | article_id={article.id} | "
                f"project_id={project.id} | error={error_msg}"
            )
            category = "auth_error" if "credential" in error_msg.lower() else "connection_error"
            retryable = any(term in error_msg.lower() for term in ("timeout", "connect", "http 5"))
            raise WordPressPublishError(
                error_msg,
                category=category,
                retryable=retryable,
            )

        wordpress_url = project.wordpress_url
        wordpress_username = project.wordpress_username
        if not wordpress_url or not wordpress_username:
            raise WordPressPublishError(
                "WordPress credentials not configured",
                category="configuration_error",
                retryable=False,
            )

        api_url = f"{wordpress_url.rstrip('/')}/wp-json/wp/v2/posts"
        auth = httpx.BasicAuth(wordpress_username, self._wordpress_password(project))

        # Convert content to Gutenberg blocks for modern WordPress
        gutenberg_content = self.convert_to_gutenberg_blocks(article.content)

        # Resolve keyword strings to WordPress tag IDs
        tag_ids, missing_tag_names, tag_warnings = await self._resolve_tag_ids(
            article.keywords if article.keywords else [],
            wordpress_url,
            auth,
        )

        slug = self._wordpress_slug(article)
        post_data = {
            "title": article.title,
            "content": gutenberg_content,
            "status": post_status,  # AP-1: controlled by caller, defaults to 'draft'
            "slug": slug,
            "tags": tag_ids,
        }
        if post_status == "future" and scheduled_at:
            scheduled_utc = scheduled_at
            if scheduled_utc.tzinfo is None:
                scheduled_utc = scheduled_utc.replace(tzinfo=timezone.utc)
            else:
                scheduled_utc = scheduled_utc.astimezone(timezone.utc)
            post_data["date_gmt"] = scheduled_utc.replace(tzinfo=None).isoformat()

        # Retry logic with exponential backoff and jitter
        import asyncio
        import random

        last_error: Exception | None = None

        for attempt in range(self.max_retries):
            try:
                # Use connection pooling with longer timeout for large content
                timeout_config = httpx.Timeout(connect=10.0, read=60.0, write=30.0, pool=5.0)
                async with httpx.AsyncClient(
                    timeout=timeout_config,
                    limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                    follow_redirects=False,
                ) as client:
                    target_post_id = wordpress_post_id
                    if not target_post_id and idempotency_key:
                        existing = await self._find_existing_wordpress_post(
                            client=client,
                            api_url=api_url,
                            auth=auth,
                            slug=slug,
                        )
                        if existing:
                            target_post_id = existing.get("id")
                            logger.warning(
                                f"WordPress duplicate prevention hit | article_id={article.id} | "
                                f"existing_post_id={target_post_id} | slug={slug}"
                            )

                    target_url = (
                        f"{wordpress_url.rstrip('/')}/wp-json/wp/v2/posts/{target_post_id}"
                        if target_post_id
                        else api_url
                    )
                    response = await client.post(target_url, json=post_data, auth=auth)

                response.raise_for_status()

                try:
                    response_data = response.json()
                except Exception as exc:
                    raise WordPressPublishError(
                        "WordPress publish response was not valid JSON",
                        category="invalid_response",
                        retryable=True,
                        retry_count=attempt,
                    ) from exc
                if not isinstance(response_data, dict):
                    raise WordPressPublishError(
                        "WordPress publish response was invalid",
                        category="invalid_response",
                        retryable=True,
                        retry_count=attempt,
                    )
                post_id = response_data.get("id")
                verified = await self._verify_wordpress_post(
                    wordpress_url=wordpress_url,
                    auth=auth,
                    post_id=post_id,
                    expected_slug=slug,
                    expected_status=post_status,
                )
                post_url = str(verified.get("link") or "")
                created_tag_ids, created_tag_warnings = await self._create_missing_tag_ids(
                    missing_tag_names,
                    wordpress_url,
                    auth,
                )
                final_tag_ids = list(dict.fromkeys([*tag_ids, *created_tag_ids]))
                warnings: list[dict[str, str]] = [*tag_warnings, *created_tag_warnings]
                schema_stored = False
                seo_meta_stored = False

                # Optional plugin/custom metadata is deliberately separated from the
                # primary post write. Sites that do not expose custom meta through
                # REST must not cause an otherwise valid article publish to fail.
                schema_markup = self.generate_schema_markup(article, post_url)
                optional_meta = {
                    "_yoast_wpseo_metadesc": article.meta_description or "",
                    "_smarlux_idempotency_key": idempotency_key or "",
                    "_schema_json_ld": schema_markup,
                }
                optional_update: dict[str, Any] = {"meta": optional_meta}
                if final_tag_ids != tag_ids:
                    optional_update["tags"] = final_tag_ids
                try:
                    update_url = f"{wordpress_url.rstrip('/')}/wp-json/wp/v2/posts/{post_id}"
                    async with httpx.AsyncClient(timeout=10.0, follow_redirects=False) as client:
                        metadata_response = await client.post(
                            update_url, json=optional_update, auth=auth
                        )
                    if metadata_response.status_code < 400:
                        schema_stored = True
                        seo_meta_stored = True
                    else:
                        warnings.append(
                            {
                                "category": "optional_metadata_storage_failed",
                                "message": f"Optional SEO/schema metadata was not stored by WordPress (HTTP {metadata_response.status_code})",
                            }
                        )
                except Exception:
                    warnings.append(
                        {
                            "category": "optional_metadata_storage_failed",
                            "message": "Optional SEO/schema metadata could not be stored by WordPress",
                        }
                    )

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
                    "timestamp": _dt.now(timezone.utc).isoformat().replace("+00:00", "Z"),
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
                    "schema_stored": schema_stored,
                    "seo_meta_stored": seo_meta_stored,
                    "remote_verified": True,
                    "remote_verified_at": _dt.now(timezone.utc).isoformat(),
                    "warnings": warnings,
                    "attempts": attempt + 1,
                    "post_status": post_status,
                }

            except httpx.HTTPStatusError as e:
                last_error = e
                category, retryable = self._classify_http_status(e.response.status_code)
                safe_error = redact_text(self._safe_response_text(e.response) or str(e))
                # Don't retry on non-transient client errors.
                if not retryable:
                    logger.error(
                        f"WordPress API client error (NO RETRY) | article_id={article.id} | "
                        f"status_code={e.response.status_code} | category={category} | "
                        f"error={safe_error}"
                    )
                    raise WordPressPublishError(
                        f"WordPress API error: {safe_error}",
                        category=category,
                        retryable=False,
                        status_code=e.response.status_code,
                        retry_count=attempt,
                    )

                logger.warning(
                    f"WordPress HTTP error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | "
                    f"status_code={e.response.status_code} | category={category} | "
                    f"error={redact_text(str(e))}"
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
                category, _ = self._classify_exception(e)
                logger.warning(
                    f"WordPress network error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | "
                    f"error_type={type(e).__name__} | category={category} | "
                    f"error={redact_text(str(e))}"
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

            except WordPressPublishError as e:
                last_error = e
                if not e.retryable:
                    raise
                logger.warning(
                    f"WordPress classified error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | category={e.category} | "
                    f"error={e.safe_message}"
                )
                if attempt < self.max_retries - 1:
                    base_wait = min(self.retry_delay * (2**attempt), 10.0)
                    jitter = random.uniform(0, base_wait * 0.3)
                    await asyncio.sleep(base_wait + jitter)

            except Exception as e:
                last_error = e
                category, _ = self._classify_exception(e)
                logger.error(
                    f"Unexpected WordPress error (RETRYING) | article_id={article.id} | "
                    f"attempt={attempt + 1}/{self.max_retries} | "
                    f"error_type={type(e).__name__} | category={category} | "
                    f"error={redact_text(str(e))}",
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
            f"last_error_type={type(last_error).__name__} | last_error={redact_text(str(last_error))}"
        )
        category = "unknown_error"
        retryable = True
        status_code = None
        if isinstance(last_error, WordPressPublishError):
            category = last_error.category
            retryable = last_error.retryable
            status_code = last_error.status_code
        elif isinstance(last_error, httpx.HTTPStatusError):
            category, retryable = self._classify_http_status(last_error.response.status_code)
            status_code = last_error.response.status_code
        elif isinstance(last_error, Exception):
            category, retryable = self._classify_exception(last_error)
        raise WordPressPublishError(
            f"WordPress distribution failed after {self.max_retries} attempts: {redact_text(str(last_error))}",
            category=category,
            retryable=retryable,
            status_code=status_code,
            retry_count=max(self.max_retries - 1, 0),
        )
