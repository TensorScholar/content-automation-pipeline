"""Reliable, read-only Google Search Console OAuth and performance synchronization."""

from __future__ import annotations

import hashlib
import secrets
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import quote, urlencode, urlparse
from uuid import UUID, uuid4

import httpx
from fastapi import HTTPException, status

from config.settings import Settings
from infrastructure.credential_encryption import (
    CredentialEncryptionError,
    decrypt_credential,
    encrypt_credential,
    validate_encryption_key,
)
from infrastructure.redaction import redact_text
from knowledge.search_console_repository import SearchConsoleRepository
from services.performance_feedback_service import PerformanceFeedbackService

READONLY_SCOPE = "https://www.googleapis.com/auth/webmasters.readonly"
AUTHORIZATION_ENDPOINT = "https://accounts.google.com/o/oauth2/v2/auth"
TOKEN_ENDPOINT = "https://oauth2.googleapis.com/token"
REVOKE_ENDPOINT = "https://oauth2.googleapis.com/revoke"
SITES_ENDPOINT = "https://www.googleapis.com/webmasters/v3/sites"
SEARCH_ANALYTICS_ENDPOINT = (
    "https://www.googleapis.com/webmasters/v3/sites/{site_url}/searchAnalytics/query"
)


class SearchConsoleError(RuntimeError):
    """Classified Google integration failure safe for persistence and API surfaces."""

    def __init__(
        self,
        message: str,
        *,
        category: str,
        retryable: bool,
        status_code: int = 502,
    ):
        super().__init__(message)
        self.safe_message = redact_text(message)
        self.category = category
        self.retryable = retryable
        self.status_code = status_code


class SearchConsoleService:
    """Own OAuth state, encrypted refresh tokens, property selection and durable syncs."""

    def __init__(
        self,
        *,
        repository: SearchConsoleRepository,
        performance_service: PerformanceFeedbackService,
        settings: Settings,
    ):
        self.repository = repository
        self.performance = performance_service
        self.settings = settings

    def _require_configured(self) -> None:
        config = self.settings.search_console
        if not config.configured:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=(
                    "Google Search Console OAuth is not configured. Set client ID, "
                    "client secret, and redirect URI."
                ),
            )
        try:
            validate_encryption_key(self.settings.credential_encryption_key)
        except CredentialEncryptionError as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search Console credential encryption is not configured correctly",
            ) from exc

        redirect = urlparse(str(config.redirect_uri or ""))
        frontend_return = urlparse(str(config.frontend_return_url or ""))
        if redirect.scheme not in {"http", "https"} or not redirect.netloc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search Console redirect URI is invalid",
            )
        if frontend_return.scheme not in {"http", "https"} or not frontend_return.netloc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search Console frontend return URL is invalid",
            )
        if self.settings.environment == "production" and (
            redirect.scheme != "https" or frontend_return.scheme != "https"
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Search Console OAuth URLs must use HTTPS in production",
            )

    async def create_authorization_url(self, *, project_id: UUID, user_id: UUID) -> dict[str, Any]:
        self._require_configured()
        if not await self.performance.repository.project_exists(project_id):
            raise HTTPException(status_code=404, detail="Project not found")

        state_value = secrets.token_urlsafe(48)
        state_hash = hashlib.sha256(state_value.encode("utf-8")).hexdigest()
        expires_at = datetime.now(timezone.utc) + timedelta(
            seconds=self.settings.search_console.oauth_state_ttl_seconds
        )
        await self.repository.create_oauth_state(
            state_hash=state_hash,
            project_id=project_id,
            user_id=user_id,
            expires_at=expires_at,
        )
        params = {
            "client_id": self.settings.search_console.client_id or "",
            "redirect_uri": self.settings.search_console.redirect_uri or "",
            "response_type": "code",
            "scope": READONLY_SCOPE,
            "access_type": "offline",
            "prompt": "consent",
            "state": state_value,
        }
        return {
            "authorization_url": f"{AUTHORIZATION_ENDPOINT}?{urlencode(params)}",
            "expires_at": expires_at,
            "scope": READONLY_SCOPE,
        }

    async def handle_oauth_callback(
        self,
        *,
        state_value: str,
        code: str | None,
        error: str | None,
    ) -> dict[str, Any]:
        self._require_configured()
        if not state_value:
            raise SearchConsoleError(
                "OAuth state is missing",
                category="invalid_state",
                retryable=False,
                status_code=400,
            )
        state_hash = hashlib.sha256(state_value.encode("utf-8")).hexdigest()
        state_row = await self.repository.consume_oauth_state(state_hash=state_hash)
        if not state_row:
            raise SearchConsoleError(
                "OAuth state is invalid, expired, or already used",
                category="invalid_state",
                retryable=False,
                status_code=400,
            )
        project_id = UUID(str(state_row["project_id"]))
        user_id = UUID(str(state_row["user_id"]))
        if error:
            raise SearchConsoleError(
                f"Google authorization was not completed: {error}",
                category="consent_denied",
                retryable=False,
                status_code=400,
            )
        if not code:
            raise SearchConsoleError(
                "Google authorization code is missing",
                category="missing_code",
                retryable=False,
                status_code=400,
            )

        token_payload = await self._exchange_authorization_code(code)
        refresh_token = str(token_payload.get("refresh_token") or "").strip()
        if not refresh_token:
            existing = await self.repository.get_connection(project_id)
            if existing:
                try:
                    refresh_token = str(
                        decrypt_credential(
                            existing["encrypted_refresh_token"],
                            self.settings.credential_encryption_key,
                        )
                        or ""
                    )
                except CredentialEncryptionError as exc:
                    raise SearchConsoleError(
                        "Stored Search Console credential cannot be decrypted",
                        category="credential_error",
                        retryable=False,
                        status_code=500,
                    ) from exc
        if not refresh_token:
            raise SearchConsoleError(
                "Google did not return an offline refresh token. Reconnect and grant consent.",
                category="missing_refresh_token",
                retryable=False,
                status_code=400,
            )

        try:
            encrypted = encrypt_credential(
                refresh_token,
                self.settings.credential_encryption_key,
            )
        except CredentialEncryptionError as exc:
            raise SearchConsoleError(
                "Search Console refresh token could not be encrypted",
                category="credential_error",
                retryable=False,
                status_code=500,
            ) from exc
        if not encrypted:
            raise SearchConsoleError(
                "Refresh token encryption failed",
                category="credential_error",
                retryable=False,
                status_code=500,
            )
        granted_scope = str(token_payload.get("scope") or READONLY_SCOPE)
        granted_scopes = {scope for scope in granted_scope.split() if scope}
        if granted_scopes != {READONLY_SCOPE}:
            raise SearchConsoleError(
                "Search Console authorization must grant only the read-only scope",
                category="scope_mismatch",
                retryable=False,
                status_code=403,
            )

        await self.repository.upsert_connection(
            project_id=project_id,
            encrypted_refresh_token=encrypted,
            scope=granted_scope,
            connected_by=user_id,
        )
        properties = await self.refresh_properties(project_id)
        return {
            "project_id": str(project_id),
            "connected": True,
            "property_count": len(properties),
        }

    async def get_status(self, project_id: UUID) -> dict[str, Any]:
        await self._require_project_exists(project_id)
        connection = await self.repository.get_connection(project_id)
        properties = await self.repository.list_properties(project_id) if connection else []
        runs = await self.repository.list_sync_runs(project_id, limit=5) if connection else []
        return {
            "configured": self.settings.search_console.configured,
            "connected": bool(connection and connection.get("status") == "connected"),
            "status": connection.get("status") if connection else "disconnected",
            "selected_site_url": connection.get("selected_site_url") if connection else None,
            "permission_level": connection.get("permission_level") if connection else None,
            "last_sync_at": connection.get("last_sync_at") if connection else None,
            "last_error_category": connection.get("last_error_category") if connection else None,
            "last_error_message": connection.get("last_error_message") if connection else None,
            "properties": [self._public_property(item) for item in properties],
            "recent_sync_runs": [self.public_sync_run(item) for item in runs],
            "scope": READONLY_SCOPE,
        }

    async def refresh_properties(self, project_id: UUID) -> list[dict[str, Any]]:
        connection = await self._require_connection(project_id)
        try:
            access_token = await self._refresh_access_token(connection)
            try:
                async with httpx.AsyncClient(
                    timeout=self.settings.search_console.request_timeout_seconds
                ) as client:
                    response = await client.get(
                        SITES_ENDPOINT,
                        headers={"Authorization": f"Bearer {access_token}"},
                    )
            except httpx.TimeoutException as exc:
                raise SearchConsoleError(
                    "Search Console property request timed out",
                    category="timeout",
                    retryable=True,
                    status_code=504,
                ) from exc
            except httpx.NetworkError as exc:
                raise SearchConsoleError(
                    "Search Console property request failed because of a network error",
                    category="network_error",
                    retryable=True,
                    status_code=503,
                ) from exc
            self._raise_for_google_response(response, operation="list properties")
            try:
                payload = response.json()
            except Exception as exc:
                raise SearchConsoleError(
                    "Search Console returned an invalid property response",
                    category="invalid_response",
                    retryable=True,
                ) from exc
            properties = []
            for item in payload.get("siteEntry", []) if isinstance(payload, dict) else []:
                if not isinstance(item, dict):
                    continue
                site_url = str(item.get("siteUrl") or "").strip()
                permission = str(item.get("permissionLevel") or "").strip()
                if site_url and permission and permission != "siteUnverifiedUser":
                    properties.append({"site_url": site_url, "permission_level": permission})
            cached = await self.repository.replace_properties(
                connection_id=UUID(str(connection["id"])),
                project_id=project_id,
                properties=properties,
            )
            await self.repository.clear_connection_error(project_id)
            return cached
        except SearchConsoleError as exc:
            await self.repository.set_connection_error(
                project_id=project_id,
                category=exc.category,
                message=exc.safe_message,
                status_value="revoked" if exc.category == "refresh_token_revoked" else None,
            )
            raise

    async def select_property(self, *, project_id: UUID, site_url: str) -> dict[str, Any]:
        await self._require_connection(project_id)
        normalized = site_url.strip()
        if not normalized:
            raise HTTPException(status_code=400, detail="Search Console property is required")
        prop = await self.repository.select_property(project_id=project_id, site_url=normalized)
        if not prop:
            await self.refresh_properties(project_id)
            prop = await self.repository.select_property(project_id=project_id, site_url=normalized)
        if not prop:
            raise HTTPException(
                status_code=403,
                detail="The selected Search Console property is not accessible to this connection",
            )
        return self._public_property(prop)

    async def disconnect(self, project_id: UUID) -> dict[str, Any]:
        await self._require_project_exists(project_id)
        connection = await self.repository.get_connection(project_id)
        if not connection:
            return {"project_id": str(project_id), "connected": False}
        try:
            refresh_token = decrypt_credential(
                connection.get("encrypted_refresh_token"),
                self.settings.credential_encryption_key,
            )
        except CredentialEncryptionError:
            refresh_token = None
        if refresh_token:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    await client.post(REVOKE_ENDPOINT, data={"token": refresh_token})
            except Exception:
                # Local disconnect must remain available even if Google is unavailable.
                pass
        await self.repository.disconnect(project_id)
        return {"project_id": str(project_id), "connected": False}

    async def queue_sync(
        self,
        *,
        project_id: UUID,
        date_from: date | None = None,
        date_to: date | None = None,
    ) -> dict[str, Any]:
        connection = await self._require_connection(project_id)
        site_url = str(connection.get("selected_site_url") or "").strip()
        if not site_url:
            raise HTTPException(status_code=400, detail="Select a Search Console property first")

        date_from, date_to = self._resolve_sync_window(date_from=date_from, date_to=date_to)
        task_id = str(uuid4())
        claim = await self.repository.claim_sync(
            project_id=project_id,
            connection_id=UUID(str(connection["id"])),
            site_url=site_url,
            date_from=date_from,
            date_to=date_to,
            task_id=task_id,
        )
        if not claim.claimed:
            return {
                "status": claim.run["status"],
                "idempotent": True,
                "reason": claim.reason,
                "sync_run": self.public_sync_run(claim.run),
            }

        try:
            from orchestration.celery_app import app

            app.send_task(
                "content_automation.sync_search_console",
                kwargs={"run_id": str(claim.run["id"])},
                task_id=task_id,
                queue="integrations",
            )
        except Exception as exc:
            await self.repository.mark_sync_failure(
                run_id=UUID(str(claim.run["id"])),
                category="queue_unavailable",
                message="Search Console sync could not be queued",
                retry_count=int(claim.run.get("retry_count") or 0),
                task_id=task_id,
            )
            raise HTTPException(
                status_code=503,
                detail="Search Console sync queue is unavailable",
            ) from exc
        claim.run["task_id"] = task_id
        return {
            "status": "queued",
            "idempotent": False,
            "sync_run": self.public_sync_run(claim.run),
        }

    async def execute_sync(self, run_id: UUID, *, task_id: str) -> dict[str, Any]:
        run = await self.repository.get_sync_run(run_id)
        if not run:
            raise SearchConsoleError(
                "Search Console sync run was not found",
                category="not_found",
                retryable=False,
                status_code=404,
            )
        if run["status"] == "succeeded":
            return self.public_sync_run(run)
        project_id = UUID(str(run["project_id"]))
        connection = await self._require_connection_for_worker(project_id)
        acquired = await self.repository.mark_sync_running(run_id, task_id=task_id)
        if not acquired:
            current = await self.repository.get_sync_run(run_id)
            return self.public_sync_run(current or run)
        try:
            rows, pages_fetched, truncated = await self._fetch_search_analytics(
                connection=connection,
                site_url=run["site_url"],
                date_from=run["date_from"],
                date_to=run["date_to"],
            )
            await self.performance.import_records(
                project_id=project_id,
                records=rows,
                source="search_console_api",
            )
            persisted = await self.repository.mark_sync_success(
                run_id=run_id,
                task_id=task_id,
                row_count=len(rows),
                pages_fetched=pages_fetched,
                truncated=truncated,
            )
            if persisted is False:
                current = await self.repository.get_sync_run(run_id)
                return self.public_sync_run(current or run)
            await self.repository.clear_connection_error(project_id, synced=True)
        except SearchConsoleError as exc:
            status_value = "revoked" if exc.category == "refresh_token_revoked" else None
            await self.repository.set_connection_error(
                project_id=project_id,
                category=exc.category,
                message=exc.safe_message,
                status_value=status_value,
            )
            raise
        completed = await self.repository.get_sync_run(run_id)
        return self.public_sync_run(completed or run)

    async def _fetch_search_analytics(
        self,
        *,
        connection: dict[str, Any],
        site_url: str,
        date_from: date,
        date_to: date,
    ) -> tuple[list[dict[str, Any]], int, bool]:
        access_token = await self._refresh_access_token(connection)
        endpoint = SEARCH_ANALYTICS_ENDPOINT.format(site_url=quote(site_url, safe=""))
        row_limit = self.settings.search_console.row_limit
        max_rows = self.settings.search_console.max_rows_per_sync
        start_row = 0
        pages_fetched = 0
        all_rows: list[dict[str, Any]] = []
        truncated = False
        headers = {"Authorization": f"Bearer {access_token}"}

        async with httpx.AsyncClient(
            timeout=self.settings.search_console.request_timeout_seconds
        ) as client:
            while start_row < max_rows:
                body = {
                    "startDate": date_from.isoformat(),
                    "endDate": date_to.isoformat(),
                    "dimensions": ["page"],
                    "aggregationType": "byPage",
                    "dataState": "final",
                    "rowLimit": min(row_limit, max_rows - start_row),
                    "startRow": start_row,
                }
                try:
                    response = await client.post(endpoint, headers=headers, json=body)
                except httpx.TimeoutException as exc:
                    raise SearchConsoleError(
                        "Search Console analytics request timed out",
                        category="timeout",
                        retryable=True,
                        status_code=504,
                    ) from exc
                except httpx.NetworkError as exc:
                    raise SearchConsoleError(
                        "Search Console analytics request failed because of a network error",
                        category="network_error",
                        retryable=True,
                        status_code=503,
                    ) from exc
                self._raise_for_google_response(response, operation="query analytics")
                try:
                    payload = response.json()
                except Exception as exc:
                    raise SearchConsoleError(
                        "Search Console returned an invalid analytics response",
                        category="invalid_response",
                        retryable=True,
                    ) from exc
                raw_rows = payload.get("rows", []) if isinstance(payload, dict) else []
                if not isinstance(raw_rows, list):
                    raise SearchConsoleError(
                        "Search Console returned an invalid analytics response",
                        category="invalid_response",
                        retryable=True,
                    )
                pages_fetched += 1
                for item in raw_rows:
                    if not isinstance(item, dict):
                        continue
                    keys = item.get("keys") or []
                    if not keys:
                        continue
                    url = str(keys[0] or "").strip()
                    if not url:
                        continue
                    try:
                        clicks = max(0, int(round(float(item.get("clicks") or 0))))
                        impressions = max(0, int(round(float(item.get("impressions") or 0))))
                        ctr = max(0.0, min(float(item.get("ctr") or 0), 1.0))
                        position = max(0.0, float(item.get("position") or 0))
                    except (TypeError, ValueError) as exc:
                        raise SearchConsoleError(
                            "Search Console returned invalid analytics metrics",
                            category="invalid_response",
                            retryable=True,
                        ) from exc
                    all_rows.append(
                        {
                            "url": url,
                            "date_from": date_from,
                            "date_to": date_to,
                            "clicks": clicks,
                            "impressions": impressions,
                            "ctr": ctr,
                            "average_position": position,
                        }
                    )
                if len(raw_rows) < body["rowLimit"]:
                    break
                start_row += len(raw_rows)
            else:
                truncated = True
        if len(all_rows) >= max_rows:
            truncated = True
        return all_rows[:max_rows], pages_fetched, truncated

    async def _exchange_authorization_code(self, code: str) -> dict[str, Any]:
        secret = self.settings.search_console.client_secret
        data = {
            "client_id": self.settings.search_console.client_id or "",
            "client_secret": secret.get_secret_value() if secret else "",
            "code": code,
            "grant_type": "authorization_code",
            "redirect_uri": self.settings.search_console.redirect_uri or "",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.search_console.request_timeout_seconds
            ) as client:
                response = await client.post(TOKEN_ENDPOINT, data=data)
        except httpx.TimeoutException as exc:
            raise SearchConsoleError(
                "Google token exchange timed out",
                category="timeout",
                retryable=True,
                status_code=504,
            ) from exc
        except httpx.NetworkError as exc:
            raise SearchConsoleError(
                "Google token exchange failed because of a network error",
                category="network_error",
                retryable=True,
                status_code=503,
            ) from exc
        self._raise_for_google_response(response, operation="exchange authorization code")
        try:
            payload = response.json()
        except Exception as exc:
            raise SearchConsoleError(
                "Google returned an invalid token response",
                category="invalid_response",
                retryable=True,
            ) from exc
        if not isinstance(payload, dict):
            raise SearchConsoleError(
                "Google returned an invalid token response",
                category="invalid_response",
                retryable=True,
            )
        return payload

    async def _refresh_access_token(self, connection: dict[str, Any]) -> str:
        try:
            refresh_token = decrypt_credential(
                connection.get("encrypted_refresh_token"),
                self.settings.credential_encryption_key,
            )
        except CredentialEncryptionError as exc:
            raise SearchConsoleError(
                "Stored Search Console credential cannot be decrypted",
                category="credential_error",
                retryable=False,
                status_code=500,
            ) from exc
        if not refresh_token:
            raise SearchConsoleError(
                "Stored Search Console refresh token is unavailable",
                category="credential_error",
                retryable=False,
                status_code=500,
            )
        secret = self.settings.search_console.client_secret
        data = {
            "client_id": self.settings.search_console.client_id or "",
            "client_secret": secret.get_secret_value() if secret else "",
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        }
        try:
            async with httpx.AsyncClient(
                timeout=self.settings.search_console.request_timeout_seconds
            ) as client:
                response = await client.post(TOKEN_ENDPOINT, data=data)
        except httpx.TimeoutException as exc:
            raise SearchConsoleError(
                "Google access-token refresh timed out",
                category="timeout",
                retryable=True,
                status_code=504,
            ) from exc
        except httpx.NetworkError as exc:
            raise SearchConsoleError(
                "Google access-token refresh failed because of a network error",
                category="network_error",
                retryable=True,
                status_code=503,
            ) from exc
        if response.status_code >= 400:
            error_name = ""
            try:
                payload = response.json()
                error_name = str(payload.get("error") or "") if isinstance(payload, dict) else ""
            except Exception:
                pass
            if error_name == "invalid_grant":
                raise SearchConsoleError(
                    "Search Console access was revoked or expired; reconnect the project",
                    category="refresh_token_revoked",
                    retryable=False,
                    status_code=401,
                )
            self._raise_for_google_response(response, operation="refresh access token")
        try:
            payload = response.json()
        except Exception as exc:
            raise SearchConsoleError(
                "Google returned an invalid access-token response",
                category="invalid_response",
                retryable=True,
            ) from exc
        access_token = str(payload.get("access_token") or "") if isinstance(payload, dict) else ""
        if not access_token:
            raise SearchConsoleError(
                "Google token response did not contain an access token",
                category="invalid_response",
                retryable=True,
            )
        return access_token

    async def list_sync_runs(self, project_id: UUID, *, limit: int = 20) -> list[dict[str, Any]]:
        await self._require_project_exists(project_id)
        rows = await self.repository.list_sync_runs(project_id, limit=limit)
        return [self.public_sync_run(row) for row in rows]

    async def _require_project_exists(self, project_id: UUID) -> None:
        if not await self.performance.repository.project_exists(project_id):
            raise HTTPException(status_code=404, detail="Project not found")

    async def _require_connection(self, project_id: UUID) -> dict[str, Any]:
        self._require_configured()
        connection = await self.repository.get_connection(project_id)
        if not connection:
            raise HTTPException(status_code=404, detail="Search Console is not connected")
        if connection.get("status") in {"revoked", "disconnected"}:
            raise HTTPException(status_code=401, detail="Search Console is disconnected; reconnect")
        return connection

    async def _require_connection_for_worker(self, project_id: UUID) -> dict[str, Any]:
        if not self.settings.search_console.configured:
            raise SearchConsoleError(
                "Search Console OAuth is not configured",
                category="configuration_error",
                retryable=False,
                status_code=503,
            )
        connection = await self.repository.get_connection(project_id)
        if not connection:
            raise SearchConsoleError(
                "Search Console connection was removed",
                category="not_connected",
                retryable=False,
                status_code=404,
            )
        if connection.get("status") in {"revoked", "disconnected"}:
            raise SearchConsoleError(
                "Search Console access was revoked or disconnected; reconnect the project",
                category="refresh_token_revoked",
                retryable=False,
                status_code=401,
            )
        return connection

    def _resolve_sync_window(
        self,
        *,
        date_from: date | None,
        date_to: date | None,
    ) -> tuple[date, date]:
        latest_final_date = datetime.now(timezone.utc).date() - timedelta(
            days=self.settings.search_console.data_lag_days
        )
        resolved_to = date_to or latest_final_date
        resolved_from = date_from or (
            resolved_to - timedelta(days=self.settings.search_console.default_sync_days - 1)
        )
        if resolved_to > latest_final_date:
            raise HTTPException(
                status_code=400,
                detail=(
                    "Search Console sync end date is too recent for finalized data; "
                    f"latest allowed date is {latest_final_date.isoformat()}"
                ),
            )
        if resolved_from > resolved_to:
            raise HTTPException(status_code=400, detail="date_from must be on or before date_to")
        if (resolved_to - resolved_from).days > 365:
            raise HTTPException(status_code=400, detail="Search Console sync window cannot exceed 366 days")
        return resolved_from, resolved_to

    @staticmethod
    def _raise_for_google_response(response: httpx.Response, *, operation: str) -> None:
        if response.status_code < 400:
            return
        code = response.status_code
        reason = ""
        try:
            payload = response.json()
            errors = payload.get("error", {}).get("errors", []) if isinstance(payload, dict) else []
            if errors and isinstance(errors[0], dict):
                reason = str(errors[0].get("reason") or "")
        except Exception:
            pass
        quota_reasons = {
            "rateLimitExceeded",
            "userRateLimitExceeded",
            "dailyLimitExceeded",
            "quotaExceeded",
        }
        if code == 401:
            category, retryable, status_code = "auth_error", False, 401
        elif code == 403 and reason in quota_reasons:
            category, retryable, status_code = "quota_limited", True, 429
        elif code == 403:
            category, retryable, status_code = "permission_denied", False, 403
        elif code == 429:
            category, retryable, status_code = "rate_limited", True, 429
        elif 500 <= code < 600:
            category, retryable, status_code = "google_5xx", True, 502
        elif 400 <= code < 500:
            category, retryable, status_code = "google_4xx", False, 400
        else:
            category, retryable, status_code = "google_error", True, 502
        raise SearchConsoleError(
            f"Google Search Console could not {operation} (HTTP {code})",
            category=category,
            retryable=retryable,
            status_code=status_code,
        )

    @staticmethod
    def _public_property(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "site_url": item["site_url"],
            "permission_level": item["permission_level"],
            "last_seen_at": item.get("last_seen_at"),
        }

    @staticmethod
    def public_sync_run(item: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": str(item["id"]),
            "project_id": str(item["project_id"]),
            "site_url": item["site_url"],
            "date_from": item["date_from"],
            "date_to": item["date_to"],
            "status": item["status"],
            "task_id": item.get("task_id"),
            "row_count": int(item.get("row_count") or 0),
            "pages_fetched": int(item.get("pages_fetched") or 0),
            "truncated": bool(item.get("truncated") or False),
            "retry_count": int(item.get("retry_count") or 0),
            "error_category": item.get("error_category"),
            "error_message": item.get("error_message"),
            "started_at": item.get("started_at"),
            "finished_at": item.get("finished_at"),
            "created_at": item.get("created_at"),
        }
