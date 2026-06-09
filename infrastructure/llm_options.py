"""LLM provider/model option catalog.

This module is intentionally small and read-only: it exposes which providers
are usable from the current process without ever returning API key material.
API routes use it to validate per-request model overrides before queueing
expensive Celery work.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from config.settings import get_settings


@dataclass(frozen=True)
class LLMModelOption:
    provider: str
    model: str
    label: str
    enabled: bool
    recommended: bool = False
    reason: str | None = None


@dataclass(frozen=True)
class LLMProviderOption:
    provider: str
    label: str
    configured: bool
    active: bool
    models: list[LLMModelOption]


PROVIDER_LABELS = {
    "anthropic": "Anthropic",
    "openai": "OpenAI",
    "gemini": "Google Gemini",
    "local": "Local LLM",
}

CATALOG_MODELS: dict[str, list[tuple[str, str, bool]]] = {
    "gemini": [
        ("gemini-2.5-flash-lite", "Gemini 2.5 Flash-Lite", True),
        ("gemini-2.5-flash", "Gemini 2.5 Flash", False),
    ],
    "anthropic": [
        ("claude-haiku-4-5-20251001", "Claude Haiku 4.5", True),
        ("claude-sonnet-4-5-20250514", "Claude Sonnet 4.5", False),
    ],
    "openai": [
        ("gpt-4o-mini", "GPT-4o mini", True),
        ("gpt-4o", "GPT-4o", False),
    ],
}


def infer_provider(model: str) -> str | None:
    """Infer provider from an app-level or LiteLLM-prefixed model name."""
    value = model.lower().strip()
    if value.startswith("openai/") or value.startswith("gpt-"):
        return "openai"
    if value.startswith("anthropic/") or value.startswith("claude-"):
        return "anthropic"
    if value.startswith("gemini/") or value.startswith("gemini-"):
        return "gemini"
    if value.startswith("local-") or value.startswith("ollama/"):
        return "local"
    return None


def normalize_model_name(model: str) -> str:
    """Strip provider prefixes for display, pricing, and catalog matching."""
    value = model.strip()
    if "/" in value:
        return value.split("/", 1)[1]
    return value


def has_provider_credentials(provider: str) -> bool:
    """Return whether the current process can attempt a provider call."""
    settings = get_settings()
    normalized = provider.lower()

    if normalized == "anthropic":
        return bool(
            settings.llm.anthropic_api_key
            or os.getenv("ANTHROPIC_API_KEY")
            or os.getenv("LLM_ANTHROPIC_API_KEY")
        )
    if normalized == "openai":
        return bool(
            settings.llm.openai_api_key
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("LLM_OPENAI_API_KEY")
        )
    if normalized == "gemini":
        return bool(
            settings.llm.gemini_api_key
            or os.getenv("GEMINI_API_KEY")
            or os.getenv("GOOGLE_API_KEY")
            or os.getenv("LLM_GEMINI_API_KEY")
        )
    if normalized == "local":
        return bool(os.getenv("LOCAL_LLM_URL"))
    return False


def _unique(items: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for item in items:
        if not item:
            continue
        normalized = normalize_model_name(item)
        if normalized in seen:
            continue
        seen.add(normalized)
        result.append(normalized)
    return result


def _configured_models_for_provider(provider: str) -> list[str]:
    settings = get_settings()
    active_provider = infer_provider(settings.llm.primary_model)

    if provider == "gemini":
        configured = [
            settings.llm.gemini_model,
            os.getenv("LLM_GEMINI_FALLBACK_MODEL"),
            settings.llm.primary_model if active_provider == "gemini" else None,
            settings.llm.fallback_model if infer_provider(settings.llm.fallback_model or "") == "gemini" else None,
        ]
    elif provider == "anthropic":
        configured = [
            settings.llm.anthropic_model,
            os.getenv("LLM_ANTHROPIC_FALLBACK_MODEL"),
            settings.llm.primary_model if active_provider == "anthropic" else None,
            settings.llm.fallback_model if infer_provider(settings.llm.fallback_model or "") == "anthropic" else None,
        ]
    elif provider == "openai":
        configured = [
            os.getenv("LLM_OPENAI_MODEL", "gpt-4o-mini"),
            os.getenv("LLM_OPENAI_FALLBACK_MODEL"),
            settings.llm.primary_model if active_provider == "openai" else None,
            settings.llm.fallback_model if infer_provider(settings.llm.fallback_model or "") == "openai" else None,
        ]
    else:
        configured = [
            settings.llm.primary_model if active_provider == "local" else None,
            settings.llm.secondary_model if infer_provider(settings.llm.secondary_model or "") == "local" else None,
            settings.llm.fallback_model if infer_provider(settings.llm.fallback_model or "") == "local" else None,
            os.getenv("LLM_LOCAL_FALLBACK_MODEL"),
        ]

    catalog = [model for model, _label, _recommended in CATALOG_MODELS.get(provider, [])]
    return _unique([*configured, *catalog])


def get_llm_provider_options() -> list[LLMProviderOption]:
    """Build redacted provider/model options for API responses."""
    settings = get_settings()
    active_provider = infer_provider(settings.llm.primary_model) or "unknown"

    providers: list[LLMProviderOption] = []
    for provider in ("gemini", "anthropic", "openai", "local"):
        configured = has_provider_credentials(provider)
        catalog_by_model = {
            model: (label, recommended)
            for model, label, recommended in CATALOG_MODELS.get(provider, [])
        }
        models: list[LLMModelOption] = []

        for model in _configured_models_for_provider(provider):
            label, recommended = catalog_by_model.get(model, (model, False))
            models.append(
                LLMModelOption(
                    provider=provider,
                    model=model,
                    label=label,
                    enabled=configured,
                    recommended=recommended,
                    reason=None if configured else f"{PROVIDER_LABELS.get(provider, provider)} API key is not configured.",
                )
            )

        providers.append(
            LLMProviderOption(
                provider=provider,
                label=PROVIDER_LABELS.get(provider, provider),
                configured=configured,
                active=active_provider == provider,
                models=models,
            )
        )

    return providers


def get_selectable_models() -> list[LLMModelOption]:
    """Return models that can be selected safely by users."""
    return [
        model
        for provider in get_llm_provider_options()
        for model in provider.models
        if model.enabled
    ]


def validate_model_available(model: str | None) -> tuple[bool, str | None]:
    """Validate a per-request model override without calling the provider."""
    if not model:
        return True, None

    provider = infer_provider(model)
    if provider is None:
        return False, f"Unsupported LLM model '{model}'. Use a Gemini, Anthropic, OpenAI, or local model."

    if not has_provider_credentials(provider):
        return (
            False,
            f"{PROVIDER_LABELS.get(provider, provider)} is not configured. Add its API key before selecting {model}.",
        )

    return True, None


def build_llm_warnings() -> list[str]:
    """Return configuration warnings safe for manager-facing UI."""
    settings = get_settings()
    warnings: list[str] = []

    active_provider = infer_provider(settings.llm.primary_model)
    if active_provider is None:
        warnings.append(f"Active model '{settings.llm.primary_model}' has an unknown provider prefix.")
    elif not has_provider_credentials(active_provider):
        warnings.append(
            f"Active model '{settings.llm.primary_model}' belongs to {PROVIDER_LABELS.get(active_provider, active_provider)}, but that provider has no configured API key."
        )

    fallback = settings.llm.fallback_model
    fallback_provider = infer_provider(fallback or "") if fallback else None
    if fallback and fallback_provider and not has_provider_credentials(fallback_provider):
        warnings.append(
            f"Fallback model '{fallback}' is configured, but {PROVIDER_LABELS.get(fallback_provider, fallback_provider)} has no API key."
        )

    if not any(has_provider_credentials(provider) for provider in ("gemini", "anthropic", "openai", "local")):
        warnings.append("No LLM provider is configured. Content generation will fail until an API key or local endpoint is available.")

    return warnings
