"""Authenticated encryption for persisted third-party credentials."""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from cryptography.fernet import Fernet, InvalidToken
from pydantic import SecretStr

ENCRYPTED_VALUE_PREFIX = "enc:v1:"


class CredentialEncryptionError(ValueError):
    """Raised when credential encryption configuration or ciphertext is invalid."""


def _secret_value(value: str | SecretStr | None) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, SecretStr):
        value = value.get_secret_value()
    normalized = str(value)
    return normalized if normalized else None


def validate_encryption_key(value: str | SecretStr | None) -> str:
    """Validate and return a Fernet key without exposing it in errors."""
    key = _secret_value(value)
    if not key:
        raise CredentialEncryptionError("CREDENTIAL_ENCRYPTION_KEY is required")
    try:
        Fernet(key.encode("ascii"))
    except (ValueError, TypeError, UnicodeEncodeError) as exc:
        raise CredentialEncryptionError(
            "CREDENTIAL_ENCRYPTION_KEY must be a valid Fernet key"
        ) from exc
    return key


@lru_cache(maxsize=4)
def _fernet(key: str) -> Fernet:
    return Fernet(key.encode("ascii"))


def is_encrypted_credential(value: str | SecretStr | None) -> bool:
    raw = _secret_value(value)
    return bool(raw and raw.startswith(ENCRYPTED_VALUE_PREFIX))


def encrypt_credential(
    value: str | SecretStr | None,
    encryption_key: str | SecretStr | None,
) -> Optional[str]:
    """Encrypt plaintext once; preserve valid versioned ciphertext unchanged."""
    raw = _secret_value(value)
    if raw is None:
        return None

    key = validate_encryption_key(encryption_key)
    if is_encrypted_credential(raw):
        # Validate the token before accepting it as already encrypted.
        decrypt_credential(raw, key)
        return raw

    token = _fernet(key).encrypt(raw.encode("utf-8")).decode("ascii")
    return f"{ENCRYPTED_VALUE_PREFIX}{token}"


def decrypt_credential(
    value: str | SecretStr | None,
    encryption_key: str | SecretStr | None,
) -> Optional[str]:
    """Decrypt versioned ciphertext; preserve legacy plaintext for compatibility."""
    raw = _secret_value(value)
    if raw is None:
        return None
    if not is_encrypted_credential(raw):
        return raw

    key = validate_encryption_key(encryption_key)
    token = raw.removeprefix(ENCRYPTED_VALUE_PREFIX)
    try:
        return _fernet(key).decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise CredentialEncryptionError(
            "Stored credential cannot be decrypted with CREDENTIAL_ENCRYPTION_KEY"
        ) from exc
