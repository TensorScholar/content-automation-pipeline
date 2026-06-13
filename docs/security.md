# Security Baseline

## Credential Encryption

WordPress application passwords are encrypted before persistence with Fernet
(AES-128-CBC plus HMAC authentication as implemented by `cryptography`).
Ciphertext is versioned with the `enc:v1:` prefix.

Generate the production key once:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Store the value as `CREDENTIAL_ENCRYPTION_KEY` in the deployment secret store.
Do not place it in Git. Losing this key makes stored WordPress credentials
unrecoverable. Rotating it requires decrypting with the previous key and
re-encrypting with the new key.

Unprefixed database values are treated as legacy plaintext for backward
compatibility. The next project credential update writes the value encrypted.
API response models never include the application password.

## Secret Redaction

Standard logging, structlog, Loguru, request logging, and Sentry events pass
through `infrastructure/redaction.py`. It redacts authorization/cookie headers,
JWT and API-key patterns, connection-string passwords, WordPress passwords,
SMTP passwords, and encryption keys.

Redaction is defense in depth, not permission to log secrets. New code must not
log request bodies, provider prompts, environment dumps, or decrypted
credentials.

## Sentry

Sentry is optional:

```dotenv
SENTRY_DSN=
SENTRY_ENVIRONMENT=production
SENTRY_TRACES_SAMPLE_RATE=0
```

When configured, FastAPI and Celery integrations capture unhandled exceptions.
Default PII collection is disabled and events are redacted before transmission.
Production emits a warning, but does not fail startup, when Sentry is absent.
