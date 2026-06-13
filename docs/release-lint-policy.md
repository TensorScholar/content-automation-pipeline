# Release Lint Policy

## Blocking checks

Production releases require the repository-wide Ruff lint check to pass:

```bash
ruff check .
```

CI pins Ruff `0.5.7`, matching `poetry.lock`, so local and hosted checks use the
same rule behavior. Lint failures in backend code, migrations, tests, or scripts
block the release.

The formatter check is blocking for the Phase 2 security and runtime boundary
files listed in `.github/workflows/ci.yml`. These files handle credentials,
LLM cost controls, error reporting, application startup, and publishing.

## Existing formatter debt

The repository contains legacy files that do not yet match `ruff format`.
Formatting all of them in Phase 2.5 would produce a large review with no runtime
benefit, so the full-repository formatter check is informational in CI.

This is not an exemption from lint correctness: `ruff check .` remains
repository-wide and blocking. Full formatter adoption should be completed in a
dedicated mechanical change, with tests run before and after, rather than mixed
with production feature or security work.
