# AGENTS.md

## Core Architecture
- This project uses Next.js for the frontend and FastAPI for the backend.
- IGNORE any claims in the README.md that state the UI is built with Streamlit.

## Rules of Engagement
- Strictly operate within the `frontend/` directory unless instructed otherwise.
- Priority 1: System Reliability.
- Do NOT implement complex UI/UX architectures (like macOS glassmorphism) until the core runtime is stable and crash-free.

## Technical Mandates
- Fix memory leaks by properly cleaning up event listeners and unmounted states.
- Stop overlapping API polling. Implement proper AbortControllers or connection deduplication.
- Ensure strict Error Boundaries are wrapping critical React components.

## Low-Token UI Polish Workflow
- Work in small, reviewable patches.
- Use short task briefs.
- Do not broaden scope.
- Do not implement adjacent phases unless explicitly asked.
- For UI tasks, inspect only files relevant to the stated issue.
- Prefer minimal frontend-only diffs.
- If visual verification is needed, ask the user for exact manual screenshots instead of running broad automated screenshot flows.
- Screenshot requests must specify exact screen, state, language/theme, window size if relevant, and what must be visible.
- Report before commit.

## Closed-Loop Launch-Critical Workflow
- Launch-critical work must use a closed loop: discovery → planning → execution → verification → bounded iteration. Do not perform open-ended exploration unless explicitly requested.
- Start discovery with `git status -sb`; inspect relevant files and render paths before changing code, and do not edit during discovery.
- For non-trivial tasks, report a planning table before patching: `Failure / Goal → file or render path → minimal planned change → verification method`.
- Make surgical changes only: every changed line must map to the explicit goal or acceptance criteria. Do not broaden scope, refactor adjacent code, or add speculative abstractions/features unless explicitly requested.
- Respect safety boundaries: no backend, schema, migration, dependency, Docker, CI, or lockfile changes unless explicitly allowed. If backend files appear modified during a frontend-only task, stop and report.
- Do not use `git add .`; do not stage, commit, or push unless explicitly approved; do not stash pop/apply; never touch `artifacts/`.
- UI changes are not accepted by lint/build alone. Require targeted manual screenshots or explicit visual approval, and do not claim a visual pass without seeing the UI.
- Use relevant `rg` checks for raw strings, localization leaks, and acceptance criteria. Run validation commands appropriate to the touched area; for frontend changes, run lint, typecheck, and build unless the user explicitly narrows validation.
- Self-review loops must have a maximum iteration count. If still failing after the limit, stop and report remaining blockers instead of expanding scope.
- Final reports must include exact modified files, diff stat, validation commands/results, remaining blockers, and safety confirmations: no backend changes, no artifacts touched, no stash pop/apply, and nothing staged/committed/pushed.

## Commit and Screenshot Rules
- Do not use `git add .`.
- Stage exact files only.
- Do not stage screenshots.
- Do not stage `artifacts/`.
- Do not print `.env` values.
- Do not expose secrets.
- Do not commit or push unless explicitly approved.
- Before commit, run relevant validation:
  - `git diff --check`
  - `cd frontend && npm run lint`
  - `cd frontend && npm run typecheck`
  - `cd frontend && npm run build`
- After push, verify GitHub Actions on the pushed SHA:
  - `CI/CD Pipeline`
  - `Phase 1 Verification Gate`
