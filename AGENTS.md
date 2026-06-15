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
