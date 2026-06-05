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