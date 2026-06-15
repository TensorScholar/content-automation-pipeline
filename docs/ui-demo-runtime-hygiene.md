# UI Demo Runtime Hygiene

This checklist is for CEO/pilot screenshots and local Tauri demos only. It must not be run against production data.

## Runtime

- Use a clean local database or a disposable staging database.
- Start the backend with local/staging-safe environment values only.
- Start the macOS app through Tauri after the backend is healthy.
- Confirm the Tauri window does not show the Next.js dev indicator or issue badge.
- Do not include screenshots that show local stack traces, raw database errors, API keys, WordPress app passwords, or Authorization headers.

## Demo Data

Use realistic names in the visible demo path:

- Project name: `Northstar Editorial`
- Domain: `northstar.example`
- Article title: `2026 Content Operations Readiness`
- Manager: `Admin`

Do not show internal/test names in CEO or pilot screenshots:

- `Canary Smoke`
- `Test Project`
- `Launch readiness smoke`
- `Smoke`
- `QA`
- raw task ids as the main visible label

## Safe Local Cleanup Guidance

Do not delete or rewrite existing data automatically. If a local demo database already contains smoke-test rows, create or select a clean demo project instead of removing historical test records.

Recommended local-only approach:

1. Create one clean demo project from the UI.
2. Select that project in the sidebar.
3. Generate or import one clean demo article under that project.
4. Capture dashboard, project, task detail, review, publishing safety, and performance screenshots.
5. Leave production/staging data untouched.

If a reset is required, use a disposable local database or a fresh Docker volume prepared only for screenshots. Do not run destructive cleanup against shared staging or production databases.
