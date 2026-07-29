# Release Status

- **Current classification:** Staging only
- **Production approval:** Not granted
- **Last reviewed:** 2026-07-19

This repository contains production-oriented components, deployment definitions,
reliability controls, and test coverage. Their presence does not constitute a
production release decision. The current build may be used for local development
and controlled staging validation only.

Do not use the current build for:

- public general-availability traffic;
- unattended public publishing;
- production data without an approved backup and recovery procedure; or
- a production deployment built from an unreviewed or dirty working tree.

## Production Approval Gates

Production approval requires recorded evidence that all of the following gates
have passed for one immutable release commit and image digest:

1. The release scope is reviewed and the working tree is clean.
2. Backend, frontend, security, contract, integration, and browser end-to-end CI
   gates pass from a clean checkout.
3. Production credentials are stored securely and any exposed credentials have
   been rotated.
4. Database backups are automated, stored off-site, and proven by a restore
   drill against the approved RPO and RTO.
5. Production error tracking, metrics, logs, alert routing, and operational
   runbooks are active.
6. The selected LLM provider passes multilingual reliability, latency, cost, and
   failure-mode qualification.
7. The versioned article-quality and SEO benchmark passes deterministic,
   holdout, and human-review gates.
8. Critical FA, AR, and EN workflows pass browser, accessibility, responsive,
   RTL/LTR, and visual acceptance checks.
9. Migration, canary promotion, rollback, and post-release smoke procedures are
   exercised successfully.
10. No unresolved severity-1 or severity-2 issue remains, and the CTO records
    the final production decision.

Until every gate is satisfied, documentation and user communication must refer
to the system as a staging candidate, not as production ready.
