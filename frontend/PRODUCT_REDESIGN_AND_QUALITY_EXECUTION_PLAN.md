# Smarlux Product, Quality, and UI Execution Plan

- Status: Proposed execution contract
- Date: 2026-07-11
- Scope: Provider transition, generated-article quality, product capability clarity, and frontend redesign
- Execution model: One phase at a time, with a closed verification loop and a work-in-progress limit of one

## 1. Purpose

This document converts the current application into a sequence of bounded engineering phases. It has four goals:

1. Replace the local Qwen dependency only after the prepared provider is proven reliable.
2. Make generated-article quality measurable, reproducible, and release-gated.
3. Clarify what each existing feature actually does, especially Social Media and Task actions.
4. Redesign the UI as a coherent operational product instead of applying isolated visual patches.

This is a plan, not evidence that any phase has passed. Existing runtime validation and UI screenshots remain the source of truth until each phase is completed.

## 2. Governing Decisions

- Reliability and truthful system state precede visual redesign.
- Article quality must be evaluated with a frozen benchmark, not inferred from one generated example or a single synthetic score.
- Background jobs are implementation details. The product should organize work around projects, articles, review, and distribution.
- Social Media Management is not an accurate name for the current feature. The existing behavior is Social Drafts.
- A control must not imply persistence, publication, or automation unless that behavior exists and is verified.
- Public publishing remains fail-closed for risk, approval, and connection readiness.
- Local Qwen is deprecated only after provider qualification, and removed only after a successful cutover and rollback window.
- UI visual acceptance is deferred to the final visual gate, but every UI phase still requires focused functional verification.
- Each phase has at most two self-review iterations. A phase that still fails after the second iteration stops with a blocker report.

## 3. Current Product Truth

The following inventory is based on the current render and service paths, not UI labels.

| Surface | Current behavior | Evidence path | Classification | Product decision |
|---|---|---|---|---|
| Task History | Lists generation jobs, filters/searches them, polls active jobs, and opens article details | frontend/src/components/panels/tasks-panel.tsx | Real but overloaded | Redesign as a Work or Articles workspace; keep technical task state in Activity |
| Content reader | Displays persisted generated article content | frontend/src/components/panels/tasks-panel.tsx and api/routes/content.py | Real | Keep |
| Edit mode | Changes only component-local text; there is no save action in this render path | frontend/src/components/panels/tasks-panel.tsx | Misleading and incomplete | Hide or relabel until connected to a deliberate revision workflow |
| Article revision | A persisted revision request and revision-history path exists, but the Task edit control does not use it | api/routes/content.py and services/content_service.py | Real backend capability, missing UI integration | Use this workflow as the basis for Create revision after contract review |
| Manager review | Persists approve, request-changes, or reject decisions and gates public publishing | frontend/src/components/panels/tasks-panel.tsx and api/routes/content.py | Real safety layer | Keep and make more prominent |
| SEO tab | Displays stored and computed checks, recommendations, readability, and score components | frontend/src/components/panels/tasks-panel.tsx and services/content_service.py | Real calculation with validity limits | Rename to Quality and SEO; replace language-inappropriate metrics |
| Quality score | Combines readability, structure, SEO, and semantic-coherence calculations | services/content_service.py | Incomplete quality proxy | Do not present as comprehensive article quality until calibrated |
| WordPress draft/public actions | Calls real publishing paths; public publishing is gated by risk, approval, and readiness | frontend/src/components/panels/tasks-panel.tsx and api/routes/content.py | Real, with incomplete UX gating | Keep; make every unavailable state explicit and neutral |
| Export | Produces TXT and HTML/Markdown downloads and copy actions | frontend/src/components/panels/tasks-panel.tsx and api/routes/content.py | Real | Keep under Distribution |
| Structured data | Builds deterministic JSON-LD and can include it in HTML export | frontend/src/components/panels/content-studio-panel.tsx and api/routes/content.py | Real, narrow capability | Keep under advanced output settings |
| Social Media Management | Automatically creates fixed LinkedIn, X/Twitter, and Instagram text templates from title/topic, then displays read-only copy fields | frontend/src/components/panels/content-studio-panel.tsx and orchestration/tasks.py | Misnamed and limited | Rename to Social Drafts; do not claim management or publishing |
| Social account publishing | No account connection, schedule, queue, analytics, or direct social-post action exists in the inspected UI path | Current Studio and Task render paths | Not implemented | Defer unless approved as a separate product capability |
| Bulk generation | Starts multiple generation jobs and tracks their states | frontend/src/components/panels/content-studio-panel.tsx | Real | Keep after reliability and UX review |

### 3.1 Current Social Draft Flow

Current behavior:

1. Main article generation completes.
2. The generation task dispatches a secondary social task and returns a social_task_id.
3. The frontend polls that task.
4. The backend constructs deterministic platform templates using the title, topic, and language.
5. Studio renders read-only text with Copy actions.

The social task does not analyze the full article, adapt to a campaign, connect an account, schedule a post, publish content, or collect analytics. It must not be marketed as Social Media Management.

### 3.2 Current Task Detail Risks

- Edit mode suggests saved editing, but changes are temporary component state.
- Social drafts are not integrated into the selected article's durable distribution workspace.
- A user can understand generation job state but not the full article lifecycle at a glance.
- Quality and SEO indicators can appear more authoritative than the underlying language-dependent calculations justify.
- Publishing controls and technical readiness consume the same visual hierarchy as article work.

## 4. Target Product Model

### 4.1 Primary Information Architecture

| Area | User question answered |
|---|---|
| Overview | What needs attention now? |
| Projects | Which brand, site, provider, and publishing configuration applies? |
| Create | What content should be generated, and with which constraints? |
| Work | Which articles are running, ready, blocked, under review, or published? |
| Operations | Are workers, queues, providers, and integrations healthy? |
| Team | Who can review, approve, configure, and publish? |

Task IDs remain available for diagnostics, but they should not be the primary product object.

### 4.2 Target Article Workspace

The current Task detail becomes an Article workspace:

| Workspace area | Contents |
|---|---|
| Header | Article title, project, lifecycle state, owner, language, and primary next action |
| Article | Reader view and a deliberate revision action |
| Quality and SEO | Evidence-backed score dimensions, blockers, recommendations, and evaluation version |
| Distribution | Social drafts, exports, WordPress draft/publication, and connection state |
| Review | Approval state, checklist, reviewer notes, and auditable actions |
| Activity | Generation events, retries, task IDs, timestamps, and technical diagnostics |

### 4.3 Capability Terminology

| Current label | Target label | Reason |
|---|---|---|
| Task History | Work or Articles | Users manage content outcomes, not queue identifiers |
| Social Media Management | Social Drafts | Current behavior generates copy only |
| Edit mode | Create revision | Editing must have persistence, validation, and history |
| SEO | Quality and SEO | SEO is one quality dimension, not the whole quality model |
| Export | Distribution | Publishing, social copy, export, and delivery belong together |

Final display names must be reviewed in English, Persian, and Arabic before implementation.

## 5. Article Quality Engineering Standard

### 5.1 What Quality Means

The release-quality model must score distinct dimensions. A single readability or SEO score is insufficient.

| Dimension | Initial weight | Required evidence |
|---|---:|---|
| Factual and source fidelity | 20 | Claims supported by supplied sources or explicitly marked as uncertain |
| Search/user intent coverage | 15 | Article answers the requested purpose and required subtopics |
| Language fluency and locale fit | 15 | Natural FA/EN/AR grammar, terminology, directionality, and punctuation |
| Depth and usefulness | 15 | Specific, actionable, non-generic information |
| Structure and readability | 10 | Logical heading hierarchy, paragraphs, lists, and transitions |
| SEO naturalness | 10 | Search intent and keyword use without stuffing or mechanical phrasing |
| Brand and instruction adherence | 5 | Tone, audience, format, and project constraints followed |
| Originality and repetition | 5 | No excessive duplication, templated filler, or circular sections |
| Safety and compliance | 5 | No prohibited or unsafe claims; high-risk topics handled appropriately |

These weights are initial hypotheses. Phase 2 calibrates them against human ratings before they become a release gate.

### 5.2 Hard Blockers

An article fails regardless of aggregate score when it contains:

- Fabricated citations, sources, quotations, or claimed evidence.
- Unsupported high-impact factual or quantitative claims.
- The wrong requested language or materially broken locale output.
- Empty, truncated, malformed, or internally contradictory content.
- Prompt leakage, system instructions, secrets, or unsafe content.
- Severe keyword stuffing, repeated sections, or template artifacts.
- Missing mandatory sections or constraints from the generation request.

### 5.3 Evaluation Dataset

Create a versioned, immutable benchmark before tuning:

- 40 initial cases: 24 Persian, 8 English, and 8 Arabic.
- Include informational, commercial, comparison, tutorial, and editorial intent.
- Include short and long form, low and elevated risk, sparse and rich source material.
- Record project settings, prompt version, model/provider, sampling settings, source snapshot, latency, token usage, and cost.
- Keep a hidden holdout subset to detect tuning against visible examples.
- Store expected requirements and forbidden failure modes, not one canonical prose answer.

### 5.4 Evaluation Layers

1. Deterministic checks: schema, language, length, headings, duplicates, required terms, invalid HTML, and prohibited patterns.
2. Grounding checks: claim-to-source coverage and citation validity where sources are supplied.
3. Model-assisted review: rubric-based scoring and blinded pairwise comparison; advisory until calibrated.
4. Human review: sampled bilingual review, with disagreement recorded and adjudicated.
5. Runtime metrics: success rate, retry rate, latency, timeout, token use, cost, and quality-gate pass rate.

An LLM judge must never be the sole release authority.

### 5.5 Initial Release Gate

After calibration, the candidate provider/prompt must meet all of the following:

- Zero hard blockers in the release sample.
- Aggregate median at least 80/100.
- No quality dimension median below 70/100.
- Factual/source-fidelity median at least 85/100 for grounded cases.
- Human acceptance at least 90 percent on the sampled set.
- No statistically or practically material regression against the accepted baseline.
- Runtime success and latency satisfy the SLO defined during provider qualification.

Thresholds may be changed only with benchmark evidence and a recorded decision.

## 6. Provider Transition and Local Qwen Retirement

Local Qwen must be treated as a runtime dependency, not removed by search-and-delete.

### 6.1 Qualification

Without printing credentials:

1. Identify the prepared provider from configured variable names and current provider abstractions.
2. Verify DNS/TLS, authentication, account/quota, model availability, and a minimal completion.
3. Verify structured output, long-form output, FA/EN/AR generation, cancellation, timeout, and retry behavior.
4. Classify failures accurately: configuration, authentication, quota, rate limit, model, timeout, transport, or malformed response.
5. Record p50/p95 latency, error rate, token accounting, and cost.

Credential presence is not provider health. Readiness must not report generation available solely because a key exists.

### 6.2 Cutover

1. Map every role-specific model: keyword, planning, writing, verification, fallback, and any evaluation model.
2. Run the Phase 2 benchmark against the candidate provider.
3. Make the new provider primary behind existing configuration boundaries.
4. Keep local Qwen as a disabled rollback path during the soak window.
5. Run representative end-to-end article generations, including worker dispatch and persisted results.
6. Confirm monitoring and user-visible readiness agree with actual generation capability.

### 6.3 Removal Gate

Remove local Qwen only when:

- The candidate passes the quality and runtime gates.
- At least one representative run exists for each supported language.
- Cutover has completed without an unresolved provider-classification error.
- Rollback instructions have been tested.
- No project configuration still selects the local provider.
- A code-reference and environment-reference audit is clean.

Physical adapter/configuration removal is a separately reviewed change. It must include migration behavior for stale project settings and must not expose secrets in logs or reports.

## 7. UI Design Contract

The target is a quiet, modern, operational SaaS interface.

### 7.1 Structural Rules

- Use one clear page hierarchy: page title, primary action, status summary, work surface.
- Prefer tables, split panes, tabs, and action rails over repeated dashboard cards.
- Do not nest cards or style every section as a floating card.
- Keep individual cards at 8px radius or less.
- Use icons for familiar actions and tooltips for unfamiliar icon-only controls.
- Reserve color for status and primary actions; blocked actions use neutral styling.
- Keep system diagnostics available but visually secondary to user work.
- Make dimensions stable so status changes do not shift the layout.

### 7.2 Visual System

- Neutral base surfaces with distinct semantic colors for success, warning, danger, and information.
- A restrained type scale with compact headings inside operational panels.
- Spacing based on a small consistent scale such as 4, 8, 12, 16, 24, and 32px.
- Strong contrast in both dark and light themes; light mode is an explicit acceptance target, not an inverted afterthought.
- RTL-first validation for Persian and Arabic, including icon direction, mixed Latin text, numbers, and long labels.
- No gradients, decorative blobs, oversized marketing typography, or ornamental glass effects.

### 7.3 Interaction Rules

- Every loading state identifies the operation in progress.
- Every empty state offers the next valid action.
- Every disabled action states the blocking reason.
- Destructive actions require clear confirmation and preserve context.
- Refresh is scoped to the stale resource; polling cannot overlap.
- User edits must either persist with confirmation or be explicitly labeled as temporary preview state.
- Technical errors are translated into actionable user language while retaining a diagnostic reference.

## 8. Closed-Loop Execution Protocol

Every phase follows the same loop:

1. Preflight: verify repository, branch, status, staged files, and allowed paths.
2. Discovery: trace current render/service paths without editing.
3. Plan: publish a Failure / Goal -> path -> minimal change -> verification table.
4. Execute: make only changes mapped to an acceptance criterion.
5. Automated verification: targeted tests, lint, typecheck, build, contracts, and diff checks as appropriate.
6. Functional verification: exercise the changed behavior and negative paths.
7. Self-review: inspect the complete diff and repeat at most twice.
8. Gate report: exact files, diff stat, evidence, blockers, and safety confirmations.
9. Acceptance: do not start the next phase until the current phase is accepted.

Rules:

- Work-in-progress limit is one phase.
- No opportunistic refactoring.
- No mixed provider, quality, publishing, and visual patches.
- Backend/API changes require an explicit file allowlist for that phase.
- UI work requires targeted screenshots before visual acceptance.
- A blocker is reported; it is not bypassed by weakening a safety gate.

## 9. Ordered Delivery Plan

### Phase 0 - Capability Contract and Baseline

Goal: establish the current product truth and freeze scope.

Deliverables:

- This document.
- Current feature and render-path inventory.
- Known dirty-tree baseline.
- Explicit terminology and capability decisions.

Exit gate:

- Every existing Task/Studio feature is classified.
- No unsupported claim of publishing, editing, quality, or social management remains in the plan.

UI required: No.

### Phase 1 - Provider Runtime Truth and Candidate Qualification

Goal: prove the prepared provider can generate content and make readiness truthful.

Scope:

- Provider diagnostics and error classification.
- Model-role inventory.
- Minimal multilingual completion and structured-output smoke tests.
- Runtime SLO proposal.

Exit gate:

- Authentication, quota, model, and completion are verified.
- Readiness evidence distinguishes configured from operational.
- Candidate provider and rollback path are documented.
- No production cutover yet.

UI required: No.

### Phase 2 - Article Quality Benchmark

Goal: make quality measurable before changing prompts or providers.

Scope:

- Versioned benchmark cases and rubric.
- Deterministic checks.
- Human review sheet/protocol.
- Baseline report from existing persisted articles and candidate runs.

Exit gate:

- Benchmark is reproducible.
- Quality dimensions and hard blockers are measurable.
- Current score limitations are documented with evidence.

UI required: No.

### Phase 3 - Provider Cutover and Soak

Goal: make the qualified provider primary without deleting the rollback path.

Scope:

- Role-specific model configuration.
- End-to-end worker generation.
- Observability and failure classification.
- Benchmark comparison.

Exit gate:

- Runtime and quality gates pass.
- Readiness, monitoring, and real generation agree.
- Rollback is tested.

UI required: No, except an optional final readiness confirmation.

### Phase 4 - Generation Quality Remediation

Goal: correct failures demonstrated by the benchmark.

Scope is evidence-driven and may include:

- Prompt contracts.
- Language-specific planning and quality checks.
- Source-grounding behavior.
- Retry/regeneration criteria.
- Replacing English-centric readability assumptions.

Exit gate:

- Candidate passes the calibrated release gate on visible and holdout cases.
- No hard blocker is hidden by aggregate scores.
- Prompt/model/evaluator versions are recorded.

UI required: No.

### Phase 5 - Product Information Architecture Specification

Goal: approve structure before visual implementation.

Deliverables:

- Page map and navigation labels.
- Article lifecycle/state model.
- Studio and Article workspace wire-level layouts.
- Capability terminology in EN/FA/AR.
- Action and permission matrix.

Exit gate:

- Every action has an owner, precondition, result, and failure state.
- Social Drafts and real social publishing are separated.
- Task internals are moved to Activity/Operations.

UI required: No running UI; review is document-based.

### Phase 6 - Shell and Navigation

Goal: establish the shared professional hierarchy without redesigning feature internals.

Scope:

- Navigation, project context, page headers, spacing, and shared status hierarchy.
- Error boundaries and non-overlapping refresh behavior where touched.

Exit gate:

- Navigation is predictable in EN/FA/AR.
- Desktop and mobile layout constraints are stable.
- No feature behavior changes.

UI required: Targeted screenshots.

### Phase 7 - Create and Content Studio

Goal: make generation a focused, trustworthy workflow.

Scope:

- Brief/configuration.
- Readiness and connection state.
- Generation progress and results.
- Bulk and structured-data placement.
- Remove misleading labels.

Exit gate:

- WordPress does not block generation.
- Worker/provider readiness is fresh and truthful.
- The primary generation path has one obvious next action.
- Social output is labeled as draft copy.

UI required: Targeted screenshots and one end-to-end generation.

### Phase 8 - Work and Article Workspace

Goal: replace the overloaded Task detail with an article-centered workspace.

Scope:

- Work list/state model.
- Article, Quality and SEO, Distribution, Review, and Activity areas.
- Deliberate revision path.
- Timestamp consistency audit.

Exit gate:

- No unsaved Edit mode is presented as persistence.
- Review and publication blockers are clear.
- Technical task details remain accessible without dominating the screen.

UI required: Targeted screenshots and persisted-state reload tests.

### Phase 9 - Social Drafts

Goal: make the existing feature useful and accurately scoped.

Minimum approved capability:

- Generate drafts from the persisted article, project voice, language, and platform constraints.
- Persist or rediscover drafts from the Article workspace.
- Allow deliberate editing and copying with clear save state.
- Validate platform-specific length and required fields.
- Never imply account publishing or scheduling.

Separate future capability, not included by default:

- Account connections.
- Scheduling/calendar.
- Direct publishing.
- Approval workflows per channel.
- Analytics and campaign reporting.

Exit gate:

- The UI label matches actual behavior.
- Draft output is article-aware and durable.
- Unsupported management capabilities are absent.

UI required: Targeted screenshots and reload/persistence tests.

### Phase 10 - Review, Publishing, and Secondary Screens

Goal: finish safety-critical distribution and then apply the system to Projects, Operations, Team, and login.

Exit gate:

- Public publishing fails closed.
- Draft/public distinctions are unambiguous.
- WordPress configuration errors are prevented before action where possible.
- Secondary screens use the same hierarchy and status language.

UI required: Targeted screenshots and negative-path publishing tests.

### Phase 11 - Final Visual and Release Gate

Goal: perform the deferred complete UI review and release decision.

Required screenshot matrix:

- Persian RTL, dark, 1440x900: Overview, Create readiness, generation running/success, Work list, Article, Quality, Social Drafts, and blocked publication.
- Persian RTL, light, 1440x900: the same critical states.
- English LTR, dark, 1440x900: Create, Article, and Distribution.
- Persian RTL, 390x844: navigation, Create, Article, and blocked action states.

Required checks:

- No overlap, clipping, layout shift, untranslated copy, raw object rendering, or false primary action.
- Light mode has hierarchy and contrast equal to dark mode.
- Browser console has no product-caused hydration, runtime, or request errors.
- Keyboard focus, labels, and contrast meet the agreed accessibility baseline.
- Final end-to-end generation, review, draft publication, and safely blocked public publication pass.

Exit gate:

- Explicit visual approval.
- Automated checks pass.
- Runtime and article-quality release gates pass.
- Remaining issues are documented and intentionally deferred.

## 10. Test Strategy

| Layer | Purpose |
|---|---|
| Unit | Formatters, gates, language logic, scoring, and state transitions |
| Contract | API schemas, provider error mapping, task results, review, and readiness |
| Integration | Provider calls, Celery task flow, persistence, quality evaluation, and WordPress validation |
| End-to-end | Generate -> inspect -> review -> distribute, including blocked paths |
| Visual | RTL/LTR, dark/light, desktop/mobile, long copy, loading/error/empty states |
| Resilience | Timeout, rate limit, quota loss, worker loss, stale readiness, retry, and cancellation |
| Regression | Frozen quality benchmark and approved screenshot states |

Tests must assert user-visible outcomes and safety properties, not only implementation details.

## 11. Reliability and Observability Requirements

Before release, dashboards or structured logs must make these distinguishable:

- Provider configured versus provider operational.
- Worker registered versus worker currently available.
- Generation queued, running, retrying, failed, and persisted.
- Provider authentication, quota, rate-limit, model, timeout, and transport failures.
- Quality-gate failure versus runtime failure.
- WordPress disconnected, invalid configuration, authorization failure, and publication failure.

Initial SLOs must be baselined before enforcement. At minimum, track:

- Generation acceptance and completion rate.
- p50 and p95 end-to-end generation latency.
- Retry and timeout rate.
- Stuck-task count and recovery time.
- Quality hard-blocker rate and human acceptance rate.
- Publish success rate by draft/public mode.
- Cost and token usage per accepted article.

## 12. Change and Safety Controls

- Preserve the existing dirty working tree; never overwrite unrelated modifications.
- Each phase starts with git status and an exact allowed-file list.
- Do not touch artifacts.
- Do not stage, commit, push, stash, reset, restore, merge, or rebase without explicit approval.
- Do not change schemas, dependencies, Docker, CI, or lockfiles as an incidental part of another phase.
- Do not print environment values or secrets.
- Stop when unexpected files change.
- Report visual work as unaccepted until the required screenshots are reviewed.

## 13. Risk Register

| Risk | Consequence | Control |
|---|---|---|
| Removing local Qwen before replacement soak | No generation rollback | Deprecate, cut over, soak, then remove |
| Treating credential presence as health | UI says ready while generation fails | Live operational probe and accurate error classes |
| Optimizing one synthetic quality score | High scores with weak or false articles | Multidimensional benchmark plus human calibration |
| Calling drafts social management | User expects publishing/scheduling that does not exist | Rename and define capability boundary |
| Preserving unsaved Edit mode | Users lose work | Hide until revision persistence is implemented |
| Redesigning all screens at once | Large regressions and unreviewable diff | One bounded surface per phase |
| Mixing visual and behavior changes | Root cause becomes unclear | Separate functional, structural, and visual gates |
| English-centric evaluation in FA/AR | Invalid quality judgments | Locale-specific checks and bilingual human review |

## 14. Per-Phase Report Template

Every phase report must include:

1. Repository, branch, status, and staged-state preflight.
2. Planning table used.
3. Exact files modified.
4. Exact behavior changed.
5. Diff stat and targeted search results.
6. Test commands and results.
7. Functional or visual evidence.
8. Known limitations and remaining blockers.
9. Rollback notes.
10. Confirmation that unrelated files, artifacts, and Git history were untouched.

## 15. Immediate Next Step

The next executable unit is Phase 1: Provider Runtime Truth and Candidate Qualification.

Its first action is inspection only:

1. Inventory provider names, model-role settings, and health/readiness paths without printing values.
2. Identify the prepared target provider.
3. Produce a phase-specific allowed-file list and planning table.
4. Run non-destructive diagnostics.
5. Report qualification evidence before any cutover or removal.

No UI is required for Phase 1. This allows provider reliability and article-quality work to proceed while the final visual review remains intentionally deferred.
