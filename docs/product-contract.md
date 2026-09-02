# Product Contract

**Status:** Canonical product-direction contract  
**Scope:** Initial commercial wedge and product boundaries  
**Evidence status:** Product hypothesis informed by repository audit and public market research; not customer validated

## 1. Product Thesis

This product is a **content-operations control loop for lean B2B SaaS teams**.

It is not a generic AI writer, agent builder, SEO suite, social-media manager, CMS, or generalized marketing-automation platform.

The product exists to reduce the repetitive operational work between a content decision and a safely reconciled publication while preserving editorial judgment, factual integrity, brand consistency, bounded AI cost, and recoverable execution.

The initial loop is:

```text
Campaign objective
    -> Brief
    -> Source context
    -> Content asset + immutable revision
    -> Deterministic and semantic validation
    -> Exception-driven review
    -> WordPress schedule/publish
    -> Publication reconciliation
    -> Search Console performance
    -> Evidence-backed next action
```

The durable differentiation hypothesis is **safe operational leverage**, not raw generation quality or feature breadth.

## 2. Initial Customer

### Target customer

A lean B2B SaaS company operating an owned-content program with:

- a small content/growth team;
- WordPress as a primary publishing surface;
- Google Search Console as an important first-party performance source;
- recurring educational, product-led, comparison, or search-oriented content;
- enough publishing volume that handoffs, QA, scheduling, status tracking, and repurposing create recurring operational load;
- insufficient scale or desire to staff a dedicated content-operations function.

The initial product is optimized for one internal team and one or a small number of brands/sites. Multi-tenant agency operation is not an initial requirement.

### Primary user

**Content Marketing Lead / Content Lead / Growth Lead** responsible for shipping an owned-content program with limited operational headcount.

Secondary users are editors/reviewers and an operator with publishing/configuration permissions.

### Job to be done

> Keep a high-quality B2B content program moving from campaign intent to publication and measurable outcome without manually coordinating every routine transition, while retaining human control over genuinely risky or creative decisions.

### Primary triggers

- a new campaign or content objective;
- an approved topic/backlog item that needs production;
- an existing publication that requires refresh or repurposing based on first-party performance evidence.

## 3. Value Contract

The product should reduce **necessary human operational effort**, not merely generation time.

Primary value metrics are measured, never assumed:

- human touches per asset;
- manual operational minutes per asset;
- review-required rate;
- exception rate;
- rework rate;
- failure-recovery interventions;
- percentage of assets reaching a reconciled publication without manual operational intervention;
- accepted-brief-to-reconciled-publication elapsed time;
- AI cost per asset and per campaign;
- publication ambiguity/duplicate incidents;
- source-support coverage for material factual claims where source-backed generation is required.

No improvement target is a product claim until prospectively measured.

## 4. Canonical Product Objects

These are product concepts. Implementation names may remain temporarily different during migration, but new behavior must not create competing definitions.

### Workspace

The team-level operating boundary. V1 may remain a single-workspace deployment; speculative SaaS multi-tenancy is explicitly deferred.

### Brand

The operational content identity: audience, positioning, terminology, voice, formatting preferences, product naming, CTA conventions, and bounded claim restrictions.

The current `Project` model partially combines brand, site, and integration configuration. Phase 3 must decide how to migrate or narrow it rather than creating a second parallel brand model.

### Campaign

A bounded content objective with an owner, audience, objective, time context, and success signals. Campaigns organize decisions; they are not generic project-management containers.

### Brief

The execution contract for one intended content asset: purpose, audience, angle, required questions/sections, source context, constraints, target channel, and acceptance requirements.

A brief must be inspectable before expensive generation starts.

### Content Asset

The durable logical content item independent of any single generated draft or external publication.

### Revision

An immutable version of a Content Asset. Content, approval, validation evidence, and publication intent must resolve to a specific revision identity.

An edit creates a new revision. It must never silently mutate the revision that was previously reviewed, scheduled, or published.

### Review Decision

A human or policy decision tied to exactly one revision. Approval of revision `N` never implies approval of revision `N+1`.

### Publication

A channel-specific external side effect tied to exactly one revision. It owns scheduling, idempotency, external identity, ambiguous outcomes, retries, reconciliation, and terminal status.

### Performance Snapshot

Source-aware metrics associated with a reconciled publication and campaign context. Performance data is evidence for future decisions, not an automatic claim of causality.

## 5. State Contract

Editorial state and external-publication state are separate lifecycles. They must not be collapsed into a boolean flag set or one overloaded status enum.

### Revision editorial lifecycle

A minimal conceptual lifecycle is:

```text
DRAFT
  -> VALIDATING
  -> REVIEW_REQUIRED | APPROVED

REVIEW_REQUIRED -> APPROVED | REJECTED
APPROVED -> superseded only by creation of a newer revision
```

Exact persisted states may change during Phase 4, but these invariants are mandatory:

1. An edit creates or identifies a new revision.
2. Approval is revision-bound.
3. Validation evidence is revision-bound.
4. A newer revision cannot inherit approval silently.
5. Concurrent review/edit operations cannot approve content that the reviewer did not evaluate.

### Publication lifecycle

A separate minimal lifecycle is:

```text
UNSCHEDULED
  -> SCHEDULED
  -> PUBLISHING
  -> PUBLISHED | AMBIGUOUS | FAILED

SCHEDULED -> CANCELLED
FAILED -> retry/reconcile only according to the side-effect contract
AMBIGUOUS -> RECONCILING -> PUBLISHED | FAILED | ACTION_REQUIRED
```

Mandatory invariants:

1. Every publication references one immutable revision.
2. Public/scheduled publication requires the applicable approval policy for that revision.
3. A timeout after a possibly successful side effect is reconciled before re-creation.
4. Idempotency identity and external publication identity are durable.
5. Editing a content asset cannot silently change an already scheduled or published remote payload.

## 6. V1 Workflow

### 6.1 Plan

A campaign owner provides:

- objective;
- audience;
- topic or content opportunity;
- relevant source material;
- brand context;
- channel intent;
- explicit constraints;
- optional cost limit.

The system produces a brief. It may automate routine decomposition, but the resulting brief remains inspectable.

### 6.2 Produce

The system assembles trusted context and generates a candidate revision using bounded AI calls.

Deterministic work remains deterministic: schema checks, required fields, URL rules, budget enforcement, revision identity, state transitions, and publication identity do not use an LLM.

### 6.3 Validate

Quality remains multidimensional. Initial dimensions are:

- factual/source support;
- brief adherence;
- audience fit;
- brand fit;
- usefulness/depth;
- originality/redundancy;
- structural/readability checks appropriate to the language;
- channel requirements;
- claim risk.

No uncalibrated aggregate grade is a release authority.

### 6.4 Review by exception

The system should route a revision to a human when deterministic policy or calibrated semantic checks identify material risk or ambiguity.

Initial examples include:

- unsupported material factual claims;
- sensitive or high-impact claims;
- substantial brief deviation;
- low-confidence brand adherence;
- validation failure;
- explicit campaign-owner policy.

The product does not assume that every low-risk routine asset always requires a manual approval. Automatic approval policies may be introduced only after quality evidence supports them.

### 6.5 Publish and reconcile

WordPress is the only initial direct publishing connector.

The system owns:

- scheduling;
- idempotency;
- publish attempt state;
- ambiguous-outcome handling;
- external post identity;
- reconciliation;
- retryable recovery;
- operator-visible failure semantics.

### 6.6 Measure and recommend

Google Search Console is the initial first-party performance integration.

The system may surface evidence-backed opportunities such as refresh, expansion, consolidation, or repurposing. It must retain source, time window, publication, campaign objective, and uncertainty context.

It must not claim causal optimization from small observational datasets.

## 7. Initial Channels and Integrations

### Direct publishing

- **WordPress:** supported and hardened first.

### Adaptation without direct publication

- **LinkedIn:** channel-specific adaptation/export is justified for B2B SaaS, but direct account publishing is deferred until a real connector is selected and reliability requirements are met.

A generated copy block is not a successful publication.

### Performance

- **Google Search Console:** read-only performance ingestion and reconciliation with owned web content.

### AI execution

The application consumes an inference abstraction and records product-level usage/cost. Generalized provider routing, pricing intelligence, SLO optimization, and provider reliability are not product-owned differentiators; those belong to InferenceLedger.

## 8. Human / Automation Boundary

### Humans own

- campaign strategy and objective choice;
- distinctive creative direction;
- genuinely ambiguous editorial judgment;
- sensitive/high-impact claims;
- policy exceptions;
- decisions whose business or reputational risk exceeds the configured automation policy.

### The system should own when safe

- brief scaffolding;
- source/context assembly;
- routine drafting;
- metadata and formatting;
- deterministic validation;
- routine semantic QA;
- duplicate/repetition checks;
- channel adaptation;
- scheduling mechanics;
- retryable recovery;
- publication reconciliation;
- analytics ingestion;
- status synchronization;
- surfacing next-action candidates.

The user should manage exceptions and decisions, not Celery jobs, Redis queues, worker identities, or provider implementation details.

## 9. V1 Product Surface

The initial customer-facing information architecture should converge toward:

- **Overview / Attention Queue** — what needs human action and what the system is already handling;
- **Campaigns** — objective, audience, assets, status, and outcome context;
- **Content** — briefs, assets, immutable revisions, validation, and review;
- **Publishing** — schedule, publication state, reconciliation, and failures;
- **Performance** — first-party outcome evidence and next-action candidates;
- **Brand Settings** — operational brand rules and integrations;
- **Operations** — diagnostic surface for authorized operators, visually secondary to content work.

Exact navigation names are a Phase 9 design decision. Internal task/job concepts must not become the primary IA.

## 10. Explicit Non-Goals

V1 does **not** attempt to provide:

- a generic AI writer or prompt playground;
- a generic agent builder or agent marketplace;
- a general-purpose inference router or provider optimizer;
- a full SEO/AEO/GEO platform;
- a headless CMS or DAM;
- a CRM or marketing-automation suite;
- a full social-media scheduler/engagement suite;
- a broad publishing connector catalog;
- direct LinkedIn/X/Instagram publishing before connector reliability is proven;
- email-campaign delivery infrastructure;
- image/video generation pipelines;
- speculative multi-tenant agency infrastructure;
- enterprise SSO/SCIM/governance breadth before customer demand;
- autonomous public publishing without a validated policy;
- compliance/trust-claim governance;
- generalized behavioral evaluation infrastructure;
- model training on small performance datasets presented as optimization intelligence.

## 11. Language Scope

Commercial validation starts with **English B2B content** to keep the quality surface measurable.

Existing Persian/RTL capability may remain available as staging functionality, but it is not part of the initial commercial quality claim. Arabic, Persian, and broader localization become supported commercial workflows only after locale-specific generation, validation, UI, and human-review evidence exists.

Language expansion must not reuse English-only readability or SEO heuristics as if they were universal.

## 12. Evidence and Claim Discipline

Product evidence must use explicit labels:

- ENGINEERING VERIFIED
- SYNTHETICALLY OBSERVED
- RETROSPECTIVELY OBSERVED
- PROSPECTIVELY OBSERVED
- CUSTOMER VALIDATED
- NOT VALIDATED

Current commercial wedge and value proposition are **NOT CUSTOMER VALIDATED**.

Green CI proves engineering checks, not content quality, customer value, labor reduction, conversion lift, or production deployment readiness.

## 13. Architecture Consequences

This contract constrains subsequent engineering work:

1. Phase 3 must remove or isolate generic distribution paths that can report side-effect success without a real connector.
2. Phase 4 must establish revision-bound review and publication semantics before adding campaign automation.
3. Existing `Project`, article, review, and publishing representations must be consolidated rather than duplicated with parallel models.
4. Generalized LLM infrastructure must be treated as an adapter boundary, not expanded as a product differentiator.
5. Frontend restructuring follows the canonical product objects and attention-driven workflow, not backend task topology.
6. Campaign/Brief persistence is added only after revision/publication integrity is fixed; correctness precedes workflow breadth.

## 14. Product Contract Exit Gate

This contract is considered implemented only when a realistic prospective workflow can demonstrate, with preserved evidence:

```text
Campaign objective
-> inspectable brief
-> source-backed revision
-> validation
-> exception review only where policy requires
-> revision-bound WordPress schedule/publish
-> reconciliation
-> Search Console performance ingestion
-> evidence-backed next-action candidate
```

The implementation does not need every transition to be autonomous initially. It must, however, preserve truthful state, bounded side effects, and a clear path to reducing human operational touches without weakening editorial control.