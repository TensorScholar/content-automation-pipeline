# Article Quality Human Review Protocol v1

## Purpose

This protocol governs human review of the `phase2-v1` article benchmark. It
supplements deterministic checks; it does not replace the release gate or
authorize production release by itself.

## Review Inputs

Each review packet must contain the benchmark case ID, anonymized candidate
output, requested language, target word count, required sections and keywords,
source snapshot when applicable, provider/model alias, prompt version, and
generation timestamp. Do not show provider identity during scoring.

## Reviewer Requirements

- Persian and Arabic outputs require a fluent reviewer for the requested locale.
- High-risk financial, health, or legal cases require a qualified subject-matter
  reviewer in addition to the language reviewer.
- A reviewer must not score an output they authored or manually corrected.

## Scoring

Score every rubric dimension from 0 to 100 using the weights in
`tests/fixtures/article_quality_benchmark_v1.json`. Record evidence for any
dimension below 70. Record factual/source-fidelity evidence for every grounded
case, regardless of score.

The weighted score is advisory when a hard blocker exists. Any hard blocker
forces `reject`, even when the weighted score is otherwise high.

## Decisions

Use exactly one decision:

- `accept`: no hard blocker and the reviewer considers the output releasable.
- `reject`: at least one hard blocker or the output is materially unusable.
- `needs_adjudication`: evidence is incomplete or reviewers materially disagree.

Every review record must include `case_id`, pseudonymous `reviewer_id`,
`review_round`, all dimension scores, hard blockers, decision, and concise notes.
Do not place customer secrets or unpublished source text in review notes.

## Sampling And Adjudication

Review all high-risk and source-grounded cases. For low-risk ungrounded cases,
review a stratified sample covering every language and intent. Two independent
reviewers score the release sample. A third qualified reviewer adjudicates any
hard-blocker disagreement, acceptance disagreement, or dimension difference of
20 or more points.

## Release Evidence

The release record must identify the immutable application commit, provider and
model aliases, prompt version, benchmark and rubric versions, case IDs, review
round, deterministic results, reviewer decisions, adjudications, latency, token
usage, and task-attributed cost. Raw credentials and prompt secrets must never
be included.

Production quality approval requires zero hard blockers, the calibrated score
thresholds in the launch-readiness plan, at least 90 percent human acceptance,
and no material regression against the accepted baseline. Missing reviews,
missing source snapshots, or unattributed telemetry block approval.
