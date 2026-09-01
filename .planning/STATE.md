# Project State: Aura Rehabilitation

**Updated:** 2026-09-01
**Active phase:** Phase 3 — Memory Integrity and Data Lifecycle
**Status:** Phase 2 verified 16/16; Phase 3 Plans 03-01 and 03-02 complete (2/9 plans)

## Verified So Far

- Evidence-based codebase map committed as `7a79048`.
- Local-only runtime boundary committed as `541b5da`.
- Deterministic Python suite: 510 passing tests, 2 expected skips, 1 live deselection.
- Active TypeScript type-check and Vite production build pass.
- Required GitHub CI is green at commit `1daf3ba` (Run `33288903971`), including
  clean-install Pyright, frontend type-check/build, backend tests, and lint.
- `npm audit --package-lock-only --audit-level=high` reports 0 vulnerabilities.
- `ornith:latest` is installed locally and available for marked live-model checks.
- Git tracks 59 grandfathered runtime/generated artifact paths totaling exactly
  153,612,467 bytes in the content-free tracked-runtime baseline.
- No database, backup, archive, or Git history has been deleted or rewritten.
- All 662 inventoried files have an outside-Git immutable backup with exact
  source-before/source-after/destination parity.
- A disposable restore passed SQLite integrity, exact FK parity, Chroma counts,
  and deterministic opaque retrieval for every non-empty collection.
- Phase 3's new SQLite ledger gate passes 43 focused storage tests: atomic
  rollback, idempotency, concurrency, canonical hashes, provenance, correction,
  retraction, and cycle rejection.
- The sanitized memory instrument passes 30 focused tests across exact corpus,
  privacy, provenance, scope, stale-fact, resource, and adoption controls.
- The complete deterministic non-live suite passes at 585 tests with 2 expected
  skips and 1 live deselection after Plan 03-02.

## Current Position

Phase 1 is independently verified at 30/30 and Phase 2 at 16/16. Phase 3 now has
both the explicit SQLite truth owner and the independent measurement instrument:
complete turns converge, derived memories remain sourced and supersedable, and
an invented hash-bound 10,000-event corpus distinguishes faithful retrieval from
leaks, stale facts, forged provenance, and inconclusive resource outcomes. No
production retriever has been measured or promoted. No historical store has been
opened, migrated, repaired, or deleted. Projection ownership and the retained
eight-row FK anomalies remain gated Phase 3 work before migration or cleanup.

## Locked Decisions

- Aura is private and local; no mandatory sign-in.
- Loopback is the default boundary; LAN exposure is explicit opt-in.
- Ollama is first-class, but deterministic tests do not depend on a running model.
- Preserve behavior and data before refactoring.
- Optimize against measurements, not claims.
- Treat simulated neurochemical/brainwave state as bounded computational control,
  not measured human biology; require causal tests against a fixed control.
- Keep the durable event ledger authoritative; Memvid, Chroma, temporal graphs,
  and RLM readers are replaceable projections or experiments.
- Bind request identity to version, scope, session, and exact user content before
  durable write or side effects; bind response/event hashes to exact stored text.
- Derive current memories from provenance, supersession, and retraction edges;
  never mutate a drifting current-state flag.
- Freeze invented benchmark inputs and exact kill gates before retrieval
  optimization; timeout, truncation, resource limits, and missing controls are
  inconclusive non-pass outcomes.
- Keep neutral and constant-salience arms separately named with zero salience
  contribution throughout Phase 3.
- Remote Git-history rewriting requires separate explicit approval.

## Execution Metrics

| Plan | Duration | Tasks | Files | Result |
|---|---:|---:|---:|---|
| 03-01 | 15 min | 3 | 9 | 555 passed, 2 skipped, 1 deselected |
| 03-02 | 13h 3m | 2 | 4 | 585 passed, 2 skipped, 1 deselected |

## Last Session

**Stopped at:** Completed 03-02-PLAN.md
**Resume file:** None

## Working Tree Note

`.trunk/` predates this rehabilitation and remains untracked and untouched.
