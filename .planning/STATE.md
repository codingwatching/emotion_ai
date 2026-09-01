# Project State: Aura Rehabilitation

**Updated:** 2026-09-01
**Active phase:** Phase 3 — Memory Integrity and Data Lifecycle
**Status:** Phase 2 verified 16/16; Phase 3 Plan 03-01 complete (1/9 plans)

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
- The complete deterministic non-live suite passes at 555 tests with 2 expected
  skips and 1 live deselection after Plan 03-01.

## Current Position

Phase 1 is independently verified at 30/30 and Phase 2 at 16/16. Phase 3 Plan
03-01 now provides the explicit SQLite truth owner: complete turns commit once,
canonical retries converge, and derived memories remain sourced and explicitly
supersedable/retractable. Plan 03-02 can build the sanitized memory benchmark on
these identities. No historical store has been opened, migrated, repaired, or
deleted. Root ownership and the retained eight-row FK anomalies remain gated
Phase 3 work before any migration or cleanup.

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
- Remote Git-history rewriting requires separate explicit approval.

## Execution Metrics

| Plan | Duration | Tasks | Files | Result |
|---|---:|---:|---:|---|
| 03-01 | 15 min | 3 | 9 | 555 passed, 2 skipped, 1 deselected |

## Last Session

**Stopped at:** Completed 03-01-PLAN.md
**Resume file:** None

## Working Tree Note

`.trunk/` predates this rehabilitation and remains untracked and untouched.
