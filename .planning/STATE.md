# Project State: Aura Rehabilitation

**Updated:** 2026-09-07
**Active phase:** Phase 3 — Memory Integrity and Data Lifecycle
**Status:** Phase 2 verified 16/16; Phase 3 Plans 03-01 through 03-06 complete (6/9 plans)

## Latest Independent Improvement

Ty authorized an autonomous improvement pass on 2026-09-07. The active
conversation path now uses versioned, source-checked emotion proposals with
explicit abstention/invalid/unavailable states, durable uncertainty, and honest
simulation labels. A shared provider defect accepting non-streaming truncated
responses was repaired, and bounded Ollama analysis can disable optional
reasoning without changing ordinary chat defaults. Existing Python typing
errors were corrected without package/lock changes.

Verification: 763 deterministic tests passed, 2 skipped, 1 live deselected;
Python/frontend type checks, Ruff, and production build pass. The local Ornith
smoke evaluation completed 12 cases and matched 11, with one false-positive
label on ambiguous sarcasm: its strict gate **failed**. Browser interaction
remains unverified because no browser was connected. Evidence and limitations
are in [the emotion assessment report](../docs/emotion-assessment.md).

This does not complete Phase 3 or the Phase 4 affective controller. Historical
data, backups, migration/read-owner gates, and dependency locks were untouched.

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
- Chroma 1.5.9 is now a stable-ID cosine projection whose candidates must match
  active SQLite scope, origin, content hash, schema, generation, model, and config.
- Fresh projection generations switch only after exact count/ID/hash/config and
  deterministic fixture parity; post-commit and partial-upsert faults reconcile.
- The legacy importer rejects unauthorized paths before Chroma opens, preserves
  exact source triples and typed fragments, and reads only operation copies of
  authorized disposable restores.
- The complete deterministic non-live suite passes at 604 tests with 2 expected
  skips and 1 live deselection after Plan 03-03.
- Neutral retrieval now applies fixed 50-plus-50 FTS5/cosine caps, normalized
  RRF, and SQLite-authoritative scope/provenance/freshness/relevance gates.
- Retrieval cursors page stored frozen runs; history cursors use an event-PK
  watermark plus the `(observed_at,event_id)` keyset; both fail closed on tamper.
- Candidate traces are content-free, prompt retrieval remains untrusted data,
  and the shipped Phase 3 salience seam accepts only exact zero.
- The complete deterministic non-live suite passes at 616 tests with 2 expected
  skips and 1 live deselection after Plan 03-04.
- Online SQLite snapshots now publish only after bounded backup, fsync, hash,
  canonical manifest, and atomic-rename gates; isolated restore additionally
  requires integrity/FK/schema/count/digest/FTS/projection/retrieval parity.
- Scoped export is truthful JSON only, with deterministic allowlisted ledger and
  profile records, sanitization, content hashes, and live-ledger round-trip proof.
- Deletion now requires an expiring HMAC-bound exact inventory, explicit one-use
  confirmation, transactional canonical deletion, active-target verification,
  retry truth, and explicit retained historical/archive/backup copies.
- The complete deterministic non-live suite passes at 647 tests with 2 expected
  skips and 1 live deselection after Plan 03-05.
- The active runtime now commits complete idempotent SQLite turns before lazy
  Chroma projection; projection failure preserves durable truth and retry identity.
- Search/history, JSON export, and deletion now cross the injected storage owner;
  legacy reads remain explicit until a verified SQLite read-owner marker exists.
- The operational CLI fails closed on absent human evidence, writes canonical
  exclusive receipts, and requires the exact five deterministic current-SHA CI jobs.
- The complete deterministic non-live suite passes at 665 tests with 2 expected
  skips and 1 live deselection after Plan 03-06; Pyright reports zero errors.

## Current Position

Phase 1 is independently verified at 30/30 and Phase 2 at 16/16. Phase 3 now has
the SQLite truth owner, independent measurement instrument, rebuildable cosine
projection/import boundary, bounded affect-neutral retrieval/history service,
verified snapshot/restore/export/deletion lifecycle, and the live runtime/API/CLI
storage boundary. Every new turn writes only SQLite durable truth; existing
installations remain on explicit legacy reads until a later verified marker.
Retrieval eligibility cannot be weakened by caller filters or salience, and
destructive authority cannot be inferred from vague, expired, replayed, tampered,
or changed inventory. No real conversation-quality benchmark has been run, and
no historical store has been opened, migrated, repaired, or deleted. The retained
eight-row FK anomalies remain gated work before any authorized real import, read
switch, cleanup, or deletion.

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
- Open legacy Chroma only from an operation-owned temporary copy of an exact,
  fingerprint-bound authorized disposable restore; never open the authorized
  source itself because ordinary Chroma access may mutate its internal files.
- Preserve every legacy source triple independently as a typed fragment when
  pair, timestamp, scope, or semantic structure is not directly evidenced.
- Keep raw BM25/cosine values as trace facts only; use fixed normalized RRF and
  SQLite-authoritative gates to form the affect-neutral eligible set.
- Bind opaque cursors to frozen run/history state, exact scope/query/config/
  generation, expiry, and stable ordering boundaries; reject non-zero salience.
- Treat SQLite as snapshot truth and rebuild FTS/Chroma during isolated restore;
  never require a live Chroma directory copy for recoverability.
- Advertise JSON export only, and require exact plan-confirm-execute-verify
  deletion with one-use confirmation, failed-target retries, residual-copy truth,
  and no forensic-erasure claim.
- Commit each complete turn to SQLite before projection and perform projection
  reconciliation only after the repository writer boundary releases.
- Default existing installations to legacy reads; SQLite reads require a verified
  marker or explicit clean-install selection, while new writes remain SQLite-only.
- Keep Chroma, embedding, retrieval, and lifecycle construction lazy so base-only
  startup does not require optional SDKs or initialize a model/database at import.
- Treat later-checkpoint refusal as an authorization gate: autonomous tasks cannot
  infer real migration, snapshot, benchmark, publication, or CI authority.

## Execution Metrics

| Plan | Duration | Tasks | Files | Result |
|---|---:|---:|---:|---|
| 03-01 | 15 min | 3 | 9 | 555 passed, 2 skipped, 1 deselected |
| 03-02 | 13h 3m | 2 | 4 | 585 passed, 2 skipped, 1 deselected |
| 03-03 | 19 min | 2 | 7 | 604 passed, 2 skipped, 1 deselected |
| 03-04 | 4h 39m | 2 | 6 | 616 passed, 2 skipped, 1 deselected |
| 03-05 | 22 min | 3 | 6 | 647 passed, 2 skipped, 1 deselected |
| 03-06 | 6h 46m | 2 | 8 | 665 passed, 2 skipped, 1 deselected |

## Last Session

**Stopped at:** Completed 03-06-PLAN.md
**Resume file:** None

## Working Tree Note

`.trunk/` predates this rehabilitation and remains untracked and untouched.
