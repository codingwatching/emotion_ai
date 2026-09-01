---
phase: 03-memory-integrity-and-data-lifecycle
plan: 01
subsystem: storage
tags: [sqlite, fts5, atomicity, idempotency, provenance, supersession]

requires:
  - phase: 01-preservation-and-trusted-baseline
    provides: immutable preservation evidence and temporary-only test discipline
  - phase: 02-provider-and-runtime-core
    provides: import-safe local runtime and typed failure conventions
provides:
  - one explicit SQLite durable-truth boundary for complete conversation turns
  - stable idempotent replay and content-free conflict outcomes
  - provenance-bearing derived memories with append-only correction and retraction edges
  - transactionally aligned external-content FTS5 indexes and deterministic rebuild
affects: [03-02, 03-03, 03-04, 03-05, 03-06, memory, retrieval, lifecycle]

tech-stack:
  added: []
  patterns:
    - explicit absolute-path SQLite ownership with no import-time I/O
    - BEGIN IMMEDIATE atomic commands with fixed-code content-free failures
    - immutable events plus edge-derived current memory state

key-files:
  created:
    - aura_backend/storage/__init__.py
    - aura_backend/storage/models.py
    - aura_backend/storage/connection.py
    - aura_backend/storage/schema.py
    - aura_backend/storage/repository.py
    - tests/storage/conftest.py
    - tests/storage/test_atomic_ledger.py
    - tests/storage/test_idempotency.py
    - tests/storage/test_provenance_supersession.py
  modified: []

key-decisions:
  - "SQLite is opened only through an explicitly supplied absolute path and owns canonical truth; FTS is a transactionally maintained projection."
  - "A canonical request hash binds version, scope, session, and user content; response and event hashes bind exact stored text."
  - "Current memory is derived from provenance, supersession, and retraction edges rather than a mutable current flag."

patterns-established:
  - "Public storage records are frozen and slotted; routine failures expose stable codes and identifiers only."
  - "Corrections append a complete new turn and supersession edge in one transaction while preserving original rows."

requirements-completed: [TEST-03, DATA-01, DATA-06]

duration: 15 min
completed: 2026-09-01
status: complete
---

# Phase 3 Plan 01: Atomic Event Ledger and Provenance Summary

**A versioned SQLite ledger now commits complete turns exactly once, preserves immutable source events, and exposes only provenance-valid current memories through explicit supersession and retraction edges.**

## Performance

- **Duration:** 15 min
- **Started:** 2026-09-01T04:11:34Z
- **Completed:** 2026-09-01T04:26:14Z
- **Tasks:** 3
- **Files modified:** 9

## Accomplishments

- Added an import-safe, explicit-path SQLite owner with foreign keys, WAL, finite busy bounds, ordered schema versioning, and external-content FTS5 rebuild support.
- Proved complete turns either commit with two ordered events and all source edges or leave zero durable rows at every injected pre-commit boundary.
- Made same-key retries return stable original identities, reject changed canonical requests without mutation, and recover a committed turn after post-commit interruption.
- Enforced typed, bounded, same-scope derivations and append-only correction/retraction semantics while rejecting orphan, cross-scope, self, and cyclic transitions.
- Bound request, response, and event hashes to their canonical command data before storage or injected side effects.

## Task Commits

Each task followed a RED then GREEN cycle; no refactor commit was needed because the smallest GREEN implementations remained clear and lint-clean.

1. **Task 03-01-01: explicit schema and all-or-nothing turns**
   - `9a65d52` — RED atomic-ledger contract
   - `c408df5` — GREEN versioned schema, atomic transaction, and FTS behavior
2. **Task 03-01-02: convergent retries and races**
   - `49579c3` — RED idempotency/race/interruption contract
   - `45f64dc` — GREEN repository replay, conflict, and reconciliation outcomes
3. **Task 03-01-03: provenance and append-only correction**
   - `256205e` — RED provenance/supersession/retraction contract
   - `455f7d5` — GREEN provenance validation and edge transitions
4. **Correctness hardening discovered at the plan gate**
   - `5952ef0` — RED canonical-hash tampering regression
   - `19c503d` — GREEN command-hash validation

## Files Created/Modified

- `aura_backend/storage/models.py` — frozen typed commands, results, enums, and content-free failures.
- `aura_backend/storage/connection.py` — explicit connection setup, transaction ownership, derivation validation, and failure injection.
- `aura_backend/storage/schema.py` — versioned strict schema, ownership tables, provenance edges, trace/profile scaffolding, and FTS triggers.
- `aura_backend/storage/repository.py` — sole durable repository with idempotency, replay, correction, retraction, and current-memory queries.
- `tests/storage/conftest.py` — deterministic IDs, times, canonical hashes, and temporary database paths.
- `tests/storage/test_atomic_ledger.py` — rollback, complete-turn, FTS, import-safety, and content-free error proof.
- `tests/storage/test_idempotency.py` — sequential/concurrent replay, conflict, post-commit recovery, scope, and hash-tampering proof.
- `tests/storage/test_provenance_supersession.py` — typed derivation, two-scope, immutability, correction, retraction, and cycle proof.

## Decisions Made

- The repository validates canonical hashes before it opens the database or invokes any supplied effect callback, preventing a forged identity from reaching durable state.
- Aura events may remain secondary context sources, but a derived claim requires a user-authored primary source; Aura's own output alone cannot establish a durable claim about Ty.
- Relationship is presently only a typed memory kind with the same provenance discipline as other derivations; affect, salience, and relationship-state updates remain deferred.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test defect] Replaced class-reloading import probe**
- **Found during:** Task 03-01-01 GREEN
- **Issue:** Reloading the model module changed exception class identity and caused injected `StorageFailure` instances to be treated as unknown errors.
- **Fix:** Moved the import-side-effect assertion into an isolated subprocess and retained exact temporary-directory parity.
- **Files modified:** `tests/storage/test_atomic_ledger.py`
- **Verification:** `12 passed` for the Task 03-01-01 focused command.
- **Committed in:** `c408df5`

**2. [Rule 1 - Test defect] Corrected an ambiguous FTS token assertion**
- **Found during:** Task 03-01-01 GREEN
- **Issue:** The token `Synthetic` legitimately matched both fixture events while the assertion expected one row.
- **Fix:** Asserted a user-event-specific term and used the documented FTS `delete-all`/`rebuild` control path.
- **Files modified:** `tests/storage/test_atomic_ledger.py`
- **Verification:** Insert, update, delete, and rebuild assertions all pass.
- **Committed in:** `c408df5`

**3. [Rule 2 - Missing critical functionality] Bound stored hashes to canonical command data**
- **Found during:** Plan-wide must-have review after Task 03-01-03
- **Issue:** Caller-supplied request, response, and event hashes were persisted but were not independently checked against the command content.
- **Fix:** Added canonical request construction and constant-time validation before database creation or effect callbacks, with four tampering regressions.
- **Files modified:** `aura_backend/storage/repository.py`, `tests/storage/conftest.py`, `tests/storage/test_idempotency.py`, `tests/storage/test_provenance_supersession.py`
- **Verification:** RED produced four expected failures; GREEN produced `43 passed` across all storage suites.
- **Committed in:** `5952ef0`, `19c503d`

---

**Total deviations:** 3 auto-fixed (2 test defects, 1 missing critical integrity check).
**Impact on plan:** All changes directly strengthen the planned atomicity, repudiation, and privacy contracts; no historical store, route, dependency, manifest, or lock was touched.

## Issues Encountered

- The installed GSD helper under `/home/ty/.claude/gsd-core` cannot currently load because its expected parent `package.json` is absent. Execution and verification were unaffected; summary and planning-state updates use the repository files directly.

## TDD Gate Compliance

- Task 03-01-01: RED `9a65d52` precedes GREEN `c408df5` — PASS.
- Task 03-01-02: RED `49579c3` precedes GREEN `45f64dc` — PASS.
- Task 03-01-03: RED `256205e` precedes GREEN `455f7d5` — PASS.
- Canonical-hash hardening: RED `5952ef0` precedes GREEN `19c503d` — PASS.

## Verification

- Focused storage gate: `43 passed` in 1.18s.
- Full deterministic non-live suite: `555 passed, 2 skipped, 1 deselected` in 26.22s.
- Ruff: all changed storage and storage-test files pass.
- Import probe: `aura_backend.storage` imports successfully without creating data.
- Git hygiene: clean after task commits; no SQLite, WAL, profile, trace, export, manifest, or lock artifact was produced.

## Known Stubs

None. No placeholder or not-implemented branch remains in the created or modified plan files.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- The durable identities and provenance contracts required by Plan 03-02's sanitized benchmark now exist.
- Real historical roots remain unopened and unchanged; migration, backup, and read-switch work remains behind later explicit gates.

## Self-Check: PASSED

- All nine declared implementation/test files exist.
- All eight RED/GREEN/hardening commits resolve in Git history.
- Every task acceptance command and the plan-level full non-live command passed.

---
*Phase: 03-memory-integrity-and-data-lifecycle*
*Completed: 2026-09-01*
