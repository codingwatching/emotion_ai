---
phase: 03-memory-integrity-and-data-lifecycle
plan: 06
subsystem: runtime-storage
tags: [sqlite, idempotency, retrieval, lifecycle, cli, read-owner, evidence]

requires:
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 04
    provides: neutral hybrid retrieval, stable cursors, traces, and zero salience
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 05
    provides: verified snapshot, JSON export, and deletion lifecycle owner
provides:
  - SQLite-first idempotent conversation persistence with post-commit projection reconciliation
  - lifespan-owned lazy repository, projection, retriever, and lifecycle composition
  - bounded neutral search and history routes with cursors and trace identity
  - JSON lifecycle export and explicit plan-confirm-execute-verify deletion routes
  - fail-closed read-owner, authorization, publication, and CI evidence CLI seams
affects: [03-07, 03-08, 03-09, runtime-api, storage-migration, phase-03-acceptance]

tech-stack:
  added: []
  patterns:
    - commit SQLite truth before any rebuildable projection work
    - defer Chroma and embedding imports until the selected storage operation
    - explicit legacy-or-sqlite read-owner marker with fail-closed fallback
    - canonical exclusive evidence receipts and fixed non-pass exit semantics

key-files:
  created:
    - aura_backend/storage/cli.py
  modified:
    - aura_backend/conversation_persistence_service.py
    - aura_backend/main.py
    - aura_backend/aura_internal_tools.py
    - aura_backend/runtime/config.py
    - tests/api/test_api_compatibility.py
    - tests/runtime/test_base_install_startup.py
    - tests/storage/test_storage_owner.py

key-decisions:
  - "SQLite commits the complete turn before projection; projection failure degrades status but never creates a second durable truth owner."
  - "Existing installations default to the explicit legacy read owner; SQLite reads require a verified marker or an explicit clean-install selection."
  - "Chroma, embedding, retrieval, and lifecycle integrations remain lazy so base-only startup imports no optional Google or model SDK."
  - "Operational commands reject real migration, snapshot, and benchmark execution until later checkpoint evidence is supplied; this plan grants no real-data authority."

patterns-established:
  - "Stable retry identity: one caller or generated idempotency key survives immediate persistence and its bounded background retry."
  - "Lifecycle API: vague DELETE only returns a confirmation-required inventory; mutation requires plan, confirm, execute, and verify."
  - "Evidence boundary: private receipts may contain exact paths, while public receipts expose only allowlisted statuses, IDs, times, and digests."

requirements-completed: [PRES-04, TEST-03, DATA-01, DATA-02, DATA-03, DATA-04, DATA-05, DATA-06, DATA-09]

duration: 6h46m
completed: 2026-09-02
status: complete
---

# Phase 3 Plan 06: Runtime Storage Boundary Summary

**Aura now commits complete idempotent turns to SQLite before projection, serves bounded neutral retrieval and lifecycle operations through one injected owner, and exposes fail-closed evidence commands without changing the local companion/provider contract.**

## Performance

- **Duration:** 6h46m wall time across a resumed execution
- **Started:** 2026-09-01T23:16:30Z
- **Completed:** 2026-09-02T06:02:00Z
- **Tasks:** 2
- **Files modified:** 8

## Accomplishments

- Replaced the legacy Chroma/file-system persistence implementation with a compatibility service that builds one canonical `TurnCommand`, commits it atomically through `StorageRepository`, and projects only after the ledger returns durable truth.
- Added absolute disjoint `AURA_LEDGER_DIRECTORY` settings with the repository-resolved `aura_data_v2` default while leaving legacy `AURA_DATA_DIRECTORY` unchanged.
- Composed repository, projection, retriever, lifecycle, and internal-tool resources inside lifespan with no import-time database/model work, legacy backup writer, or required optional SDK.
- Preserved the seven-field conversation response, HTTP-200 typed provider fallback, answer-preserving degraded persistence, loopback/no-sign-in boundary, provider-neutral tools, and untrusted memory delimiters.
- Routed SQLite-selected search/history through fixed 1-100 pages, opaque cursors, hard scope gates, neutral results, and trace identity; optional Memvid never supplies the primary eligible set.
- Routed SQLite-selected JSON export and deletion through `LifecycleService`; vague legacy DELETE returns a non-mutating confirmation-required plan, while explicit endpoints expose plan, confirm, execute, and verify.
- Added a standard-library, import-lazy storage CLI with exact read-owner markers, exclusive authorization receipts, local-only publication proposals, and five-job current-SHA CI evidence evaluation. Real operations remain blocked until later human checkpoint evidence exists.

## Task Commits

Both tasks followed committed RED then GREEN cycles. One post-GREEN correctness fix was committed separately.

1. **Task 03-06-01: Wire SQLite-first turn persistence without public drift**
   - `b3896cb` — RED SQLite-first runtime and idempotency contract
   - `ccd6dfd` — GREEN SQLite owner, post-commit projection, and lazy base composition
2. **Task 03-06-02: Route reads, export, deletion, and operations through the storage boundary**
   - `6b67ff6` — RED storage-boundary and operational evidence contract
   - `4151211` — GREEN bounded read/lifecycle routes and fail-closed CLI
3. **Post-GREEN correctness**
   - `f66def8` — align lazy lifecycle factories with their typed protocols

## Files Created/Modified

- `aura_backend/storage/cli.py` — standard-library command grammar, read-owner markers, canonical authorization receipts, local publication proposals, and deterministic CI evidence parser.
- `aura_backend/conversation_persistence_service.py` — SQLite-first turn adapter, stable retry identity, explicit read-only legacy seam, and compatibility results.
- `aura_backend/main.py` — lazy lifespan storage composition plus bounded retrieval, history, JSON export, and explicit deletion endpoints.
- `aura_backend/aura_internal_tools.py` — injected neutral retriever for `aura.search_memories`; Memvid remains optional and non-primary.
- `aura_backend/runtime/config.py` — absolute/disjoint v2 ledger, database, and projection settings.
- `tests/api/test_api_compatibility.py` — idempotency, projection recovery, route bounds, lifecycle routing, and public-shape coverage.
- `tests/runtime/test_base_install_startup.py` — base-only startup and no-legacy-writer composition proof.
- `tests/storage/test_storage_owner.py` — internal-tool owner, marker rollback, CLI command, receipt, and CI evidence gates.

## Decisions Made

- A projection callback cannot run while the repository writer lock is held because the projection marks completion through that repository. The runtime therefore commits/replays first, releases the lock, then projects and reloads durable status.
- Base startup owns ordinary lazy wrappers rather than constructing `ProjectionAdapter`, `HybridRetriever`, or `LifecycleService` eagerly. This preserves base-only operation when optional Google/Chroma transitive imports are unavailable.
- Missing, malformed, or non-pass read-owner evidence selects legacy; only a verified marker or explicit clean-install flag selects SQLite. New writes remain SQLite-owned either way.
- CLI benchmark/snapshot/import commands refuse with `later_checkpoint_required` during this autonomous plan. Plans 03-08 and 03-09 must supply the separately approved receipts and explicit disposable paths before those operations can execute.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Safety bug] Removed legacy protection construction after a RED test exposed a live-root side effect**
- **Found during:** Task 03-06-01 RED
- **Issue:** The old positional constructor interpreted repository/projection test doubles as legacy collaborators, constructed the database-protection singleton, and pruned two tracked `auto_backups` SQLite files.
- **Fix:** Restored those exact two tracked files immediately without inspecting their contents, then replaced the constructor with typed repository/projection dependencies. No deletion entered a commit.
- **Files modified:** `aura_backend/conversation_persistence_service.py`; the two affected tracked backup files were restored byte-for-byte and are absent from every plan commit.
- **Verification:** Clean status after restoration; all later tests and final diff show no historical/data/backup/archive file changes.
- **Commit:** `ccd6dfd`

**2. [Rule 1 - Correctness bug] Removed repository/projection lock re-entry**
- **Found during:** Task 03-06-01 first GREEN gate
- **Issue:** Repository callbacks ran under the writer lock while projection completion re-entered the same repository, causing a deterministic deadlock.
- **Fix:** Persist/replay under the repository boundary, release the lock, project afterward, and reload status. Failure returns durable `stored` plus projection `pending`; retry reconciles the same turn.
- **Files modified:** `aura_backend/conversation_persistence_service.py`
- **Verification:** Projection interruption/retry test and Task 1 exact gate pass without timeout or duplicate rows.
- **Commit:** `ccd6dfd`

**3. [Rule 2 - Missing critical startup behavior] Deferred optional Chroma/Google imports**
- **Found during:** Task 03-06-01 base-only GREEN probe
- **Issue:** Eager lifecycle/projection imports pulled a transitive Google package during required base startup, violating the base-only install contract.
- **Fix:** Added lazy projection, retriever, and lifecycle wrappers; construction remains import/resource-light until a selected operation needs the optional implementation.
- **Files modified:** `aura_backend/main.py`
- **Verification:** Base-only child proves zero attempted optional imports and zero forbidden network/database/subprocess effects.
- **Commit:** `ccd6dfd`

**4. [Rule 1 - Type contract] Corrected lazy factory parameter names**
- **Found during:** Post-GREEN Python type gate
- **Issue:** Runtime-compatible factory callables did not match the named parameters in the lifecycle protocols.
- **Fix:** Matched the projection factory and fixture verifier protocol signatures without changing behavior.
- **Files modified:** `aura_backend/main.py`
- **Verification:** Project-wide Pyright reports `0 errors, 0 warnings, 0 informations`; affected runtime/API slice reports `193 passed`.
- **Commit:** `f66def8`

---

**Total deviations:** 4 auto-fixed correctness/safety issues (3 Rule 1, 1 Rule 2).
**Impact on plan:** Every fix tightened the planned single-owner, base-only, or typed lifecycle boundary. No dependency, manifest, lock, real-data authority, public response field, authentication, provider, companion, prompt, or frontend contract changed.

## Issues Encountered

- The `/home/ty/.Codex` GSD helper resolves to an incomplete installation missing its expected package metadata. The intact `/home/ty/.hermes/gsd-core/bin/gsd-tools.cjs` fallback was used for planning-state queries and prescribed tracking; no package was installed or changed.
- One command-runner window truncated the first full-suite stream at 64%. That partial run was discarded and rerun in a persistent session to a complete exit and final summary.

## TDD Gate Compliance

- Task 03-06-01 RED `b3896cb`: 10 intended failures with 44 existing passes. GREEN `ccd6dfd`: exact command completed with `54 passed`.
- Task 03-06-02 RED `6b67ff6`: collection failed because `aura_backend.storage.cli` did not yet exist. GREEN `4151211`: exact cross-contract command completed with `260 passed` after the final added lifecycle-route assertion.
- Commit order is RED -> GREEN for both behavior-adding tasks; no separate refactor commit was required.

## Verification

- Task 1 exact compatibility gate: `54 passed` in 9.36s.
- Task 2 exact API/storage/runtime gate: `260 passed` in 17.18s.
- Complete deterministic non-live suite after the final fix: `665 passed, 2 skipped, 1 deselected` in 37.51s.
- Affected post-type-fix API/runtime slice: `193 passed` in 12.86s.
- Project-wide Python type check: `0 errors, 0 warnings, 0 informations`.
- Focused Ruff check and `git diff --check`: pass.
- Dependency hygiene: all Python tests used `uv run --locked --no-sync`; no install, package manifest, or lock file changed.
- Data safety: all new databases, exports, markers, receipts, and lifecycle fixtures were temporary/synthetic. No real historical, data, backup, archive, Chroma, Memvid, profile, or secret root was inspected or mutated; the accidental RED-side tracked backup change was restored immediately and never committed.

## Known Stubs

None. The CLI's `later_checkpoint_required` result for benchmark, snapshot, and migration execution is an intentional authorization gate consumed by Plans 03-08/03-09, not placeholder success behavior. Optional Memvid-unavailable and empty-history results are bounded runtime states.

## Threat Flags

None. The new API lifecycle and file/evidence operations are the exact trust surfaces registered as T-03-06-01 through T-03-06-06 and T-03-06-SC; tests cover scope, bounds, idempotency, confirmation, exclusive creation, public privacy, current-SHA CI jobs, and no dependency mutation.

## User Setup Required

None - no dependency, credential, model, external service, manifest, lock, or system configuration was added.

## Next Phase Readiness

- Plan 03-07 can perform metadata-only preservation readiness against separately approved exact paths; this plan itself grants no observation or backup authority.
- Plans 03-08/03-09 can consume the CLI receipt/marker seams only after their explicit human checkpoints, durable-copy restore proof, neutral benchmark, rollback rehearsal, and exact-current CI evidence.
- Existing installations remain on legacy reads until a verified switch marker. SQLite remains the only owner of every new durable turn, and Memvid remains optional copy-only.

## Self-Check: PASSED

- All eight declared implementation/test files and this summary exist.
- All five RED/GREEN/fix commits resolve in Git history in the required order.
- Exact task gates, complete non-live regression, Ruff, Pyright, and diff hygiene pass.
- The implementation worktree was clean before this summary, and no manifest, lock, historical/data/backup/archive, or generated runtime artifact is part of the plan diff.

---
*Phase: 03-memory-integrity-and-data-lifecycle*
*Completed: 2026-09-02*
