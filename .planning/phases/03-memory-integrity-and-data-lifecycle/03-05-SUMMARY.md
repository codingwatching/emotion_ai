---
phase: 03-memory-integrity-and-data-lifecycle
plan: 05
subsystem: storage
tags: [sqlite, snapshot, restore, export, deletion, lifecycle, git-hygiene]

requires:
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 01
    provides: authoritative SQLite ledger, provenance, correction, and table digests
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 03
    provides: disposable cosine projection generations and exact reconciliation
provides:
  - bounded online SQLite snapshots with durable canonical manifests
  - isolated exact restore with FTS/projection rebuild and retrieval fixtures
  - truthful scope-local JSON export with sanitization and round-trip parity
  - expiring HMAC-bound one-use deletion confirmation and exact active-target verification
  - residual-copy honesty and Phase 3 generated-artifact Git gates
affects: [03-06, 03-07, 03-08, 03-09, runtime-api, preservation, storage-operations]

tech-stack:
  added: []
  patterns:
    - SQLite Connection.backup with bounded progress and atomic manifest publication
    - isolated restore from canonical SQLite with rebuildable FTS and Chroma projection
    - deterministic JSON allowlists with counts, record hashes, and manifest HMAC comparison
    - plan-confirm-execute-verify deletion with exact inventories and injected active targets

key-files:
  created:
    - aura_backend/storage/lifecycle.py
    - tests/storage/test_snapshot_restore.py
    - tests/storage/test_export_delete.py
    - tests/storage/test_git_hygiene.py
  modified:
    - tests/api/test_filesystem_contract.py
    - .gitignore

key-decisions:
  - "SQLite is the recoverable snapshot authority; FTS and Chroma are rebuilt and verified from restored SQLite rather than copied as required truth."
  - "Portable export is JSON only and contains deterministic allowlisted scoped rows; trace, embedding, credential, secret, and internal-path fields remain excluded."
  - "Deletion authority is an expiring HMAC-bound inventory plus explicit confirmation; execution consumes it once and complete requires exact post-delete absence across every approved active target."
  - "Normal active deletion never opens retained historical, Memvid archive, or backup roots and always reports their content-free aliases as residual copies."

patterns-established:
  - "Lifecycle publication: private exclusive staging, fsync, atomic rename, parent fsync, then a complete status."
  - "Deletion truth: canonical rows commit together, projection/cache/export targets report failures separately, and forensic erasure is always false."

requirements-completed: [PRES-04, TEST-03, DATA-01, DATA-02, DATA-03, DATA-04]

duration: 22m
completed: 2026-09-01
status: complete
---

# Phase 3 Plan 05: Truthful Storage Lifecycle Summary

**Bounded SQLite snapshots now restore through an exact isolated gate, scoped JSON exports round-trip real allowlisted records, and destructive actions require one-use inventory-bound confirmation with explicit residual and retry truth.**

## Performance

- **Duration:** 22m
- **Started:** 2026-09-01T22:48:32Z
- **Completed:** 2026-09-01T23:10:54Z
- **Tasks:** 3
- **Files modified:** 6

## Accomplishments

- Added incremental online SQLite snapshots with absolute/disjoint path checks, resource and timeout bounds, mode-0600 exclusive staging, partial cleanup, hashes, exact table/FTS facts, atomic publication, and durable manifests.
- Added restore into a new isolated root only. A complete result requires manifest/database hashes, the exact required check set, integrity and foreign-key gates, schema/count/digest parity, rebuilt FTS, a fresh cosine projection, five fixed retrieval fixtures, and unchanged active-source hash.
- Added deterministic JSON-only export of actual sessions, turns, events, derived memories, provenance, supersession/retraction edges, and sanitized profile versions from one scoped read transaction. Unsupported formats and unsafe paths fail before creation.
- Added explicit deletion planning, confirmation, execution, and verification for the six exact action names. Plan inventories are HMAC-bound to action, scope, IDs, hashes, counts, policy, and expiry; confirmation is single-use.
- Added exact parameterized canonical deletion in one transaction, followed by injected projection/cache/export operations, projection reconciliation, retained-copy reporting, and incomplete/retry results whenever an approved target remains or fails.
- Extended ignore coverage and synthetic staged-artifact tests for the Phase 3 ledger, projection generations, snapshots, backups, exports, profiles, traces, logs, secrets, and Memvid archives without removing any grandfathered tracked artifact.

## Task Commits

Each task followed a committed RED then GREEN cycle. No separate refactor commit was required.

1. **Task 03-05-01: Prove consistent snapshot and exact isolated restore**
   - `ddfa2cf` — RED concurrent, partial, corrupt, tampered, path, FK, and active-invariance contract
   - `dadd44e` — GREEN bounded online snapshot and exact isolated restore gate
2. **Task 03-05-02: Export actual scoped JSON and reject false formats**
   - `a4ffaef` — RED real-row export, isolation, sanitization, containment, and round-trip contract
   - `67f95d6` — GREEN truthful deterministic scoped JSON export
3. **Task 03-05-03: Require deletion plan, confirmation, execution, verification, and residual truth**
   - `dc14d6a` — RED tamper/expiry/replay/scope, target-failure, residual, Memvid, and Git-hygiene contract
   - `b22ca78` — GREEN exact verified deletion protocol and generated-artifact boundary

## Files Created/Modified

- `aura_backend/storage/lifecycle.py` — lifecycle owner for snapshot, restore, export, deletion inventory, confirmation, execution, verification, and residual reporting.
- `tests/storage/test_snapshot_restore.py` — concurrent snapshot, interruption/resource/path, corruption/tamper/FK, projection/fixture, and source-invariance cases.
- `tests/storage/test_export_delete.py` — actual scoped export plus explicit destructive-action and active-target retry cases over synthetic SQLite/in-memory stores.
- `tests/storage/test_git_hygiene.py` — ignore coverage, force-staged synthetic artifact rejection, and exact grandfathered-index preservation.
- `tests/api/test_filesystem_contract.py` — JSON-only and final-symlink filesystem characterization for subsequent API wiring.
- `.gitignore` — explicit Phase 3 generated ledger/projection/snapshot/export/profile/trace/log/secret/archive patterns.

## Decisions Made

- Snapshot success is publication of a complete SQLite/manifest pair, not successful file copying. Restore success is the entire ordered check set, not a subset.
- Export hashes the sanitized portable representation and compares it against current scoped ledger truth. It makes no claim to export embeddings, retrieval traces, operational secrets, or unrelated scopes.
- Deletion confirmation state is deliberately process-local: a restart invalidates outstanding confirmations rather than reviving destructive authority. Later API wiring must preserve the explicit human confirmation step.
- Active non-SQLite stores are injected as exact inventory/delete/remaining adapters. The lifecycle service never discovers a real Chroma, provider cache, export, archive, backup, or historical path.
- Archive and backup-generation purge are distinct exact actions and require a full independent restore proof. This plan exercised no real purge and granted no such approval.
- Application-level deletion and best-effort physical purge are reported separately; forensic erasure remains false in every outcome.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 3 - Blocking issue] Added the missing database opener import**
- **Found during:** Task 03-05-03 first GREEN run
- **Issue:** The new transactional executor referenced the established `open_database` helper without importing it, producing four deterministic `NameError` failures.
- **Fix:** Imported the existing storage connection helper; no database or dependency contract changed.
- **Files modified:** `aura_backend/storage/lifecycle.py`
- **Commit:** `b22ca78`

**2. [Rule 1 - Safety bug] Moved active-target inventory revalidation before canonical mutation**
- **Found during:** Task 03-05-03 pre-commit review
- **Issue:** A changed projection/cache/export inventory would have been detected only after the canonical SQLite transaction committed.
- **Fix:** Revalidate the full injected active-target inventory before entering the destructive SQLite transaction, so changed confirmation-bound inventory mutates nothing.
- **Files modified:** `aura_backend/storage/lifecycle.py`
- **Commit:** `b22ca78`

---

**Total deviations:** 2 auto-fixed correctness issues.
**Impact on plan:** Both changes enforce the intended fail-closed boundary; scope, architecture, dependencies, and user authority were unchanged.

## Issues Encountered

- The installed GSD helper remains unusable because its resolved installation is missing the expected parent `package.json`. No package or manifest was altered to conceal that environment defect; sequential summary, state, and roadmap tracking were updated directly, matching the established Phase 3 fallback.

## TDD Gate Compliance

- Task 03-05-01: RED `ddfa2cf` failed on the missing lifecycle module; GREEN `dadd44e` passed the exact snapshot/restore contract with `10 passed`.
- Task 03-05-02: RED `a4ffaef` failed on missing export lifecycle types; GREEN `67f95d6` passed export/filesystem verification with `39 passed`.
- Task 03-05-03: RED `dc14d6a` failed on missing deletion lifecycle types; GREEN `b22ca78` passed deletion/Git-hygiene/Memvid verification with `21 passed`.
- Commit order is RED -> GREEN for all three behavior-adding tasks.

## Verification

- Required plan integration gate: `59 passed` in 3.44s.
- Complete deterministic non-live suite: `647 passed, 2 skipped, 1 deselected` in 33.17s.
- Project-wide Python type check: `0 errors, 0 warnings, 0 informations`.
- Focused Ruff check and format check: pass for lifecycle and Plan 03-05 tests.
- `git diff --check`: pass.
- Dependency hygiene: Python/test commands used `uv run --locked --no-sync`; no install, dependency, package manifest, or lock file changed.
- Data safety: every database, export, projection target, staged artifact, and residual alias was temporary, synthetic, or in-memory. No real historical, data, backup, archive, Chroma, Memvid, profile, or secret root was inspected or mutated.

## Known Stubs

None. Empty export collections, optional lifecycle adapters, and zero-item exact inventories are bounded runtime states rather than placeholder behavior.

## Threat Flags

None. Snapshot/file publication, scoped export, destructive intent, active-target verification, residual reporting, and generated-artifact Git boundaries are the exact threat surfaces registered and tested by this plan.

## User Setup Required

None - no dependency, credential, model, external service, or system configuration was added.

## Next Phase Readiness

- Plan 03-06 can wire the lifecycle boundary into runtime endpoints while preserving JSON-only export, explicit human deletion confirmation, and typed fail-closed results.
- Real snapshot, restore, import, read-switch, archive/backup purge, cleanup, and historical-root deletion remain separately authorized operations; this plan grants none of them.
- Memvid remains copy-only. Archive creation cannot delete its active source records.

## Self-Check: PASSED

- All six declared implementation/test files and this summary exist.
- All six RED/GREEN task commits resolve in Git history in the required order.
- Exact task gates, plan integration, full non-live regression, Ruff, Pyright, and diff hygiene pass.
- The worktree is clean before documentation tracking, and no runtime database, WAL, snapshot, export, profile, trace, log, secret, archive, manifest, or lock artifact was generated.

---
*Phase: 03-memory-integrity-and-data-lifecycle*
*Completed: 2026-09-01*
