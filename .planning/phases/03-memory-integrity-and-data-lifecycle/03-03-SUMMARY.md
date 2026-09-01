---
phase: 03-memory-integrity-and-data-lifecycle
plan: 03
subsystem: storage
tags: [sqlite, chromadb, cosine, projection, migration, provenance]

requires:
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 01
    provides: authoritative SQLite origins, provenance, supersession, and generation tables
  - phase: 01-preservation-and-trusted-baseline
    provides: immutable-history and disposable-restore evidence discipline
provides:
  - stable-ID cosine Chroma projection derived only from active SQLite origins
  - idempotent reconciliation and verified fresh-generation rebuild switching
  - exact-scope legacy importer that cannot open original or durable-backup roots
  - typed unpaired fragments with exact source triples and canonical raw metadata
affects: [03-04, 03-06, 03-08, 03-09, retrieval, migration, lifecycle]

tech-stack:
  added: []
  patterns:
    - SQLite-owned projection generations with public-API-only Chroma adapters
    - stable event and memory projection IDs with hash/config/generation validation
    - authorized restore copied again into an operation-owned Chroma read workspace
    - exact legacy source triples with idempotent typed-fragment insertion

key-files:
  created:
    - aura_backend/storage/projection.py
    - aura_backend/storage/migration.py
    - tests/storage/test_projection_rebuild.py
    - tests/storage/test_legacy_migration.py
    - tests/storage/test_storage_owner.py
  modified:
    - aura_backend/storage/repository.py
    - aura_backend/storage/schema.py

key-decisions:
  - "Chroma generations become current only when public-API count, ID, content-hash, configuration, and deterministic fixture parity pass."
  - "Legacy Chroma is opened only from an operation-owned temporary copy of an authorized disposable restore because locked Chroma 1.5.9 mutates a store during ordinary client access."
  - "Unpaired legacy records remain separate typed fragments keyed by the exact root fingerprint, collection, and legacy ID; equal content never merges migration evidence."

patterns-established:
  - "Projection candidate gate: every Chroma hit must resolve to an active scope/hash/config/generation-matching SQLite origin."
  - "Migration authorization gate: evidence, allowlist, containment, forbidden-root, fingerprint, and symlink checks precede client construction."

requirements-completed: [TEST-03, DATA-01, DATA-06]

duration: 19 min
completed: 2026-09-01
status: complete
---

# Phase 3 Plan 03: Rebuildable Projection and Safe Legacy Import Summary

**A stable-ID cosine Chroma projection now rebuilds entirely from SQLite truth, while historical Chroma import is restricted to verified operation copies of explicitly authorized disposable restores.**

## Performance

- **Duration:** 19 min
- **Started:** 2026-09-01T17:38:35Z
- **Completed:** 2026-09-01T17:57:30Z
- **Tasks:** 2
- **Files modified:** 7

## Accomplishments

- Added an ordinary injected projection adapter with no import-time I/O, singleton capture, implicit root, or direct third-party SQLite access.
- Froze Chroma's locked 1.5.9 collection configuration to cosine and kept raw distance separate from clamped cosine similarity.
- Projected committed events and active provenance-bearing memories under stable `event:{id}` and `memory:{id}` identities with scope, content hash, SQLite schema, generation, model, and embedding-config metadata.
- Made post-commit interruption, partial upsert, retry interruption, stale metadata, orphan rows, supersession, and projection loss converge through idempotent reconciliation.
- Built fresh generation directories in deterministic SQLite pages and switched current generation only after exact count, ID, document, metadata, configuration, and retrieval-fixture parity.
- Added a source-safe legacy importer that rejects original, durable-backup, outside-workspace, root-symlink, nested-symlink, missing-evidence, alias, and fingerprint attacks before Chroma client construction.
- Preserved both synthetic legacy roots independently, including equal content from different origins, missing/malformed timestamps, unproven pairs, unmapped metadata, content-free anomaly codes, and exact rerun identity.

## Task Commits

Each behavior-adding task followed a committed RED then GREEN cycle. No separate refactor commit was needed.

1. **Task 03-03-01: Make cosine Chroma a rebuildable projection**
   - `d4ad44e` — RED stable-ID, fault, tamper, rebuild, and ownership contract
   - `4771b2f` — GREEN SQLite-derived cosine projection and generation switching
2. **Task 03-03-02: Make historical import idempotent and source-immutable**
   - `242cfd4` — RED authorization, two-root, fragment, rerun, and immutability contract
   - `275ae19` — GREEN authorized operation-copy importer and typed legacy schema
   - `731a2a1` — nested-symlink containment hardening
3. **Post-wave verification repair**
   - `8b78eba` — narrow the optional Chroma client before collection creation

## Files Created/Modified

- `aura_backend/storage/projection.py` — public-API cosine projection, candidate validation, reconciliation, and verified rebuild.
- `aura_backend/storage/migration.py` — exact authorization object, source-tree fingerprinting, staged public Chroma reader, mapping, and content-free result.
- `aura_backend/storage/repository.py` — SQLite origin paging, generation state, projection acknowledgement, and idempotent legacy-fragment/source mappings.
- `aura_backend/storage/schema.py` — schema version 2 with typed legacy fragments and content-free import evidence.
- `tests/storage/test_projection_rebuild.py` — locked API, metadata, failure, tamper, supersession, and rebuild parity proof.
- `tests/storage/test_legacy_migration.py` — divergent synthetic roots, duplicates, fragments, reruns, path attacks, and source immutability proof.
- `tests/storage/test_storage_owner.py` — static rejection of direct SQLite internals, old singleton ownership, and default-path capture.

## Decisions Made

- Generation state and watermarks remain in SQLite. A failed directory is diagnostic disposable state and never becomes current.
- A rebuild marks pending turns complete only after the fresh generation is verified and selected; failed acknowledgements remain safely retryable.
- The importer does not open even the authorized restored source directly. It hash-verifies that source, copies it beneath the authorized temporary workspace, and opens only the operation copy, insulating source bytes from Chroma's ordinary access-time mutations.
- No legacy pair, timestamp, scope, fact, or relationship is inferred. Missing scope receives an explicit root-bound `legacy-unscoped` classification; missing or malformed time remains null with a fixed reason code.
- Schema version 2 stores unpaired fragments separately instead of fabricating complete two-event turns merely to fit the canonical conversation schema.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 2 - Missing critical functionality] Added typed fragment persistence**
- **Found during:** Task 03-03-02 GREEN
- **Issue:** The existing version-1 schema had only complete two-event turns, so persisting an unmatched legacy row would require fabricating a pair or leaving it outside the ledger.
- **Fix:** Added a forward schema migration for `legacy_fragments` and content-free `legacy_import_evidence`, while retaining the exact existing `legacy_sources` triple.
- **Files modified:** `aura_backend/storage/schema.py`, `aura_backend/storage/repository.py`
- **Verification:** Both-root, fragment, duplicate, raw-metadata, rerun, and all prior atomic-ledger tests pass.
- **Committed in:** `275ae19`

**2. [Rule 2 - Missing critical functionality] Isolated authorized sources from Chroma access-time mutation**
- **Found during:** Task 03-03-02 GREEN
- **Issue:** Locked Chroma 1.5.9 changed database bytes during ordinary public-client access, violating the byte-identical authorized-source contract even without an explicit write.
- **Fix:** Hash and copy the authorized disposable restore into an operation-owned temporary child, open only that child, and verify the authorized source digest after import.
- **Files modified:** `aura_backend/storage/migration.py`, `tests/storage/test_legacy_migration.py`
- **Verification:** A mutation spy changes only the operation copy while the authorized source retains exact before/after SHA-256 parity.
- **Committed in:** `275ae19`

**3. [Rule 2 - Security hardening] Rejected nested source symlinks**
- **Found during:** Plan threat-boundary review
- **Issue:** Rejecting only a symlinked root did not prevent an internal source-tree symlink from escaping containment during hashing or copying.
- **Fix:** Reject every symlink entry before opening files or constructing a Chroma client.
- **Files modified:** `aura_backend/storage/migration.py`, `tests/storage/test_legacy_migration.py`
- **Verification:** Fourteen focused migration/ownership tests pass, including nested path forgery before client construction.
- **Committed in:** `731a2a1`

**4. [Rule 1 - Test defects] Corrected numeric and array assumptions in Chroma assertions**
- **Found during:** Task 03-03-01 GREEN
- **Issue:** Chroma can return a tiny floating-point-negative cosine distance around zero, and NumPy embedding arrays cannot be truth-tested with Python `or`.
- **Fix:** Used a `1e-6` distance tolerance and explicit `is not None` array checks while preserving the raw distance value.
- **Files modified:** `tests/storage/test_projection_rebuild.py`
- **Verification:** Seven projection/ownership tests and all plan gates pass.
- **Committed in:** `4771b2f`

**5. [Rule 1 - Typing defect] Narrowed the optional Chroma client**
- **Found during:** Independent post-wave Python typecheck
- **Issue:** Pyright retained the defensive `Any | None` type after `PersistentClient` construction and rejected `create_collection` as a possible access on `None`.
- **Fix:** Added an explicit unavailable-client guard before collection creation, preserving the existing cleanup path and storage error contract.
- **Files modified:** `aura_backend/storage/projection.py`
- **Verification:** Eight focused projection/ownership tests pass; project-wide Pyright reports zero errors; focused Ruff and diff hygiene pass.
- **Committed in:** `8b78eba`

---

**Total deviations:** 5 auto-fixed (3 missing-critical/security controls, 1 test correction, 1 typing correction).
**Impact on plan:** Every deviation was necessary to satisfy the declared immutable-source, typed-fragment, containment, or locked-Chroma behavior. No dependency, manifest, lock, production route, or real data operation was added.

## Issues Encountered

- The installed GSD helper remains broken because its expected parent `package.json` is absent. No package or manifest was created to mask the tooling defect; summary and sequential tracking use direct repository file updates consistent with Plans 03-01 and 03-02.
- Chroma's access-time mutation made direct read-open incompatible with immutable-source proof. The operation-copy design above resolved the issue without touching a real or durable source.

## TDD Gate Compliance

- Task 03-03-01: RED `d4ad44e` failed on the missing projection module; GREEN `4771b2f` passes the exact task command.
- Task 03-03-02: RED `242cfd4` failed on the missing migration module; GREEN `275ae19` passes the exact task command.
- Security follow-up `731a2a1` remains after the Task 2 GREEN commit and passes the complete plan gate.

## Verification

- Task 03-03-01 exact gate: `7 passed`.
- Task 03-03-02 exact gate after hardening: `14 passed`.
- Plan storage gate: `53 passed` across projection, migration, ownership, atomic ledger, and provenance/supersession.
- Complete storage suite: `91 passed` before the final nested-symlink hardening; the affected focused and plan gates were rerun afterward.
- Complete deterministic non-live suite after final code: `604 passed, 2 skipped, 1 deselected` in 31.40s.
- Post-wave focused projection/ownership gate: `8 passed`; project-wide Python typecheck: `0 errors, 0 warnings, 0 informations`.
- Ruff and `git diff --check`: pass for every changed implementation and test file.
- Dependency hygiene: every Python/test command used `uv run --locked --no-sync`; no package manifest or lock file changed.
- Data safety: every Chroma root used by new tests was created and copied beneath pytest temporary directories. No real historical Chroma root, durable backup, archive, profile, Memvid store, or data store was passed to Chroma or migration code.

## Known Stubs

None. Empty local accumulators and optional callbacks are populated runtime structures, not placeholder behavior.

## Threat Flags

None. Projection generation, restore authorization, legacy mapping, evidence privacy, schema, and package-supply surfaces are the exact trust boundaries covered by the plan's threat model and adversarial tests.

## User Setup Required

None - no dependency, credential, model, external service, or system configuration was added.

## Next Phase Readiness

- Plan 03-04 can consume validated affect-neutral vector candidates while reapplying scope, provenance, freshness, relevance, and generation gates in SQLite.
- Plans 03-08 and 03-09 now have an importer that requires their later exact authorization object; this plan performed no real historical import and grants no migration or read-switch authority.
- The preserved eight-row historical FK anomalies remain untouched and unresolved by design.

## Self-Check: PASSED

- All five declared created files and both modified storage-owner files exist.
- All five RED/GREEN/hardening commits and post-wave typing repair resolve in Git history in the required order.
- Every task acceptance command, the plan-level combined gate, Ruff, diff hygiene, and the complete deterministic non-live suite pass.
- No generated database, WAL, manifest, lock, profile, export, or runtime artifact is present in Git status.

---
*Phase: 03-memory-integrity-and-data-lifecycle*
*Completed: 2026-09-01*
