---
phase: 03-memory-integrity-and-data-lifecycle
plan: 04
subsystem: storage
tags: [sqlite, fts5, chroma, retrieval, pagination, provenance, benchmark]

requires:
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 01
    provides: authoritative SQLite origins, provenance, supersession, and retrieval-run tables
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 02
    provides: frozen synthetic benchmark instrument and fail-closed competence gates
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 03
    provides: bounded cosine projection candidates with stable generation and content hashes
provides:
  - affect-neutral FTS5 and cosine candidate fusion with authoritative hard gates
  - content-free candidate traces and HMAC-protected frozen-run cursors
  - bounded immutable ledger history using frozen keyset pagination
  - exact-zero Phase 4 salience seam and public-retriever benchmark adapter
affects: [03-06, 03-08, 03-09, retrieval, history, runtime-api, lifecycle]

tech-stack:
  added: []
  patterns:
    - configuration-owned 50-plus-50 candidate caps and normalized reciprocal-rank fusion
    - SQLite revalidation before admission of every projection candidate
    - opaque server-side cursor bindings over HMAC-authenticated random identifiers
    - frozen retrieval runs and observed-at/event-id history keysets

key-files:
  created:
    - aura_backend/storage/retrieval.py
    - tests/storage/test_hybrid_retrieval.py
  modified:
    - aura_backend/storage/models.py
    - aura_backend/storage/benchmark.py
    - tests/storage/test_memory_benchmark.py
    - tests/memory/test_prompt_memory_boundary.py

key-decisions:
  - "Candidate eligibility is a mandatory SQLite-authoritative conjunction; caller filters and later scorers cannot weaken scope, provenance, supersession, hash, generation, or neutral relevance gates."
  - "Retrieval pages read a stored frozen run, while history pages freeze both an event primary-key watermark and the observed-at/event-id horizon."
  - "Phase 3 accepts only an exact-zero salience scorer; any non-zero or non-numeric contribution fails closed before a page is returned."
  - "The benchmark adapter consumes only the public retrieval-page contract and preserves the frozen instrument controls and explicit non-pass outcomes."

patterns-established:
  - "Neutral ranking: normalized RRF first, exact match second, observed time third, stable origin ID last."
  - "Cursor boundary: opaque token plus stored scope, query, config, generation, run, expiry, offset, and last stable sort tuple."

requirements-completed: [TEST-03, DATA-05, DATA-06, DATA-09]

duration: 4h 39m
completed: 2026-09-01
status: complete
---

# Phase 3 Plan 04: Neutral Hybrid Retrieval and Stable Pages Summary

**Affect-neutral SQLite FTS5 and cosine candidates now pass fixed evidence gates before normalized RRF ranking, with content-free traces, frozen pages, exact-zero salience, and a fail-closed benchmark boundary.**

## Performance

- **Duration:** 4h 39m
- **Started:** 2026-09-01T18:03:59Z
- **Completed:** 2026-09-01T22:43:10Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments

- Added fixed configuration-owned caps of 50 FTS5 and 50 cosine candidates, normalized RRF with `k=60`, canonical-content deduplication, and deterministic neutral ordering.
- Re-resolved candidates against SQLite truth and rejected missing origins, cross-scope rows, absent provenance, superseded/retracted memories, content/hash mismatches, stale projection generations, and insufficient neutral similarity with stable content-free codes.
- Recorded complete selected, rejected, and duplicate-collapsed traces containing IDs, hashes, ranks, raw channel values, normalized components, every gate decision, contributors, provenance, and selected rank without query or remembered text.
- Added default-20/maximum-100 retrieval pages over retained frozen runs with opaque HMAC-authenticated cursors bound to scope, query, configuration, projection generation, run, expiry, offset, and the last stable sort tuple.
- Added bounded ledger history over the `(observed_at, event_id)` keyset, protected by an initial event-PK watermark and horizon so later writes cannot create gaps or duplicates in an active traversal.
- Kept retrieved instructions inside the exact `<untrusted_memory_context>` prompt delimiter and proved formatting creates no event or derivation.
- Added a public retrieval-page benchmark adapter. The committed synthetic positive control clears every frozen gate; deliberately missed direct recall remains the explicit `recall_gate_failed` outcome.

## Task Commits

Each task followed a committed RED then GREEN cycle. No separate refactor commit was needed.

1. **Task 03-04-01: Enforce neutral candidate gates and normalized RRF**
   - `3be6033` — RED adversarial scope, provenance, freshness, raw-score, duplicate, and ordering contract
   - `63ab5d7` — GREEN authoritative neutral retrieval gates and fixed fusion
2. **Task 03-04-02: Add stable traces/cursors and prove the neutral benchmark**
   - `951ce6c` — RED cursor, history, trace, salience, prompt-boundary, and connected-benchmark contract
   - `9b7db3d` — GREEN frozen retrieval pages, keyset history, complete traces, zero salience, and benchmark adapter

## Files Created/Modified

- `aura_backend/storage/retrieval.py` — authoritative hybrid retrieval, hard gates, RRF/deduplication, traces, frozen cursor pages, history pages, retention, and zero salience.
- `aura_backend/storage/models.py` — immutable retrieval/history page and trace value objects.
- `aura_backend/storage/benchmark.py` — public retriever adapter that preserves the frozen benchmark protocol.
- `tests/storage/test_hybrid_retrieval.py` — two-scope adversarial gates, deterministic ranking, tamper/expiry/binding, frozen-page mutation, trace privacy, history, and non-zero salience rejection.
- `tests/storage/test_memory_benchmark.py` — connected public-page positive and missed-gate controls over the committed synthetic corpus.
- `tests/memory/test_prompt_memory_boundary.py` — exact untrusted-memory delimiter and no-write proof against stored instructions.

## Decisions Made

- FTS5 BM25 and cosine raw values remain trace facts only. Ranking uses normalized channel ranks, preventing incomparable raw score scales from affecting order.
- Event provenance is its immutable event ID; derived memories require non-empty source-event provenance and must remain neither superseded nor retracted.
- Frozen retrieval continuation does not consult a changed live projection. It verifies the cursor against the original stored run and original projection generation instead.
- Cursor payloads contain no scope, query, content, or run metadata. A random token ID is authenticated with HMAC-SHA256 and resolves to bounded server-side binding state.
- History combines keyset ordering with a primary-key watermark because a timestamp horizon alone cannot exclude a later insert carrying an older timestamp.
- The benchmark connector is an instrument/protocol proof over committed synthetic data, not a claim about real conversation quality or a production-data benchmark.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Test defect] Repaired the Task 2 RED test imports and scorer signature**
- **Found during:** Task 03-04-02 GREEN static checks
- **Issue:** The committed RED tests referenced `StorageFailure` without importing it, and the deliberately non-zero scorer used a protocol-incompatible parameter annotation/name.
- **Fix:** Imported the existing exception and typed the scorer against `RetrievalItem` without weakening the non-zero rejection assertion.
- **Files modified:** `tests/storage/test_hybrid_retrieval.py`
- **Verification:** Focused Ruff and changed-file Pyright pass; the exact 44-test Task 2 gate remains green.
- **Committed in:** `9b7db3d`

---

**Total deviations:** 1 auto-fixed test defect.
**Impact on plan:** The correction made the intended RED contract executable and type-valid; no behavior, threshold, corpus, dependency, manifest, or lock was changed.

## Issues Encountered

- The installed GSD helper remains unusable because its resolved installation is missing the expected parent `package.json`. No package or manifest was created to conceal that environment defect; summary and sequential tracking were updated directly, following the established Phase 3 fallback.
- Project-wide Pyright currently reports 18 pre-existing errors in unrelated autonomic, Memvid, restore, Gemini, embedding, migration, and projection files. The changed production retrieval/model/benchmark files and the changed hybrid contract report zero errors; the plan's full runtime gates are green.

## TDD Gate Compliance

- Task 03-04-01: RED `3be6033` failed on the missing retrieval module; GREEN `63ab5d7` passes the exact focused contract with `5 passed`.
- Task 03-04-02: RED `951ce6c` failed on missing `ZeroSalienceScorer` and `ConnectedBenchmarkRetriever`; GREEN `9b7db3d` passes the exact task command with `44 passed`.
- Commit order is RED → GREEN for each behavior-adding task.

## Verification

- Task 03-04-01 exact focused gate: `5 passed`.
- Task 03-04-02 exact gate: `44 passed`.
- Plan integration gate across retrieval, benchmark, prompt boundary, projection rebuild, and provenance/supersession: `71 passed` in 4.80s.
- Complete deterministic non-live suite: `616 passed, 2 skipped, 1 deselected` in 32.71s.
- Focused Ruff: pass for all six plan implementation/test files.
- Changed production files plus hybrid retrieval contract Pyright: `0 errors, 0 warnings, 0 informations`.
- `git diff --check`: pass.
- Dependency hygiene: every Python/test command used `uv run --locked --no-sync`; no dependency, package manifest, or lock file changed.
- Data safety: all new ledger/projection behavior was exercised only with pytest temporary SQLite databases and deterministic in-memory/synthetic projection fixtures. No real historical, data, backup, archive, Chroma, Memvid, or profile root was opened or mutated.

## Known Stubs

None. Empty collections and optional cursor fields are bounded runtime state, not placeholder UI or unwired behavior.

## Threat Flags

None. Scope admission, candidate revalidation, stored prompt text, trace privacy, cursor integrity, and package-supply boundaries are the exact surfaces covered by the plan threat model and adversarial tests.

## User Setup Required

None - no dependency, credential, model, external service, or system configuration was added.

## Next Phase Readiness

- Later runtime/API wiring can consume one public neutral page boundary without gaining any authority to weaken scope/evidence gates or add dynamic affect.
- Phase 4 can replace only the salience implementation after its own evaluation gate; Phase 3 rejects every non-zero scorer exactly.
- Lifecycle plans can build on bounded history and retrieval pages, while real migration/read-switch/cleanup authority remains explicitly ungranted.
- Real conversation quality remains unmeasured by design; this plan proves deterministic contracts and frozen synthetic instrument behavior only.

## Self-Check: PASSED

- All six declared plan implementation/test files and this summary exist.
- All four RED/GREEN commits resolve in Git history in the required order.
- Both exact task gates, the plan integration gate, Ruff, changed-file typing, diff hygiene, and the complete deterministic non-live suite pass.
- No generated database, WAL, manifest, lock, profile, export, or runtime artifact is present in Git status.

---
*Phase: 03-memory-integrity-and-data-lifecycle*
*Completed: 2026-09-01*
