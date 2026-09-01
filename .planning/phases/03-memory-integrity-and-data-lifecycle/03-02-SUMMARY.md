---
phase: 03-memory-integrity-and-data-lifecycle
plan: 02
subsystem: testing
tags: [benchmark, retrieval, privacy, provenance, kill-gates, synthetic-corpus]

requires:
  - phase: 03-memory-integrity-and-data-lifecycle
    plan: 01
    provides: typed scope, event, memory, and provenance identities
provides:
  - versioned invented retrieval corpus with a static 10,000-event load slice
  - hash-bound manifest with exact cases, controls, thresholds, bounds, and arms
  - fail-closed benchmark runner with content-free evidence and typed outcomes
  - fixed competence, containment, resource, and alternative-adoption gates
affects: [03-04, 03-09, retrieval, benchmarking, affective-memory]

tech-stack:
  added: []
  patterns:
    - injected retriever protocol driven by immutable versioned fixtures
    - pass, fail, and inconclusive remain distinct at every benchmark gate
    - aggregate reports contain IDs, hashes, counts, timings, and codes only

key-files:
  created:
    - aura_backend/storage/benchmark.py
    - tests/storage/test_memory_benchmark.py
    - tests/fixtures/memory_eval/corpus.jsonl
    - tests/fixtures/memory_eval/manifest.json
  modified: []

key-decisions:
  - "The Phase 3 benchmark uses only committed invented fixtures; it never opens, samples, paraphrases, or hashes a historical Aura store."
  - "Timeout, resource limit, truncation, missing controls, invalid measurements, and incomplete repetitions return inconclusive with no fabricated metrics."
  - "Neutral and constant-salience arms are separate named interfaces whose salience contribution remains exactly zero in Phase 3."
  - "Alternative adoption requires five absolute quality points or a thirty-percent storage or latency reduction with no correctness, privacy, provenance, stale-fact, duplicate, or restore regression."

patterns-established:
  - "Fixture contract: canonical corpus digest plus exact case/control/threshold inventory before any result is accepted."
  - "Candidate verification: supplied scope and provenance are checked against frozen origin and event facts rather than trusted as labels."

requirements-completed: [TEST-03, DATA-05, DATA-09]

duration: 13h 3m
completed: 2026-09-01
status: complete
---

# Phase 3 Plan 02: Sanitized Memory Benchmark Summary

**A hash-bound invented corpus and fail-closed benchmark now distinguish faithful retrieval from scope leaks, stale facts, forged provenance, missing controls, and resource-limited non-results before production retrieval is measured.**

## Performance

- **Duration:** 13h 3m, including the continuation pause
- **Started:** 2026-09-01T04:31:38Z
- **Completed:** 2026-09-01T17:35:04Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Froze 10,031 invented JSONL records, including all eleven required case classes and a materialized fixed-seed 10,000-event load slice, under corpus SHA-256 `7c50a8a4b6db28c52d4ae1397f2dfdd3e85bea53d31055f19a898df265e17d51`.
- Bound the manifest to exact case IDs and expected origins, tokenizer, cosine metric slot, candidate caps, RRF `k=60`, admission thresholds, page bounds, three timing repetitions, runtime/hardware labels, controls, arms, and kill gates.
- Added a bounded loader that rejects empty, malformed, truncated, oversized, digest-mismatched, privacy-marked, duplicate, unknown, incomplete, or non-deterministic inputs with content-free codes.
- Added an injected benchmark runner that derives metrics from fixed origin, scope, provenance, freshness, timing, and completeness facts; reports never include query or memory text.
- Proved faithful controls pass while scope-free, scope-spoofed, stale, provenance-free, forged-provenance, unknown-origin, truncated, timed-out, resource-limited, missing-control, and incomplete-repetition variants cannot pass.
- Encoded the existing five-point / thirty-percent alternative-adoption rule and three-cycle / one-workday stopping condition without implementing or claiming a production alternative.

## Task Commits

Each task followed RED then GREEN. No separate refactor commit was needed because the final implementation remained typed, documented, lint-clean, and green.

1. **Task 03-02-01: Freeze the sanitized corpus and manifest contract**
   - `93891c1` — RED corpus, manifest, privacy, digest, truncation, and bound contract
   - `2f78cc6` — GREEN complete invented corpus and fail-closed loader
2. **Task 03-02-02: Prove benchmark controls and truthful gate outcomes**
   - `2efecff` — RED injected runner, negative controls, reports, and adoption gates
   - `12bd6c1` — GREEN metric aggregation, typed outcomes, containment checks, and stopping semantics

## Files Created/Modified

- `aura_backend/storage/benchmark.py` — immutable instrument records, bounded loader, injected retriever protocol, content-free reporting, metrics, competence gates, and adoption gate.
- `tests/storage/test_memory_benchmark.py` — thirty corpus, privacy, integrity, faithful-control, adversarial-control, arm-separation, report-safety, and alternative-gate tests.
- `tests/fixtures/memory_eval/corpus.jsonl` — invented fixed retrieval cases and static 10,000-event load slice.
- `tests/fixtures/memory_eval/manifest.json` — canonical digest, exact inventory, thresholds, controls, bounds, labels, and zero-salience arm definitions.

## Decisions Made

- The static load slice is committed rather than generated during a benchmark, so the input cannot drift with runtime state, a model, the network, or an unrecorded generator version.
- The benchmark accepts only the two pre-registered Phase 3 arms; both have zero salience contribution, and neither can change eligibility or rank.
- A run with incomplete evidence has `metrics: null` and `status: inconclusive`; the runner never extrapolates partial observations into a passing score.
- Nonempty provenance supplied by a retriever is insufficient: every source ID must resolve to an invented event in the same scope.
- Reports retain per-case selected origin IDs, repetitions, timings, and fixed codes while excluding fixture query/text fields.

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Closed the forged-provenance and scope-spoof aggregate blind spot**
- **Found during:** Task 03-02-02 GREEN pre-commit review
- **Issue:** The first GREEN draft recorded invalid provenance and source-scope mismatches in per-case codes but aggregated only whether the retriever supplied a nonempty provenance tuple and its declared scope. A forged nonempty source ID or a cross-scope origin carrying a spoofed same-scope label could therefore avoid the aggregate containment gate.
- **Fix:** Derived containment metrics from validation codes checked against the frozen origin/event inventory, added an invalid-candidate count, and made recall or duplicate regressions block alternative adoption.
- **Files modified:** `aura_backend/storage/benchmark.py`, `tests/storage/test_memory_benchmark.py`
- **Verification:** Added forged-provenance, scope-spoof, unknown-origin, and recall-regression controls; all 30 benchmark tests pass.
- **Committed in:** `12bd6c1`

---

**Total deviations:** 1 auto-fixed bug.
**Impact on plan:** The fix directly strengthens the planned provenance, scope, and no-regression gates. It adds no dependency, production retrieval path, or scope beyond the benchmark instrument.

## Issues Encountered

- The installed GSD helper under `/home/ty/.claude/gsd-core` still cannot load because its expected parent `package.json` is absent. No package or manifest was created to mask the tooling defect; planning-state files are updated directly, matching Plan 03-01's established fallback.

## TDD Gate Compliance

- Task 03-02-01: RED `93891c1` precedes GREEN `2f78cc6` — PASS.
- Task 03-02-02: RED `2efecff` precedes GREEN `12bd6c1` — PASS.
- Both RED runs failed for their intended unimplemented contracts; both exact GREEN commands pass.

## Verification

- Task 03-02-01 exact gate: `14 passed, 16 deselected` in 0.24s.
- Task 03-02-02 exact gate: `30 passed` in 1.09s.
- Plan storage gate: `64 passed` in 2.10s across the benchmark, atomic ledger, and provenance/supersession suites.
- Complete deterministic non-live suite: `585 passed, 2 skipped, 1 deselected` in 27.42s.
- Ruff: all changed Python and test files pass.
- Dependency hygiene: no package manifest or lock file changed; every Python/test command used the existing locked environment with `--no-sync`.
- Data safety: no historical store was opened, read, migrated, repaired, copied, or deleted; the only corpus content is invented and versioned.

## Known Stubs

None. `embedding_fingerprint_slot` is intentionally bound to `not-measured-phase-03-02` because this plan validates the ruler before Plan 03-04 measures a real neutral retriever; it is not a production ranking claim or missing implementation path.

## Threat Flags

None. The new bounded fixture-file and injected-retriever surfaces are the exact trust boundaries covered by the plan threat model; no network, authentication, historical-store, schema, or production write surface was introduced.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness

- Plan 03-04 can connect a real affect-neutral retriever only after its own projection dependencies are complete, using this immutable instrument without changing thresholds.
- No production retrieval competence or performance claim exists yet; only the benchmark ruler and its controls have passed.

## Self-Check: PASSED

- All four declared implementation/fixture files exist.
- All four RED/GREEN task commits resolve in Git history and occur in the required order.
- Every task acceptance command and the plan-level full non-live command passed.
- The worktree contains no uncommitted task files, generated databases, manifests, locks, or runtime artifacts.

---
*Phase: 03-memory-integrity-and-data-lifecycle*
*Completed: 2026-09-01*
