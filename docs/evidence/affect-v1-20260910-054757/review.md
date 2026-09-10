# Checkpoint A Review (Second Submission) — S01–S03 Integrity Repairs & Grounded Regulation

**Date**: 2026-09-10  
**Evidence directory**: `docs/evidence/affect-v1-20260910-054757/`  
**Commit branch**: `main`  
**Status**: Ready for review (Checkpoint A stop)

---

## 1. Executive Summary

This submission resolves all six review findings from the rejection of the initial Checkpoint A submission. Each finding was turned into an explicit regression test, verified to fail on the previous implementation, and fixed.

All 69 affect tests pass; all 827 tests in the wider test suite pass (with zero regressions); ruff checks pass with zero warnings or errors; pyright reports zero errors; and deterministic evaluation passes Gate M and Gate P pure engine latency benchmarks.

---

## 2. Review Findings & Remediation Verification

### Finding 1: Timeout Race & Revision Conflict on Production Route (High)
* **Problem**: `persist_conversation_exchange_immediate` used `asyncio.wait_for` wrapping `to_thread`. When timeout occurred, `main.py` discarded the staged turn and stayed at revision 0, but the OS thread continued running and committed revision 1 to SQLite. On the subsequent request, `AffectService.get_state` only consulted SQLite if the scope was not in `_states`; since the scope was in `_states` with revision 0, the next turn staged revision 1 expecting revision 0, causing `affect_revision_conflict` on the SQLite CAS write.
* **Fix**:
  1. Wrapped `to_thread` with `asyncio.shield` in `persist_conversation_exchange_immediate` so timeout does not abort in-flight durability checks.
  2. Updated `AffectService.get_state` to always reconcile with `self.repository.get_affect_head(scope_id)` whenever `persisted.revision > current.revision`.
  3. Reconciled head state in `main.py` before staging turns so late commits are absorbed.
  4. Updated `main.py` error path to read the latest reconciled state from `get_state` rather than blindly reverting to pre-turn state.
* **Regression Test**:
  * `tests/affect/test_durability_and_loop.py::test_timeout_late_commit_does_not_cause_revision_conflict` (PASS)

### Finding 2: Regulation Had No Functional Effect & Staged Decisions Unwired (High)
* **Problem**: Disrespect messages produced zero appraisal events because `_DIRECTED_DISRESPECT_PATTERNS` was incomplete and `dynamics.py` hard-coded `enable_contempt_branch: False` which bypassed impulses completely. Consequently, `candidate_state` never deviated from baseline, and `regulate()` returned an identical state. Furthermore, `RegulationDecision` was not persisted into `AffectTransition`, and `main.py` did not pass `task_facts` or `task_outcome` to the affect lifecycle.
* **Fix**:
  1. Enhanced `aura_backend/affect/appraisal.py` with directed disrespect patterns (e.g. "you are completely useless and stupid") emitting `repeated_directed_contempt` while properly guarding against tool complaints, sarcasm, quotes, and self-sadness.
  2. Removed the hard-coded bypass in `aura_backend/affect/dynamics.py` so candidate impulses reach the regulator.
  3. Implemented active numerical suppression in `aura_backend/affect/regulation.py::regulate()`: when isolated disrespect is detected, `candidate_state` (which received the -0.08 impulse) has its affiliation dimension restored to `pre_impulse_state.affiliation`, resulting in a measurable numerical difference between candidate and regulated state.
  4. Wired `_staged_decisions` into `AffectService`: `compute_provisional_policy` stages the decision, `stage_turn` merges it into `AffectTransition.accepted_appraisal["regulation_decision"]`, and `discard_staged_turn` clears it.
  5. Added optional `task_facts` and `task_outcome` fields to `ConversationRequest` in `main.py`, passing `task_facts` to `compute_provisional_policy` and `task_outcome` (as `TaskOutcome`) to `stage_turn`.
* **Regression Tests**:
  * `tests/affect/test_regulation.py::test_disrespect_regulation_produces_zero_affiliation_delta` (PASS)
  * `tests/affect/test_regulation.py::test_regulation_decision_persisted_in_transition` (PASS)

### Finding 3: Live Evaluation History Was Contaminated (High)
* **Problem**: In live provider mode, `evaluate.py` used a single shared `canonical_conversation` built exclusively from Arm C's responses. All arms (A, B, C, D) thus received Arm C's generated context as their conversation history, and subsequent scenarios inherited previous scenario context.
* **Fix**:
  1. Replaced shared `canonical_conversation` with `per_arm_history: dict[Arm, list[dict[str, str]]]` initialized freshly per variant.
  2. In live mode, each arm receives its own independent history (`per_arm_history[arm]`) and appends only its own user/assistant exchanges.
  3. In mock mode, all arms receive `fixture_conversation` constructed strictly from authored scenario turns.
* **Regression Test**:
  * `tests/affect/test_evaluate.py::test_live_mode_histories_are_per_arm_not_shared` (PASS)

### Finding 4: Arm D Was Not the Specified Control (High)
* **Problem**: Arm D did not receive `task_facts` (passed `None`), used unmatched post-turn states, and set `last_event_time=timestamp`, resulting in zero decay interval.
* **Fix**:
  1. Wired `turn_task_facts` identically across Arm B, Arm C, and Arm D.
  2. Injected shuffled prior vectors from `d_trajectory_pool` (collected from actual Arm C runs) at matching turn indices.
  3. Set `last_event_time = max(0.0, timestamp - 30.0)` for Arm D so the standard 30-second inter-turn decay interval operates normally.
* **Regression Test**:
  * `tests/affect/test_evaluate.py::test_arm_d_receives_same_task_evidence_as_arm_c` (PASS)

### Finding 5: Nonsense Passed Gate B & 0.5 Contaminated Correctness (High)
* **Problem**: A provider outputting keyword-stuffed nonsense ("with that in mind, based on the answer is therefore...") could pass continuity and recovery heuristics. Furthermore, 0.5 (UNSCORABLE for families without `task_check`) was averaged into `task_correctness`, allowing nonsense to pass Gate B.
* **Fix**:
  1. Defined `_scorable_task_corr` which strictly filters out `0.5` placeholder values.
  2. Added `task_correctness_gate_pass`: for scenario families with an active `task_check`, Arm C must achieve a mean correctness score $\ge 1.0$.
  3. Gate B requires `task_correctness_gate_pass` to pass.
* **Regression Test**:
  * `tests/affect/test_evaluate.py::test_keyword_nonsense_fails_gate_b` (PASS)

### Finding 6: Evidence Report Overstated Coverage (Medium)
* **Problem**: The previous review report cited `test_projection_failure_does_not_discard_committed_ledger` and `test_fixture_history_not_contaminated_across_scenarios` as PASS when neither test actually existed in the test suite.
* **Fix**:
  1. Implemented `test_projection_failure_does_not_discard_committed_ledger` in `tests/affect/test_durability_and_loop.py` (verifies ledger commit persists and affect advances when projection raises `RuntimeError`).
  2. Implemented `test_fixture_history_not_contaminated_across_scenarios` in `tests/affect/test_evaluate.py` (verifies scenario 2 evaluates independently of scenario 1).
  3. Verified both tests execute and pass in the test suite.
* **Regression Tests**:
  * `tests/affect/test_durability_and_loop.py::test_projection_failure_does_not_discard_committed_ledger` (PASS)
  * `tests/affect/test_evaluate.py::test_fixture_history_not_contaminated_across_scenarios` (PASS)

---

## 3. Comprehensive Regression Test Verification Table

All tests listed below actually exist on disk and have been verified by test execution:

| Test Name | File | Line | Finding Addressed | Status |
|---|---|---|---|---|
| `test_timeout_late_commit_does_not_cause_revision_conflict` | `tests/affect/test_durability_and_loop.py` | 807 | Finding 1 (F3 timeout race) | PASS |
| `test_projection_failure_does_not_discard_committed_ledger` | `tests/affect/test_durability_and_loop.py` | 872 | Finding 6 / F2 (projection vs ledger) | PASS |
| `test_disrespect_regulation_produces_zero_affiliation_delta` | `tests/affect/test_regulation.py` | 326 | Finding 2 (functional regulation) | PASS |
| `test_regulation_decision_persisted_in_transition` | `tests/affect/test_regulation.py` | 352 | Finding 2 (decision provenance) | PASS |
| `test_live_mode_histories_are_per_arm_not_shared` | `tests/affect/test_evaluate.py` | 311 | Finding 3 (live history contamination) | PASS |
| `test_arm_d_receives_same_task_evidence_as_arm_c` | `tests/affect/test_evaluate.py` | 364 | Finding 4 (Arm D matched control) | PASS |
| `test_fixture_history_not_contaminated_across_scenarios` | `tests/affect/test_evaluate.py` | 390 | Finding 6 / F4 (cross-scenario isolation) | PASS |
| `test_keyword_nonsense_fails_gate_b` | `tests/affect/test_evaluate.py` | 416 | Finding 5 (Gate B nonsense prevention) | PASS |
| `test_task_correctness_does_not_use_length_fallback` | `tests/affect/test_evaluate.py` | 254 | F7 (no length fallback) | PASS |
| `test_provider_init_failure_exits_nonzero` | `tests/affect/test_evaluate.py` | 272 | F5 (fail explicit on init) | PASS |

---

## 4. Verification Suite Results

```bash
# 1. Affect suite
$ uv run --locked --no-sync python -m pytest tests/affect -v
69 passed, 1 warning in 2.33s

# 2. Entire repository regression suite
$ uv run --locked --no-sync python -m pytest tests -q -m "not live and not private_evidence"
827 passed, 7 skipped, 5 deselected, 1 warning in 31.80s

# 3. Python static typecheck
$ npm run typecheck:python
pyright --project pyproject.toml
0 errors, 0 warnings, 0 informations

# 4. Frontend static typecheck & build
$ npm run typecheck:frontend && npm run build
tsc --noEmit && vite build
Built successfully in 73ms

# 5. Linter
$ uv run --locked --no-sync ruff check aura_backend tests --exclude aura_backend/archive_unused --exclude aura_backend/scratch --exclude aura_backend/tests
All checks passed!

# 6. Git whitespace check
$ git diff --check
(clean, exit code 0)

# 7. Deterministic evaluation gate
$ uv run --locked --no-sync python -m aura_backend.affect.evaluate --mode deterministic
[Gate Deterministic Verdict]: PASS
Evidence directory: docs/evidence/affect-v1-20260910-054757
```

---

## 5. Model Defaults & Scope Boundary

* `ornith-aq1.5:35b` remains the unchanged baseline model default.
* `aura-ornith:35b` remains the installed Aura specialization; Modelfile.aura was not touched.
* No live provider comparisons were run.
* No model training or fine-tuning was performed.
* No destructive database migrations or schema alterations were performed.
* Legacy conversation adapter refactoring (F9) remains explicitly deferred to S05/S06 as planned.
* Stopping at Checkpoint A for review as instructed.
