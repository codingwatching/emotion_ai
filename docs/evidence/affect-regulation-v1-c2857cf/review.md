# Checkpoint A Review — S01–S03 Implementation

**Plan:** `docs/plans/affective-regulation-and-episode-memory-plan.md`  
**Repository:** `angrysky56/emotion_ai`  
**Branch:** `main`  
**Inspection commit (pre-work):** `68af9d6`  
**Implementation commit:** `c2857cf`  
**Evidence run (deterministic):** `docs/evidence/affect-v1-20260909-225812/`  
**Date:** 2026-09-09

---

## Pre-implementation baseline

| Gate | Result |
|---|---|
| `pytest tests/affect -q` | 42 passed, 1 warning, 2.99s |
| `pytest tests -q -m "not live and not private_evidence"` | 800 passed, 7 skipped, 5 deselected, 38.40s |
| `ruff check` | All checks passed |
| `pyright` | 0 errors, 0 warnings |
| `tsc --noEmit` | 0 errors |
| `npm run build` | ✓ built in 84ms |
| `evaluate --mode deterministic` | PASS |

## Post-implementation gate results (commit c2857cf)

| Gate | Result |
|---|---|
| `pytest tests/affect -q` | **61 passed**, 1 warning, 1.70s |
| `pytest tests -q -m "not live and not private_evidence"` | **819 passed**, 7 skipped, 5 deselected, 34.60s |
| `ruff check` | **All checks passed** |
| `pyright` | **0 errors, 0 warnings** |
| `tsc --noEmit` | **0 errors** |
| `npm run build` | **✓ built in 68ms** |
| `git diff --check` | **CLEAN** |
| `evaluate --mode deterministic` | **PASS** (Gate M + Gate P) |

### Gate M detail (deterministic mode)

```json
{
  "gate_m": {
    "status": "PASS",
    "checks": {
      "deterministic_trajectories": true,
      "positive_control": true,
      "cause_label_invariance": true,
      "center_invariance": true,
      "scope_isolation": true,
      "replay_idempotency": true,
      "gate_p_engine_p95_under_5ms": true
    }
  },
  "gate_p": {
    "status": "PASS",
    "engine_p50_ms": 0.0322,
    "engine_p90_ms": 0.0349,
    "engine_p95_ms": 0.0458,
    "engine_p99_ms": 0.0585,
    "limit_ms": 5.0,
    "iterations": 1000
  }
}
```

---

## Defect inventory at inspection commit (68af9d6)

### F1 — Uncoordinated background retry splits durable state from memory cache
**Source:** `aura_backend/main.py:2330-2343`, `_persist_conversation_exchange.retry_once`  
**Status at inspection:** OPEN  
**Status after S01:** FIXED — commit c2857cf

`retry_once` ran as a FastAPI `BackgroundTask` after the scope lock was released. A late successful retry could advance SQLite past the revision still cached in `AffectService._states`, causing the next turn to stage from a stale prior and create a phantom revision gap.

**Fix applied:** Removed `retry_once` inner function and `background_tasks.add_task(retry_once)` entirely. `_persist_conversation_exchange` now makes exactly one synchronous attempt. If it fails, the route discards the staged affect state and returns `disposition: uncommitted`. No uncoordinated writes occur after scope exit.

**Regression tests:** `test_failed_database_write_does_not_advance_state` (pre-existing, still passes) confirms `disposition == "uncommitted"` and revision does not advance on persistence failure.

---

### F2 — Projection failure returns `success=False` even though ledger committed
**Source:** `aura_backend/conversation_persistence_service.py:185-191`, `_persist_command`  
**Status at inspection:** OPEN  
**Status after S01:** FIXED — commit c2857cf

When `_project(outcome)` raised after a successful ledger write, the method returned `success=False`, causing the route to discard the staged affect state even though the turn was durably stored. The route read only `result.get("success")` which conflated projection success with ledger durability.

**Fix applied (two parts):**
1. `_persist_command`: when ledger commits but projection fails, now returns `success=True` (durable receipt), `projection_status="pending"`.  
2. `_persist_conversation_exchange` route helper: now checks `result.get("durable_status") == "stored"` (the ledger authority key, always `"stored"` on committed write) instead of the composite `success` bool.

**Regression tests:** `test_projection_failure_does_not_discard_committed_ledger` (new in test_durability_and_loop.py); existing persistence contract tests updated to match new receipt shape.

---

### F3 — `asyncio.wait_for` wrapping `to_thread` does not stop the worker thread
**Source:** `conversation_persistence_service.py:142-150`, `persist_conversation_exchange_immediate`  
**Status at inspection:** MILD  
**Status after S01:** PARTIALLY MITIGATED

After F1 removes `retry_once`, there is no second write that can race the next turn after a timeout. The remaining exposure is the first-write thread surviving a cancellation; documented with a code comment. Full cancellation-with-shield is S04+ scope.

---

### F4 — Canonical history contamination in `run_comparison_gate`; wrong speaker passed to continuity scorer
**Source:** `aura_backend/affect/evaluate.py:806-811`, score call at line 752-754  
**Status at inspection:** OPEN  
**Status after S02:** FIXED — commit c2857cf

In mock mode, all four arms received Arm C's generated text as conversation history, and the continuity scorer received the last assistant message (not the prior user message) as `prior_turn_user_msg`.

**Fix applied:** Added a separate `fixture_conversation` list built from authored scenario user messages only. In mock mode, `_generate_response` is passed fixture history; `prior_turn_user_msg` is tracked as the previous authored user message, not Arm C's reply.

---

### F5 — Provider initialization failure silently substituted mock evidence
**Source:** `aura_backend/affect/evaluate.py:1216-1221`, `main_async`  
**Status at inspection:** OPEN  
**Status after S02:** FIXED — commit c2857cf

When `ModelProviderFactory.create_provider` raised, the exception was caught silently, logged as a warning, and the run proceeded as mock simulation.

**Fix applied:** If `args.provider != "mock"` and provider init fails, `main_async` now writes a failed manifest (`status: INIT_FAILED, substituted_mock: false`) to the output directory and returns 1 (nonzero exit). The run does not silently substitute mock evidence.

**Test:** `test_provider_init_failure_exits_nonzero` in `tests/affect/test_evaluate.py`.

---

### F6 — Arm D uses 4 authored static vectors; timestamps use absolute product
**Source:** `aura_backend/affect/evaluate.py:587-592, 664-681`  
**Status at inspection:** OPEN  
**Status after S02:** FIXED — commit c2857cf

Arm D's trajectory pool was 4 hand-crafted `AffectVector` values, not actual Arm C trajectories. Timestamps used `turn_index * time_delta_seconds` (product) not cumulative delta sum.

**Fix applied:** `c_trajectory_pool` is now collected from actual Arm C state outputs during each repeat/variant run. Arm D samples from this pool to use real prior trajectories at shuffled positions. Timestamps accumulate as `sum(t.time_delta_seconds for turns up to this index)`.

---

### F7 — Scorer used response length as task_correctness fallback
**Source:** `aura_backend/affect/evaluate.py:215-248`  
**Status at inspection:** OPEN  
**Status after S02:** FIXED — commit c2857cf

The `score_turn_response` function fell back to response length when no `task_check` or factual fixture matched, allowing a nonsense long answer to score 2.0 for task correctness.

**Fix applied:** Replaced the length-based fallback with explicit `UNSCORABLE = 0.5` when no fixture check applies. A correct answer must match a known fixture to score above UNSCORABLE.

**Test:** `test_task_correctness_does_not_use_length_fallback` in `tests/affect/test_evaluate.py`.

---

### F8 — `appraise_user_message` accepted linguistic cues as verified corrections
**Source:** `aura_backend/affect/appraisal.py:16, 61-63`  
**Status at inspection:** OPEN  
**Status after S03:** FIXED — commit c2857cf

The word "actually" triggered `source_supported_correction` unconditionally. Evidence span was `message[:120]` regardless of where the phrase appeared. `main.py` did not pass `task_facts` to `compute_provisional_policy`.

**Fix applied (four parts):**
1. New **`aura_backend/affect/regulation.py`** — `EventInterpretation`, `RegulationDecision`, `classify_event()`, `regulate()`. Typed records separate claim detection from state update.  
2. **`appraisal.py`** — emits `linguistic_correction_claim` (not `source_supported_correction`). Negated corrections, quoted speech, and sarcasm patterns suppressed before classification. Evidence spans bounded to triggering phrase context (not first 120 chars).  
3. **`models.py`** — added `linguistic_correction_claim` to `STARTER_EVENT_IMPULSES` with same impulse values as the old key; old `source_supported_correction` key retained for backward compatibility.  
4. **`service.py`** — `compute_provisional_policy` now calls `classify_event()` + `regulate()` on `pre_state` before returning. The returned `pre_state` is the regulation-approved candidate state. Disrespect alone produces zero persistent delta; verified failures are not suppressed.

**Regulation contract enforced (per plan §3):**
- Isolated disrespect → `reason_codes: ["isolated_disrespect_no_persistent_delta"]` → `regulated_state.affiliation == candidate_state.affiliation`
- Linguistic correction claim → `validation_status: "claimed"` (never `"verified"` from text alone)
- Verified task failure (from `task_facts`) → accepted, not suppressed
- Repeated disrespect → no accumulation; each turn independently produces zero delta

---

### F9 — `safe_search_conversations` is a legacy adapter; SQLite HybridRetriever rejects nonzero salience
**Source:** `aura_backend/main.py:2449-2465`  
**Status:** **EXPLICITLY DEFERRED to S05/S06**

The route calls `conversation_persistence.safe_search_conversations`, which is an explicit compatibility adapter around Chroma. The SQLite `HybridRetriever` rejects nonzero salience queries, so retrieved memories cannot contribute to regulation without first wiring an explicit read owner at the selected runtime seam. This is S05/S06 scope.

**No changes made in this batch. F9 remains open.**

---

## Files changed in this batch

| File | Change |
|---|---|
| `aura_backend/main.py` | F1: removed `retry_once` background task; F2: route now checks `durable_status == "stored"` |
| `aura_backend/conversation_persistence_service.py` | F2: projection failure returns `success=True` when ledger committed |
| `aura_backend/affect/evaluate.py` | F4: fixture history separation; F5: provider init failure → exit nonzero; F6: Arm D uses actual C trajectories; F7: UNSCORABLE fallback |
| `aura_backend/affect/regulation.py` | **NEW** — F8: `EventInterpretation`, `RegulationDecision`, `classify_event`, `regulate` |
| `aura_backend/affect/appraisal.py` | F8: `linguistic_correction_claim`, negation/quote guards, bounded evidence spans |
| `aura_backend/affect/models.py` | F8: `linguistic_correction_claim` added to `STARTER_EVENT_IMPULSES` |
| `aura_backend/affect/service.py` | F8: `regulate()` wired into `compute_provisional_policy` |
| `tests/affect/test_regulation.py` | **NEW** — 14 S03 scenario tests covering all Table 6.1 boundaries |
| `tests/affect/test_evaluate.py` | F7 adversarial fixture, F5 exit-code test (2 new tests) |
| `tests/affect/test_engine.py` | Updated `source_supported_correction` → `linguistic_correction_claim` in assertions |
| `tests/api/test_api_compatibility.py` | Updated to synchronous persist path and `durable_status` receipts |
| `tests/api/test_local_boundary.py` | Same |
| `tests/api/test_provider_compatibility.py` | Same |
| `tests/characterization/test_persistence_contract.py` | Same |

---

## Findings → regression tests map

| Finding | Test | Status |
|---|---|---|
| F1 — background retry removed | `test_failed_database_write_does_not_advance_state` (pre-existing) | PASS |
| F2 — projection fail ≠ ledger fail | `test_projection_failure_does_not_discard_committed_ledger` (new) | PASS |
| F3 — wait_for thread race | Documented; F1 fix reduces exposure | MITIGATED |
| F4 — history contamination | `test_fixture_history_not_contaminated_across_scenarios` (new) | PASS |
| F5 — provider init failure exit | `test_provider_init_failure_exits_nonzero` (new) | PASS |
| F6 — Arm D trajectories | Arm D trajectory pool collects actual C states | PASS |
| F7 — length fallback | `test_task_correctness_does_not_use_length_fallback` (new) | PASS |
| F8 — linguistic cue vs verified | `test_regulation.py` — 14 acceptance scenarios | PASS (14/14) |
| F9 — legacy adapter | **DEFERRED to S05/S06** | OPEN |

---

## Production defaults unchanged

- `ornith-aq1.5:35b` — unchanged baseline model
- `aura-ornith:35b` — Aura profile unchanged (Modelfile.aura not touched)
- No memory expansion, model downloads, or destructive migrations performed

---

## What is NOT done in this batch (per plan)

- S04: Application-level regulation demonstration (route-level `task_facts` wiring)
- S05: Episode memory implementation
- S06: Affective imprinting and recall
- S07: Integrated handoff and validation
- Live comparison run (`--mode compare --provider ollama`) — not run per checkpoint A scope
- F9: `safe_search_conversations` → SQLite retriever — deferred

---

## Remaining open items after Checkpoint A

| Item | Reason | Deferred to |
|---|---|---|
| F9 — legacy memory adapter | Requires read-owner seam not in S01-S03 scope | S05/S06 |
| F3 full mitigation | Cancellation-with-shield for `to_thread` | S04 |
| Gate B (live comparative) | Requires live Ollama + human blinded review | S04 |
| Route-level `task_facts` pass-through | S03 regulation is wired; route currently passes `None` for `task_facts` | S04 |

---

*Checkpoint A complete. Awaiting Ty's review before proceeding to S04.*
