# Checkpoint A repair evidence — 2026-09-10

## Disposition

**Critical integrity repairs verified; Checkpoint A remains OPEN.**
This report supersedes the completion claims in the previous walkthrough, not its historical artifacts.
It does not approve S04, demonstrate behavioral superiority, or claim the whole S01–S03 plan is complete.

Base commit: `1631f94d562304de10517c16a07a0ef6f12b2efa`.
Implementation is an uncommitted working-tree change; nothing was pushed.
Tracked diff SHA256 (before this new report): `f87f319db0754c7b6482fafad82c40df39506c771ec9a1d2984c9dadbf02c998`.
That digest covers tracked changes only, not the new receipts module or frontend test.
Fresh temporary SQLite roots were used. No personal database migration, live model comparison, training, or model installation was performed.

## Verified repairs

### Durable ownership and request identity

- The persistence service retains and shields the exact writer task. Timeout/disconnect does not release same-scope sequencing while that writer is unresolved. Other scopes remain usable.
- A known rollback permits one retry of the same constructed command, without regenerating the model response. Unknown outcomes are not asserted to be rolled back.
- Typed receipts distinguish committed, replayed, rejected, pending, unknown, and ephemeral states. Successful task outcomes are not durability receipts.
- Stored receipts verify exact transition identity and revision. SQLite reconciles stale cache heads, including equal-revision identity mismatches.
- Shutdown drains retained writers. Unexpected worker failures become unknown receipts rather than unowned task exceptions.
- User event and turn identifiers now match the stored appraisal provenance.
- Frontend transport/manual retries of the same request object retain both idempotency key and session. New request objects get new identities; conflicts are not transport-retried.

Production-route/service regression tests in `tests/affect/test_durability_and_loop.py` include:

- `test_pending_writer_blocks_same_scope_before_provider`
- `test_cancelled_waiter_retains_exact_writer` (both service entry points)
- `test_http_disconnect_pending_replay_and_independent_scope`
- `test_rolled_back_write_retries_exact_command_without_regeneration`
- `test_exhausted_rollback_retry_reports_rejected`
- `test_unexpected_writer_failure_is_unknown_not_an_unowned_task`
- `test_five_overlapping_http_turns_commit_once_each`
- `test_projection_failure_does_not_discard_committed_ledger`
- `test_no_storage_is_explicitly_ephemeral`

The five-request test exercises the HTTP endpoint, retains revisions 1–5, verifies five distinct transition identities, and replays all requests without additional provider calls. It supplements rather than substitutes for the cancellation/barrier tests.

### Restart clock and numerical state

`test_restart_preserves_the_transition_clock` reproduced a mismatch between the engine's pre-generation event time and storage's post-generation exchange time. The route now persists the same event time (within datetime microsecond precision).

Extending this test also reproduced loss of numerical state through four-decimal display serialization. SQLite now stores full-precision dataclass vectors, while UI-oriented rounding remains separate. Reloaded fast and mood vectors match the published vectors exactly.

This repairs newly written records; it cannot recover precision already lost in historical rows.

### Evidence-grounded regulation

- Ordinary disrespect produces no persistent grievance impulse by default; the optional experimental branch is not silently enabled.
- Counterfactual regulation removes grievance contributions without dropping legitimate simultaneous observations.
- Mixed criticism/correction/task observations retain separate interpretations.
- Unsupported corrections remain claims and select a verification policy; an old negative state alone no longer invents a current setback.
- Public request fields cannot certify task success/failure. Tests cover boolean and misleading string values.
- Appraisal source IDs resolve to the actual committed user event; bounded matching spans and source SHA256 are recorded.
- Regulation records include `grounded-regulation-v2`, accepted/discarded meanings, and action.
- The one-outcome-per-turn compatibility seam avoids applying the same pre-appraised outcome again after response. Whole-turn impulses, including post-response outcomes, remain bounded.
- Nonboolean task outcomes are rejected.

Important limit: selecting a verification instruction is not the same as executing a trusted fact-check. The production observer connection remains outstanding below.

### Evaluation integrity and honest quarantine

- All arms receive the same frozen fixture history; generated replies never become later matched-comparison inputs.
- D uses precomputed whole compatible prior trajectories with cumulative clocks, preserved mood, explicit donor identity, and identical current task evidence. Identity donors are disclosed.
- Seeded arm order, explicit provider warmup, cumulative deadlines, input bounds, and complete unique trace pairing are checked.
- Paired statistics use exact sign-flip enumeration for small samples and SciPy's permutation implementation; invalid/nonfinite pairs are rejected. The original effect threshold and Holm comparisons are not relaxed.
- Blinded packets omit condition/state/automatic scores and include conversation history.
- In-memory state commit timing is no longer called SQLite persistence. Full-route latency is NOT_RUN.
- **Gate B is deliberately quarantined as FAIL with semantic validation NOT_VALIDATED.** Heuristic thresholds remain diagnostics. An adversarial test makes C's keyword scores meet the old thresholds, then confirms this cannot certify behavior.

This is a containment repair, not a completed semantic evaluator. Historical live FAIL results remain unchanged.

## Fresh verification after the final code edits

| Surface | Result |
| --- | --- |
| Full deterministic/nonprivate regression lane | **863 passed, 7 skipped, 5 deselected**, 35.35 s |
| Affect suite | **105 passed**, 4.19 s |
| Ruff | All checks passed |
| Pyright | 0 errors, 0 warnings |
| Frontend retry test | 1 passed |
| Frontend TypeScript | exit 0 |
| Vite production build | exit 0, 77 ms |
| git diff --check | exit 0 |
| Deterministic mechanism + pure-engine benchmark | PASS; engine p95 0.0584 ms |
| Live behavioral comparison / full-route performance | NOT_RUN |

Commands and tool output are retained in `verification.txt`.
Deterministic artifacts are in `deterministic/`.
Pure-engine timing is not evidence for SQLite, provider, or total-route latency.

## Outstanding acceptance work — do not silently skip

1. **Versioned old-head cutover:** configuration still identifies the original affect-v1 coefficients; the regulator has a separate version marker, but a tested explicit old-head conversion/reset protocol is not implemented. Storage also retains the legacy config-version field. Do not deploy as an approved v1-state migration. Author/test the conversion on a fresh root, preserving original evidence, before proposing a live cutover.
2. **Trusted runtime observations:** request-supplied facts are now correctly ignored. Actual server-owned task/check outcomes must be connected with source identity and validation. Resolve contradictory success/failure observations explicitly; the current compatibility seam is one outcome per turn, not a general fact ledger.
3. **Independent semantic evaluation:** author substantive per-turn fixture replies and expected facts/actions, including negative controls. The current frozen fallback “Understood.” prevents cross-arm contamination but is not an adequate full fixture corpus. Add independently validated semantic ratings/verdict inputs before allowing Gate B acceptance.
4. **Protocol completeness:** replace/disclose first-N family selection explicitly, improve actual provider/model configuration provenance, and treat identity-only D strata as noninformative controls. Measure the real route separately when S04 is authorized.
5. **Consequence continuity:** event-aware wording is not yet a full unresolved-task or episode-memory implementation. Do not infer that a calm reply resolved a consequence. S05/S06 memory work remains deferred.

## Implementation references

- `aura_backend/affect/receipts.py`
- `aura_backend/conversation_persistence_service.py`
- `aura_backend/main.py`
- `aura_backend/storage/connection.py`, `repository.py`
- `aura_backend/affect/service.py`, `regulation.py`, `policy.py`, `appraisal.py`
- `aura_backend/affect/evaluate.py`
- `src/services/auraApi.ts`
- `tests/affect/`, `tests/frontend/aura-retry.test.mjs`

Technical references consulted: [Python task cancellation and shielding](https://docs.python.org/3/library/asyncio-task.html), [SciPy paired permutation testing](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html), [browser UUID generation](https://developer.mozilla.org/en-US/docs/Web/API/Crypto/randomUUID).

