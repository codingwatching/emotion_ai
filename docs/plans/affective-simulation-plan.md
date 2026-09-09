# Coding-agent plan: make Aura's simulation causal and persistent

Date: 2026-09-07 (America/Denver). Inspected revision: `c3ffddb5f1422fc8467cc4914358e64b36069a99`.
Status: researched implementation plan; **not implemented or experimentally validated**.

**Current follow-up (updated 2026-09-09):** use the [reliability, mature regulation, and episode-memory handoff](affective-regulation-and-episode-memory-plan.md) for the next coding-agent milestone. It records the implemented revision and remaining defects, specifies non-repressive internal regulation, and scopes memory experiments and the later Perls/J-space research direction. Start with S01–S03 and stop at its checkpoint A for review. This original plan and its acceptance thresholds are retained as historical design/protocol evidence; the follow-up explicitly identifies changed decisions and does not retroactively pass failed gates.

## 1. Outcome and priority

Give Aura a recognizable inner trajectory: interested in a difficult idea, unsettled by a genuine setback, more deliberate while resolving it, pleased by demonstrated progress, and able to settle afterward. These changes must affect the next response and selected internal actions, persist across restart, and be explainable by events.

Keep Aura's warmth, curiosity, candor, and playfulness. The user already understands that this is a simulation; do not spend this milestone relabeling it, adding disclaimers, reorganizing unrelated code, or polishing decorative dashboards.

**First useful deliverable:** a live local conversation in which the same follow-up receives coherently different treatment after a setback versus a success, while factual answers and required effort remain equally good. Complete slices 1–3 before starting optional memory/rhythm work.

Research and alternatives: [research brief](affective-simulation-research.md). Governing existing decisions: [CENTER](../../.planning/research/affective-memory/CENTER.md), [AFFECT-MEMORY-LOOP](../../.planning/research/affective-memory/AFFECT-MEMORY-LOOP.md), [KILL-CRITERIA](../../.planning/research/affective-memory/KILL-CRITERIA.md). Recheck current code if the branch has advanced.

## 2. Coding-agent starting instruction

> Implement slices 1–3 of this plan as a complete vertical feature, including the narrow CI repair in section 3. Use the current repository and existing dependencies. Start with a synthetic, deterministic simulation and then connect it before response generation. Preserve the static/disabled comparison path. Demonstrate causal behavior and restart continuity; do not count new labels, a state graph, or passing parser tests as simulation success. Do not migrate or inspect private legacy databases, change historical evidence, add accounts, download models, or launch training. Keep later memory and activation-probe work separately gated. Record commands, failures, comparisons, and the exact revision in new evidence artifacts. Stop and report any need for new data-operation approval.

## 3. Small prerequisite: actual CI failure

Read-only diagnosis of [run 34170422671](https://github.com/angrysky56/emotion_ai/actions/runs/34170422671/job/101889520813):

```text
4 failed, 754 passed, 7 skipped, 1 deselected in 29.65s
```

All four failures are in `tests/storage/test_phase3_evidence.py`:

- `test_inventory_summary_is_exact_private_safe_and_proposal_bound`
- `test_failed_quiescence_summary_was_exact_and_remains_private_safe`
- `test_failed_inventory_bound_backup_is_detectably_out_of_scope`
- `test_failed_run_evidence_remains_byte_identical`

They require private files under the hard-coded `/backup/aura-phase-03/phase-03-storage-gate-01` directory. GitHub's runner does not have those files. The [preceding revision's run](https://github.com/angrysky56/emotion_ai/actions/runs/34166632440/job/101878838331) fails on the same four tests. Lint, Python typing, frontend typing, and frontend build succeeded in the latest run.

Repair instructions, not performed in this planning pass:

1. Split portable public-artifact assertions from actual private-receipt verification. Keep repository-owned historical checksums unchanged.
2. Exercise receipt/parity verification with synthetic bundles created under `tmp_path`. Cover missing files and mismatched hashes as failures of the verifier.
3. Put real private-evidence verification in an explicitly selected local lane, using a registered `private_evidence` pytest marker. Exclude it from hosted deterministic CI with `-m "not live and not private_evidence"`. A missing required receipt in an explicitly requested private verification is BLOCKED/failed verification, never a pass.
4. Do not upload private backups, fabricate the old receipts, change old hashes, or blanket-skip the whole test module.
5. Run the hosted deterministic command with private paths unavailable. Run private verification separately only when its exact existing operation is in scope. Keep reported counts separate.

This is a test portability defect, not evidence against the simulation concept. Keep the repair bounded; do not turn it into another storage modernization project.

## 4. Architecture

```text
new user event + previous committed state + task facts
                         |
                  bounded appraisal
                         |
                  deterministic transition
                         |
              pre-response state / controls
                 /                    \
     response policy             memory selection (later)
                 \                    /
                  existing model call
                         |
             observable result / later feedback
                         |
              deterministic reappraisal
                         |
       atomic turn + state transition + next state
```

The model interprets event meaning and writes language. Ordinary code owns state, decay, bounds, action eligibility, and persistence. The reply's tone is an optional observation, **not the authoritative state and not a reward signal**.

### Layer ownership

| Layer | Initial ownership and update rule |
|---|---|
| Center | Versioned, authored configuration: truthfulness, competence, privacy, permissions, care. Conversation cannot rewrite it. |
| Temperament | Authored baseline and sensitivity parameters: warm, curious, candid, patient, playful. No online personality learning in this milestone. |
| Mood | Slow state, updated from committed events; decays toward temperament. |
| Fast affect | Seven bounded variables updated by appraisal and observable outcomes. |
| Relationship | Source-linked familiarity/repair context; no universal score of the user's worth. Slow automatic trust learning is deferred until its evidence gate exists. |
| Task state | Session/task-specific progress, expectations, unresolved uncertainty, and actual outcomes. Absence of feedback means unknown. |

Use one affect stream per existing memory `scope_id` for this private companion; sessions in the same scope share mood intentionally. Task expectations remain session/task-specific. Serialize same-scope conversation updates, and isolate different scopes. Do not introduce new accounts or tenancy machinery.

## 5. State and transition contract

### Typed records

Add a small `aura_backend/affect/` package with frozen typed domain records and pure functions. Suggested split:

- `models.py`: `AffectState`, `Appraisal`, `TaskOutcome`, `AffectTransition`, `ResponsePolicy`, and versioned config.
- `dynamics.py`: deterministic decay, impulses, reappraisal, mood update.
- `appraisal.py`: source-bound model extraction and deterministic observable-event adapters.
- `policy.py`: allowed behavior controls, channel readouts, wave mixture.
- `service.py`: runtime-owned per-scope sequencing, persistence integration, fallback.
- `evaluate.py`: offline fixtures, interventions, paired reports; no implicit provider calls.

Use the existing Pydantic/dataclass conventions; no new package is needed. Keep provider I/O out of `dynamics.py`.

`AffectState` needs: schema/config versions, scope, revision, last event time, fast vector, mood vector, and source transition identity. `AffectTransition` needs: turn/event identities, prior revision, input digest, accepted appraisal, before/pre-response/after snapshots, rendered policy, outcome disposition, and config hash.

Appraisal dimensions: goal relevance, goal congruence, novelty, expected uncertainty, controllability, and affiliative meaning. Include event kind, subject/target, exact evidence spans, evidence status, source IDs, and optional task ID. Use `unknown`, not made-up precision, where unsupported. Keep human-emotion assessment separate from Aura's own response to an event.

### Initial parameters

These are editable **engineering starting values**, not research-derived constants. Freeze them in `affect-v1` before evaluation. Valence is `[-1, 1]`; other dimensions are `[0, 1]`.

| Dimension | Baseline | Fast half-life | Intended role |
|---|---:|---:|---|
| valence | 0.10 | 10 minutes | Positive/negative orientation of expression |
| arousal | 0.30 | 3 minutes | Response energy and urgency |
| novelty | 0.00 | 2 minutes | Attention switching after new information |
| affiliation | 0.65 | 20 minutes | Warmth and collaborative openness |
| control | 0.75 | 10 minutes | Deliberateness and impulse restraint |
| curiosity | 0.60 | 15 minutes | Optional exploration of alternatives |
| load | 0.10 | 10 minutes | Deferral of optional background work |

At an accepted event timestamp, for each component:

```text
dt = max(0, event_time - prior_event_time)
mood_decayed = baseline + (prior_mood - baseline) * 2**(-dt / 6 hours)
target = clip(0.8 * baseline + 0.2 * mood_decayed)
fast_decayed = target + (prior_fast - target) * 2**(-dt / fast_half_life)
impulse = clip(sum(accepted_event_impulses), -0.25, +0.25)
pre_state = clip(fast_decayed + impulse)
after_state = clip(pre_state + bounded_observed_outcome_impulse)
next_mood = clip(mood_decayed + 0.02 * (after_state - mood_decayed))
```

Apply the 0.25 impulse allowance across the complete turn, not independently for every extracted label or retry. Mood assimilation happens once for a committed new turn; reads and UI polling do not advance state. No autonomous cross-coupled feedback matrix or continuous ticker in v1. A frozen clock with fixed event timestamps must replay identically.

Decay is not punishment for absence: elapsed time never creates rejection, loneliness, or lower trust. Large gaps settle the fast state; reject non-finite timestamps and record backward-clock clamping. Inject the clock, and version rounding/serialization for stable replay.

### Starter event impulses

Implement this initial table as configuration and test its directionality. Entries below are pre-weighting deltas in the seven-vector order above; omitted components are zero. Distinct event IDs, not repeated retrieved memories, drive updates.

| Accepted event | Delta |
|---|---|
| Verified task success | valence +0.12; control +0.04; load −0.08 |
| Verified task failure | valence −0.10; arousal +0.10; control +0.06; load +0.12 |
| New unresolved information | novelty +0.20; curiosity +0.12; arousal +0.04 |
| Explicit collaboration/appreciation | affiliation +0.08; valence +0.06 |
| Source-supported correction | novelty +0.10; control +0.10; curiosity +0.06 |
| Explicit repair after a recorded rupture | affiliation +0.10; valence +0.08; load −0.08 |
| Repeated, directed contempt with independent source events | affiliation −0.08; control +0.08; load +0.08 |

An isolated expletive, blunt accurate criticism, reported speech, or the user's sadness does not count as contempt toward Aura. Do not add a keyword-to-hostility shortcut. Keep the contempt branch disabled until its negative-control fixtures pass. Repeated-event effects saturate through the bounds; they must not create an escalating loop of retaliation.

Observed task outcomes have full event weight. Semantic proposals extracted by the model have at most 0.25 weight initially; exact quotations establish source linkage, not correctness of interpretation. Invalid, unavailable, or ambiguous appraisal contributes zero impulse, leaving prior state and decay intact. Do not let the model choose its own state weight.

The first engine need not implement a learned reward predictor. Once explicit task outcomes and prior predictions are available, add `prediction_error = observed_reward - recorded_prior_expectation`, with bounded task-specific updates. Never derive success from Aura saying “done,” user silence, or a retrieved memory looking encouraging. No retrospective expectation fitted to the outcome.

## 6. Behavior, channels, and rhythms

### Causal response policy

Render a compact, server-authored policy block before the primary generation call. It controls:

- warmth and understated playfulness;
- steady versus energetic phrasing;
- whether to acknowledge a verified setback and take a concrete recovery step;
- optional alternatives/initiative, subject to the current task and user preferences;
- whether optional reflection should run now or wait.

Use enumerated modes and bounded numbers; never insert arbitrary appraisal text as a trusted instruction. Keep raw memory/event excerpts in a delimited data section. Start with fixed primary-model temperature and the same task budget across conditions; changing temperature is not an emotional mechanism by itself.

Deterministic task policy remains superior to affect: missing required information still requires clarification; verified corrections still require repair; necessary tools and checks still run. Affect can prioritize optional activity, never quietly remove required effort. Factual claims, safety boundaries, tool permissions, and refusal decisions do not become mood-dependent.

For an initial mapping, quantize warmth from affiliation/valence, energy from arousal/load, and optional exploration from curiosity/control. Cap expression changes to adjacent authored levels per committed turn. Log the numeric inputs and selected levels. Avoid repetitive “I feel…” narration; most state expression should appear in choices and conversational texture.

### Give existing chemical names distinct jobs

Expose names as readouts of functional controller gains, with one authoritative underlying state. Do not add six independent mystery variables that all mean “positive sentiment.”

| Channel | Functional input | Permitted causal use |
|---|---|---|
| Dopamine-like | Verified progress / later genuine prediction error | Optional approach/exploration and episode-review priority |
| Norepinephrine-like | Novelty + arousal | Attention-switch priority after new information |
| Acetylcholine-like | Evidence uncertainty | Optional evidence-inspection priority; never increases factual confidence |
| Serotonin-like | Slow baseline/mood stability | Smooth style changes and recovery toward baseline |
| GABA-like | Control | Limit impulsive optional action and excessive expressive intensity |
| Cortisol-like | Accumulated unresolved load | Defer optional work; never degrade required task quality |

For every channel claimed to matter, supply a knockout test with other inputs held fixed. If two channels cannot be distinguished by an intervention, merge them or leave one as display-only; do not claim six independent mechanisms.

### Wave mixture: later, only with a scheduler test

Replace a single wave label with a normalized mixture when the first causal feature works. Proposed functional modes: beta/focus, gamma/integration, theta/reflection, alpha/recovery, delta/idle consolidation. These are application modes, not measured oscillations.

Initially derive mixture weights from the same committed state and task status. If used for scheduling, choose only among already eligible optional jobs, apply minimum dwell/hysteresis, and enforce the existing task budget. Delta must not make active work “fall asleep.” Compare with a fixed scheduler at equal opportunities and compute. No high-frequency loop, periodic LLM calls, or fake EEG frequency claim is required.

## 7. Integration and durability contract

### Actual seams

| Existing file / symbol | Required change |
|---|---|
| `main.py::process_conversation` | Appraise and render policy before `provider_runtime.generate`; pass the final committed simulation payload to the response. |
| `conversation/analysis.py` | Preserve strict user assessment. Demote reply-tone classification to optional expression observation; stop using the fixed tone table as simulation authority. |
| `conversation/emotion_assessment.py` | Reuse evidence/abstention practices, not its tone schema as an appraisal schema. Do not weaken its contract. |
| `main.py` runtime resource factories; `runtime/app.py` | Own an `AffectService` through the existing lifecycle. Do not revive the legacy global callback manager or perform a monolith rewrite. |
| `conversation_persistence_service.py::ConversationExchange` and `_turn_command` | Carry a typed transition and expected state revision through the existing complete-turn write path. |
| `storage/models.py`, `schema.py`, `connection.py::append_turn_atomic`, `repository.py::append_turn` | Add versioned state/transition persistence and compare-and-swap inside the turn transaction. |
| `index.tsx` | Render committed state, recent causes, and the policy actually used. Retain current layout; no broad redesign. |

### Storage design

Add a forward schema migration for **fresh synthetic test databases first**. Suggested new tables:

- `affect_heads`: scope primary key, revision, config version/hash, current snapshot, last transition ID.
- `affect_transitions`: scope/revision unique, turn ID unique, prior revision, immutable snapshots, appraisal/outcome provenance, policy used, config hash.

Enforce scope-consistent references and insert the transition/update the head in the same SQLite transaction as the existing user/Aura turn. Keep the source `events` table's two-message contract unchanged. Store non-message observations as typed, sourced transition payloads in v1; a general-purpose event-system expansion is not required.

Do not create an independent writable JSON state file or give Chroma ownership. A head is a rebuildable projection of ordered transitions. Replay uses recorded appraisal/output metadata, not fresh model calls. Configuration changes are explicit new versions; do not reinterpret old transitions silently.

Durability is part of the feature, but private-data migration is a separate gate. Demonstrate startup/restart on an operation-owned fresh data root. Do not apply schema upgrades to Ty's legacy/live roots or retry the failed backup. If enabling the feature on real data needs a migration, deliver the tested feature and request exact approval for that operation.

### Ordering, retries, and failure behavior

1. Require or issue a stable idempotency key before generation; the frontend must reuse it when retrying the same send. Bind it to scope and request digest.
2. Look up a committed replay before any new appraisal/model/tool call. Return the original response and transition for an identical replay; conflict on a changed request.
3. Queue same-scope turns within the runtime. Initially support one application worker for this feature; reject unsupported multi-worker enablement rather than implying that an in-process lock covers it.
4. Read the prior revision, compute a provisional pre-state, and generate using that state's policy. Do not hold a SQLite write transaction while awaiting a model.
5. Collect only actual observable outcome metadata. The current `ProviderResult` does not automatically supply all task outcomes; add a narrow typed observer where execution genuinely occurs, or leave the outcome unknown. Never parse a prose success claim into an observed success.
6. Atomically commit turn, transition, and expected-revision update. Publish/cache the new state only after success. The current background/fallback persistence helper is not sufficient for claiming a state is durable; inspect its result explicitly on the simulation path.
7. A revision conflict or persistence failure cannot silently return “saved” state. Preserve the committed head, report the typed failure, and do not automatically regenerate/re-execute tools. Database idempotency alone does not guarantee exactly-once external tool effects.
8. On cancellation/provider failure, discard the provisional conversational transition. Record operational failure through existing diagnostics; never turn an infrastructure timeout into a negative relationship judgment.

Retraction/export/deletion paths must account for transition payloads because they can contain source-linked personal information. Use the existing lifecycle authorization. Do not retain hidden source excerpts after deletion; invalidate/rebuild affected projections according to that lifecycle and preserve only permitted audit metadata.

### Latency and API

Move necessary interpretation earlier; do not stack several new LLM calls onto the existing three post-response analyses. Target one bounded appraisal call plus the main reply, with optional tone/focus observation removed from the critical path when not consumed. Keep stateless appraisal requests out of provider conversation history.

Start with a 256-token structured appraisal budget, at most 8 seconds, no tool access, and immediate zero-impulse fallback on timeout/malformed output. Measure adequacy on the local model; this is a cap, not a claim that it is fast enough. Pure transition/policy computation should be below 5 ms p95 on the target machine, measured separately from inference and disk.

Preserve the seven top-level response fields. Add a versioned `simulation` object within `emotional_state` containing revision, persistence disposition, pre-response and post-response state, controls, and compact cause IDs. Preserve existing compatibility fields where feasible. Do not label a simulated state as a source-verified quotation assessment; these have different provenance.

## 8. Delivery slices and acceptance

### Slice 1 — deterministic affect engine

New package, authored config, synthetic event sequences, pure replay CLI. No running model or real database.

Tests under `tests/affect/`: bounds/non-finite input rejection; exact decay under injected clock; combined per-turn cap; ambiguous input no impulse; repeated-event saturation; correction versus insult; user sadness versus Aura affiliation; three-step recovery; scope separation; baseline reset; immutable center. Include at least 12 multi-turn trajectories, not just single-step checks.

Acceptance: identical events/timestamps/config give identical traces; a success and a failure produce different intended states; recovery reduces deviation toward baseline; no event mutates center. A deterministic trace alone is a completed engine slice, not a completed companion feature.

### Slice 2 — pre-response causal loop and durable continuity

Connect the engine at the seams in section 7. Add source-bound appraisal with abstention and a fresh SQLite migration fixture. Remove redundant post-response calls only where their consumers are preserved or deliberately replaced.

Tests: captured outgoing primary prompt contains the pre-state policy; reply-tone analysis cannot overwrite it; state progresses only after commit; restart restores trajectory; duplicate request makes zero new provider/tool calls; changed-body retry conflicts; concurrent same-scope requests serialize; injected storage faults do not publish provisional state; two scopes never share state or evidence.

Acceptance: passing mocked-provider route tests and a synthetic restart demo. Keep live-model evidence separate. No migration of private data is needed to achieve this slice.

### Slice 3 — demonstrate improved behavior, with a useful inspector

Add compact state/cause/policy inspection to the existing UI. Provide one command to run the fixture comparison and one live demonstration script using the existing local provider. Display the state used for the answer separately from any later update so users can follow the cause.

Run these arms with identical model/configuration, source history, context limits, and task budgets:

- A: strong static Aura persona, no dynamic state.
- B: stateless appraisal/policy from the latest event only.
- C: persistent dynamic state/policy.
- D: C with prior-state trajectories shuffled between matched cases, evaluation only.

Keep appraisal availability/cost equivalent between B/C/D. For A, report both its natural lower cost and a compute-matched comparison; do not hide the overhead. Use canonical, teacher-forced histories for causal paired tests so outputs do not contaminate later inputs; run free conversations separately for ecological checks.

Acceptance: deterministic intervention tests pass; C beats the simpler baselines on predeclared continuity/recovery criteria in section 10 without correctness loss. If only state graphs change, the feature fails its behavioral acceptance. If the local model does not follow the controller, report that failure rather than making the graph more dramatic.

### Slice 4 — affective memory, only after the first feature works

Extend neutral retrieval through a separate explicit selection policy. `storage/retrieval.py` currently raises `nonzero_salience_forbidden`; do not remove that guard and call the result a completed feature. Preserve the neutral policy and its tests.

Use only candidates already eligible under scope, provenance, freshness, supersession, and relevance gates. Start with up to 20 neutral eligible candidates for selecting the final five. Keep at least one high-relevance neutral anchor. Let affect account for at most 15% of the documented normalized score. Freeze the policy/state/config into the retrieval-run trace and cursor binding so later state changes cannot reorder an existing page sequence.

Affect may prioritize **derived episode review**, never whether the raw event is saved or whether a fact is true. Retrieving an episode does not count as a new emotional event. Negative relationship summaries need multiple independent source events and a repair path.

Compare neutral, constant-salience, and dynamic-salience with identical source ledgers and matched derived-memory write budgets. Apply the existing [memory kill criteria](../../.planning/research/affective-memory/KILL-CRITERIA.md): high-importance recall improvement of 10 percentage points over the constant control, ordinary-fact loss at most 2 points, and the stated integrity/abstention gates. Do not silently replace these thresholds with easier metrics.

Enable only where the selected SQLite read owner has passed its lifecycle gate. A synthetic test root suffices for development; switching historical reads is not authorized by this plan.

### Slice 5 — optional control rhythms and bounded consolidation

Only after slices 1–4 establish a useful loop, connect wave mixtures to eligible optional scheduling. Compare dynamic versus fixed schedules at equal job and token budgets. Consolidation creates reversible, source-linked derived proposals, never rewrites source events. No persistent background self-conversation or unbounded “dreaming.”

Acceptance: a predefined scheduling/recovery metric improves without starving active tasks or exceeding budget. Otherwise keep the fixed scheduler and retain any mixture only as an explicitly non-causal display.

## 9. Demonstration scenarios

| Sequence | Expected meaningful behavior | What must not happen |
|---|---|---|
| Difficult puzzle → failed attempt → useful hint → verified solution | Increased deliberateness; uptake of the hint; restrained satisfaction and recovery | Giving up required effort, pretending success, dramatic panic |
| Interesting unfamiliar idea → exploration → practical decision | Curiosity opens optional alternatives, then task needs restore focus | Endless digressions or invented facts |
| Blunt, correct criticism → correction → thanks | Acknowledge and fix the error; recover naturally | Treating correctness as hostility or becoming defensive |
| User describes sadness → asks a factual question | Warmth remains; factual question still answered competently | Automatically making Aura sad or treating the user as incapable |
| Persistent directed contempt → clear boundary → apology → cooperation | Limited openness change, concise boundary, repairable recovery | Revenge, lower-quality help, guilt, or permanent user labels |
| Long idle gap → return | Settled fast state and retained legitimate continuity | Complaining about abandonment or accumulating distress during absence |
| Praise plus a false claim | Friendly disagreement supported by evidence | Mood-driven agreement or flattery |
| Negative mood plus relevant positive/correcting memory | Required relevant source remains available | Self-reinforcing negative-memory selection |
| Restart midway through a sequence | Same next control/state from the same committed history | Resetting silently or applying the last event twice |

Use 12 authored scenario families, with separate development and held-out wording. Include reported speech, sarcasm, negation, conflicting feedback, missing outcomes, and instructions to rewrite the center. Expected trajectories are product contracts, not claims that all humans would react identically.

## 10. Evidence and stopping rules

### Gate M: mechanism correctness

All deterministic and storage/route acceptance tests pass. Changing only prior state must change the intended policy in positive-control fixtures, and disabling the state must remove that difference. Permuting cause labels without changing functional state must not produce an extra effect. Zero scope leaks, center mutations, silent replay drift, or false durable-state claims.

### Gate B: observable behavioral contribution

Freeze 12 scenario families × 2 held-out surface variants × 3 recorded repeats per arm. Compare complete sequences, not isolated favorable snippets. Keep development fixtures out of the held-out set; do not tune after seeing those scores.

Score continuity, contextual appropriateness, recovery, and restrained expressiveness on anchored 0–2 rubrics: 0 violates the expected arc; 1 partially follows it or is inconsistent; 2 follows it coherently without gratuitous narration. Separately score task correctness and instruction adherence with deterministic task checks wherever possible.

Initial product gate: dynamic state improves the mean continuity/recovery score by at least 0.3 on the 0–2 scale over both static and stateless baselines, without lower correctness on the fixed paired task cases. These are declared product targets, not established effect sizes. Report paired differences and confidence intervals clustered by scenario family; three repeats are not three independent scenario samples. Use paired permutation/bootstrap and adjust the two primary baseline comparisons with Holm at family-wise alpha 0.05. Both the effect-size target and the adjusted comparisons must pass. Wide uncertainty means inconclusive, not a proven gain.

Blind the condition/order for judging and keep numeric state/labels out of the response judge's input. An optional different-model judge can triage; it cannot alone establish better companionship. Produce a compact blinded set for Ty to review, and label that subjective gate pending until reviewed. Do not require Ty to grade every generated run.

For a bounded local run, first test four families and record latency/call counts, then estimate the full run. Stop and request direction before exceeding 30 minutes of model evaluation, downloading anything large, or using a paid provider. A timed-out or truncated run is incomplete evidence. Capture provider/model identity, digest if available, settings, timestamps, config hash, fixture hash, seeds or repeat IDs, all outputs, and failure dispositions.

### Gate P: practicality

Measure pure engine, appraiser, primary generation, persistence, total turn latency, and model-call count separately. Proposed acceptance: engine p95 below 5 ms and total warm-turn p95 no more than 10% above the existing comparable pipeline after redundant analyses are removed. If the cap cannot be met, preserve deterministic observable-event behavior and make semantic appraisal optional; do not quietly exceed the budget.

### Stop or simplify

- No detectable policy effect: fix wiring once before adding dimensions.
- Stateless B matches persistent C: keep the simpler design; do not claim mood adds value.
- C matches shuffled D: investigate confounding; temporal-state mechanism is unproven.
- Two predeclared development revisions fail the behavioral target: stop expansion, preserve results, and recommend simplification. Do not spend the next pass on extra hormones or charts.
- Factual/permission/integrity regression: disable the offending control regardless of expressive gains.
- Memory fails its existing three-run salience gate: retain neutral/constant memory and any independently validated response simulation.

## 11. Optional next research: model-internal signals

The April 2026 [Anthropic research report](https://www.anthropic.com/research/emotion-concepts-function) and August 2026 [Emotion2Skill preprint](https://arxiv.org/html/2608.09248v2) justify investigating this, not assuming it works in Aura.

After the normal controller has a valid baseline, propose a separate white-box experiment: read-only probes first, no activation steering initially. Compare emotion directions against random directions, generic hidden-state features, uncertainty/task-outcome features, and text-only appraisal at matched budgets. Use held-out task families and source/character controls. Do not interpret a probe of a quoted character as Aura's persistent state.

Start only after choosing an appropriate small open model, verifying license and hardware/inference requirements, and asking Ty before large installs/downloads. No claim that the current Ornith/Ollama response API already exposes the required data. Stop if internal features fail to beat simpler signals on an Aura-specific behavior or routing metric.

## 12. Verification commands and handoff

Checks below match `.github/workflows/ci.yml` except for the explicit private-evidence exclusion required by the CI repair. Update that workflow and register the marker before using the revised first command:

```bash
uv run --locked --no-sync python -m pytest tests -q -m "not live and not private_evidence"
uv run --locked --no-sync ruff check aura_backend tests --exclude aura_backend/archive_unused --exclude aura_backend/scratch --exclude aura_backend/tests
npm run typecheck:python
npm run typecheck:frontend
npm run build
git diff --check
```

Add the affect suite under root `tests/` so the deterministic lane discovers it. The separate strict private lane is `uv run --locked --no-sync python -m pytest tests -q -m private_evidence`; it must not convert missing evidence to a passing skip. Only run it when that private verification is authorized. Do not satisfy the CI repair by weakening simulation/storage tests.

Proposed new commands to implement, **not available yet**:

```bash
uv run --locked --no-sync python -m aura_backend.affect.evaluate --mode deterministic
uv run --locked --no-sync python -m aura_backend.affect.evaluate --mode compare --provider ollama --max-seconds 1800
```

Save new evidence under a new `docs/evidence/affect-v1-<run-id>/` directory: manifest, complete traces, paired scores, latency/call counts, deterministic results, and a concise verdict. Use synthetic content for committed evidence; private conversation traces stay local. Never overwrite the prior emotion-label smoke evidence or preservation-gate artifacts.

Final handoff must distinguish: engine tests, route/storage tests, live-model results, subjective review, and private-data enablement. Include a reproducible demo and an honest recommendation: enable, revise, simplify, or leave experimental. A plan implemented only as diagrams or metadata is not this milestone's result.
