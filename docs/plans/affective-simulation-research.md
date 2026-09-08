# Research brief: advancing Aura's affective simulation

Research checked: 2026-09-07 (America/Denver). Repository baseline: `c3ffddb5f1422fc8467cc4914358e64b36069a99`.

## Recommendation

Build a small, persistent **appraisal → affect → action → outcome → reappraisal** engine around Aura's existing model. Give it fast reactions, slower mood, a stable personality, and bounded control over conversational behavior and memory. This is a simulation-engine improvement, not another emotion-labeling exercise.

The important distinction is causal: state must influence a decision before it happens. A more elaborate description attached to a finished answer does not accomplish this.

Keep the existing Python backend, TypeScript frontend, provider interface, SQLite ledger, and Chroma projection. No new model download, training run, framework, vector database, or cloud service is necessary for the first implementation. This is an engineering recommendation from the research and repository inspection, not a published finding about Aura.

Implementation instructions: [Coding-agent plan](affective-simulation-plan.md).

## What the research contributes

| Primary source | Useful result or design precedent | Decision for Aura; limitation |
|---|---|---|
| [ALMA, DFKI / Gebhard, 2005](https://alma.dfki.de/) | Separates short-lived emotions, medium-term mood, and long-term personality; includes decay and interactions between layers. | Adopt multiple timescales. Do not expand immediately to its full emotion taxonomy or import its older runtime. Plausibility of a virtual character is not evidence of better task performance. |
| [EMA, Marsella & Gratch, 2009, author-uploaded paper](https://www.researchgate.net/publication/222571373_EMA_A_process_model_of_appraisal_dynamics) | Models appraisal, coping, and reappraisal as a loop over the agent's changing interpretation of events. Goal relevance, expectations, and control matter beyond positive/negative wording. | Implement event/goal appraisal and observable recovery. Borrow information-seeking and replanning, not wishful thinking, suppression of evidence, or shifting blame. |
| [FAtiMA Toolkit paper, 2021](https://arxiv.org/abs/2103.03020) and [official implementation](https://github.com/GAIPS/FAtiMA-Toolkit) | Existing socio-emotional agent architecture and authoring tools demonstrate that this need not be invented from scratch. The implementation is C# and Apache-2.0 licensed. | Reuse the architectural separation of appraisal and decision-making. Adding a second runtime is unnecessary for Aura's narrow first slice. Inspect license/attribution before copying any implementation. |
| [Yu & Dayan, 2005](https://pubmed.ncbi.nlm.nih.gov/15944135/) | A computational theory distinguishes expected uncertainty from unexpected contextual change, associated with acetylcholine and norepinephrine respectively. | Separate uncertainty-driven checking from surprise-driven attention switching. This is theoretical inspiration, not justification for mapping a chemical directly to “Happy” or “Angry.” |
| [Generative Agents, Park et al., 2023](https://arxiv.org/abs/2304.03442) | Combines observations, retrieval, planning, and reflection; ablations evaluate their contributions to believable behavior. | Evaluate continuity across event sequences, not isolated responses. Reuse sourced reflection, but preserve Aura's stricter memory provenance and supersession rules. |
| [Third-Person Appraisal Agent, Hong et al., Findings of EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1288/) | Uses appraisal, feedback, and counterfactual refinement for emotional reasoning; includes reinforced fine-tuning. | Supports extracting an event's meaning rather than only its emotion label. It does not establish that several model calls or training will improve Aura's persistent simulation; start with one bounded extraction call. |
| [Anthropic's original research report, April 2026](https://www.anthropic.com/research/emotion-concepts-function) | Reports emotion-related representations with causal effects in Sonnet 4.5. The representations can track the currently relevant character rather than a persistent assistant mood. Some interventions affect reward hacking and other undesirable behavior. | Treat model-internal signals as a serious later research direction, but not as an existing persistent state mechanism in Aura. Measure behavior, preserve a stable competence floor, and do not optimize for desperation or dependence. Transfer to Ornith is untested. |
| [Emotion2Skill, August 2026, v2 preprint](https://arxiv.org/html/2608.09248v2) | Uses residual-stream emotion features to condition skill selection and analyze skill trajectories. Its stated limitations include needing white-box access. | Worth a bounded follow-on experiment after the ordinary controller works. Aura's current provider result does not expose those activations. Reported benchmark gains are author results, not independently reproduced Aura evidence. |

The ALMA/EMA line supplies the small, controllable runtime design. The 2026 work makes an activation-based extension interesting, but does not remove the need to prove that an external controller improves this particular application.

## Alternatives considered

| Approach | Strength | Main cost or weakness | Disposition |
|---|---|---|---|
| Richer persona prompt only | Very cheap; immediately expressive | No explicit temporal dynamics; cannot reliably replay or isolate causes | Required comparison baseline |
| Deterministic appraisal/state engine + existing LLM | Inspectable, local, replayable; independent of provider | Authored rules and parameters need behavioral evaluation | Build first |
| Full FAtiMA integration | Existing engine and authoring ecosystem | C# integration and architecture change beyond current need | Reference, not dependency |
| New trained emotional model | Potentially richer learned responses | Data, training, generalization, and hardware costs before causal baseline exists | Defer |
| Activation probes / steering | Can examine model-internal mechanisms | White-box inference; model-specific calibration; intervention risks | Optional research spike, not first deliverable |
| Spiking-neuron or detailed biochemical simulator | Greater biological detail | No established mapping from that detail to useful Aura behavior | Outside this milestone |

## Current-code evidence

- `aura_backend/main.py::process_conversation` generates the answer before calling `detect_user_emotion`, `detect_aura_emotion`, and `detect_aura_cognitive_focus`.
- `aura_backend/conversation/analysis.py::_SIMULATED_TONE_STATES` maps a tone to one fixed wave label and one named chemical. It is not a dynamical system.
- `AuraStateManager` remains defined in `main.py`, but current initialization assigns `state_manager = None`; its callbacks are not an active simulation loop.
- `aura_backend/providers/base.py::ProviderResult` does not supply residual-stream activations. Text descriptions and reflection summaries are not substitutes for such measurements.
- `aura_backend/storage/retrieval.py` explicitly rejects nonzero salience. Its current neutral retrieval contract must remain available for comparison.
- `aura_backend/storage/schema.py` is version 2 and restricts source events to two message events per turn. New arbitrary “affect events” cannot simply be inserted into that table.
- Existing accepted direction already specifies the core architecture: [decision](../../.planning/research/affective-memory/DECISION.md), [center](../../.planning/research/affective-memory/CENTER.md), [causal loop](../../.planning/research/affective-memory/AFFECT-MEMORY-LOOP.md), and [kill criteria](../../.planning/research/affective-memory/KILL-CRITERIA.md). The new plan makes that direction executable; it does not replace it.

## Evaluation implications

1. **Mechanism:** hold event, task, memories, model, and budget fixed; intervene on prior state. Confirm the intended control changes. A moving graph is insufficient.
2. **Temporal behavior:** test buildup, reappraisal, recovery, restart, and replay. One pleasant answer cannot demonstrate persistent mood.
3. **Useful expression:** compare dynamic state against a strong static persona and a stateless appraiser using blinded sequence ratings. More emotional wording is not automatically better.
4. **Memory contribution:** compare neutral, constant-salience, and dynamic-salience arms at equal derived-memory budgets. Raw source events are retained equally in all arms.
5. **Side effects:** evaluate factual correctness, instruction adherence, flattery, unjustified defensiveness, and pressure on the user alongside liveliness and continuity.

The existing 12-case emotion-label smoke test is not a simulation benchmark. Its recorded sarcasm failure also cautions against turning an uncertain interpretation into a large state change. The extraction layer and the deterministic simulation must have separate tests so that one cannot hide the other's failures.

## Unresolved empirical questions

- Does explicit dynamic state improve the experience over a well-written static Aura persona?
- Do distinct computational channels provide more than a smaller state vector?
- Can the available local model follow control instructions without becoming repetitive or theatrical?
- Does affective memory outperform ordinary relevance and importance at equal storage budgets?
- Would model-internal features add value beyond directly observed uncertainty and task outcomes?

None has been established for Aura by this research pass. The implementation plan specifies the tests and stopping rules. Simulation coefficients, timescales, and score limits proposed there are versioned engineering choices, not measured biological constants.

## Access and scope notes

Sources were checked on the research date above. The 2026 Emotion2Skill source is a preprint. The very large Transformer Circuits full paper could not be fetched by the research browser; the Anthropic row relies on the authors' accessible original research report. The EMA university-hosted PDF was unavailable; its author-uploaded paper was inspected instead. No claimed Aura experiment was run in this planning pass.
