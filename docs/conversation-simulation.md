# Conversation-driven simulation

Updated 2026-09-15.

The stuck indicators had two independent causes. The controller recognized only
short exact phrases and never consumed the separate emotion assessments. The
header then used an analysis of Aura's reply style as its emotion, so supportive
prose could appear calm while the user's message expressed something very
different. Failed reply analyses produced Unknown. The previous display repair
showed saved values correctly but did not test a changing conversation.

## Current path

1. The selected conversation provider proposes up to two conversational events:
   exploration, celebration, distress, affection, settling, or overload.
2. The application validates the complete structured response and every exact
   source quote. Repeated evidence for one kind counts that event once. The
   model cannot set channel levels, authorize task success, or create grievances.
3. Authored, capped impulses change the seven-dimensional controller state.
   Elapsed time decays the previous state toward its baseline. The resulting
   policy is included in the outgoing conversation prompt.
4. State and its appraisal provenance commit atomically with the conversation.
   Failed persistence retains the prior published state. Retries replay the
   committed turn without another appraisal or generation.
5. The header's **Simulated Emotion**, activation band, and chemical channels
   all derive from that committed state. Reload restores the same readouts.

The header shows activation as a percentage as well as a band. Alpha covers an
interval, so small state changes need not change its name. The largest chemical
channel can also stay the same while other channels change; the simulation panel
now displays all six values. GABA-like control may remain stable in ordinary
conversation because the gain table does not change every dimension on every turn.

The provider contract now accepts an immutable optional output schema. Ollama
and OpenRouter translate it to `response_format`; Gemini translates it to JSON
output configuration. The existing user/reply emotion classifier keeps prompt
JSON with strict parsing: enforcing its nullable schema caused excessive
abstention in the local model and that change was rejected. This follows the provider's
[structured-output interface](https://docs.ollama.com/capabilities/structured-outputs).
No provider switch, new model, or additional package is required.

## Failure behavior and limits

An unavailable or invalid appraisal supplies no semantic impulse; exact-phrase
fallbacks remain available and the UI reports analysis failure. The new analysis
is one tool-free call per new turn, capped at 20 seconds and 384 output tokens.
It uses no persistent provider session. It can add latency to a turn.

These are authored software dynamics, not a physiological model or evidence of
subjective feelings. Source quotes establish attribution, not correct meaning;
sarcasm, ambiguity, and cultural context remain interpretation risks. A stateful
simulation also carries earlier tendencies forward, so its label need not mirror
the latest sentence. Nothing adds random movement to make meters look active.

The API's legacy `emotional_state.name` and `assessment` retain the optional
reply-style characterization for stored-analysis compatibility. The UI reads
`emotional_state.simulation.display.emotion`. User-emotion assessments remain
separate and never receive biological labels. Existing historical analyses are
not rewritten.

## Verification

Deterministic coverage includes meaningful state changes before generation,
source and schema rejection, duplicate-event handling, non-punitive distress,
provider failure/cancellation, atomic publication, actual HTTP replay and
restoration, and visible changes within an unchanged band/dominant channel.

Run the full deterministic suite with:

```sh
MEMVID_TELEMETRY=0 uv run --locked --no-sync python -m pytest tests -q -m 'not live and not private_evidence'
npm run test:frontend
npm run typecheck:frontend
npm run typecheck:python
```

Live evidence is recorded separately from deterministic tests. Small synthetic
checks establish those observed cases, not general emotion-recognition accuracy.

Recorded checks on the selected local `aura-ornith:35b` model:

- [Four actual browser chat turns](evidence/conversation-simulation-2026-09-15.json)
  showed Curious/Alpha 40%, Excited/Beta 56%, Concerned/Beta 68%, and
  Warm/Alpha 49% after relief. All four sets of chemical values differed; reload
  preserved them. The distress-policy pacing issue found in that run was
  subsequently corrected and checked separately.
- The final [live support-policy check](evidence/conversation-support-2026-09-15.json)
  produced Concerned/Alpha 44%, a steady and focused response policy, and
  changed channel values. Reload and reduced-motion rendering passed.
- The existing 12-case [emotion-classifier check](evidence/emotion-classifier-2026-09-15.json)
  passed 11 cases, including all five explicit feelings and mixed-emotion
  abstention. Ambiguous sarcasm was misclassified as Sad. Its strict all-case
  acceptance gate remains failed.
- The [rejected classifier-format experiment](evidence/emotion-structured-output-rejected-2026-09-15.json)
  scored only 6/12. That configuration is not used by the emotion classifier.
- Deterministic verification: 924 passed, 5 skipped, 5 deselected; six frontend
  tests passed, plus Python/TypeScript checking, Ruff, and the production build.
