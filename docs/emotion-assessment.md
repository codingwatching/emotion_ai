# Emotion assessment: implementation and evidence

Updated 2026-09-07. This is a focused improvement to the active conversation path,
independent of the remaining Phase 3 historical-data operations. It does not
complete Phase 3 or implement Phase 4's time-evolving affective controller.

## Scorecard

| Surface | Observed result |
| --- | --- |
| Original deterministic backend baseline | 692 passed; 2 skips; 1 live deselection |
| Updated deterministic backend suite | 763 passed; 2 skips; 1 live deselection |
| Python and frontend type checks | Both pass; Python reports 0 errors |
| Ruff and frontend production build | Both pass |
| Local Ornith synthetic evaluation | 12/12 attempted, 11/12 matched expectations |
| Explicit-emotion controls | 5/5 matched allowed labels |
| Abstention controls | 6/7; one false-positive label on ambiguous sarcasm |
| Strict live acceptance gate | **Failed**; requires every case to match |
| Browser interaction | Unverified; browser discovery returned no connected browser |
| Historical stores, backups, migration | No operations performed |

The [recorded evaluation](evidence/2026-09-07-emotion-smoke.json) preserves the
observed application-level results, including the failure. The corpus and
expected outcomes were defined before the first live call. After repairing the
transport, the prompt and expectations were not tuned to eliminate the failed
case. This small English corpus is a smoke check, not a calibrated accuracy
estimate, a cultural or clinical validation, or a comparative quality benchmark.

## What changed

Previously the emotion parser accepted free-form `Happy (Medium)` strings and
converted unsupported output into `Normal (Medium)`. That made a parsing failure
look like a positive observation of neutrality. It also assigned fixed brainwave
and chemical names to the user's inferred emotion.

The new `emotion-assessment-v1` contract requires a complete, strictly validated
JSON object. An emotion needs an allowed name, a valid intensity, and one to three
distinct quotations that occur verbatim in the analyzed text. The application
adds the source SHA-256, subject, schema version, and epistemic status. Unknown
keys, duplicate keys, contradictory fields, invented quotations, excessive output,
and non-JSON prose are rejected. There is no silent response repair or invented
confidence percentage.

| Status | Meaning |
| --- | --- |
| `inferred` | A source-checked, tentative user-emotion interpretation |
| `simulated` | A source-checked label for Aura's own response tone |
| `abstained` | No single supported label, or the input could not be assessed |
| `invalid` | Output did not satisfy the response or source contract |
| `unavailable` | Provider error, resource limit, or timeout |
| `unverified` | Compatibility payload for a legacy DTO with no assessment metadata |

For abstention, invalidity, and unavailability, the displayed/stored name and
intensity are `Unknown`, not `Normal`/`Medium`. User assessments contain empty
biological fields. Aura's optional fixed indicator mappings are explicitly
simulated. Its analysis sees only its visible reply, preventing the validator
from accepting a quotation copied solely from the user's message as Aura's tone.
Quotation membership cannot validate semantics: a model can still misread a
negation, sarcasm, or reported speech. The failed sarcasm case demonstrates this.

The existing seven top-level conversation fields remain. `emotional_state`
additionally contains `description` and `assessment`. Complete SQLite turns
preserve both user and Aura assessments in their event payloads without a schema
migration. The UI continues to display Aura's state; it does not expose a new
user-emotion dashboard or correction endpoint. The system prompt tells Aura to
respect the user's account and corrections. This is guidance, not a proven
behavioral guarantee or automatic rewriting of historical assessments.

## Bounded execution and provider repair

Each emotion assessment makes at most one tool-free request, with no persistent
provider session. It has a 20-second deadline, 384 output-token budget, 16,000
source-character cap, and 4,096 output-character validation cap. Oversized input
is not silently truncated. Cancellation propagates. Failures retain the chat
reply and store an unknown assessment rather than turning it into neutrality.

Live diagnosis found Ornith exhausted the 384-token budget in its reasoning
field, returning empty visible content with `finish_reason="length"`. Two initial
runs stopped after two unavailable assessments each; neither established any
emotion accuracy. The initial generic provider error was refined to report the
safe terminal error category. A separate synthetic transport probe established
the actual `length`/empty-content failure.

Emotion requests now opt out of optional reasoning. The Ollama adapter maps the
hint to `reasoning_effort="none"`; ordinary conversation settings are unchanged.
Other provider adapters currently retain their reasoning defaults and may exhaust
the bounded assessment budget. No live Gemini/OpenRouter quality claim is made.

The shared non-streaming compatible adapter now checks terminal status before
accepting content or executing tools. `length` and `content_filter` are resource
failures even when content looks complete. Missing/unknown terminal reasons and
tool/finish mismatches are malformed. This repairs a separate defect that could
otherwise persist truncated replies or execute incomplete tool work.

Existing Python type errors were also repaired by describing the bridge's actual
read-only tool-call interface, normalizing numeric query embeddings, accepting
read-only Chroma result mappings, and correcting optional SDK boundaries. Chroma
1.5.9's actual Client supports `close()` although its `ClientAPI` annotation omits
it; cleanup still calls the public method and does not silently skip it. No
packages or lockfiles were changed.

The verification workflow used fresh checks of the changed implementation and
saved the transport troubleshooting procedure as a reusable local agent skill.

## Reproduce

Deterministic checks use temporary data and no live models:

```bash
uv run --locked --no-sync python -m pytest tests -q -m 'not live'
uv run --locked --no-sync ruff check aura_backend tests --exclude aura_backend/archive_unused --exclude aura_backend/scratch --exclude aura_backend/tests
npm run typecheck:python
npm run typecheck:frontend
npm run build
```

Explicit optional live check, using only the installed local `ornith:latest`:

```bash
uv run --locked --no-sync python -m aura_backend.conversation.evaluate_emotion --live-ollama
```

The command prints JSON and exits nonzero for any failed or missing case. It
stops after two consecutive unavailable responses. It uses no personal memories,
tools, cloud credentials, or model downloads. Do not replace a historical evidence
file with a later run. Always-abstaining, always-labeling, missing, invalid, and
unavailable controls test that the evaluation instrument cannot award a false pass.

## Research and implementation references

- [Wu et al., ACL 2024: Handling Ambiguity in Emotion](https://aclanthology.org/2024.acl-long.114/)
  examines ambiguous labels and uncertainty. It supports taking ambiguity
  seriously; its experiments do not validate Aura's implementation or local model.
- [Pydantic strict validation](https://docs.pydantic.dev/latest/concepts/strict_mode/)
  documents the validation behavior used at the model-output boundary.
- [Ollama OpenAI compatibility](https://docs.ollama.com/api/openai-compatibility)
  documents the reasoning control used for local analysis calls.
- [Python structural protocols](https://typing.python.org/en/latest/reference/protocols.html)
  documents the read-only protocol used by the neutral/Gemini tool bridge.

The next research step is an independently reviewed corpus with broader language,
context, and ambiguity coverage. The next product step is a user-visible way to
inspect and correct an interpretation with explicit provenance. Neither should
be represented as already implemented or validated by this change.
