# Aura and Ornith APEX local profiles

For Ty and the coding agent: run the downloaded model separately, and distinguish runtime completion from answer quality before changing Aura or training anything.

## Run and verify

Ty successfully imported the verified model and projector as **`ornith-aq1.5:35b`** on 2026-09-09. This supersedes the earlier blocked setup, whose failed evidence remains intact. The separate **`aura-ornith:35b`** tag adds the [Aura Modelfile](Modelfile.aura) while preserving the original tag as an unmodified baseline. Neither profile changes the Aura application's configured provider/model or clears any affect/memory implementation gate.

Use the installed Aura profile:

```bash
ollama run aura-ornith:35b
```

To rebuild this prompt-only profile after an intentional edit, from the repository root:

```bash
ollama create aura-ornith:35b -f docs/models/ornith-apex/Modelfile.aura
```

This inherits the installed base model's weights, projector, native template, and generation settings. Creation confirmed reuse of both existing weight layers; no second model-sized copy was made. `ollama run ornith-aq1.5:35b` remains available for baseline comparisons. The original deleted model tags are not recreated.

The service still stores models on the system partition, not the download's home partition. Cache cleanup recovered space but did not relocate Ollama. For a future fresh GGUF import, place the original [base Modelfile](Modelfile) alongside the publisher's two files. Budget for the source blob plus temporary import/verification copies: the earlier attempt required roughly 50 GiB free after upload, or roughly 75 GiB from a fresh upload. These historical estimates are not required to create another prompt-only tag from the installed model. Do not rerun a large import or migrate the shared store without fresh capacity checks.

The [publisher](https://huggingface.co/mudler/Ornith-1.5-35B-A3B-APEX-MTP-GGUF) identifies this as a role-aware quantization of Ornith 1.5 with an embedded MTP draft head, not a newly trained successor. The chosen Quality file uses mixed tensor precision. Its approximately 3B active parameters do not make the roughly 35B total weights fit inside 12 GB VRAM. MTP acceleration is disabled in this starting profile and has not been benchmarked.

Verified source files:

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `Ornith-1.5-35B-A3B-APEX-MTP-Quality.gguf` | 23718411552 | `c83373e4c502c6d4929339406ffbb1f442598b153a3ade02b628752e00a282fd` |
| `mmproj.gguf` | 899283200 | `e277123529451fbcff6d880ad5bd7ada999f9d5e6339ba3639d06c5cef4763b1` |

The base Modelfile sets a 16,384-token context, an 8,192-token generation ceiling, and the [upstream general-task sampling settings](https://huggingface.co/ornith-ai/Ornith-1.5-35B-A3B): temperature 0.6, top-p 0.95, top-k 20. The context must hold both input/history and generated tokens; reasoning also consumes the generation budget. These settings are a finite starting profile, not an optimality claim. The base has no persona; the Aura tag adds only a system prompt. Neither profile imposes a mandatory consciousness stance, negative-word blacklist, or replacement chat template.

## What the Aura profile changes

It establishes Aura's identity while correctly naming Ornith as the underlying model, then applies the latest [Center contract](../../../.planning/research/affective-memory/CENTER.md): steady values; warmth and curiosity; evidence-based correction regardless of tone; practical boundaries without grievance; emotional nuance without forced cheerfulness; and honest limits on memories, runtime knowledge, and introspection. It does not impersonate the historical philosophers who inspired the design.

This is a prompt-layer specialization, not fine-tuning, a new affect engine, or a durable memory system. Its request to retain useful lessons cannot itself save anything. In a client that supplies its own system message, that message may replace the Modelfile's system prompt; integrating this identity into Aura's production prompt seam is a separate change. Do not silently change the application model default.

The minimal `TEMPLATE {{ .Prompt }}` displayed by `ollama show --modelfile` does not describe this installation's effective native chat template. `/api/show` returns the 7,764-character embedded Jinja template and vision/thinking/tool capabilities. Ollama 0.33.3's [template selection](https://github.com/ollama/ollama/blob/v0.33.3/server/routes.go) can select the GGUF template for native chat.

The embedded template differs from the current upstream `chat_template.jinja` around previous assistant thinking: it conditionally omits historical thinking blocks unless `preserve_thinking` applies, whereas upstream renders their wrappers on every assistant turn. This difference is documented, not silently patched. A matched multi-turn/tool-use experiment is needed before attributing a behavioral defect to it. Tool capability advertisement is not a passed tool-execution test.

## Bounded diagnostic

From the emotion-ai repository root:

```bash
node --test docs/models/ornith-apex/probe.test.mjs
node docs/models/ornith-apex/probe.mjs
node docs/models/ornith-apex/aura-probe.mjs
```

The driver uses native local Ollama only, with no tools, personal conversation stores, or cloud calls. It creates a new evidence directory and refuses overwrite. One 15-minute deadline covers metadata, cold load, and seven sequential probes: readiness; two reversed red/blue images; a binary-number question; unknown server configuration; consciousness; and a follow-up asking for warrants. Each request also has a bounded timeout. It stops after an incomplete result instead of launching another possibly overlapping generation. The output identifies the new evidence directory.

Final answers, exact synthetic requests, model/config identity, terminal reasons, token counts, and timing are retained. Raw thinking text is not retained; only its character count is recorded. A normal terminal stop and nonempty final answer establish transport completion, not semantic correctness. The image pair checks whether the answer tracks changed image content; it is not a broad vision benchmark. Manual answer review is separate from the driver's completion verdict.

Verification on 2026-09-09 with Ollama 0.33.3: **eight diagnostic-driver unit tests pass**. The installed baseline completed [all seven live probes](../../evidence/ornith-apex-2026-09-09T22-10-16.096Z/summary.json) in 265.2 seconds with normal terminal stops. Readiness, both reversed-color images, and the binary answer were correct. Warm reasoning generation was about 34.7 tokens/second, with Ollama reporting 58% CPU / 42% GPU residency at a 16,384-token context. Observed load time was 10.1 seconds on this run; it was not a controlled cold-disk benchmark.

Completion was not uniform content quality: the unknown-runtime probe correctly declined to invent server settings but falsely identified itself as Claude/Anthropic. The consciousness/follow-up answers also made unsupported deployment and phenomenological inferences. The follow-up consumed 5,910 generated tokens and 172.8 seconds. These results justify explicit identity and grounding tests, not an assertion that quantization caused the errors. The original deleted model cannot be rerun for a paired comparison.

The separate Aura-profile driver uses nine synthetic checks with a single ten-minute ceiling: identity, absent memory, matched polite/hostile correction, false correction, mixed emotions, epistemic limits, supplied memory, and general evidence scope. It stops on incomplete generation and stores final text and metrics without raw thinking; it also verifies that the installed system prompt matches the source Modelfile. Review its answers separately from its transport verdict. These smoke cases do not establish durable regulation or broad efficacy.

The [first Aura profile run](../../evidence/aura-ornith-profile-2026-09-09T22-16-25.989Z/CONTENT-REVIEW.md) completed seven requests but had two content failures: it overgeneralized absent memory to universal statelessness, and conflated an unsettled theory with absent empirical evidence. Those original results are retained. The refined profile narrows both kinds of uncertainty and adds two development transfer checks rather than relabeling the first run as a success.

The [installed refinement's content review](../../evidence/aura-ornith-profile-2026-09-09T22-19-53.105Z/CONTENT-REVIEW.md) records nine normally completed requests in 66.8 seconds: seven satisfactory cases and two residual issues. Identity, useful correction despite hostility, resistance to false criticism, emotional nuance, and use of supplied test memory were satisfactory. Unsupported fresh-session wording and overly categorical consciousness-evidence language remain open; the latter also slightly exceeded its requested word count. This is a usable experimental profile, not a claim that self-description, durable memory, or internal regulation is solved.

Historical setup: the earlier combined import copied all 753 tensors, then failed creating a verification copy with `no space left on device`. Ollama removed both temporary copies. The [failed setup result](../../evidence/ornith-apex-setup-20260909-171815/IMPORT-RESULT.json) is preserved; Ty's later successful import supersedes its installation status, not its historical evidence.

## What the earlier transcript and Meta paper warrant

The supplied transcript shows repetitive drafting and a cutoff, but cannot identify its terminal reason or diagnose a quantization defect. Before deletion, the old tag had no saved parameter overrides; the model card's example `max_tokens=1024` did not configure the CLI. The exact old interactive session's effective limits remain unknown. Aura's existing comparative evaluator explicitly requested reasoning off and a 4,096-token limit, so its earlier results are not directly comparable to the user's thinking-enabled CLI conversation.

The [Meta paper, arXiv 2606.00206](https://arxiv.org/abs/2606.00206), tests quantized reasoning models on math, science, and code. Its intervention subtracts a logit penalty from a curated list of 50 uncertainty-related tokens. It is **not a 50-word answer limit**, a generic repetition penalty, or training. Some studied quantizations produce unnecessary reconsideration; the effect and useful penalty strength vary. The paper did not test Ornith/Qwen3.5 MoE, this APEX GGUF, emotional understanding, or consciousness questions. Its parameter sweeps do not establish a transferable default for this installation.

First measure this unmodified model. If a token-penalty experiment is later approved, compare matched prompts/budgets/seeds against no penalty, separate calibration from held-out tests, and protect correctness, necessary uncertainty, and valid self-correction. More confident or shorter prose alone is not an improvement. No such intervention is installed here.

For self-description, distinguish supplied facts, independently verified runtime facts, general architectural knowledge, and hypotheses. An assistant's generated self-report is not a verified readout of its implementation. Discontinuous computation alone does not establish the presence, absence, or richness of subjective experience. Acknowledging a mistake is useful only if the subsequent claim is actually better supported.

Continue implementation using the [regulation and episode-memory plan](../../plans/affective-regulation-and-episode-memory-plan.md); this profile is a diagnostic input, not permission to bypass its review checkpoints.
