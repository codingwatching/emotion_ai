# Autonomic work and memory

Aura's background worker maintains the searchable index using committed SQLite
records. It runs at startup, after saved exchanges, and every five minutes by
default. Multiple triggers coalesce while maintenance is queued or running.
Maintenance reconciles pending turns and repairs missing or stale index entries.
It does not call a language model, alter source conversation text, or generate
personal facts.

Explicit analysis, generation, and tool tasks remain available through the
`/autonomic/submit-task` endpoint. Model tasks share the selected conversation
provider and model (Ollama, Gemini, or OpenRouter); there is no separate Gemini
requirement. The old `AURA_AUTONOMIC_MODEL` setting was ignored and has been
removed from active configuration.

One worker is the local default. The bounded priority queue supports FIFO when
`AUTONOMIC_QUEUE_PRIORITY_ENABLED=false`. Forced admission preserves task type
and does not bypass queue capacity. Stopping the system cancels and awaits its
owned tasks and marks queued work cancelled. Tool errors are failures, and the
execution deadline covers tools and rate-limit waiting too. A synchronous index
operation is allowed to settle on cancellation before storage is released;
cancelling its await cannot stop a Python thread's writes.

Task results and queue history are in-process diagnostics and reset on restart.
The conversation ledger and search index persist. Restart runs maintenance again.

## What memory does

- SQLite retains complete user/Aura exchanges and their session identity.
- Recent exchanges from the same session provide conversation continuity.
- Hybrid keyword/embedding retrieval supplies relevant historical evidence across
  sessions for the same user, including one-word queries. Prompt context includes
  source IDs and timestamps within a 12,000-character recall budget.
- Affective state is saved with the conversation's durable receipt.
- Optional Memvid archives are separate copies; normal recall does not require them.

There is no automatic derived-summary or personal-fact extraction in this
configuration. Correction recall currently depends on retrieving the source
correction and the model interpreting it; it is not a general fact-revision
engine. A passing small recall test does not establish reliable long-term recall
at scale. Long documents that exceed the embedding model's context remain an
indexing limitation; failed indexing is visible as pending and retains the
SQLite source.

`GET /memory/status` reports source counts, pending indexing and the most recent
maintenance outcome. It deliberately does not label memory quality as proven.
`GET /autonomic/status` reports worker state, queue occupancy, failures and the
actual selected model. The UI's System Health area refreshes these facts every
30 seconds and after replies.

## Configuration

`.env.example` is the supported starting point. `.env` uses the same settings
with local model choices and optional features enabled. Shell variables override
`.env`. The local setup keeps Ornith and embeddinggemma, with one background
worker and copy-only archives enabled.

`AURA_MAX_OUTPUT_TOKENS` now reaches the primary conversation request, with an
8,192-token default. Background model work defaults to 2,048 tokens. These are
output limits; `AURA_HISTORY_MAX_CHARS` is a separate character budget for recent
history. Model context allocation must still accommodate history, recalled
memories, instructions, tools and output.

Removed active settings that were misleading or obsolete:

| Old setting | Current behavior |
| --- | --- |
| `HOST` | Use `AURA_HOST`; local default is `127.0.0.1`. |
| `CORS_ORIGINS` | Use comma-separated `ALLOWED_ORIGINS`. |
| `AURA_AUTONOMIC_MODEL` | Background generation shares the conversation provider/model. |
| `AFC_*`, `MAIN_MODEL_RATE_LIMIT_*` | These did not control the selected provider runtime. |
| `ENABLE_*`, legacy thinking-debug switches | These did not switch the modern conversation path on or off. |

Provider SDK/service limits still apply. Autonomic RPM/RPD settings apply only to
background model requests, not chat or index maintenance.

## Verification

Offline tests cover queue limits, unique IDs, priority/FIFO ordering, tool failure,
timeout, shutdown/restart, coalesced maintenance, actual SQLite/Chroma repair,
short-query recall, source-labelled context and post-commit-only scheduling.

Run the opt-in live check using the installed models and Memvid extra:

```bash
uv run --locked --no-sync python scripts/verify_local_memory.py --output /tmp/aura-memory-check.json
```

It starts an isolated backend with synthetic facts, tests within-session and
cross-session recall, a correction, absent-fact abstention, copy-only archives,
user isolation, and restart. It cleans up only its own temporary backend.
The recorded result for this repair is
[autonomic-memory-2026-09-13.json](evidence/autonomic-memory-2026-09-13.json):
all 17 checks passed. The first run found an unknown-user search error; its
[failed result](evidence/autonomic-memory-2026-09-13-failed-01.json) is retained.
Unknown users now get an empty result without a profile write or embedding call.

The final offline suite passed 902 tests, with 5 skipped and 5 live/private tests
deselected. Active-code Ruff, Python/TypeScript typing, the frontend API regression
and Vite build passed. Both the restarted backend and frontend returned HTTP 200;
startup maintenance completed with all three existing exchanges indexed.
An explicit background model task also completed through Ornith in 14.4 seconds
([model-task evidence](evidence/autonomic-model-2026-09-14.json)).
[Verification summary](evidence/aura-repair-verification-2026-09-14.json).

Implementation references checked during this repair:
[Python queue ownership](https://docs.python.org/3.12/library/asyncio-queue.html)
and [Ollama's non-truncating embedding API](https://docs.ollama.com/api/embed).
