# Model and Memvid functional repair — 2026-09-12, finalized 2026-09-13

## Installed model and configuration

- Rebuilt `aura-ornith:35b` from `docs/models/ornith-apex/Modelfile.aura`.
- Reused the installed quantized weights and projector; no weight download.
- Previous model retained as `aura-ornith:pre-memvid-20260911` (shared blobs).
- Explicit context: 131072; generation budget: 8192.
- Ollama reported context_length 131072 and approximately 10.5 GB GPU allocation.
- A synthetic direct request consumed 30000 prompt tokens and returned the code
  placed at its beginning (`AMBER-7429`), with `done_reason=stop`.
- This verifies recall beyond the previous 16384 limit, NOT full-128K accuracy.
- Local `.env`: archive enabled, embeddinggemma via Ollama, SDK telemetry off;
  application history budget 96000 characters (distinct from token capacity).

## Real memory workflow

Installed the locked `memvid-sdk==2.0.160` extra (95.1 MiB wheel), with permission.
The current application uses SQLite for committed records, Chroma for disposable
active vectors, and Memvid `.mv2` copies with independently computed vectors.
The old Chroma-bound archival facade is no longer the runtime archive backend.

`uv run --locked --no-sync python scripts/verify_local_memory.py` completed:

```json
{
  "continuity": "AMBER-7429",
  "messages_archived": 4,
  "chunks": 4,
  "archive_size_bytes": 115561,
  "embedding_model": "embeddinggemma:latest",
  "archive_search_after_start_0": 4,
  "archive_search_after_start_1": 4,
  "status": "PASS"
}
```

The smoke used the real launcher, real model, real local embeddings, a fresh
temporary ledger, actual archive files, shutdown and immediate restart. Every
archived chunk's original text and metadata were compared after reopening.
Sources were retained; another user scope returned no archive records.

An additional real-model request explicitly requested archive search. Server logs
confirmed `aura.search_archives` execution and success; the model reported both
synthetic archived entries and the conversation committed. Merely mentioning a
tool in the answer was not used as proof of execution.

## Browser verification

An isolated Chromium session used the real frontend and backend with a synthetic
committed session. Selected the chat, clicked **Archive this chat**, observed one
archive, disabled Active Memory, and searched. Two exact synthetic message records
were displayed with `Source: Memvid archive`. A combined search also returned
active and archived records. Screenshot: [browser](memvid-browser-20260912.png).

## Failures encountered and corrected

1. Original live recall probe failed: the model interpreted untrusted historical
   memory as forbidden-to-recall data and refused the fictional code. Added a
   distinction between using facts and obeying embedded instructions. The SAME
   recall prompt then passed; this does not prove universal refusal calibration.
2. Memvid's frame API takes a URI and exposes metadata, not original message text.
   Read-back tests failed before the adapter stored exact text explicitly and
   decoded the SDK's JSON-encoded metadata correctly.
3. Internal registration did not refresh the provider's immutable tool catalog.
   Added catalog refresh, a request-level regression, and the real tool test.
4. Immediate restart failed because preflight lacked socket reuse semantics and
   rejected TIME_WAIT connections. Reproduced independently, fixed to match
   uvicorn, and tested that live listeners still fail the check.
5. UI search requested 50000 results against a maximum of 100, ignored source
   checkboxes, and rendered incompatible scores as NaN percentages. Corrected
   request bounds/source selection and removed misleading confidence percentages.
6. Stopping npm left its Vite child alive, causing the next frontend to move to
   5174. POSIX launches now own a separate process group, and cleanup signals the
   group. Vite uses strict port 5173. A real grandchild-listener regression passes;
   an actual full-stack stop released both ports, removed the Vite child, and
   restarted successfully on 8000/5173.

## Final regression gate

- Non-live pytest: **885 passed, 5 skipped, 5 deselected** (44.32 seconds).
- Real SDK archive tests execute; CI now installs the Memvid extra.
- Ruff: all checks passed.
- Pyright: 0 errors, 0 warnings.
- Frontend tsc: exit 0; Vite build: exit 0.
- Frontend retry/search/archive request contract: 1 passed.
- `git diff --check`: exit 0.

The user's old runtime was gracefully stopped and the updated stack started.
Live checks returned `/ready` 200 with Memvid ready, `/memvid/status` operational,
and frontend HTTP 200 on the usual port 5173. No source chat data was removed.

## Limits

Archive creation is an explicit session snapshot, not automatic consolidation.
It copies at most the latest 100 exchanges. Existing historical video/Chroma
archives were not imported, and no source memory was deleted. Changing archive
embedding configuration fails explicitly rather than mixing incompatible vectors.
The legacy facade remains for legacy callers; it is not evidence of functioning
runtime archival. Affective-evaluation and consolidation milestones are separate
from these functional repairs. No training or claims about subjective experience.
