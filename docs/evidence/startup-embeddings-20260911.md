# Startup and embedding configuration repair

## Changes

- CLI preflight and application composition share an explicit repository `.env`
  loader. Exported shell values take precedence and children inherit the result.
- `AURA_MODEL` selects the model for any provider unless a provider-specific model
  setting is supplied. Explicit mapping-based tests remain isolated from `.env`.
- Disabled the later, conflicting Gemini model assignment in the local `.env`.
  The selected chat model is now Ollama `aura-ornith:35b`.
- Added `AURA_EMBEDDING_PROVIDER` / `AURA_EMBEDDING_MODEL` for active SQLite-derived
  search, separate from Memvid archival settings. This installation now selects
  Ollama `embeddinggemma:latest`; unconfigured installations preserve MiniLM's
  existing index identity.
- Ollama embeddings use bounded batches, a 60-second request timeout, no silent
  truncation, and model/count/dimension/finite-vector validation. There is no
  fallback that silently changes embedding models.
- On a changed embedding configuration, the runtime builds a fresh derived index,
  verifies it, and only then switches. Original records and old generations are
  retained. This work did not run a rebuild over the user's historical records;
  the configured path performs it on first use if needed.

## Verification

- Regression tests reproduced ignored `.env`, ignored shared OpenRouter model,
  acceptance of a wrong embedding model, and failure to rebuild after model change.
- Updated full non-live/nonprivate suite: **875 passed, 7 skipped, 5 deselected**.
- Ruff and Python typecheck passed. Frontend retry test, TypeScript check, and Vite
  build passed. `git diff --check` passed.
- Actual preflight: all checks PASS with `aura-ornith:35b`.
- Actual Ollama embedding probe: one synthetic input, one valid 768-dimensional
  vector from `embeddinggemma:latest`.
- Actual launcher/backend smoke test: `/ready` returned HTTP 200; SIGINT shut down
  the owned launcher/backend with exit 0. Used a temporary clean SQLite root and
  an unused loopback port. No chat generation or visual browser test was performed.

## Normal launch

From the repository root: `./start_full_system.sh`.
No extra `--env-file` flag or separate preflight command is required.

The embedding integration follows the [Ollama embedding API](https://docs.ollama.com/api/embed).
Index and query embeddings must use the same model; existing vectors are not
silently relabelled as belonging to a different model.
