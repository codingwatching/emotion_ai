# First-conversation and continuity repair

## Reproduced defects

1. The selected SQLite ledger's parent directory did not exist. Startup only
   constructed a repository object, so readiness passed before any database open.
2. The frontend requested 200,000 history entries (the UI passed 2,000,000) while
   the backend accepts at most 100. HTTP 400 was then caught and converted to 500.
3. Provider requests contained only the latest user message. Passing a session ID
   did not supply conversational history to the stateless transport.
4. The SQLite chat-history adapter returned event/turn IDs as session IDs and
   mixed individual events into what the UI expected to be session summaries.
5. The UI replaced conversation titles with every new message and labelled any
   successful HTTP response as optimal system health.
6. Runtime memory context still called the legacy reader despite SQLite read
   ownership. The profile service was not constructed, though its tool was exposed.

## Repairs

- Startup creates/opens the explicitly configured ledger before readiness.
  Preflight remains read-only. The isolated startup test now exercises real SQLite
  in its own temporary root rather than forbidding the required startup open.
- Frontend requests conform to the backend bound; HTTP exceptions retain status.
- Recent committed user/assistant exchanges from the same scope and session are
  included in the provider request. Complete pairs are retained up to 100 turns
  and the AURA_HISTORY_MAX_CHARS budget (default 24,000 characters, not tokens).
- Session listing groups actual session IDs, counts both speakers, and uses the
  first user message as a stable title. Session retrieval filters both scope and ID.
- SQLite-owned context retrieval uses the actual hybrid retriever/Chroma path.
- The profile service is initialized under the selected ledger's profile-files
  directory; missing profile is now not_found, not an initialization error.
- Tools with absent backends are not advertised as available. Health text says
  backend/reply reachable, not that all subsystems have been verified.
- This installation explicitly selects the current SQLite read path for new
  chats. No legacy database import or Memvid archive restoration was performed.

## Evidence

New production-composition test: tests/runtime/test_first_conversation.py.
It covers readiness from a nonexistent root, durable first/second turns, exact
outbound conversational history, profile save/read, session identity/title, and
cross-scope/session exclusion. Model and embedding I/O are synthetic in CI.
The existing frontend retry test also reproduces and checks history limit bounds.

During the repair, an isolated real-model two-turn check using aura-ornith:35b
returned a committed acknowledgement of the synthetic codename Cedar Lantern,
then correctly answered Cedar Lantern on the next turn. Revisions were 1 and 2,
both projections completed, and history returned one real session with 4 messages.
The launcher shut down with exit 0. This live check preceded the final profile
connection; that final connection is covered by the production-composition test.

The live tone proposal validated for the first reply and abstained for the bare
codename on the second reply. Abstention is not a failed call or evidence of a
particular emotion. No broad classifier-accuracy claim follows from this smoke test.

## Remaining boundaries

- Memvid SDK/archive functionality remains unavailable; this repair does not
  restore the old archive workflow or historical data.
- The inspected Ollama model tag has num_ctx=16384 and num_predict=8192. Its
  theoretical capacity does not mean 120k tokens are allocated locally. No model
  context resize was performed. The history character bound is not a tokenizer.
- The health label is now narrower, not a new comprehensive subsystem monitor.
- Session history is bounded; full UI history pagination is not implemented here.
- Backend restart and browser refresh are needed for the running app to load
  these edits. The user's existing server was not killed or restarted remotely.
