"""Slice 2 characterization and acceptance tests: causal loop, durability, and replay."""

from __future__ import annotations

import asyncio
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from starlette.testclient import TestClient

from aura_backend import main
from aura_backend.affect.appraisal import build_appraisal_record
from aura_backend.affect.dynamics import (
    apply_pre_state,
    calculate_turn_impulse,
)
from aura_backend.affect.models import (
    AffectConfig,
    AffectState,
    AffectTransition,
    TaskOutcome,
)
from aura_backend.affect.policy import render_policy
from aura_backend.affect.service import AffectService
from aura_backend.conversation_persistence_service import ConversationPersistenceService
from aura_backend.providers.base import (
    ProviderRequest,
    ProviderResult,
)
from aura_backend.providers.runtime import ProviderRuntime
from aura_backend.providers.tools import ToolCatalog
from aura_backend.storage.connection import append_turn_atomic, open_database
from aura_backend.storage.models import (
    EventInput,
    StorageFailure,
    TurnCommand,
)
from aura_backend.storage.repository import StorageRepository, canonical_request_hash
from aura_backend.storage.schema import apply_migrations


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


class _Projection:
    def upsert_committed(self, _turn_id: str) -> int:
        return 0


class _CaptureProvider:
    def __init__(self, response_text: str = "Synthetic reply from model.") -> None:
        self.requests: list[ProviderRequest] = []
        self.response_text = response_text

    async def generate(self, request: ProviderRequest) -> ProviderResult:
        self.requests.append(request)
        if len(self.requests) == 1:
            return ProviderResult(content=self.response_text)
        return ProviderResult(
            content=json.dumps(
                {
                    "emotion": "Sad",
                    "intensity": "High",
                    "evidence": ["Synthetic test"],
                    "reason": "Synthetic reply tone override attempt",
                }
            )
        )

    async def stream(self, _request: ProviderRequest) -> Any:
        raise AssertionError("non-streaming route")

    async def clear_session(self, _session_id: str) -> None:
        pass

    async def health(self) -> Any:
        return None

    async def aclose(self) -> None:
        pass


@dataclass
class _TestAppRuntime:
    provider_runtime: ProviderRuntime
    tool_catalog: ToolCatalog

    async def start(self) -> _TestAppRuntime:
        return self

    async def aclose(self) -> None:
        await self.provider_runtime.aclose()

    def resource(self, name: str) -> object:
        if name != "legacy_services":
            raise LookupError(name)
        return SimpleNamespace(mcp_router=None, tool_catalog=self.tool_catalog)


@pytest.fixture
def clean_db(tmp_path: Path) -> Path:
    db_path = tmp_path / "affect_test_ledger.sqlite3"
    connection = open_database(db_path)
    try:
        apply_migrations(connection)
    finally:
        connection.close()
    return db_path


def _make_turn_command(
    scope_id: str,
    session_id: str,
    turn_id: str,
    idempotency_key: str,
    user_text: str,
    aura_text: str,
    affect_transition: AffectTransition | None = None,
    expected_state_revision: int | None = None,
) -> TurnCommand:
    user_event = EventInput(
        event_id=f"evt_user_{turn_id}",
        actor="user",
        content=user_text,
        observed_at="2026-09-08T12:00:00Z",
        content_sha256=_digest(user_text),
    )
    aura_event = EventInput(
        event_id=f"evt_aura_{turn_id}",
        actor="aura",
        content=aura_text,
        observed_at="2026-09-08T12:00:01Z",
        content_sha256=_digest(aura_text),
    )
    return TurnCommand(
        scope_id=scope_id,
        session_id=session_id,
        turn_id=turn_id,
        idempotency_key=idempotency_key,
        request_hash_version=1,
        request_hash=canonical_request_hash(scope_id, session_id, user_text, version=1),
        response_hash=_digest(aura_text),
        occurred_at="2026-09-08T12:00:00Z",
        user_event=user_event,
        aura_event=aura_event,
        derived_memories=(),
        affect_transition=affect_transition,
        expected_state_revision=expected_state_revision,
    )


# 1. Primary outgoing prompt contains the pre-state policy block
def test_primary_prompt_contains_pre_state_policy_block(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepository(clean_db)
    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_Projection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("I am here and ready to help.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    with TestClient(app) as client:
        response = client.post(
            "/conversation",
            json={
                "user_id": "test-user-policy-prompt",
                "message": "Hello, can we solve this difficult puzzle?",
                "session_id": "sess-prompt-001",
            },
        )
    assert response.status_code == 200
    assert len(provider.requests) >= 1
    primary_request = provider.requests[0]

    instruction = primary_request.system_instruction
    assert "### Simulated Behavioral Posture (affect-v1)" in instruction
    assert "Warmth: warm" in instruction
    assert "Energy & Pace: balanced" in instruction
    assert "Initiative & Exploration: balanced" in instruction


# 2. Reply-tone analysis cannot overwrite authoritative simulation state
def test_reply_tone_analysis_cannot_overwrite_simulation_state(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepository(clean_db)
    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_Projection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("We made steady progress on the task.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    with TestClient(app) as client:
        response = client.post(
            "/conversation",
            json={
                "user_id": "test-user-tone-isolation",
                "message": "Here is a hint for the puzzle.",
                "session_id": "sess-tone-001",
            },
        )
    assert response.status_code == 200
    body = response.json()

    # The simulation payload must be governed by appraisal and dynamics, not the "Sad" tone
    sim = body["emotional_state"]["simulation"]
    assert sim["revision"] == 1
    assert sim["post_state"]["valence"] >= 0.1
    head = repo.get_affect_head("test-user-tone-isolation")
    assert head is not None
    assert head.revision == 1
    assert head.fast_state.valence == pytest.approx(sim["post_state"]["valence"])


# 3. CAS revision conflict failure and rollback
def test_cas_revision_conflict_and_storage_fault_rollback(clean_db: Path) -> None:
    config = AffectConfig()
    t0 = 1700000000.0
    s0 = AffectState.initial("scope-cas", config, t0)
    appraisal = build_appraisal_record(
        event_id="evt_cas_1",
        message="turn 1",
        accepted_events=["verified_task_success"],
    )
    policy = render_policy(s0.fast_state)
    impulse = calculate_turn_impulse(["verified_task_success"], config)
    pre = apply_pre_state(s0.fast_state, impulse)
    fast_next = pre
    trans1 = AffectTransition(
        transition_id="trans_cas_1",
        scope_id="scope-cas",
        revision=1,
        turn_id="turn-cas-1",
        idempotency_key="key-cas-1",
        prior_revision=0,
        input_digest=_digest("turn 1"),
        accepted_appraisal=appraisal.to_dict(),
        pre_state=pre,
        after_state=fast_next,
        rendered_policy=policy,
        outcome_disposition="observed",
        config_hash=config.config_hash,
        timestamp=t0 + 1.0,
    )

    cmd1 = _make_turn_command(
        scope_id="scope-cas",
        session_id="sess-cas",
        turn_id="turn-cas-1",
        idempotency_key="key-cas-1",
        user_text="turn 1",
        aura_text="turn 1 reply",
        affect_transition=trans1,
        expected_state_revision=0,
    )
    conn = open_database(clean_db)
    try:
        append_turn_atomic(conn, cmd1)
    finally:
        conn.close()

    repo = StorageRepository(clean_db)
    head = repo.get_affect_head("scope-cas")
    assert head is not None
    assert head.revision == 1

    # Attempt turn 2 with stale expected revision 0 (should conflict with revision 1)
    trans2 = AffectTransition(
        transition_id="trans_cas_2",
        scope_id="scope-cas",
        revision=2,
        turn_id="turn-cas-2",
        idempotency_key="key-cas-2",
        prior_revision=1,
        input_digest=_digest("turn 2"),
        accepted_appraisal=appraisal.to_dict(),
        pre_state=pre,
        after_state=fast_next,
        rendered_policy=policy,
        outcome_disposition="observed",
        config_hash=config.config_hash,
        timestamp=t0 + 2.0,
    )
    cmd2_stale = _make_turn_command(
        scope_id="scope-cas",
        session_id="sess-cas",
        turn_id="turn-cas-2",
        idempotency_key="key-cas-2",
        user_text="turn 2",
        aura_text="turn 2 reply",
        affect_transition=trans2,
        expected_state_revision=0,  # STALE: current is 1!
    )
    conn = open_database(clean_db)
    try:
        with pytest.raises(StorageFailure) as exc_info:
            append_turn_atomic(conn, cmd2_stale)
        assert exc_info.value.code == "affect_revision_conflict"
    finally:
        conn.close()

    # Head is untouched, revision remains 1
    head_after = repo.get_affect_head("scope-cas")
    assert head_after is not None
    assert head_after.revision == 1
    assert repo.get_affect_transition("scope-cas", 2) is None

    # Test storage fault rollback: fault hook after affect_transition before affect_head
    cmd2_fault = _make_turn_command(
        scope_id="scope-cas",
        session_id="sess-cas",
        turn_id="turn-cas-2-fault",
        idempotency_key="key-cas-2-fault",
        user_text="turn 2 fault",
        aura_text="turn 2 fault reply",
        affect_transition=trans2,
        expected_state_revision=1,
    )
    def fault_hook(stage: str) -> None:
        if stage == "after_affect_transition":
            raise StorageFailure("fault_sim")

    conn = open_database(clean_db)
    try:
        with pytest.raises(StorageFailure) as fault_exc:
            append_turn_atomic(
                conn,
                cmd2_fault,
                fault_hook=fault_hook,
            )
        assert fault_exc.value.code == "fault_sim"
    finally:
        conn.close()

    # Verify atomic rollback: neither transition 2 nor head update was committed
    assert repo.get_affect_transition("scope-cas", 2) is None
    head_final = repo.get_affect_head("scope-cas")
    assert head_final is not None
    assert head_final.revision == 1


# 4. SQLite restart restores exact trajectory and revision
@pytest.mark.asyncio
async def test_restart_restores_exact_trajectory_and_revision(clean_db: Path) -> None:
    repo = StorageRepository(clean_db)
    service1 = AffectService(repository=repo)
    scope_id = "restart-user"

    # Turn 1:
    t1 = 1700000000.0
    prior_s, policy1, events1, app1, pre1 = await service1.compute_provisional_policy(
        scope_id=scope_id,
        message="Let's investigate this unexpected test failure.",
        timestamp=t1,
    )
    next_s1, trans1 = await service1.commit_turn(
        scope_id=scope_id,
        turn_id="turn_r_1",
        idempotency_key="key_r_1",
        input_digest=_digest("turn 1"),
        prior_state=prior_s,
        pre_state=pre1,
        policy=policy1,
        appraisal=app1,
        outcome=TaskOutcome(task_id="task_r_1", success=True),
        timestamp=t1 + 5.0,
    )

    # Persist turn 1 to SQLite
    cmd1 = _make_turn_command(
        scope_id=scope_id,
        session_id="sess-r",
        turn_id="turn_r_1",
        idempotency_key="key_r_1",
        user_text="Let's investigate this unexpected test failure.",
        aura_text="I'll look at the test logs.",
        affect_transition=trans1,
        expected_state_revision=0,
    )
    conn = open_database(clean_db)
    try:
        append_turn_atomic(conn, cmd1)
    finally:
        conn.close()

    # Turn 2:
    t2 = t1 + 60.0
    prior_s2, policy2, events2, app2, pre2 = await service1.compute_provisional_policy(
        scope_id=scope_id,
        message="We found the bug and fixed it, thank you!",
        timestamp=t2,
    )
    next_s2, trans2 = await service1.commit_turn(
        scope_id=scope_id,
        turn_id="turn_r_2",
        idempotency_key="key_r_2",
        input_digest=_digest("turn 2"),
        prior_state=prior_s2,
        pre_state=pre2,
        policy=policy2,
        appraisal=app2,
        outcome=TaskOutcome(task_id="task_r_2", success=True),
        timestamp=t2 + 5.0,
    )

    # Persist turn 2 to SQLite
    cmd2 = _make_turn_command(
        scope_id=scope_id,
        session_id="sess-r",
        turn_id="turn_r_2",
        idempotency_key="key_r_2",
        user_text="We found the bug and fixed it, thank you!",
        aura_text="Glad we solved it!",
        affect_transition=trans2,
        expected_state_revision=1,
    )
    conn = open_database(clean_db)
    try:
        append_turn_atomic(conn, cmd2)
    finally:
        conn.close()

    saved_head = repo.get_affect_head(scope_id)
    assert saved_head is not None
    assert saved_head.revision == 2

    # --- SIMULATE SERVER RESTART ---
    # Create brand-new repo and AffectService instance
    new_repo = StorageRepository(clean_db)
    service2 = AffectService(repository=new_repo)

    # In-memory dictionary is completely empty
    assert scope_id not in service2._states

    # Fetch state on the fresh service instance
    restored_state = service2.get_state(scope_id)
    assert restored_state.revision == 2
    assert restored_state.config_version == saved_head.config_version
    assert restored_state.config_hash == saved_head.config_hash
    assert restored_state.fast_state.valence == pytest.approx(saved_head.fast_state.valence)
    assert restored_state.fast_state.arousal == pytest.approx(saved_head.fast_state.arousal)
    assert restored_state.mood_state.valence == pytest.approx(saved_head.mood_state.valence)
    assert restored_state.source_transition_id == saved_head.source_transition_id

    # Turn 3 on the restarted service continues seamlessly from revision 2
    t3 = t2 + 30.0
    prior_s3, policy3, events3, app3, pre3 = await service2.compute_provisional_policy(
        scope_id=scope_id,
        message="What's our next task?",
        timestamp=t3,
    )
    assert prior_s3.revision == 2

    next_s3, trans3 = await service2.commit_turn(
        scope_id=scope_id,
        turn_id="turn_r_3",
        idempotency_key="key_r_3",
        input_digest=_digest("turn 3"),
        prior_state=prior_s3,
        pre_state=pre3,
        policy=policy3,
        appraisal=app3,
        outcome=None,
        timestamp=t3 + 5.0,
    )
    assert next_s3.revision == 3
    assert trans3.prior_revision == 2


# 5. Duplicate request with identical idempotency key returns cached replay (0 new provider calls)
def test_idempotent_replay_makes_zero_extra_provider_calls(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepository(clean_db)
    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_Projection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("Deterministic replayable answer.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    payload = {
        "user_id": "replay-user",
        "message": "Replayable question",
        "session_id": "sess-replay-1",
        "idempotency_key": "idem-key-stable-101",
    }

    with TestClient(app) as client:
        # First request: processed normally
        resp1 = client.post("/conversation", json=payload)
        assert resp1.status_code == 200
        body1 = resp1.json()
        assert body1["response"] == "Deterministic replayable answer."
        initial_provider_calls = len(provider.requests)
        assert initial_provider_calls >= 1

        # Second request with identical payload and idempotency_key
        resp2 = client.post("/conversation", json=payload)
        assert resp2.status_code == 200
        body2 = resp2.json()

        # Zero extra calls to provider
        assert len(provider.requests) == initial_provider_calls
        # Exact response returned
        assert body2["response"] == body1["response"]
        assert body2["session_id"] == body1["session_id"]
        assert body2["emotional_state"]["simulation"]["revision"] == 1


# 6. Changed-body retry with the same idempotency key returns 409 conflict
def test_changed_body_retry_with_same_idempotency_key_returns_409(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepository(clean_db)
    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_Projection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("Initial answer.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    with TestClient(app) as client:
        # First request
        resp1 = client.post(
            "/conversation",
            json={
                "user_id": "conflict-user",
                "message": "Original message body",
                "session_id": "sess-conflict",
                "idempotency_key": "idem-key-conflict-202",
            },
        )
        assert resp1.status_code == 200

        # Retry with SAME idempotency_key but DIFFERENT message body
        resp2 = client.post(
            "/conversation",
            json={
                "user_id": "conflict-user",
                "message": "Completely different message body",
                "session_id": "sess-conflict",
                "idempotency_key": "idem-key-conflict-202",
            },
        )
        assert resp2.status_code == 409
        assert "Idempotency conflict" in resp2.json()["detail"]


# 7. Concurrent same-scope requests serialize cleanly
@pytest.mark.asyncio
async def test_concurrent_same_scope_requests_serialize(clean_db: Path) -> None:
    repo = StorageRepository(clean_db)
    service = AffectService(repository=repo)
    scope_id = "concurrent-user"
    t_base = 1700000000.0

    async def execute_turn(turn_idx: int) -> int:
        async with service.scope_lock(scope_id):
            prior, pol, evs, app, pre = await service.compute_provisional_policy(
                scope_id=scope_id,
                message=f"Concurrent message {turn_idx}",
                timestamp=t_base + turn_idx,
            )
            # Simulate in-flight async delay (e.g. model generation)
            await asyncio.sleep(0.01)
            next_state, trans = await service.stage_turn(
                scope_id=scope_id,
                turn_id=f"turn_conc_{turn_idx}",
                idempotency_key=f"key_conc_{turn_idx}",
                input_digest=_digest(f"Concurrent message {turn_idx}"),
                prior_state=prior,
                pre_state=pre,
                policy=pol,
                appraisal=app,
                outcome=None,
                timestamp=t_base + turn_idx + 0.5,
            )
            service.publish_turn(scope_id, next_state, trans)
            return next_state.revision

    # Execute 5 overlapping concurrent turns simultaneously via asyncio.gather
    revisions = await asyncio.gather(*(execute_turn(i) for i in range(1, 6)))

    assert sorted(revisions) == [1, 2, 3, 4, 5]
    final_state = service.get_state(scope_id)
    assert final_state.revision == 5


# 8. Two separate scopes never share state or leak evidence
@pytest.mark.asyncio
async def test_two_scopes_are_completely_isolated(clean_db: Path) -> None:
    repo = StorageRepository(clean_db)
    service = AffectService(repository=repo)
    scope_alice = "user-alice"
    scope_bob = "user-bob"
    t_start = 1700000000.0

    # Alice experiences a puzzle success and gratitude
    p_alice, pol_a, evs_a, app_a, pre_a = await service.compute_provisional_policy(
        scope_id=scope_alice,
        message="Thank you! We solved the puzzle successfully!",
        timestamp=t_start,
    )
    s_alice, trans_a = await service.commit_turn(
        scope_id=scope_alice,
        turn_id="turn_a_1",
        idempotency_key="key_a_1",
        input_digest=_digest("alice message"),
        prior_state=p_alice,
        pre_state=pre_a,
        policy=pol_a,
        appraisal=app_a,
        outcome=TaskOutcome(task_id="task_a_1", success=True),
        timestamp=t_start + 1.0,
    )

    # Bob experiences a difficult failure
    p_bob, pol_b, evs_b, app_b, pre_b = await service.compute_provisional_policy(
        scope_id=scope_bob,
        message="That attempt failed completely.",
        timestamp=t_start + 2.0,
    )
    s_bob, trans_b = await service.commit_turn(
        scope_id=scope_bob,
        turn_id="turn_b_1",
        idempotency_key="key_b_1",
        input_digest=_digest("bob message"),
        prior_state=p_bob,
        pre_state=pre_b,
        policy=pol_b,
        appraisal=app_b,
        outcome=TaskOutcome(task_id="task_b_1", success=False),
        timestamp=t_start + 3.0,
    )

    # Alice's valence must be higher than Bob's
    assert s_alice.fast_state.valence > s_bob.fast_state.valence
    assert s_alice.scope_id == "user-alice"
    assert s_bob.scope_id == "user-bob"

    # Transitions must be isolated
    assert trans_a.scope_id == "user-alice"
    assert trans_b.scope_id == "user-bob"
    assert trans_a.turn_id == "turn_a_1"
    assert trans_b.turn_id == "turn_b_1"


# 9. Failed database write rolls back state and does not advance revision
def test_failed_database_write_does_not_advance_state(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = StorageRepository(clean_db)
    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_Projection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("I am here to assist.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    user_id = "test-user-failing-db"

    with TestClient(app) as client:
        # Turn 1: Normal working persistence -> Revision 1
        resp1 = client.post(
            "/conversation",
            json={
                "user_id": user_id,
                "message": "First message, database is fine.",
                "session_id": "sess-fail-001",
            },
        )
        assert resp1.status_code == 200
        sim1 = resp1.json()["emotional_state"]["simulation"]
        assert sim1["revision"] == 1
        assert affect_svc.get_state(user_id).revision == 1
        persisted_head = repo.get_affect_head(user_id)
        assert persisted_head is not None
        assert persisted_head.revision == 1

        # Turn 2: Inject storage write failure
        async def mock_fail_persist(*args: Any, **kwargs: Any) -> dict[str, Any]:
            return {"success": False, "error": "Disk I/O error or constraint violation"}

        monkeypatch.setattr(
            persistence,
            "persist_conversation_exchange_immediate",
            mock_fail_persist,
        )
        monkeypatch.setattr(
            persistence,
            "persist_conversation_exchange",
            mock_fail_persist,
        )

        resp2 = client.post(
            "/conversation",
            json={
                "user_id": user_id,
                "message": "Second message, database write will fail.",
                "session_id": "sess-fail-001",
            },
        )
        # Route gracefully degrades (200 OK) preserving response
        assert resp2.status_code == 200
        sim2 = resp2.json()["emotional_state"]["simulation"]
        assert sim2["revision"] == 1
        assert sim2["disposition"] == "uncommitted"

        # Authority assertion: State MUST NOT advance in memory or in storage
        assert affect_svc.get_state(user_id).revision == 1
        persisted_head_after_fail = repo.get_affect_head(user_id)
        assert persisted_head_after_fail is not None
        assert persisted_head_after_fail.revision == 1

        # Turn 3: Restore working persistence and verify clean sequence continuation
        monkeypatch.undo()  # restores persist_conversation_exchange_immediate
        # Re-apply non-failing monkeypatches
        monkeypatch.setattr(main, "conversation_persistence", persistence)
        monkeypatch.setattr(main, "affect_service", affect_svc)
        monkeypatch.setattr(main, "provider", None)
        monkeypatch.setattr(main, "client", None)
        monkeypatch.setattr(main, "mcp_gemini_bridge", None)

        resp3 = client.post(
            "/conversation",
            json={
                "user_id": user_id,
                "message": "Third message, database is healthy again.",
                "session_id": "sess-fail-001",
            },
        )
        assert resp3.status_code == 200
        sim3 = resp3.json()["emotional_state"]["simulation"]
        # Must be revision 2 (not 3), because turn 2 failed and never committed
        assert sim3["revision"] == 2
        assert affect_svc.get_state(user_id).revision == 2
        persisted_head_3 = repo.get_affect_head(user_id)
        assert persisted_head_3 is not None
        assert persisted_head_3.revision == 2


# 10. Timeout race: slow writer committing late does NOT cause affect_revision_conflict on next turn
def test_timeout_late_commit_does_not_cause_revision_conflict(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A timed-out writer that commits late to SQLite must not cause affect_revision_conflict on next turn."""
    import time
    repo = StorageRepository(clean_db)
    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_Projection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("Synthetic reply.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    user_id = "test-user-timeout-race"

    with TestClient(app) as client:
        monkeypatch.setenv("PERSISTENCE_TIMEOUT", "0.05")

        orig_persist_command = persistence._persist_command

        def slow_persist_command(command: Any) -> Any:
            time.sleep(0.15)
            return orig_persist_command(command)

        monkeypatch.setattr(persistence, "_persist_command", slow_persist_command)

        resp1 = client.post(
            "/conversation",
            json={
                "user_id": user_id,
                "message": "Turn 1 will time out immediately.",
                "session_id": "sess-race-001",
            },
        )
        assert resp1.status_code == 200

        # Wait for the slow writer thread to finish committing to SQLite
        time.sleep(0.3)

        # Confirm the slow writer committed rev 1 to SQLite
        db_head = repo.get_affect_head(user_id)
        assert db_head is not None
        assert db_head.revision == 1

        # Turn 2: restore normal persistence speed
        monkeypatch.setattr(persistence, "_persist_command", orig_persist_command)
        monkeypatch.setenv("PERSISTENCE_TIMEOUT", "5.0")

        resp2 = client.post(
            "/conversation",
            json={
                "user_id": user_id,
                "message": "Turn 2 should build on revision 1 and advance to 2 without conflict.",
                "session_id": "sess-race-001",
            },
        )
        assert resp2.status_code == 200
        sim2 = resp2.json()["emotional_state"]["simulation"]
        # Must be revision 2, NOT conflict error!
        assert sim2["revision"] == 2
        assert sim2["disposition"] != "uncommitted"
        assert affect_svc.get_state(user_id).revision == 2
        head_after_2 = repo.get_affect_head(user_id)
        assert head_after_2 is not None
        assert head_after_2.revision == 2


# 11. Projection failure: disposable projection failure does NOT discard committed ledger
def test_projection_failure_does_not_discard_committed_ledger(
    clean_db: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When disposable projection fails, the committed ledger must NOT be discarded; affect state advances."""
    repo = StorageRepository(clean_db)

    class _FailingProjection:
        def upsert_committed(self, _turn_id: str) -> int:
            raise RuntimeError("Chroma vector index disk full or connection error")

    persistence = ConversationPersistenceService(
        repository=repo,
        projection=_FailingProjection(),
    )
    affect_svc = AffectService(repository=repo)
    provider = _CaptureProvider("I am here to assist.")
    runtime = ProviderRuntime(provider, timeout_seconds=2.0)
    app_runtime = _TestAppRuntime(runtime, ToolCatalog(()))

    monkeypatch.setattr(main, "conversation_persistence", persistence)
    monkeypatch.setattr(main, "affect_service", affect_svc)
    monkeypatch.setattr(main, "provider", None)
    monkeypatch.setattr(main, "client", None)
    monkeypatch.setattr(main, "mcp_gemini_bridge", None)
    main.active_chat_sessions.clear()

    app = main.create_app(runtime_builder=lambda: app_runtime)
    user_id = "test-user-proj-fail"

    with TestClient(app) as client:
        resp = client.post(
            "/conversation",
            json={
                "user_id": user_id,
                "message": "Projection will fail, but ledger should commit.",
                "session_id": "sess-proj-001",
            },
        )
        assert resp.status_code == 200
        sim = resp.json()["emotional_state"]["simulation"]
        # Ledger committed, so state advanced to revision 1 (not uncommitted)
        assert sim["revision"] == 1
        assert sim["disposition"] != "uncommitted"
        assert affect_svc.get_state(user_id).revision == 1
        persisted_head = repo.get_affect_head(user_id)
        assert persisted_head is not None
        assert persisted_head.revision == 1
