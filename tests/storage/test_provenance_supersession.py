"""Append-only provenance, correction, and retraction contracts."""

from __future__ import annotations

import sqlite3
import hashlib
from dataclasses import replace
from pathlib import Path

import pytest

from aura_backend.storage.models import (
    DerivedMemory,
    EpistemicStatus,
    MemoryKind,
    PersistedTurn,
    StorageFailure,
    TurnCommand,
)
from aura_backend.storage.repository import StorageRepository, canonical_request_hash


def _snapshot_memory(path: Path, memory_id: str) -> tuple[object, ...]:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT memory_id, scope_id, memory_kind, canonical_text, confidence,
                   epistemic_status, primary_source_event_id, created_at,
                   content_sha256
            FROM derived_memories WHERE memory_id = ?
            """,
            (memory_id,),
        ).fetchone()
        sources = connection.execute(
            "SELECT event_id, relation FROM memory_sources "
            "WHERE memory_id = ? ORDER BY event_id",
            (memory_id,),
        ).fetchall()
    assert row is not None
    return (*row, tuple(sources))


def _snapshot_event(path: Path, event_id: str) -> tuple[object, ...]:
    with sqlite3.connect(path) as connection:
        row = connection.execute(
            """
            SELECT event_id, scope_id, turn_id, ordinal, actor, observed_at,
                   content, payload_json, content_sha256, source_kind
            FROM events WHERE event_id = ?
            """,
            (event_id,),
        ).fetchone()
    assert row is not None
    return row


def _correction_command(original: TurnCommand) -> TurnCommand:
    user_content = "Synthetic corrected preference"
    aura_content = "Synthetic correction acknowledgement"
    user_event = replace(
        original.user_event,
        event_id="event-user-correction",
        content=user_content,
        content_sha256=hashlib.sha256(user_content.encode("utf-8")).hexdigest(),
        observed_at="2026-08-31T13:00:00Z",
    )
    aura_event = replace(
        original.aura_event,
        event_id="event-aura-correction",
        content=aura_content,
        content_sha256=hashlib.sha256(aura_content.encode("utf-8")).hexdigest(),
        observed_at="2026-08-31T13:00:01Z",
    )
    memory = replace(
        original.derived_memories[0],
        memory_id="memory-002",
        canonical_text="The synthetic user prefers detailed answers",
        confidence=0.95,
        epistemic_status=EpistemicStatus.OBSERVED,
        primary_source_event_id=user_event.event_id,
        source_event_ids=(user_event.event_id,),
        created_at="2026-08-31T13:00:02Z",
    )
    return replace(
        original,
        turn_id="turn-002",
        idempotency_key="request-002",
        request_hash=canonical_request_hash(
            original.scope_id,
            original.session_id,
            user_content,
            version=original.request_hash_version,
        ),
        response_hash=hashlib.sha256(aura_content.encode("utf-8")).hexdigest(),
        occurred_at="2026-08-31T13:00:00Z",
        user_event=user_event,
        aura_event=aura_event,
        derived_memories=(memory,),
    )


@pytest.mark.parametrize("kind", tuple(MemoryKind))
def test_every_memory_kind_requires_complete_typed_provenance(
    ledger_path: Path, turn_command: TurnCommand, kind: MemoryKind
) -> None:
    repository = StorageRepository(ledger_path)
    memory = replace(turn_command.derived_memories[0], kind=kind)
    outcome = repository.append_turn(
        turn_command.scope_id,
        replace(turn_command, derived_memories=(memory,)),
    )

    assert isinstance(outcome, PersistedTurn)
    current = repository.current_memories(turn_command.scope_id)
    assert current == (
        DerivedMemory(
            memory_id=memory.memory_id,
            scope_id=turn_command.scope_id,
            kind=kind,
            canonical_text=memory.canonical_text,
            confidence=memory.confidence,
            epistemic_status=memory.epistemic_status,
            primary_source_event_id=memory.primary_source_event_id,
            source_event_ids=tuple(sorted(memory.source_event_ids)),
            created_at=memory.created_at,
            content_sha256=current[0].content_sha256,
        ),
    )


@pytest.mark.parametrize(
    "invalid_memory",
    (
        lambda memory, command: replace(memory, source_event_ids=()),
        lambda memory, command: replace(
            memory, source_event_ids=(command.aura_event.event_id,)
        ),
        lambda memory, command: replace(
            memory, primary_source_event_id=command.aura_event.event_id
        ),
        lambda memory, command: replace(memory, confidence=-0.01),
        lambda memory, command: replace(memory, confidence=1.01),
        lambda memory, command: replace(memory, confidence=float("nan")),
        lambda memory, command: replace(
            memory,
            source_event_ids=(
                command.user_event.event_id,
                command.user_event.event_id,
            ),
        ),
        lambda memory, command: replace(memory, kind="unsupported"),
        lambda memory, command: replace(memory, epistemic_status="unsupported"),
    ),
)
def test_invalid_or_aura_authored_derivation_fails_without_partial_rows(
    ledger_path: Path,
    turn_command: TurnCommand,
    invalid_memory,
) -> None:
    repository = StorageRepository(ledger_path)
    memory = invalid_memory(turn_command.derived_memories[0], turn_command)

    with pytest.raises(StorageFailure):
        repository.append_turn(
            turn_command.scope_id,
            replace(turn_command, derived_memories=(memory,)),
        )

    with sqlite3.connect(ledger_path) as connection:
        counts = tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("turns", "events", "derived_memories", "memory_sources")
        )
    assert counts == (0, 0, 0, 0)


def test_cross_scope_derivation_source_fails_without_mutating_either_scope(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    other = replace(
        turn_command,
        scope_id="scope-beta",
        session_id="session-beta",
        turn_id="turn-beta",
        idempotency_key="request-beta",
        request_hash=canonical_request_hash(
            "scope-beta",
            "session-beta",
            turn_command.user_event.content,
            version=turn_command.request_hash_version,
        ),
        user_event=replace(turn_command.user_event, event_id="event-beta-user"),
        aura_event=replace(turn_command.aura_event, event_id="event-beta-aura"),
        derived_memories=(),
    )
    repository.append_turn(other.scope_id, other)
    invalid = replace(
        turn_command.derived_memories[0],
        primary_source_event_id=other.user_event.event_id,
        source_event_ids=(other.user_event.event_id,),
    )

    with pytest.raises(StorageFailure):
        repository.append_turn(
            turn_command.scope_id,
            replace(turn_command, derived_memories=(invalid,)),
        )

    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM derived_memories"
        ).fetchone()[0] == 0


def test_correction_appends_evidence_and_preserves_original_bytes(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    before = _snapshot_memory(ledger_path, "memory-001")
    event_before = _snapshot_event(ledger_path, turn_command.user_event.event_id)
    correction = _correction_command(turn_command)

    created = repository.correct_memory(
        turn_command.scope_id,
        correction,
        old_memory_id="memory-001",
        new_memory_id="memory-002",
        basis_event_id=correction.user_event.event_id,
        reason="explicit_user_correction",
        created_at="2026-08-31T13:00:03Z",
    )

    assert created.memory_id == "memory-002"
    assert _snapshot_memory(ledger_path, "memory-001") == before
    assert _snapshot_event(ledger_path, turn_command.user_event.event_id) == event_before
    assert tuple(memory.memory_id for memory in repository.current_memories(
        turn_command.scope_id
    )) == ("memory-002",)
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute(
            "SELECT old_memory_id, new_memory_id, basis_event_id "
            "FROM memory_supersessions"
        ).fetchall() == [
            ("memory-001", "memory-002", correction.user_event.event_id)
        ]


def test_retraction_is_edge_derived_and_does_not_mutate_memory(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    before = _snapshot_memory(ledger_path, "memory-001")

    repository.retract_memory(
        turn_command.scope_id,
        memory_id="memory-001",
        basis_event_id=turn_command.user_event.event_id,
        reason="explicit_withdrawal",
        created_at="2026-08-31T13:00:00Z",
    )

    assert repository.current_memories(turn_command.scope_id) == ()
    assert _snapshot_memory(ledger_path, "memory-001") == before


@pytest.mark.parametrize("case", ("self", "cycle", "cross_scope", "missing"))
def test_invalid_supersession_fails_closed(
    ledger_path: Path, turn_command: TurnCommand, case: str
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    correction = _correction_command(turn_command)
    repository.append_turn(correction.scope_id, correction)
    repository.supersede_memory(
        turn_command.scope_id,
        old_memory_id="memory-001",
        new_memory_id="memory-002",
        basis_event_id=correction.user_event.event_id,
        reason="first_edge",
        created_at="2026-08-31T13:00:03Z",
    )
    before = _snapshot_memory(ledger_path, "memory-001")

    if case == "self":
        arguments = ("memory-001", "memory-001", turn_command.user_event.event_id)
        scope_id = turn_command.scope_id
    elif case == "cycle":
        arguments = ("memory-002", "memory-001", correction.user_event.event_id)
        scope_id = turn_command.scope_id
    elif case == "cross_scope":
        other = replace(
            turn_command,
            scope_id="scope-beta",
            session_id="session-beta",
            turn_id="turn-beta",
            idempotency_key="request-beta",
            request_hash=canonical_request_hash(
                "scope-beta",
                "session-beta",
                turn_command.user_event.content,
                version=turn_command.request_hash_version,
            ),
            user_event=replace(turn_command.user_event, event_id="event-beta-user"),
            aura_event=replace(turn_command.aura_event, event_id="event-beta-aura"),
            derived_memories=(
                replace(
                    turn_command.derived_memories[0],
                    memory_id="memory-beta",
                    primary_source_event_id="event-beta-user",
                    source_event_ids=("event-beta-user",),
                ),
            ),
        )
        repository.append_turn(other.scope_id, other)
        arguments = ("memory-001", "memory-beta", other.user_event.event_id)
        scope_id = turn_command.scope_id
    else:
        arguments = ("memory-missing", "memory-002", correction.user_event.event_id)
        scope_id = turn_command.scope_id

    with pytest.raises(StorageFailure):
        repository.supersede_memory(
            scope_id,
            old_memory_id=arguments[0],
            new_memory_id=arguments[1],
            basis_event_id=arguments[2],
            reason="invalid_edge",
            created_at="2026-08-31T14:00:00Z",
        )

    assert _snapshot_memory(ledger_path, "memory-001") == before
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute(
            "SELECT count(*) FROM memory_supersessions"
        ).fetchone()[0] == 1


def test_failed_atomic_correction_leaves_no_new_event_or_memory(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    correction = _correction_command(turn_command)
    before = _snapshot_memory(ledger_path, "memory-001")

    with pytest.raises(StorageFailure):
        repository.correct_memory(
            turn_command.scope_id,
            correction,
            old_memory_id="memory-missing",
            new_memory_id="memory-002",
            basis_event_id=correction.user_event.event_id,
            reason="invalid_correction",
            created_at="2026-08-31T13:00:03Z",
        )

    assert _snapshot_memory(ledger_path, "memory-001") == before
    with sqlite3.connect(ledger_path) as connection:
        assert connection.execute("SELECT count(*) FROM turns").fetchone()[0] == 1
        assert connection.execute("SELECT count(*) FROM events").fetchone()[0] == 2
        assert connection.execute(
            "SELECT count(*) FROM derived_memories"
        ).fetchone()[0] == 1
