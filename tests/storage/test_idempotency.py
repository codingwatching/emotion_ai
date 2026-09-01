"""Idempotency, retry, interruption, and writer-race contracts."""

from __future__ import annotations

import sqlite3
import threading
import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from aura_backend.storage.models import (
    IdempotencyConflict,
    PersistedTurn,
    StorageFailure,
    TurnCommand,
    TurnWriteStatus,
)
from aura_backend.storage.repository import StorageRepository, canonical_request_hash


def _row_counts(path: Path) -> tuple[int, int, int]:
    with sqlite3.connect(path) as connection:
        return tuple(
            connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
            for table in ("turns", "events", "derived_memories")
        )


def test_same_key_and_hash_replays_stable_ids_without_side_effects(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    provider_calls: list[str] = []
    projection_calls: list[str] = []

    first = repository.append_turn(
        turn_command.scope_id,
        turn_command,
        provider_callback=lambda: provider_calls.append("provider"),
        projection_callback=lambda turn: projection_calls.append(turn.turn_id),
    )
    replay = repository.append_turn(
        turn_command.scope_id,
        replace(
            turn_command,
            turn_id="unused-retry-turn",
            user_event=replace(turn_command.user_event, event_id="unused-retry-user"),
            aura_event=replace(turn_command.aura_event, event_id="unused-retry-aura"),
        ),
        provider_callback=lambda: provider_calls.append("replayed-provider"),
        projection_callback=lambda turn: projection_calls.append(
            f"replayed:{turn.turn_id}"
        ),
    )

    assert isinstance(first, PersistedTurn)
    assert isinstance(replay, PersistedTurn)
    assert first.status is TurnWriteStatus.STORED
    assert replay.status is TurnWriteStatus.REPLAYED
    assert replay.turn_id == first.turn_id == turn_command.turn_id
    assert replay.user_event_id == first.user_event_id == turn_command.user_event.event_id
    assert replay.aura_event_id == first.aura_event_id == turn_command.aura_event.event_id
    assert provider_calls == ["provider"]
    assert projection_calls == [turn_command.turn_id]
    assert _row_counts(ledger_path) == (1, 2, 1)


def test_same_key_with_changed_hash_conflicts_without_mutation(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    original = repository.append_turn(turn_command.scope_id, turn_command)
    before = _row_counts(ledger_path)
    changed_content = "A different synthetic request"
    conflict = repository.append_turn(
        turn_command.scope_id,
        replace(
            turn_command,
            turn_id="conflicting-turn",
            request_hash=canonical_request_hash(
                turn_command.scope_id,
                turn_command.session_id,
                changed_content,
                version=turn_command.request_hash_version,
            ),
            user_event=replace(
                turn_command.user_event,
                event_id="conflicting-user",
                content=changed_content,
                content_sha256=hashlib.sha256(
                    changed_content.encode("utf-8")
                ).hexdigest(),
            ),
        ),
    )

    assert isinstance(original, PersistedTurn)
    assert conflict == IdempotencyConflict(
        scope_id=turn_command.scope_id,
        idempotency_key=turn_command.idempotency_key,
        existing_turn_id=turn_command.turn_id,
    )
    assert _row_counts(ledger_path) == before
    with sqlite3.connect(ledger_path) as connection:
        hashes = connection.execute(
            "SELECT request_hash, response_hash FROM turns"
        ).fetchone()
    assert hashes == (turn_command.request_hash, turn_command.response_hash)


@pytest.mark.parametrize(
    "tamper",
    (
        lambda command: replace(command, request_hash="0" * 64),
        lambda command: replace(command, response_hash="0" * 64),
        lambda command: replace(
            command,
            user_event=replace(command.user_event, content_sha256="0" * 64),
        ),
        lambda command: replace(
            command,
            aura_event=replace(command.aura_event, content_sha256="0" * 64),
        ),
    ),
)
def test_tampered_command_hash_fails_before_any_durable_row(
    ledger_path: Path, turn_command: TurnCommand, tamper
) -> None:
    repository = StorageRepository(ledger_path)

    with pytest.raises(StorageFailure) as error:
        repository.append_turn(turn_command.scope_id, tamper(turn_command))

    assert error.value.code == "command_hash_mismatch"
    assert not ledger_path.exists()


def test_post_commit_interruption_replays_and_schedules_reconciliation(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)

    def interrupt(stage: str) -> None:
        if stage == "after_commit":
            raise StorageFailure("post_commit_interrupted", identifier=turn_command.turn_id)

    with pytest.raises(StorageFailure) as error:
        repository.append_turn(
            turn_command.scope_id,
            turn_command,
            fault_hook=interrupt,
        )
    assert error.value.code == "post_commit_interrupted"
    assert _row_counts(ledger_path) == (1, 2, 1)

    reconciled: list[str] = []
    replay = repository.append_turn(
        turn_command.scope_id,
        turn_command,
        reconciliation_callback=lambda turn: reconciled.append(turn.turn_id),
    )
    assert isinstance(replay, PersistedTurn)
    assert replay.status is TurnWriteStatus.REPLAYED
    assert replay.projection_reconciliation_required is True
    assert reconciled == [turn_command.turn_id]
    assert _row_counts(ledger_path) == (1, 2, 1)


def test_concurrent_same_key_calls_converge_to_one_complete_turn(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    provider_count = 0
    count_lock = threading.Lock()

    def provider_effect() -> None:
        nonlocal provider_count
        with count_lock:
            provider_count += 1

    def append(index: int):
        command = replace(
            turn_command,
            turn_id=f"turn-race-{index}",
            user_event=replace(turn_command.user_event, event_id=f"user-race-{index}"),
            aura_event=replace(turn_command.aura_event, event_id=f"aura-race-{index}"),
            derived_memories=(),
        )
        return repository.append_turn(
            command.scope_id,
            command,
            provider_callback=provider_effect,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        outcomes = list(executor.map(append, range(8)))

    persisted = [outcome for outcome in outcomes if isinstance(outcome, PersistedTurn)]
    assert len(persisted) == 8
    assert sum(turn.status is TurnWriteStatus.STORED for turn in persisted) == 1
    assert sum(turn.status is TurnWriteStatus.REPLAYED for turn in persisted) == 7
    assert len({turn.turn_id for turn in persisted}) == 1
    assert provider_count == 1
    assert _row_counts(ledger_path) == (1, 2, 0)


def test_scope_argument_cannot_be_replaced_by_command_metadata(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    repository = StorageRepository(ledger_path)
    with pytest.raises(StorageFailure) as error:
        repository.append_turn("scope-beta", turn_command)
    assert error.value.code == "scope_mismatch"
    assert not ledger_path.exists()
