"""Explicit SQLite connection and transaction ownership."""

from __future__ import annotations

import hashlib
import json
import math
import sqlite3
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

from aura_backend.storage.models import (
    EpistemicStatus,
    MemoryKind,
    StorageFailure,
    TurnCommand,
)
from aura_backend.storage.schema import apply_migrations

FaultHook = Callable[[str], None]
TransactionHook = Callable[[sqlite3.Connection], None]


def open_database(path: Path, *, busy_timeout_ms: int = 5_000) -> sqlite3.Connection:
    """Open and migrate one explicitly located ledger database."""
    if not path.is_absolute():
        raise StorageFailure("absolute_path_required")
    if busy_timeout_ms <= 0 or busy_timeout_ms > 300_000:
        raise StorageFailure("invalid_busy_timeout")
    if not path.parent.is_dir():
        raise StorageFailure("parent_directory_missing")
    try:
        connection = sqlite3.connect(
            path,
            timeout=busy_timeout_ms / 1_000,
            isolation_level=None,
        )
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute(f"PRAGMA busy_timeout = {busy_timeout_ms}")
        connection.execute("PRAGMA journal_mode = WAL")
        apply_migrations(connection)
    except sqlite3.Error as error:
        try:
            connection.close()
        except UnboundLocalError:
            pass
        raise StorageFailure("database_open_failed") from error
    return connection


def _fault(fault_hook: FaultHook | None, stage: str) -> None:
    if fault_hook is not None:
        fault_hook(stage)


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def _validate_memory_provenance(
    connection: sqlite3.Connection,
    command: TurnCommand,
    memory_index: int,
) -> None:
    memory = command.derived_memories[memory_index]
    if not isinstance(memory.kind, MemoryKind) or not isinstance(
        memory.epistemic_status, EpistemicStatus
    ):
        raise StorageFailure("invalid_memory_type", identifier=memory.memory_id)
    if not math.isfinite(memory.confidence) or not 0.0 <= memory.confidence <= 1.0:
        raise StorageFailure("invalid_confidence", identifier=memory.memory_id)
    if (
        not memory.source_event_ids
        or len(set(memory.source_event_ids)) != len(memory.source_event_ids)
        or memory.primary_source_event_id not in memory.source_event_ids
    ):
        raise StorageFailure("incomplete_provenance", identifier=memory.memory_id)

    primary = connection.execute(
        "SELECT actor FROM events WHERE event_id = ? AND scope_id = ?",
        (memory.primary_source_event_id, command.scope_id),
    ).fetchone()
    if primary is None or primary[0] != "user":
        raise StorageFailure("invalid_primary_source", identifier=memory.memory_id)
    for event_id in memory.source_event_ids:
        source = connection.execute(
            "SELECT 1 FROM events WHERE event_id = ? AND scope_id = ?",
            (event_id, command.scope_id),
        ).fetchone()
        if source is None:
            raise StorageFailure("cross_scope_source", identifier=memory.memory_id)


def append_turn_atomic(
    connection: sqlite3.Connection,
    command: TurnCommand,
    *,
    fault_hook: FaultHook | None = None,
    before_commit: TransactionHook | None = None,
) -> None:
    """Persist a complete turn or roll back every durable row."""
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT OR IGNORE INTO memory_scopes(scope_id, created_at) VALUES (?, ?)",
            (command.scope_id, command.occurred_at),
        )
        connection.execute(
            """
            INSERT OR IGNORE INTO sessions(session_id, scope_id, created_at)
            VALUES (?, ?, ?)
            """,
            (command.session_id, command.scope_id, command.occurred_at),
        )
        connection.execute(
            """
            INSERT INTO turns(
                turn_id, scope_id, session_id, idempotency_key,
                request_hash_version, request_hash, response_hash, occurred_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                command.turn_id,
                command.scope_id,
                command.session_id,
                command.idempotency_key,
                command.request_hash_version,
                command.request_hash,
                command.response_hash,
                command.occurred_at,
            ),
        )
        _fault(fault_hook, "after_turn")

        for ordinal, event in enumerate((command.user_event, command.aura_event)):
            expected_actor = "user" if ordinal == 0 else "aura"
            if event.actor != expected_actor:
                raise StorageFailure("invalid_event_actor", identifier=event.event_id)
            connection.execute(
                """
                INSERT INTO events(
                    event_id, scope_id, turn_id, ordinal, actor, observed_at,
                    content, payload_json, content_sha256, source_kind
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    event.event_id,
                    command.scope_id,
                    command.turn_id,
                    ordinal,
                    event.actor,
                    event.observed_at,
                    event.content,
                    event.payload_json,
                    event.content_sha256,
                    event.source_kind,
                ),
            )
            _fault(fault_hook, f"after_event_{ordinal}")

        for memory_index, memory in enumerate(command.derived_memories):
            _validate_memory_provenance(connection, command, memory_index)
            connection.execute(
                """
                INSERT INTO derived_memories(
                    memory_id, scope_id, memory_kind, canonical_text, confidence,
                    epistemic_status, primary_source_event_id, created_at,
                    content_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    memory.memory_id,
                    command.scope_id,
                    memory.kind.value,
                    memory.canonical_text,
                    memory.confidence,
                    memory.epistemic_status.value,
                    memory.primary_source_event_id,
                    memory.created_at,
                    _content_hash(memory.canonical_text),
                ),
            )
            _fault(fault_hook, f"after_derivation_{memory_index}")
            for source_index, event_id in enumerate(memory.source_event_ids):
                connection.execute(
                    """
                    INSERT INTO memory_sources(memory_id, event_id, scope_id, relation)
                    VALUES (?, ?, ?, 'support')
                    """,
                    (memory.memory_id, event_id, command.scope_id),
                )
                _fault(fault_hook, f"after_source_{memory_index}_{source_index}")

        if command.affect_transition is not None:
            trans = command.affect_transition
            cursor = connection.execute(
                "SELECT revision FROM affect_heads WHERE scope_id = ?",
                (command.scope_id,),
            )
            row = cursor.fetchone()
            current_rev = int(row[0]) if row else 0
            expected_rev = (
                command.expected_state_revision
                if command.expected_state_revision is not None
                else trans.prior_revision
            )
            if current_rev != expected_rev:
                raise StorageFailure(
                    "affect_revision_conflict",
                    identifier=f"{current_rev}!={expected_rev}",
                )

            mood_json = (
                json.dumps(asdict(trans.next_mood), sort_keys=True)
                if trans.next_mood
                else json.dumps(asdict(trans.after_state), sort_keys=True)
            )

            connection.execute(
                """
                INSERT INTO affect_transitions(
                    transition_id, scope_id, revision, turn_id, prior_revision,
                    idempotency_key, input_digest, appraisal_json, pre_state_json,
                    after_state_json, policy_json, outcome_disposition, config_hash,
                    created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    trans.transition_id,
                    command.scope_id,
                    trans.revision,
                    command.turn_id,
                    trans.prior_revision,
                    trans.idempotency_key,
                    trans.input_digest,
                    json.dumps(trans.accepted_appraisal, sort_keys=True),
                    json.dumps(asdict(trans.pre_state), sort_keys=True),
                    json.dumps(asdict(trans.after_state), sort_keys=True),
                    json.dumps(trans.rendered_policy.to_dict(), sort_keys=True),
                    trans.outcome_disposition,
                    trans.config_hash,
                    command.occurred_at,
                ),
            )
            _fault(fault_hook, "after_affect_transition")

            connection.execute(
                """
                INSERT INTO affect_heads(
                    scope_id, revision, config_version, config_hash,
                    fast_state_json, mood_state_json, last_transition_id, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(scope_id) DO UPDATE SET
                    revision=excluded.revision,
                    config_version=excluded.config_version,
                    config_hash=excluded.config_hash,
                    fast_state_json=excluded.fast_state_json,
                    mood_state_json=excluded.mood_state_json,
                    last_transition_id=excluded.last_transition_id,
                    updated_at=excluded.updated_at
                """,
                (
                    command.scope_id,
                    trans.revision,
                    "affect-v1",
                    trans.config_hash,
                    json.dumps(asdict(trans.after_state), sort_keys=True),
                    mood_json,
                    trans.transition_id,
                    command.occurred_at,
                ),
            )
            _fault(fault_hook, "after_affect_head")

        if before_commit is not None:
            before_commit(connection)
        _fault(fault_hook, "before_commit")
        connection.commit()
    except StorageFailure:
        connection.rollback()
        raise
    except Exception as error:
        connection.rollback()
        raise StorageFailure("turn_write_failed", identifier=command.turn_id) from error
