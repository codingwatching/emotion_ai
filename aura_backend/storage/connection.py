"""Explicit SQLite connection and transaction ownership."""

from __future__ import annotations

import sqlite3
import hashlib
from collections.abc import Callable
from pathlib import Path

from aura_backend.storage.models import StorageFailure, TurnCommand
from aura_backend.storage.schema import apply_migrations

FaultHook = Callable[[str], None]


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


def append_turn_atomic(
    connection: sqlite3.Connection,
    command: TurnCommand,
    *,
    fault_hook: FaultHook | None = None,
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

        _fault(fault_hook, "before_commit")
        connection.commit()
    except StorageFailure:
        connection.rollback()
        raise
    except Exception as error:
        connection.rollback()
        raise StorageFailure("turn_write_failed", identifier=command.turn_id) from error
