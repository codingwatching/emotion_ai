"""Sole durable read/write owner for Aura's SQLite event ledger."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from collections.abc import Callable
from pathlib import Path

from aura_backend.storage.connection import FaultHook, append_turn_atomic, open_database
from aura_backend.storage.models import (
    DerivedMemory,
    IdempotencyConflict,
    PersistedTurn,
    StorageFailure,
    TurnCommand,
    TurnWriteStatus,
)

TurnOutcome = PersistedTurn | IdempotencyConflict
TurnCallback = Callable[[PersistedTurn], None]


def canonical_request_hash(
    scope_id: str,
    session_id: str,
    user_content: str,
    *,
    version: int = 1,
) -> str:
    """Hash one versioned canonical request without logging its content."""
    canonical = json.dumps(
        {
            "request_hash_version": version,
            "scope_id": scope_id,
            "session_id": session_id,
            "user_content": user_content,
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class StorageRepository:
    """Explicit injected owner of one absolute SQLite ledger path."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path
        self._writer_lock = threading.Lock()

    def append_turn(
        self,
        scope_id: str,
        command: TurnCommand,
        *,
        provider_callback: Callable[[], None] | None = None,
        projection_callback: TurnCallback | None = None,
        reconciliation_callback: TurnCallback | None = None,
        fault_hook: FaultHook | None = None,
    ) -> TurnOutcome:
        """Append or replay one idempotent complete turn."""
        if scope_id != command.scope_id:
            raise StorageFailure("scope_mismatch", identifier=scope_id)

        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                existing = self._existing_outcome(connection, scope_id, command)
                if existing is not None:
                    if (
                        isinstance(existing, PersistedTurn)
                        and existing.projection_reconciliation_required
                        and reconciliation_callback is not None
                    ):
                        self._run_turn_callback(
                            reconciliation_callback,
                            existing,
                            code="reconciliation_callback_failed",
                        )
                    return existing

                if provider_callback is not None:
                    self._run_provider_callback(provider_callback, command.turn_id)

                try:
                    append_turn_atomic(connection, command, fault_hook=fault_hook)
                except StorageFailure as error:
                    if error.code != "turn_write_failed":
                        raise
                    raced = self._existing_outcome(connection, scope_id, command)
                    if raced is not None:
                        return raced
                    raise

                stored = self._load_persisted_turn(
                    connection,
                    scope_id,
                    command.idempotency_key,
                    status=TurnWriteStatus.STORED,
                )
                if fault_hook is not None:
                    fault_hook("after_commit")

                if projection_callback is not None:
                    self._run_turn_callback(
                        projection_callback,
                        stored,
                        code="projection_callback_failed",
                    )
                    connection.execute("BEGIN IMMEDIATE")
                    try:
                        connection.execute(
                            "UPDATE turns SET projection_status = 'complete' "
                            "WHERE turn_id = ? AND scope_id = ?",
                            (stored.turn_id, scope_id),
                        )
                        connection.commit()
                    except sqlite3.Error as error:
                        connection.rollback()
                        raise StorageFailure(
                            "projection_status_failed", identifier=stored.turn_id
                        ) from error
                    stored = self._load_persisted_turn(
                        connection,
                        scope_id,
                        command.idempotency_key,
                        status=TurnWriteStatus.STORED,
                    )
                return stored
            finally:
                connection.close()

    def current_memories(self, scope_id: str) -> tuple[DerivedMemory, ...]:
        """Return active provenance-bearing memories for one scope."""
        del scope_id
        raise StorageFailure("not_implemented")

    def supersede_memory(
        self,
        scope_id: str,
        *,
        old_memory_id: str,
        new_memory_id: str,
        basis_event_id: str,
        reason: str,
        created_at: str,
    ) -> None:
        """Append one scope-local acyclic supersession edge."""
        del (
            scope_id,
            old_memory_id,
            new_memory_id,
            basis_event_id,
            reason,
            created_at,
        )
        raise StorageFailure("not_implemented")

    def correct_memory(
        self,
        scope_id: str,
        command: TurnCommand,
        *,
        old_memory_id: str,
        new_memory_id: str,
        basis_event_id: str,
        reason: str,
        created_at: str,
    ) -> DerivedMemory:
        """Atomically append correction evidence, derivation, and edge."""
        del (
            scope_id,
            command,
            old_memory_id,
            new_memory_id,
            basis_event_id,
            reason,
            created_at,
        )
        raise StorageFailure("not_implemented")

    def retract_memory(
        self,
        scope_id: str,
        *,
        memory_id: str,
        basis_event_id: str,
        reason: str,
        created_at: str,
    ) -> None:
        """Append an explicit scope-local retraction edge."""
        del scope_id, memory_id, basis_event_id, reason, created_at
        raise StorageFailure("not_implemented")

    @staticmethod
    def _run_provider_callback(
        callback: Callable[[], None], turn_id: str
    ) -> None:
        try:
            callback()
        except Exception as error:
            raise StorageFailure(
                "provider_callback_failed", identifier=turn_id
            ) from error

    @staticmethod
    def _run_turn_callback(
        callback: TurnCallback,
        turn: PersistedTurn,
        *,
        code: str,
    ) -> None:
        try:
            callback(turn)
        except Exception as error:
            raise StorageFailure(code, identifier=turn.turn_id) from error

    def _existing_outcome(
        self,
        connection: sqlite3.Connection,
        scope_id: str,
        command: TurnCommand,
    ) -> TurnOutcome | None:
        row = connection.execute(
            """
            SELECT turn_id, request_hash_version, request_hash
            FROM turns
            WHERE scope_id = ? AND idempotency_key = ?
            """,
            (scope_id, command.idempotency_key),
        ).fetchone()
        if row is None:
            return None
        turn_id, hash_version, request_hash = row
        if (
            int(hash_version) != command.request_hash_version
            or str(request_hash) != command.request_hash
        ):
            return IdempotencyConflict(
                scope_id=scope_id,
                idempotency_key=command.idempotency_key,
                existing_turn_id=str(turn_id),
            )
        return self._load_persisted_turn(
            connection,
            scope_id,
            command.idempotency_key,
            status=TurnWriteStatus.REPLAYED,
        )

    @staticmethod
    def _load_persisted_turn(
        connection: sqlite3.Connection,
        scope_id: str,
        idempotency_key: str,
        *,
        status: TurnWriteStatus,
    ) -> PersistedTurn:
        row = connection.execute(
            """
            SELECT
                turn_id, session_id, request_hash_version, request_hash,
                response_hash, projection_status
            FROM turns
            WHERE scope_id = ? AND idempotency_key = ?
            """,
            (scope_id, idempotency_key),
        ).fetchone()
        if row is None:
            raise StorageFailure("turn_not_found", identifier=idempotency_key)
        turn_id, session_id, version, request_hash, response_hash, projection = row
        events = connection.execute(
            "SELECT event_id FROM events WHERE turn_id = ? ORDER BY ordinal",
            (turn_id,),
        ).fetchall()
        if len(events) != 2:
            raise StorageFailure("incomplete_turn", identifier=str(turn_id))
        return PersistedTurn(
            scope_id=scope_id,
            session_id=str(session_id),
            turn_id=str(turn_id),
            idempotency_key=idempotency_key,
            request_hash_version=int(version),
            request_hash=str(request_hash),
            response_hash=str(response_hash),
            user_event_id=str(events[0][0]),
            aura_event_id=str(events[1][0]),
            status=status,
            projection_reconciliation_required=projection != "complete",
        )
