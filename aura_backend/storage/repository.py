"""Sole durable read/write owner for Aura's SQLite event ledger."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
import threading
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from aura_backend.storage.connection import FaultHook, append_turn_atomic, open_database
from aura_backend.storage.models import (
    DerivedMemory,
    EpistemicStatus,
    IdempotencyConflict,
    MemoryKind,
    PersistedTurn,
    StorageFailure,
    TurnCommand,
    TurnWriteStatus,
)
from aura_backend.storage.schema import SCHEMA_VERSION

TurnOutcome = PersistedTurn | IdempotencyConflict
TurnCallback = Callable[[PersistedTurn], None]


@dataclass(frozen=True, slots=True)
class ProjectionOrigin:
    """One active SQLite origin eligible for a derived vector projection."""

    projection_id: str
    origin_kind: str
    origin_id: str
    scope_id: str
    content: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class ProjectionGenerationRecord:
    """SQLite-owned lifecycle state for one disposable projection build."""

    generation_id: str
    embedding_model: str
    config_sha256: str
    metric: str
    sqlite_watermark: int
    build_status: str
    created_at: str
    completed_at: str | None


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
        self._validate_command_hashes(command)

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
        connection = open_database(self.database_path)
        try:
            rows = connection.execute(
                """
                SELECT memory_id
                FROM derived_memories AS memory
                WHERE memory.scope_id = ?
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_supersessions AS edge
                      WHERE edge.scope_id = memory.scope_id
                        AND edge.old_memory_id = memory.memory_id
                  )
                  AND NOT EXISTS (
                      SELECT 1 FROM memory_retractions AS retraction
                      WHERE retraction.scope_id = memory.scope_id
                        AND retraction.memory_id = memory.memory_id
                  )
                ORDER BY memory.created_at, memory.memory_id
                """,
                (scope_id,),
            ).fetchall()
            return tuple(
                self._load_memory(connection, scope_id, str(row[0])) for row in rows
            )
        finally:
            connection.close()

    def projection_origins(
        self,
        *,
        after_projection_id: str = "",
        limit: int = 100,
        turn_id: str | None = None,
    ) -> tuple[ProjectionOrigin, ...]:
        """Page committed events and active provenance-bearing memories."""
        if limit <= 0 or limit > 10_000:
            raise StorageFailure("invalid_projection_page_size")
        connection = open_database(self.database_path)
        try:
            turn_clause = "AND event.turn_id = ?" if turn_id is not None else ""
            memory_turn_clause = (
                "AND EXISTS (SELECT 1 FROM memory_sources AS turn_source "
                "JOIN events AS turn_event ON turn_event.event_id = turn_source.event_id "
                "WHERE turn_source.memory_id = memory.memory_id "
                "AND turn_event.turn_id = ?)"
                if turn_id is not None
                else ""
            )
            parameters: list[object] = []
            if turn_id is not None:
                parameters.append(turn_id)
            if turn_id is not None:
                parameters.append(turn_id)
            parameters.extend((after_projection_id, limit))
            rows = connection.execute(
                f"""
                WITH active_origins AS (
                    SELECT
                        'event:' || event.event_id AS projection_id,
                        'event' AS origin_kind,
                        event.event_id AS origin_id,
                        event.scope_id,
                        event.content,
                        event.content_sha256
                    FROM events AS event
                    JOIN turns AS turn ON turn.turn_id = event.turn_id
                    WHERE 1 = 1 {turn_clause}
                    UNION ALL
                    SELECT
                        'memory:' || memory.memory_id AS projection_id,
                        'memory' AS origin_kind,
                        memory.memory_id AS origin_id,
                        memory.scope_id,
                        memory.canonical_text AS content,
                        memory.content_sha256
                    FROM derived_memories AS memory
                    WHERE EXISTS (
                        SELECT 1 FROM memory_sources AS source
                        WHERE source.memory_id = memory.memory_id
                          AND source.scope_id = memory.scope_id
                    )
                      AND NOT EXISTS (
                        SELECT 1 FROM memory_supersessions AS edge
                        WHERE edge.old_memory_id = memory.memory_id
                          AND edge.scope_id = memory.scope_id
                    )
                      AND NOT EXISTS (
                        SELECT 1 FROM memory_retractions AS retraction
                        WHERE retraction.memory_id = memory.memory_id
                          AND retraction.scope_id = memory.scope_id
                    )
                      {memory_turn_clause}
                )
                SELECT projection_id, origin_kind, origin_id, scope_id,
                       content, content_sha256
                FROM active_origins
                WHERE projection_id > ?
                ORDER BY projection_id
                LIMIT ?
                """,
                parameters,
            ).fetchall()
            return tuple(ProjectionOrigin(*map(str, row)) for row in rows)
        finally:
            connection.close()

    def projection_origin(self, projection_id: str) -> ProjectionOrigin | None:
        """Resolve one projection ID only if its SQLite origin remains active."""
        if ":" not in projection_id:
            return None
        origin_kind, origin_id = projection_id.split(":", 1)
        if origin_kind not in {"event", "memory"} or not origin_id:
            return None
        after = ""
        while True:
            page = self.projection_origins(after_projection_id=after, limit=1_000)
            if not page:
                return None
            match = next(
                (item for item in page if item.projection_id == projection_id), None
            )
            if match is not None:
                return match
            if page[-1].projection_id > projection_id:
                return None
            after = page[-1].projection_id

    def projection_origin_ids(self) -> tuple[str, ...]:
        """Return all active stable projection IDs in deterministic order."""
        identities: list[str] = []
        after = ""
        while True:
            page = self.projection_origins(after_projection_id=after, limit=1_000)
            if not page:
                return tuple(identities)
            identities.extend(item.projection_id for item in page)
            after = page[-1].projection_id

    def pending_projection_turn_ids(self) -> tuple[str, ...]:
        """Return committed turns whose projection acknowledgement is pending."""
        connection = open_database(self.database_path)
        try:
            return tuple(
                str(row[0])
                for row in connection.execute(
                    "SELECT turn_id FROM turns WHERE projection_status = 'pending' "
                    "ORDER BY occurred_at, turn_id"
                ).fetchall()
            )
        finally:
            connection.close()

    def mark_turn_projection_complete(self, turn_id: str) -> None:
        """Acknowledge a turn only after every eligible origin was upserted."""
        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "UPDATE turns SET projection_status = 'complete' "
                    "WHERE turn_id = ? AND projection_status = 'pending'",
                    (turn_id,),
                )
                if cursor.rowcount not in {0, 1}:
                    raise StorageFailure(
                        "projection_status_failed", identifier=turn_id
                    )
                connection.commit()
            except StorageFailure:
                connection.rollback()
                raise
            except sqlite3.Error as error:
                connection.rollback()
                raise StorageFailure(
                    "projection_status_failed", identifier=turn_id
                ) from error
            finally:
                connection.close()

    def sqlite_projection_watermark(self) -> int:
        """Return a monotonic-enough rebuild watermark from canonical row IDs."""
        connection = open_database(self.database_path)
        try:
            event_max = int(
                connection.execute(
                    "SELECT COALESCE(MAX(event_pk), 0) FROM events"
                ).fetchone()[0]
            )
            memory_max = int(
                connection.execute(
                    "SELECT COALESCE(MAX(memory_pk), 0) FROM derived_memories"
                ).fetchone()[0]
            )
            return (event_max << 32) | memory_max
        finally:
            connection.close()

    def begin_projection_generation(
        self,
        *,
        generation_id: str,
        embedding_model: str,
        config_sha256: str,
        created_at: str,
    ) -> None:
        """Record a fresh building generation before derived files are created."""
        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    """
                    INSERT INTO projection_generations(
                        generation_id, embedding_model, config_sha256, metric,
                        sqlite_watermark, build_status, created_at
                    ) VALUES (?, ?, ?, 'cosine', ?, 'building', ?)
                    """,
                    (
                        generation_id,
                        embedding_model,
                        config_sha256,
                        self.sqlite_projection_watermark(),
                        created_at,
                    ),
                )
                connection.commit()
            except sqlite3.Error as error:
                connection.rollback()
                raise StorageFailure(
                    "projection_generation_start_failed", identifier=generation_id
                ) from error
            finally:
                connection.close()

    def finish_projection_generation(
        self,
        generation_id: str,
        *,
        status: str,
        completed_at: str,
    ) -> None:
        """Mark a build ready or failed; only a verified build may be ready."""
        if status not in {"ready", "failed"}:
            raise StorageFailure(
                "invalid_projection_generation_status", identifier=generation_id
            )
        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                cursor = connection.execute(
                    "UPDATE projection_generations SET build_status = ?, "
                    "completed_at = ? WHERE generation_id = ? "
                    "AND build_status = 'building'",
                    (status, completed_at, generation_id),
                )
                if cursor.rowcount != 1:
                    raise StorageFailure(
                        "projection_generation_transition_failed",
                        identifier=generation_id,
                    )
                connection.commit()
            except StorageFailure:
                connection.rollback()
                raise
            except sqlite3.Error as error:
                connection.rollback()
                raise StorageFailure(
                    "projection_generation_transition_failed",
                    identifier=generation_id,
                ) from error
            finally:
                connection.close()

    def projection_generation(
        self, generation_id: str
    ) -> ProjectionGenerationRecord:
        """Return one SQLite-owned generation record."""
        connection = open_database(self.database_path)
        try:
            row = connection.execute(
                """
                SELECT generation_id, embedding_model, config_sha256, metric,
                       sqlite_watermark, build_status, created_at, completed_at
                FROM projection_generations WHERE generation_id = ?
                """,
                (generation_id,),
            ).fetchone()
            if row is None:
                raise StorageFailure(
                    "projection_generation_not_found", identifier=generation_id
                )
            return ProjectionGenerationRecord(
                generation_id=str(row[0]),
                embedding_model=str(row[1]),
                config_sha256=str(row[2]),
                metric=str(row[3]),
                sqlite_watermark=int(row[4]),
                build_status=str(row[5]),
                created_at=str(row[6]),
                completed_at=None if row[7] is None else str(row[7]),
            )
        finally:
            connection.close()

    def current_projection_generation(self) -> ProjectionGenerationRecord | None:
        """Return the newest verified generation; failed builds never switch."""
        connection = open_database(self.database_path)
        try:
            row = connection.execute(
                """
                SELECT generation_id FROM projection_generations
                WHERE build_status = 'ready'
                ORDER BY completed_at DESC, generation_id DESC LIMIT 1
                """
            ).fetchone()
        finally:
            connection.close()
        return None if row is None else self.projection_generation(str(row[0]))

    @property
    def schema_version(self) -> int:
        """Expose the owned SQLite schema version to projection metadata."""
        return SCHEMA_VERSION

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
        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                self._insert_supersession(
                    connection,
                    scope_id,
                    old_memory_id=old_memory_id,
                    new_memory_id=new_memory_id,
                    basis_event_id=basis_event_id,
                    reason=reason,
                    created_at=created_at,
                )
                connection.commit()
            except StorageFailure:
                connection.rollback()
                raise
            except sqlite3.Error as error:
                connection.rollback()
                raise StorageFailure(
                    "supersession_write_failed", identifier=old_memory_id
                ) from error
            finally:
                connection.close()

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
        if scope_id != command.scope_id:
            raise StorageFailure("scope_mismatch", identifier=scope_id)
        self._validate_command_hashes(command)
        if sum(
            memory.memory_id == new_memory_id
            for memory in command.derived_memories
        ) != 1:
            raise StorageFailure("correction_memory_missing", identifier=new_memory_id)

        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                append_turn_atomic(
                    connection,
                    command,
                    before_commit=lambda active: self._insert_supersession(
                        active,
                        scope_id,
                        old_memory_id=old_memory_id,
                        new_memory_id=new_memory_id,
                        basis_event_id=basis_event_id,
                        reason=reason,
                        created_at=created_at,
                    ),
                )
                return self._load_memory(connection, scope_id, new_memory_id)
            finally:
                connection.close()

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
        if not reason:
            raise StorageFailure("retraction_reason_required", identifier=memory_id)
        with self._writer_lock:
            connection = open_database(self.database_path)
            try:
                connection.execute("BEGIN IMMEDIATE")
                memory_scope = connection.execute(
                    "SELECT scope_id FROM derived_memories WHERE memory_id = ?",
                    (memory_id,),
                ).fetchone()
                event_scope = connection.execute(
                    "SELECT scope_id FROM events WHERE event_id = ?",
                    (basis_event_id,),
                ).fetchone()
                if (
                    memory_scope is None
                    or event_scope is None
                    or memory_scope[0] != scope_id
                    or event_scope[0] != scope_id
                ):
                    raise StorageFailure("invalid_retraction", identifier=memory_id)
                connection.execute(
                    """
                    INSERT INTO memory_retractions(
                        memory_id, basis_event_id, scope_id, reason, created_at
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (memory_id, basis_event_id, scope_id, reason, created_at),
                )
                connection.commit()
            except StorageFailure:
                connection.rollback()
                raise
            except sqlite3.Error as error:
                connection.rollback()
                raise StorageFailure(
                    "retraction_write_failed", identifier=memory_id
                ) from error
            finally:
                connection.close()

    @staticmethod
    def _insert_supersession(
        connection: sqlite3.Connection,
        scope_id: str,
        *,
        old_memory_id: str,
        new_memory_id: str,
        basis_event_id: str,
        reason: str,
        created_at: str,
    ) -> None:
        if old_memory_id == new_memory_id or not reason:
            raise StorageFailure("invalid_supersession", identifier=old_memory_id)
        memory_scopes = connection.execute(
            """
            SELECT memory_id, scope_id FROM derived_memories
            WHERE memory_id IN (?, ?)
            """,
            (old_memory_id, new_memory_id),
        ).fetchall()
        event_scope = connection.execute(
            "SELECT scope_id FROM events WHERE event_id = ?",
            (basis_event_id,),
        ).fetchone()
        if (
            len(memory_scopes) != 2
            or any(row[1] != scope_id for row in memory_scopes)
            or event_scope is None
            or event_scope[0] != scope_id
        ):
            raise StorageFailure("invalid_supersession", identifier=old_memory_id)

        cycle = connection.execute(
            """
            WITH RECURSIVE path(memory_id) AS (
                SELECT new_memory_id
                FROM memory_supersessions
                WHERE old_memory_id = ? AND scope_id = ?
                UNION
                SELECT edge.new_memory_id
                FROM memory_supersessions AS edge
                JOIN path ON edge.old_memory_id = path.memory_id
                WHERE edge.scope_id = ?
            )
            SELECT 1 FROM path WHERE memory_id = ? LIMIT 1
            """,
            (new_memory_id, scope_id, scope_id, old_memory_id),
        ).fetchone()
        if cycle is not None:
            raise StorageFailure("supersession_cycle", identifier=old_memory_id)
        connection.execute(
            """
            INSERT INTO memory_supersessions(
                old_memory_id, new_memory_id, basis_event_id,
                scope_id, reason, created_at
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                old_memory_id,
                new_memory_id,
                basis_event_id,
                scope_id,
                reason,
                created_at,
            ),
        )

    @staticmethod
    def _load_memory(
        connection: sqlite3.Connection,
        scope_id: str,
        memory_id: str,
    ) -> DerivedMemory:
        row = connection.execute(
            """
            SELECT memory_kind, canonical_text, confidence, epistemic_status,
                   primary_source_event_id, created_at, content_sha256
            FROM derived_memories
            WHERE memory_id = ? AND scope_id = ?
            """,
            (memory_id, scope_id),
        ).fetchone()
        if row is None:
            raise StorageFailure("memory_not_found", identifier=memory_id)
        sources = tuple(
            str(source[0])
            for source in connection.execute(
                "SELECT event_id FROM memory_sources "
                "WHERE memory_id = ? AND scope_id = ? ORDER BY event_id",
                (memory_id, scope_id),
            ).fetchall()
        )
        if not sources or str(row[4]) not in sources:
            raise StorageFailure("invalid_provenance", identifier=memory_id)
        return DerivedMemory(
            memory_id=memory_id,
            scope_id=scope_id,
            kind=MemoryKind(str(row[0])),
            canonical_text=str(row[1]),
            confidence=float(row[2]),
            epistemic_status=EpistemicStatus(str(row[3])),
            primary_source_event_id=str(row[4]),
            source_event_ids=sources,
            created_at=str(row[5]),
            content_sha256=str(row[6]),
        )

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
    def _validate_command_hashes(command: TurnCommand) -> None:
        expected_request = canonical_request_hash(
            command.scope_id,
            command.session_id,
            command.user_event.content,
            version=command.request_hash_version,
        )
        expected_response = hashlib.sha256(
            command.aura_event.content.encode("utf-8")
        ).hexdigest()
        expected_user_event = hashlib.sha256(
            command.user_event.content.encode("utf-8")
        ).hexdigest()
        expected_aura_event = hashlib.sha256(
            command.aura_event.content.encode("utf-8")
        ).hexdigest()
        supplied_and_expected = (
            (command.request_hash, expected_request),
            (command.response_hash, expected_response),
            (command.user_event.content_sha256, expected_user_event),
            (command.aura_event.content_sha256, expected_aura_event),
        )
        if command.request_hash_version <= 0 or any(
            not hmac.compare_digest(supplied, expected)
            for supplied, expected in supplied_and_expected
        ):
            raise StorageFailure(
                "command_hash_mismatch", identifier=command.idempotency_key
            )

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
