"""SQLite-first conversation persistence with rebuildable projection recovery.

The service keeps Aura's characterized asynchronous method names and result keys,
but canonical truth is committed by :class:`StorageRepository`.  Chroma receives
only a post-commit projection update and can be reconciled idempotently.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import logging
import uuid
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, is_dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Protocol

from aura_backend.storage.models import (
    EventInput,
    IdempotencyConflict,
    PersistedTurn,
    StorageFailure,
    TurnCommand,
    TurnWriteStatus,
)
from aura_backend.storage.repository import StorageRepository, canonical_request_hash

logger = logging.getLogger(__name__)


class ProjectionWriter(Protocol):
    """Narrow rebuildable projection seam used after the ledger commit."""

    def upsert_committed(self, turn_id: str) -> int: ...


class LegacyReadAdapter(Protocol):
    """Explicit read-only compatibility seam for unswitched historical data."""

    async def search_conversations(
        self, *, query: str, user_id: str, n_results: int
    ) -> Sequence[Mapping[str, Any]]: ...


@dataclass(slots=True)
class ConversationExchange:
    """One normalized user/Aura exchange with one stable retry identity."""

    user_memory: Any
    ai_memory: Any
    user_emotional_state: Any | None = None
    ai_emotional_state: Any | None = None
    ai_cognitive_state: Any | None = None
    session_id: str = ""
    timestamp: datetime | None = None
    idempotency_key: str | None = None
    affect_transition: Any | None = None
    expected_state_revision: int | None = None

    def __post_init__(self) -> None:
        if self.timestamp is None:
            self.timestamp = datetime.now(UTC)
        elif self.timestamp.tzinfo is None:
            self.timestamp = self.timestamp.replace(tzinfo=UTC)
        if not self.session_id:
            candidate = getattr(self.user_memory, "session_id", None)
            if isinstance(candidate, str):
                self.session_id = candidate
        if not self.idempotency_key:
            self.idempotency_key = uuid.uuid4().hex


class ConversationPersistenceService:
    """Compatibility service over one atomic ledger and post-commit projection."""

    def __init__(
        self,
        repository: StorageRepository,
        projection: ProjectionWriter,
        *,
        legacy_reader: LegacyReadAdapter | Any | None = None,
        id_factory: Callable[[], str] | None = None,
    ) -> None:
        if not isinstance(repository, StorageRepository):
            raise TypeError("repository must be StorageRepository")
        if not callable(getattr(projection, "upsert_committed", None)):
            raise TypeError("projection must provide upsert_committed")
        self.repository = repository
        self.projection = projection
        self.legacy_reader = legacy_reader
        self._id_factory = id_factory or (lambda: uuid.uuid4().hex)
        self._metrics: dict[str, Any] = {
            "total_exchanges_stored": 0,
            "failed_stores": 0,
            "average_store_time": 0.0,
            "last_error": None,
            "retries_performed": 0,
            "backups_created": 0,
            "cleanups_performed": 0,
            "archives_created": 0,
            "failed_operations_queued": 0,
        }
        self._event_callbacks: dict[str, list[Callable[..., Any]]] = {
            "exchange_stored": [],
            "storage_failed": [],
        }

    async def persist_conversation_exchange(
        self,
        exchange: ConversationExchange,
        update_profile: bool = True,
    ) -> dict[str, Any]:
        """Commit or replay one turn, then reconcile its disposable projection."""
        del update_profile
        started = asyncio.get_running_loop().time()
        command = self._turn_command(exchange)
        result = await asyncio.to_thread(self._persist_command, command)
        result["duration_ms"] = (
            asyncio.get_running_loop().time() - started
        ) * 1000.0
        return result

    async def persist_conversation_exchange_immediate(
        self,
        exchange: ConversationExchange,
        update_profile: bool = True,
        timeout: float = 30.0,
    ) -> dict[str, Any]:
        """Run the same idempotent command behind the characterized timeout seam."""
        if timeout <= 0:
            return self._failure_result(
                exchange,
                code="invalid_persistence_timeout",
                method="immediate_invalid_timeout",
            )
        started = asyncio.get_running_loop().time()
        try:
            result = await asyncio.wait_for(
                self.persist_conversation_exchange(exchange, update_profile),
                timeout=timeout,
            )
        except TimeoutError:
            result = self._failure_result(
                exchange,
                code="persistence_timeout",
                method="immediate_timeout",
            )
        result["duration_ms"] = (
            asyncio.get_running_loop().time() - started
        ) * 1000.0
        result["method"] = result.get("method", "immediate_sqlite")
        return result

    def _persist_command(self, command: TurnCommand) -> dict[str, Any]:
        try:
            self.repository.database_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._record_failure("ledger_directory_unavailable")
            return self._failure_result_from_command(
                command, code="ledger_directory_unavailable"
            )
        try:
            outcome = self.repository.append_turn(
                command.scope_id,
                command,
            )
        except StorageFailure as error:
            self._record_failure(error.code)
            return self._failure_result_from_command(command, code=error.code)

        if isinstance(outcome, IdempotencyConflict):
            self._record_failure(outcome.code)
            return self._conflict_result(outcome)

        write_status = outcome.status
        if outcome.projection_reconciliation_required:
            try:
                self._project(outcome)
            except Exception:
                self._record_failure("projection_callback_failed")
                return self._result(
                    outcome,
                    success=False,
                    status="projection_pending",
                    projection_status="pending",
                    errors=["projection_callback_failed"],
                )
            refreshed = self.repository.append_turn(command.scope_id, command)
            if isinstance(refreshed, PersistedTurn):
                outcome = refreshed

        self._record_success(write_status)
        return self._result(
            outcome,
            success=True,
            status=write_status.value,
            projection_status=(
                "pending" if outcome.projection_reconciliation_required else "complete"
            ),
            errors=[],
        )

    def _project(self, turn: PersistedTurn) -> None:
        self.projection.upsert_committed(turn.turn_id)

    def _turn_command(self, exchange: ConversationExchange) -> TurnCommand:
        scope_id = self._required_text(exchange.user_memory, "user_id")
        if self._required_text(exchange.ai_memory, "user_id") != scope_id:
            raise StorageFailure("scope_mismatch", identifier=scope_id)
        session_id = exchange.session_id or self._required_text(
            exchange.user_memory, "session_id"
        )
        user_content = self._required_text(exchange.user_memory, "message")
        aura_content = self._required_text(exchange.ai_memory, "message")
        occurred_at = self._iso_timestamp(exchange.timestamp)
        idempotency_key = exchange.idempotency_key
        if not isinstance(idempotency_key, str) or not idempotency_key:
            raise StorageFailure("idempotency_key_missing")
        user_event_id = self._id_factory()
        aura_event_id = self._id_factory()
        return TurnCommand(
            scope_id=scope_id,
            session_id=session_id,
            turn_id=self._id_factory(),
            idempotency_key=idempotency_key,
            request_hash_version=1,
            request_hash=canonical_request_hash(scope_id, session_id, user_content),
            response_hash=self._sha256(aura_content),
            occurred_at=occurred_at,
            user_event=EventInput(
                event_id=user_event_id,
                actor="user",
                content=user_content,
                observed_at=occurred_at,
                content_sha256=self._sha256(user_content),
                payload_json=self._payload_json(
                    emotional_state=exchange.user_emotional_state
                    or getattr(exchange.user_memory, "emotional_state", None)
                ),
            ),
            aura_event=EventInput(
                event_id=aura_event_id,
                actor="aura",
                content=aura_content,
                observed_at=occurred_at,
                content_sha256=self._sha256(aura_content),
                payload_json=self._payload_json(
                    emotional_state=exchange.ai_emotional_state
                    or getattr(exchange.ai_memory, "emotional_state", None),
                    cognitive_state=exchange.ai_cognitive_state
                    or getattr(exchange.ai_memory, "cognitive_state", None),
                ),
            ),
            affect_transition=exchange.affect_transition,
            expected_state_revision=exchange.expected_state_revision,
        )

    @staticmethod
    def _required_text(value: Any, name: str) -> str:
        observed = getattr(value, name, None)
        if not isinstance(observed, str) or not observed:
            raise StorageFailure("invalid_exchange", identifier=name)
        return observed

    @staticmethod
    def _iso_timestamp(value: datetime | None) -> str:
        observed = value or datetime.now(UTC)
        if observed.tzinfo is None:
            observed = observed.replace(tzinfo=UTC)
        return observed.astimezone(UTC).isoformat().replace("+00:00", "Z")

    @staticmethod
    def _sha256(value: str) -> str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    @staticmethod
    def _payload_json(**values: Any) -> str:
        def portable(value: Any) -> Any:
            if value is None:
                return None
            if isinstance(value, Enum):
                return value.value
            if is_dataclass(value) and not isinstance(value, type):
                return portable(asdict(value))
            if isinstance(value, Mapping):
                return {str(key): portable(item) for key, item in value.items()}
            if isinstance(value, (list, tuple)):
                return [portable(item) for item in value]
            if isinstance(value, (str, int, float, bool)):
                return value
            return str(value)

        payload = {key: portable(value) for key, value in values.items() if value is not None}
        return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)

    def _result(
        self,
        turn: PersistedTurn,
        *,
        success: bool,
        status: str,
        projection_status: str,
        errors: list[str],
    ) -> dict[str, Any]:
        return {
            "success": success,
            "stored_components": ["turn", "user_event", "aura_event"] if success else [],
            "errors": errors,
            "duration_ms": 0.0,
            "retry_count": 0,
            "status": status,
            "durable_status": "stored",
            "projection_status": projection_status,
            "idempotency_key": turn.idempotency_key,
            "retry_identity": turn.idempotency_key,
            "turn_id": turn.turn_id,
        }

    def _conflict_result(self, conflict: IdempotencyConflict) -> dict[str, Any]:
        return {
            "success": False,
            "stored_components": [],
            "errors": [conflict.code],
            "duration_ms": 0.0,
            "retry_count": 0,
            "status": conflict.code,
            "durable_status": "unchanged",
            "projection_status": "unchanged",
            "idempotency_key": conflict.idempotency_key,
            "retry_identity": conflict.idempotency_key,
            "turn_id": conflict.existing_turn_id,
        }

    def _failure_result(
        self,
        exchange: ConversationExchange,
        *,
        code: str,
        method: str,
    ) -> dict[str, Any]:
        key = exchange.idempotency_key or "unavailable"
        return {
            "success": False,
            "stored_components": [],
            "errors": [code],
            "duration_ms": 0.0,
            "retry_count": 0,
            "method": method,
            "status": code,
            "durable_status": "unknown",
            "projection_status": "unknown",
            "idempotency_key": key,
            "retry_identity": key,
            "turn_id": None,
        }

    def _failure_result_from_command(
        self, command: TurnCommand, *, code: str
    ) -> dict[str, Any]:
        return {
            "success": False,
            "stored_components": [],
            "errors": [code],
            "duration_ms": 0.0,
            "retry_count": 0,
            "status": code,
            "durable_status": "unknown",
            "projection_status": "unknown",
            "idempotency_key": command.idempotency_key,
            "retry_identity": command.idempotency_key,
            "turn_id": None,
        }

    def _record_success(self, status: TurnWriteStatus) -> None:
        if status is TurnWriteStatus.REPLAYED:
            self._metrics["retries_performed"] += 1
        else:
            self._metrics["total_exchanges_stored"] += 1
        self._metrics["last_error"] = None

    def _record_failure(self, code: str) -> None:
        self._metrics["failed_stores"] += 1
        self._metrics["last_error"] = code

    async def safe_search_conversations(
        self, query: str, user_id: str, n_results: int = 20
    ) -> list[dict[str, Any]]:
        """Read only through the explicit legacy adapter before owner switch."""
        limit = min(max(int(n_results), 1), 100)
        result = await self._legacy_call(
            "search_conversations",
            query=query,
            user_id=user_id,
            n_results=limit,
            default=[],
        )
        return [dict(item) for item in result] if isinstance(result, Sequence) else []

    async def safe_get_chat_history(
        self, user_id: str, limit: int = 20
    ) -> dict[str, Any]:
        result = await self._legacy_call(
            "get_chat_history",
            user_id=user_id,
            limit=min(max(int(limit), 1), 100),
            default={"sessions": [], "total": 0},
        )
        return dict(result) if isinstance(result, Mapping) else {"sessions": [], "total": 0}

    async def get_fresh_chat_history(
        self, user_id: str, limit: int = 20
    ) -> dict[str, Any]:
        return await self.safe_get_chat_history(user_id, limit)

    async def safe_get_session_messages(
        self, user_id: str, session_id: str, limit: int = 100
    ) -> list[dict[str, Any]]:
        result = await self._legacy_call(
            "get_session_messages",
            user_id=user_id,
            session_id=session_id,
            limit=min(max(int(limit), 1), 100),
            default=[],
        )
        return [dict(item) for item in result] if isinstance(result, Sequence) else []

    async def _legacy_call(self, name: str, *, default: Any, **kwargs: Any) -> Any:
        reader = self.legacy_reader
        operation = getattr(reader, name, None) if reader is not None else None
        if not callable(operation):
            return default
        try:
            result = operation(**kwargs)
            return await result if inspect.isawaitable(result) else result
        except asyncio.CancelledError:
            raise
        except Exception:
            logger.warning("Legacy read unavailable code=legacy_read_failed")
            return default

    async def get_persistence_metrics(self) -> dict[str, Any]:
        return dict(self._metrics)

    async def register_event_callback(
        self, event_type: str, callback: Callable[..., Any]
    ) -> None:
        if event_type not in self._event_callbacks:
            raise ValueError("Unsupported event type")
        self._event_callbacks[event_type].append(callback)

    async def batch_persist_exchanges(
        self, exchanges: Sequence[ConversationExchange], batch_delay: float = 0.0
    ) -> dict[str, Any]:
        started = asyncio.get_running_loop().time()
        successful = 0
        errors: list[str] = []
        for index, exchange in enumerate(exchanges):
            result = await self.persist_conversation_exchange(exchange)
            if result["success"]:
                successful += 1
            else:
                errors.extend(str(item) for item in result["errors"])
            if batch_delay > 0 and index + 1 < len(exchanges):
                await asyncio.sleep(batch_delay)
        return {
            "total_exchanges": len(exchanges),
            "successful": successful,
            "failed": len(exchanges) - successful,
            "errors": errors,
            "duration_ms": (asyncio.get_running_loop().time() - started) * 1000.0,
        }


class PersistenceHealthCheck:
    """Content-free health view over the compatibility metrics."""

    def __init__(self, persistence_service: ConversationPersistenceService) -> None:
        self.service = persistence_service

    async def check_health(self) -> dict[str, Any]:
        metrics = await self.service.get_persistence_metrics()
        total = metrics["total_exchanges_stored"] + metrics["failed_stores"]
        error_rate = metrics["failed_stores"] / total if total else 0.0
        return {
            "healthy": error_rate <= 0.05,
            "issues": [] if error_rate <= 0.05 else ["storage_failures_observed"],
            "recommendations": [],
            "metrics": metrics,
            "timestamp": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        }
