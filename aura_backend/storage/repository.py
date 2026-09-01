"""Sole durable read/write owner for Aura's SQLite event ledger."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from aura_backend.storage.connection import FaultHook
from aura_backend.storage.models import (
    IdempotencyConflict,
    PersistedTurn,
    StorageFailure,
    TurnCommand,
)

TurnOutcome = PersistedTurn | IdempotencyConflict
TurnCallback = Callable[[PersistedTurn], None]


class StorageRepository:
    """Explicit injected owner of one absolute SQLite ledger path."""

    def __init__(self, database_path: Path) -> None:
        self.database_path = database_path

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
        del (
            scope_id,
            command,
            provider_callback,
            projection_callback,
            reconciliation_callback,
            fault_hook,
        )
        raise StorageFailure("not_implemented")
