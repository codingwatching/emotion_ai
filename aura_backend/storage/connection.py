"""Explicit SQLite connection and transaction ownership."""

from __future__ import annotations

import sqlite3
from collections.abc import Callable
from pathlib import Path

from aura_backend.storage.models import StorageFailure, TurnCommand

FaultHook = Callable[[str], None]


def open_database(path: Path, *, busy_timeout_ms: int = 5_000) -> sqlite3.Connection:
    """Open and migrate one explicitly located ledger database."""
    del path, busy_timeout_ms
    raise StorageFailure("not_implemented")


def append_turn_atomic(
    connection: sqlite3.Connection,
    command: TurnCommand,
    *,
    fault_hook: FaultHook | None = None,
) -> None:
    """Persist a complete turn or roll back every durable row."""
    del connection, command, fault_hook
    raise StorageFailure("not_implemented")
