"""Versioned schema surface for Aura's SQLite ledger."""

from __future__ import annotations

import sqlite3

from aura_backend.storage.models import StorageFailure

SCHEMA_VERSION = 1


def apply_migrations(connection: sqlite3.Connection) -> None:
    """Apply ordered forward migrations to ``connection``."""
    del connection
    raise StorageFailure("not_implemented")


def rebuild_fts(connection: sqlite3.Connection) -> None:
    """Rebuild every external-content full-text projection."""
    del connection
    raise StorageFailure("not_implemented")
