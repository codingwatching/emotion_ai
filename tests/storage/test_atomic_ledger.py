"""Atomic SQLite ledger and external-content FTS contracts."""

from __future__ import annotations

import os
import sqlite3
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from aura_backend.storage.connection import append_turn_atomic, open_database
from aura_backend.storage.models import StorageFailure, TurnCommand
from aura_backend.storage.schema import SCHEMA_VERSION, rebuild_fts

_CANONICAL_TABLES = (
    "memory_scopes",
    "sessions",
    "turns",
    "events",
    "derived_memories",
    "memory_sources",
    "memory_supersessions",
    "memory_retractions",
)


def _counts(connection: sqlite3.Connection) -> dict[str, int]:
    return {
        table: connection.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
        for table in _CANONICAL_TABLES
    }


def test_open_requires_absolute_path_and_configures_owned_database(
    ledger_path: Path,
) -> None:
    with pytest.raises(StorageFailure) as relative_error:
        open_database(Path("relative.sqlite3"))
    assert relative_error.value.code == "absolute_path_required"

    connection = open_database(ledger_path, busy_timeout_ms=1_250)
    try:
        assert connection.execute("PRAGMA foreign_keys").fetchone()[0] == 1
        assert connection.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
        assert connection.execute("PRAGMA busy_timeout").fetchone()[0] == 1_250
        assert connection.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    finally:
        connection.close()


def test_importing_storage_modules_performs_no_io(
    tmp_path: Path,
) -> None:
    before = tuple(tmp_path.iterdir())
    repository = Path(__file__).resolve().parents[2]
    environment = {**os.environ, "PYTHONPATH": str(repository)}
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import aura_backend.storage.models; "
                "import aura_backend.storage.schema; "
                "import aura_backend.storage.connection"
            ),
        ],
        cwd=tmp_path,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert tuple(tmp_path.iterdir()) == before


@pytest.mark.parametrize(
    "stage",
    (
        "after_turn",
        "after_event_0",
        "after_event_1",
        "after_derivation_0",
        "after_source_0_0",
        "after_source_0_1",
        "before_commit",
    ),
)
def test_every_precommit_fault_rolls_back_all_rows_and_fts(
    ledger_path: Path, turn_command: TurnCommand, stage: str
) -> None:
    connection = open_database(ledger_path)

    def fail_at(current: str) -> None:
        if current == stage:
            raise StorageFailure("injected_fault", identifier=current)

    try:
        with pytest.raises(StorageFailure) as error:
            append_turn_atomic(connection, turn_command, fault_hook=fail_at)
        assert error.value.code == "injected_fault"
        assert _counts(connection) == {table: 0 for table in _CANONICAL_TABLES}
        assert connection.execute("SELECT count(*) FROM event_fts").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 0
    finally:
        connection.close()


def test_success_commits_complete_ordered_turn_provenance_and_fts(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    connection = open_database(ledger_path)
    try:
        append_turn_atomic(connection, turn_command)
        assert _counts(connection) == {
            "memory_scopes": 1,
            "sessions": 1,
            "turns": 1,
            "events": 2,
            "derived_memories": 1,
            "memory_sources": 2,
            "memory_supersessions": 0,
            "memory_retractions": 0,
        }
        rows = connection.execute(
            "SELECT event_id, ordinal, scope_id FROM events ORDER BY ordinal"
        ).fetchall()
        assert rows == [
            (turn_command.user_event.event_id, 0, turn_command.scope_id),
            (turn_command.aura_event.event_id, 1, turn_command.scope_id),
        ]
        assert connection.execute("SELECT count(*) FROM event_fts").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM memory_fts").fetchone()[0] == 1
    finally:
        connection.close()


def test_fts_triggers_and_explicit_rebuild_align_existing_rows(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    connection = open_database(ledger_path)
    try:
        append_turn_atomic(connection, replace(turn_command, derived_memories=()))
        event_pk = connection.execute(
            "SELECT event_pk FROM events WHERE event_id = ?",
            (turn_command.user_event.event_id,),
        ).fetchone()[0]
        assert connection.execute(
            "SELECT rowid FROM event_fts WHERE event_fts MATCH ?", ("preference",)
        ).fetchall() == [(event_pk,)]

        connection.execute(
            "UPDATE events SET content = ? WHERE event_pk = ?",
            ("Updated synthetic phrase", event_pk),
        )
        connection.commit()
        assert connection.execute(
            "SELECT rowid FROM event_fts WHERE event_fts MATCH ?", ("Updated",)
        ).fetchall() == [(event_pk,)]

        connection.execute("INSERT INTO event_fts(event_fts) VALUES ('delete-all')")
        assert connection.execute(
            "SELECT rowid FROM event_fts WHERE event_fts MATCH ?", ("Updated",)
        ).fetchall() == []
        rebuild_fts(connection)
        assert connection.execute("SELECT count(*) FROM event_fts").fetchone()[0] == 2

        connection.execute("DELETE FROM events WHERE event_pk = ?", (event_pk,))
        connection.commit()
        assert connection.execute(
            "SELECT rowid FROM event_fts WHERE event_fts MATCH ?", ("Updated",)
        ).fetchall() == []
    finally:
        connection.close()


def test_storage_failures_do_not_echo_private_content(
    ledger_path: Path, turn_command: TurnCommand
) -> None:
    private_sentinel = "never-copy-this-private-sentinel"
    invalid = replace(
        turn_command,
        user_event=replace(turn_command.user_event, content=private_sentinel),
    )
    connection = open_database(ledger_path)
    try:
        def fail(_: str) -> None:
            raise StorageFailure("injected_fault", identifier=invalid.turn_id)

        with pytest.raises(StorageFailure) as error:
            append_turn_atomic(connection, invalid, fault_hook=fail)
        assert private_sentinel not in str(error.value)
        assert error.value.code == "injected_fault"
    finally:
        connection.close()
