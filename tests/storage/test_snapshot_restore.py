"""Synthetic proof for SQLite online snapshots and isolated restoration."""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import threading
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from aura_backend.storage.lifecycle import (
    LifecycleService,
    RestoreStatus,
    SnapshotStatus,
)
from aura_backend.storage.models import StorageFailure, TurnCommand
from aura_backend.storage.projection import ProjectionAdapter
from aura_backend.storage.repository import StorageRepository, canonical_request_hash
from tests.storage.test_projection_rebuild import FakeEmbeddingService


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _command(base: TurnCommand, ordinal: int) -> TurnCommand:
    user_text = f"Synthetic concurrent user event {ordinal:03d}"
    aura_text = f"Synthetic concurrent Aura event {ordinal:03d}"
    user = replace(
        base.user_event,
        event_id=f"event-user-{ordinal:03d}",
        content=user_text,
        content_sha256=_digest(user_text),
        observed_at=f"2026-09-01T00:00:{ordinal % 60:02d}Z",
    )
    aura = replace(
        base.aura_event,
        event_id=f"event-aura-{ordinal:03d}",
        content=aura_text,
        content_sha256=_digest(aura_text),
        observed_at=f"2026-09-01T00:01:{ordinal % 60:02d}Z",
    )
    memory = replace(
        base.derived_memories[0],
        memory_id=f"memory-{ordinal:03d}",
        canonical_text=f"Synthetic preference {ordinal:03d}",
        primary_source_event_id=user.event_id,
        source_event_ids=(user.event_id, aura.event_id),
        created_at=f"2026-09-01T00:02:{ordinal % 60:02d}Z",
    )
    return replace(
        base,
        turn_id=f"turn-{ordinal:03d}",
        idempotency_key=f"request-{ordinal:03d}",
        request_hash=canonical_request_hash(
            base.scope_id,
            base.session_id,
            user.content,
            version=base.request_hash_version,
        ),
        response_hash=_digest(aura.content),
        occurred_at=f"2026-09-01T00:00:{ordinal % 60:02d}Z",
        user_event=user,
        aura_event=aura,
        derived_memories=(memory,),
    )


def _projection_factory(root: Path, repository: StorageRepository) -> ProjectionAdapter:
    return ProjectionAdapter(
        projection_root=root,
        repository=repository,
        embedding_service=FakeEmbeddingService(),
    )


def _passing_fixtures(
    repository: StorageRepository, adapter: ProjectionAdapter
) -> dict[str, bool]:
    origins = repository.projection_origin_ids()
    page = repository.current_memories("scope-alpha")
    candidates = adapter.query_candidates(
        scope_id="scope-alpha",
        query="Synthetic preference",
        n_results=min(50, max(1, len(origins))),
    )
    return {
        "direct_retrieval": bool(candidates),
        "correction_freshness": all(item.memory_id.startswith("memory-") for item in page),
        "provenance": all(item.source_event_ids for item in page),
        "pagination": len(origins) == len(set(origins)),
        "cross_scope_isolation": not adapter.query_candidates(
            scope_id="scope-other",
            query="Synthetic preference",
            n_results=min(50, max(1, len(origins))),
        ),
    }


def _service(database_path: Path) -> LifecycleService:
    return LifecycleService(
        repository=StorageRepository(database_path),
        projection_factory=_projection_factory,
        fixture_verifier=_passing_fixtures,
        tool_commit="synthetic-test-commit",
    )


def _snapshot_and_manifest(
    service: LifecycleService, root: Path, *, snapshot_id: str = "snapshot-001"
) -> tuple[Path, Path, str]:
    result = service.create_snapshot(root, snapshot_id=snapshot_id)
    assert result.status is SnapshotStatus.PUBLISHED
    return result.database_path, result.manifest_path, result.manifest_sha256


def test_concurrent_online_snapshot_restores_exact_truth_and_fresh_projection(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """A writer may continue while one consistent SQLite image is captured."""
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    service = _service(ledger_path)
    snapshot_root = tmp_path / "snapshots"
    snapshot_root.mkdir(mode=0o700)
    restore_root = tmp_path / "isolated-restore"

    writer_started = threading.Event()
    writer_done = threading.Event()

    def writer() -> None:
        writer_started.set()
        for ordinal in range(2, 22):
            repository.append_turn(turn_command.scope_id, _command(turn_command, ordinal))
        writer_done.set()

    thread = threading.Thread(target=writer, daemon=True)
    thread.start()
    assert writer_started.wait(timeout=2.0)
    database_path, manifest_path, manifest_hash = _snapshot_and_manifest(
        service, snapshot_root
    )
    thread.join(timeout=5.0)
    assert writer_done.is_set()

    restored = service.restore_snapshot(
        database_path,
        manifest_path,
        expected_manifest_sha256=manifest_hash,
        restore_root=restore_root,
    )

    assert restored.status is RestoreStatus.COMPLETE
    assert restored.checks == LifecycleService.REQUIRED_RESTORE_CHECKS
    assert restored.table_counts == restored.manifest_table_counts
    assert restored.table_digests == restored.manifest_table_digests
    assert restored.projection_origin_count == sum(
        restored.table_counts[name] for name in ("events", "derived_memories")
    )
    assert restored.projection_ids_sha256
    assert restored.database_path.parent == restore_root
    assert restored.database_path.stat().st_mode & 0o777 == 0o600
    assert not any(path.name == "chroma.sqlite3" for path in snapshot_root.rglob("*"))


def test_interruption_timeout_and_resource_limit_publish_no_snapshot(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Partial, timed-out, and resource-bounded backup attempts fail closed."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    root = tmp_path / "snapshots"
    root.mkdir()
    service = _service(ledger_path)

    with pytest.raises(StorageFailure, match="snapshot_interrupted"):
        service.create_snapshot(
            root,
            snapshot_id="interrupted",
            fault_hook=lambda stage: (
                (_ for _ in ()).throw(StorageFailure("snapshot_interrupted"))
                if stage == "backup_progress"
                else None
            ),
        )

    ticks = iter((0.0, 10.0, 10.0, 10.0))
    with pytest.raises(StorageFailure, match="snapshot_timeout"):
        service.create_snapshot(
            root,
            snapshot_id="timed-out",
            timeout_seconds=1.0,
            monotonic=lambda: next(ticks, 10.0),
        )

    with pytest.raises(StorageFailure, match="snapshot_resource_limit"):
        service.create_snapshot(
            root,
            snapshot_id="resource-limited",
            max_database_bytes=1,
        )

    assert list(root.iterdir()) == []


def test_snapshot_rejects_overlap_symlinks_and_existing_targets(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Unsafe destinations are rejected before any target is created."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    service = _service(ledger_path)

    with pytest.raises(StorageFailure, match="snapshot_path_overlap"):
        service.create_snapshot(ledger_path.parent, snapshot_id="overlap")

    real_root = tmp_path / "real-snapshots"
    real_root.mkdir()
    linked_root = tmp_path / "linked-snapshots"
    os.symlink(real_root, linked_root, target_is_directory=True)
    with pytest.raises(StorageFailure, match="snapshot_symlink_path"):
        service.create_snapshot(linked_root, snapshot_id="linked")

    (real_root / "existing.sqlite3").write_bytes(b"claimed")
    with pytest.raises(StorageFailure, match="snapshot_target_exists"):
        service.create_snapshot(real_root, snapshot_id="existing")

    assert sorted(path.name for path in real_root.iterdir()) == ["existing.sqlite3"]


@pytest.mark.parametrize(
    "mutation",
    (
        "manifest_hash",
        "database_hash",
        "missing_check",
        "reordered_checks",
        "duplicate_check",
    ),
)
def test_restore_rejects_tampered_hash_or_non_exact_check_set(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
    mutation: str,
) -> None:
    """Manifest/check-set substitution cannot produce complete restore evidence."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    root = tmp_path / "snapshots"
    root.mkdir()
    service = _service(ledger_path)
    database_path, manifest_path, manifest_hash = _snapshot_and_manifest(service, root)
    expected_hash = manifest_hash

    if mutation == "manifest_hash":
        expected_hash = "0" * 64
    else:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if mutation == "database_hash":
            payload["database_sha256"] = "0" * 64
        elif mutation == "missing_check":
            payload["required_checks"] = payload["required_checks"][:-1]
        elif mutation == "reordered_checks":
            payload["required_checks"][0:2] = reversed(payload["required_checks"][0:2])
        elif mutation == "duplicate_check":
            payload["required_checks"][-1] = payload["required_checks"][0]
        manifest_path.write_text(
            json.dumps(payload, separators=(",", ":"), sort_keys=True),
            encoding="utf-8",
        )
        expected_hash = _digest(manifest_path.read_text(encoding="utf-8"))

    with pytest.raises(StorageFailure, match="restore_(manifest|database|check_set)"):
        service.restore_snapshot(
            database_path,
            manifest_path,
            expected_manifest_sha256=expected_hash,
            restore_root=tmp_path / f"restore-{mutation}",
        )
    assert not (tmp_path / f"restore-{mutation}").exists()


def test_corrupt_database_and_foreign_key_error_cannot_pass(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Integrity and FK gates are independent and both fail closed."""
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    service = _service(ledger_path)
    root = tmp_path / "snapshots"
    root.mkdir()
    database_path, manifest_path, manifest_hash = _snapshot_and_manifest(service, root)

    corrupt = tmp_path / "corrupt.sqlite3"
    corrupt.write_bytes(database_path.read_bytes()[:512] + b"not-a-database")
    with pytest.raises(StorageFailure, match="restore_database_hash_mismatch"):
        service.restore_snapshot(
            corrupt,
            manifest_path,
            expected_manifest_sha256=manifest_hash,
            restore_root=tmp_path / "corrupt-restore",
        )

    violating_path = tmp_path / "violating.sqlite3"
    violating_path.write_bytes(database_path.read_bytes())
    connection = sqlite3.connect(violating_path)
    try:
        connection.execute("PRAGMA foreign_keys = OFF")
        connection.execute(
            "INSERT INTO memory_sources(memory_id,event_id,scope_id,relation) "
            "VALUES ('missing-memory','missing-event','scope-alpha','support')"
        )
        connection.commit()
    finally:
        connection.close()
    violating_root = tmp_path / "violating-snapshots"
    violating_root.mkdir()
    violating_service = _service(violating_path)
    bad_db, bad_manifest, bad_manifest_hash = _snapshot_and_manifest(
        violating_service, violating_root, snapshot_id="violating"
    )
    with pytest.raises(StorageFailure, match="restore_foreign_key_failed"):
        violating_service.restore_snapshot(
            bad_db,
            bad_manifest,
            expected_manifest_sha256=bad_manifest_hash,
            restore_root=tmp_path / "violating-restore",
        )


def test_failed_projection_fixture_is_non_pass_and_active_source_is_unchanged(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Restore failure stays isolated and never changes the active ledger."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    service = _service(ledger_path)
    root = tmp_path / "snapshots"
    root.mkdir()
    database_path, manifest_path, manifest_hash = _snapshot_and_manifest(service, root)
    before = ledger_path.read_bytes()

    def failing_fixtures(
        repository: StorageRepository, adapter: ProjectionAdapter
    ) -> dict[str, bool]:
        values = _passing_fixtures(repository, adapter)
        values["cross_scope_isolation"] = False
        return values

    failing = LifecycleService(
        repository=StorageRepository(ledger_path),
        projection_factory=_projection_factory,
        fixture_verifier=failing_fixtures,
        tool_commit="synthetic-test-commit",
    )
    with pytest.raises(StorageFailure, match="restore_fixture_failed"):
        failing.restore_snapshot(
            database_path,
            manifest_path,
            expected_manifest_sha256=manifest_hash,
            restore_root=tmp_path / "fixture-failed-restore",
        )

    assert ledger_path.read_bytes() == before
    assert (tmp_path / "fixture-failed-restore").is_dir()
