"""Synthetic truthful-export and explicit-deletion lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from aura_backend.storage.connection import open_database
from aura_backend.storage.lifecycle import ExportStatus, LifecycleService
from aura_backend.storage.models import StorageFailure, TurnCommand
from aura_backend.storage.projection import ProjectionAdapter
from aura_backend.storage.repository import StorageRepository, canonical_request_hash
from tests.storage.test_projection_rebuild import FakeEmbeddingService
from tests.storage.test_snapshot_restore import _passing_fixtures


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _factory(root: Path, repository: StorageRepository) -> ProjectionAdapter:
    return ProjectionAdapter(
        projection_root=root,
        repository=repository,
        embedding_service=FakeEmbeddingService(),
    )


def _service(database_path: Path) -> LifecycleService:
    return LifecycleService(
        repository=StorageRepository(database_path),
        projection_factory=_factory,
        fixture_verifier=_passing_fixtures,
        tool_commit="synthetic-test-commit",
    )


def _other_scope(command: TurnCommand) -> TurnCommand:
    user_text = "Synthetic private data from another scope"
    aura_text = "Synthetic unrelated Aura reply"
    user = replace(
        command.user_event,
        event_id="event-user-other",
        content=user_text,
        content_sha256=_digest(user_text),
    )
    aura = replace(
        command.aura_event,
        event_id="event-aura-other",
        content=aura_text,
        content_sha256=_digest(aura_text),
    )
    memory = replace(
        command.derived_memories[0],
        memory_id="memory-other",
        canonical_text="Unrelated synthetic memory",
        primary_source_event_id=user.event_id,
        source_event_ids=(user.event_id, aura.event_id),
    )
    return replace(
        command,
        scope_id="scope-other",
        session_id="session-other",
        turn_id="turn-other",
        idempotency_key="request-other",
        request_hash=canonical_request_hash(
            "scope-other",
            "session-other",
            user_text,
            version=command.request_hash_version,
        ),
        response_hash=_digest(aura_text),
        user_event=user,
        aura_event=aura,
        derived_memories=(memory,),
    )


def _seed_export_fixture(database_path: Path, command: TurnCommand) -> None:
    repository = StorageRepository(database_path)
    repository.append_turn(command.scope_id, command)
    repository.append_turn("scope-other", _other_scope(command))
    connection = open_database(database_path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        connection.execute(
            "INSERT INTO profile_versions(scope_id,profile_version,payload_json,created_at) "
            "VALUES (?,?,?,?)",
            (
                command.scope_id,
                1,
                json.dumps(
                    {
                        "display_name": "Synthetic Ty",
                        "preference": "concise",
                        "provider_api_key": "must-not-export",
                        "embedding": [0.1, 0.2],
                        "internal_path": "/private/synthetic/path",
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "2026-09-01T00:03:00Z",
            ),
        )
        connection.execute(
            "INSERT INTO profile_versions(scope_id,profile_version,payload_json,created_at) "
            "VALUES (?,?,?,?)",
            (
                "scope-other",
                1,
                json.dumps({"display_name": "Other"}),
                "2026-09-01T00:03:00Z",
            ),
        )
        connection.execute(
            "INSERT INTO retrieval_runs(run_id,scope_id,query_sha256,config_version,created_at) "
            "VALUES (?,?,?,?,?)",
            ("secret-trace-run", command.scope_id, "f" * 64, 1, "2026-09-01T00:04:00Z"),
        )
        connection.commit()
    finally:
        connection.close()


def test_json_export_contains_exact_scoped_rows_edges_and_sanitized_profiles(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """A successful JSON export represents real allowlisted ledger records."""
    _seed_export_fixture(ledger_path, turn_command)
    export_root = tmp_path / "exports"
    export_root.mkdir(mode=0o700)
    service = _service(ledger_path)

    result = service.export_scope_json(
        export_root,
        scope_id=turn_command.scope_id,
        export_id="export-001",
        output_format="json",
        created_at="2026-09-01T12:00:00Z",
    )

    assert result.status is ExportStatus.PUBLISHED
    assert result.path.stat().st_mode & 0o777 == 0o600
    payload = json.loads(result.path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == 1
    assert payload["scope_id"] == turn_command.scope_id
    assert payload["counts"] == {
        "derived_memories": 1,
        "events": 2,
        "memory_retractions": 0,
        "memory_sources": 2,
        "memory_supersessions": 0,
        "profile_versions": 1,
        "sessions": 1,
        "turns": 1,
    }
    assert [row["event_id"] for row in payload["records"]["events"]] == [
        turn_command.user_event.event_id,
        turn_command.aura_event.event_id,
    ]
    assert payload["records"]["memory_sources"] == [
        {
            "event_id": turn_command.aura_event.event_id,
            "memory_id": turn_command.derived_memories[0].memory_id,
            "relation": "support",
        },
        {
            "event_id": turn_command.user_event.event_id,
            "memory_id": turn_command.derived_memories[0].memory_id,
            "relation": "support",
        },
    ]
    assert payload["records"]["profile_versions"][0]["payload"] == {
        "display_name": "Synthetic Ty",
        "preference": "concise",
    }
    serialized = result.path.read_text(encoding="utf-8")
    for forbidden in (
        "scope-other",
        "secret-trace-run",
        "must-not-export",
        "embedding",
        "/private/synthetic/path",
    ):
        assert forbidden not in serialized
    verification = service.verify_scope_export(
        result.path, expected_scope_id=turn_command.scope_id
    )
    assert verification.valid is True
    assert verification.counts == result.counts
    assert verification.record_hashes == result.record_hashes


@pytest.mark.parametrize("output_format", ("csv", "xml", "yaml", "../json"))
def test_unsupported_formats_create_no_path_or_file(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
    output_format: str,
) -> None:
    """A false format label fails before any filesystem side effect."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    export_root = tmp_path / "exports"
    export_root.mkdir()
    with pytest.raises(StorageFailure, match="export_format_unsupported"):
        _service(ledger_path).export_scope_json(
            export_root,
            scope_id=turn_command.scope_id,
            export_id="unsupported",
            output_format=output_format,
        )
    assert list(export_root.iterdir()) == []


@pytest.mark.parametrize("unsafe", ("../escape", "folder/name", r"folder\name", ""))
def test_export_rejects_traversal_and_unsafe_identifiers_before_creation(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
    unsafe: str,
) -> None:
    """Scope and output identifiers remain single safe components."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    export_root = tmp_path / "exports"
    export_root.mkdir()
    with pytest.raises(StorageFailure, match="export_(scope|id)_invalid"):
        _service(ledger_path).export_scope_json(
            export_root,
            scope_id=unsafe,
            export_id=unsafe,
            output_format="json",
        )
    assert list(export_root.iterdir()) == []


def test_export_publication_is_atomic_no_follow_and_deterministic(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Interrupted or redirected writes never become successful exports."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    service = _service(ledger_path)
    export_root = tmp_path / "exports"
    export_root.mkdir()

    with pytest.raises(StorageFailure, match="export_interrupted"):
        service.export_scope_json(
            export_root,
            scope_id=turn_command.scope_id,
            export_id="interrupted",
            output_format="json",
            fault_hook=lambda stage: (
                (_ for _ in ()).throw(StorageFailure("export_interrupted"))
                if stage == "before_publish"
                else None
            ),
        )
    assert list(export_root.iterdir()) == []

    outside = tmp_path / "outside.json"
    os.symlink(outside, export_root / "redirected.json")
    with pytest.raises(StorageFailure, match="export_target_exists"):
        service.export_scope_json(
            export_root,
            scope_id=turn_command.scope_id,
            export_id="redirected",
            output_format="json",
        )
    assert not outside.exists()

    first = service.export_scope_json(
        export_root,
        scope_id=turn_command.scope_id,
        export_id="deterministic-one",
        output_format="json",
        created_at="2026-09-01T12:00:00Z",
    )
    second = service.export_scope_json(
        export_root,
        scope_id=turn_command.scope_id,
        export_id="deterministic-two",
        output_format="json",
        created_at="2026-09-01T12:00:00Z",
    )
    first_payload = json.loads(first.path.read_text(encoding="utf-8"))
    second_payload = json.loads(second.path.read_text(encoding="utf-8"))
    first_payload["export_id"] = "same"
    second_payload["export_id"] = "same"
    first_payload["manifest_sha256"] = "ignored"
    second_payload["manifest_sha256"] = "ignored"
    assert first_payload == second_payload


def test_empty_scope_is_truthful_and_tampered_export_fails_round_trip(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Empty means no stored rows, while altered records cannot verify."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    service = _service(ledger_path)
    export_root = tmp_path / "exports"
    export_root.mkdir()
    empty = service.export_scope_json(
        export_root,
        scope_id="scope-empty",
        export_id="empty",
        output_format="json",
    )
    empty_payload = json.loads(empty.path.read_text(encoding="utf-8"))
    assert all(value == 0 for value in empty_payload["counts"].values())
    assert all(value == [] for value in empty_payload["records"].values())
    assert service.verify_scope_export(
        empty.path, expected_scope_id="scope-empty"
    ).valid

    populated = service.export_scope_json(
        export_root,
        scope_id=turn_command.scope_id,
        export_id="tampered",
        output_format="json",
    )
    payload = json.loads(populated.path.read_text(encoding="utf-8"))
    payload["records"]["events"][0]["content"] = "tampered"
    populated.path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(StorageFailure, match="export_(manifest|record)_hash_mismatch"):
        service.verify_scope_export(
            populated.path, expected_scope_id=turn_command.scope_id
        )
