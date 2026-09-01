"""Synthetic truthful-export and explicit-deletion lifecycle contracts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path

import pytest

from aura_backend.storage.connection import open_database
from aura_backend.storage.lifecycle import (
    DeletionAction,
    DeletionStatus,
    DeletionTarget,
    ExportStatus,
    LifecycleService,
    ResidualCopy,
)
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


class _SyntheticDeletionStore:
    """Exact in-memory active target used only by disposable deletion tests."""

    def __init__(
        self,
        name: str,
        kind: str,
        records: dict[str, str],
        *,
        fail_delete: bool = False,
        refuse_delete: bool = False,
    ) -> None:
        self.name = name
        self.kind = kind
        self.records = records
        self.fail_delete = fail_delete
        self.refuse_delete = refuse_delete
        self.reconcile_calls = 0

    def target(self) -> DeletionTarget:
        return DeletionTarget(
            name=self.name,
            kind=self.kind,
            inventory=self.inventory,
            delete=self.delete,
            remaining=self.remaining,
            reconcile=self.reconcile if self.kind == "projection" else None,
        )

    def inventory(
        self, action: DeletionAction, scope_id: str, target_id: str | None
    ) -> dict[str, str]:
        prefix = f"{scope_id}:"
        if action in {DeletionAction.SCOPE, DeletionAction.SESSION}:
            return {
                identity: digest
                for identity, digest in self.records.items()
                if identity.startswith(prefix)
            }
        if target_id is None:
            return {}
        return {
            identity: digest
            for identity, digest in self.records.items()
            if identity == target_id
        }

    def delete(self, identities: tuple[str, ...]) -> None:
        if self.fail_delete:
            raise RuntimeError("synthetic target unavailable")
        if self.refuse_delete:
            return
        for identity in identities:
            self.records.pop(identity, None)

    def remaining(self, identities: tuple[str, ...]) -> dict[str, str]:
        return {
            identity: self.records[identity]
            for identity in identities
            if identity in self.records
        }

    def reconcile(self) -> None:
        self.reconcile_calls += 1


def _deletion_service(
    database_path: Path,
    *targets: _SyntheticDeletionStore,
    clock: list[float] | None = None,
) -> LifecycleService:
    return LifecycleService(
        repository=StorageRepository(database_path),
        projection_factory=_factory,
        fixture_verifier=_passing_fixtures,
        tool_commit="synthetic-test-commit",
        deletion_secret=b"synthetic-deletion-secret-32-bytes",
        deletion_clock=(lambda: clock[0]) if clock is not None else None,
        deletion_targets=tuple(target.target() for target in targets),
        retained_copies=(
            ResidualCopy("historical_root", "synthetic-history-a"),
            ResidualCopy("memvid_archive", "synthetic-archive-a"),
            ResidualCopy("backup_generation", "synthetic-backup-a"),
        ),
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


def test_scope_deletion_is_planned_confirmed_executed_and_exactly_verified(
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Complete means every approved active target is absent after verification."""
    _seed_export_fixture(ledger_path, turn_command)
    scope = turn_command.scope_id
    projection = _SyntheticDeletionStore(
        "active-projection",
        "projection",
        {
            f"{scope}:event:{turn_command.user_event.event_id}": "a" * 64,
            f"{scope}:memory:{turn_command.derived_memories[0].memory_id}": "b" * 64,
            "scope-other:event:event-user-other": "c" * 64,
        },
    )
    cache = _SyntheticDeletionStore(
        "provider-cache",
        "cache",
        {f"{scope}:cache": "d" * 64, "scope-other:cache": "e" * 64},
    )
    generated_export = _SyntheticDeletionStore(
        "generated-exports",
        "generated_export",
        {f"{scope}:export-001": "f" * 64},
    )
    service = _deletion_service(ledger_path, projection, cache, generated_export)

    plan = service.plan_deletion(action="scope", scope_id=scope, ttl_seconds=60)
    assert plan.action is DeletionAction.SCOPE
    assert plan.counts["sqlite:events"] == 2
    assert plan.counts["sqlite:event_fts"] == 2
    assert plan.counts["sqlite:profile_versions"] == 1
    assert plan.counts["active-projection"] == 2
    assert {copy.kind for copy in plan.retained_copies} == {
        "historical_root",
        "memvid_archive",
        "backup_generation",
    }

    confirmation = service.confirm_deletion(
        plan, challenge=plan.challenge, scope_id=scope
    )
    execution = service.execute_deletion(plan, confirmation)
    result = service.verify_deletion(plan, execution)

    assert result.status is DeletionStatus.COMPLETE
    assert result.failed_targets == ()
    assert result.retry_targets == ()
    assert result.application_level_deletion is True
    assert result.physical_purge == "best_effort"
    assert result.forensic_erasure is False
    assert result.retained_copies == plan.retained_copies
    assert projection.reconcile_calls == 1
    assert set(projection.records) == {"scope-other:event:event-user-other"}
    assert set(cache.records) == {"scope-other:cache"}

    connection = open_database(ledger_path)
    try:
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_scopes WHERE scope_id=?", (scope,)
        ).fetchone()[0] == 0
        assert connection.execute(
            "SELECT COUNT(*) FROM memory_scopes WHERE scope_id='scope-other'"
        ).fetchone()[0] == 1
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()


def test_deletion_rejects_vague_tampered_expired_replayed_and_wrong_scope_intent(
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """No invalid confirmation can cross the destructive transaction boundary."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    clock = [100.0]
    service = _deletion_service(ledger_path, clock=clock)

    for vague in ("delete", "everything", "session-or-scope", ""):
        with pytest.raises(StorageFailure, match="deletion_action_invalid"):
            service.plan_deletion(action=vague, scope_id=turn_command.scope_id)

    plan = service.plan_deletion(
        action="scope", scope_id=turn_command.scope_id, ttl_seconds=10
    )
    with pytest.raises(StorageFailure, match="deletion_plan_tampered"):
        service.confirm_deletion(
            replace(plan, scope_id="scope-other"),
            challenge=plan.challenge,
            scope_id="scope-other",
        )
    with pytest.raises(StorageFailure, match="deletion_scope_mismatch"):
        service.confirm_deletion(
            plan, challenge=plan.challenge, scope_id="scope-other"
        )
    with pytest.raises(StorageFailure, match="deletion_confirmation_invalid"):
        service.confirm_deletion(
            plan, challenge="0" * 64, scope_id=turn_command.scope_id
        )

    clock[0] = 111.0
    with pytest.raises(StorageFailure, match="deletion_plan_expired"):
        service.confirm_deletion(
            plan, challenge=plan.challenge, scope_id=turn_command.scope_id
        )
    assert StorageRepository(ledger_path).projection_origin_ids()

    clock[0] = 200.0
    replay_plan = service.plan_deletion(
        action="scope", scope_id=turn_command.scope_id, ttl_seconds=10
    )
    confirmation = service.confirm_deletion(
        replay_plan,
        challenge=replay_plan.challenge,
        scope_id=turn_command.scope_id,
    )
    execution = service.execute_deletion(replay_plan, confirmation)
    assert service.verify_deletion(replay_plan, execution).status is DeletionStatus.COMPLETE
    with pytest.raises(StorageFailure, match="deletion_confirmation_replayed"):
        service.execute_deletion(replay_plan, confirmation)


def test_partial_projection_failure_is_incomplete_and_retryable_with_new_plan(
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """Canonical success cannot hide a failed projection or a retained exact ID."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    scope = turn_command.scope_id
    projection = _SyntheticDeletionStore(
        "active-projection",
        "projection",
        {f"{scope}:event:{turn_command.user_event.event_id}": "a" * 64},
        fail_delete=True,
    )
    service = _deletion_service(ledger_path, projection)
    plan = service.plan_deletion(action="scope", scope_id=scope)
    confirmation = service.confirm_deletion(
        plan, challenge=plan.challenge, scope_id=scope
    )
    execution = service.execute_deletion(plan, confirmation)
    result = service.verify_deletion(plan, execution)

    assert result.status is DeletionStatus.INCOMPLETE
    assert result.failed_targets == ("active-projection",)
    assert result.retry_targets == ("active-projection",)
    assert result.retained_copies == plan.retained_copies
    assert result.forensic_erasure is False

    projection.fail_delete = False
    retry_plan = service.plan_deletion(action="scope", scope_id=scope)
    retry_confirmation = service.confirm_deletion(
        retry_plan, challenge=retry_plan.challenge, scope_id=scope
    )
    retry_execution = service.execute_deletion(retry_plan, retry_confirmation)
    retry_result = service.verify_deletion(retry_plan, retry_execution)
    assert retry_result.status is DeletionStatus.COMPLETE
    assert projection.records == {}


def test_verification_detects_target_that_claims_delete_but_retains_exact_record(
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    """A successful callback is not evidence when the exact approved ID remains."""
    StorageRepository(ledger_path).append_turn(turn_command.scope_id, turn_command)
    scope = turn_command.scope_id
    cache = _SyntheticDeletionStore(
        "provider-cache",
        "cache",
        {f"{scope}:cache": "d" * 64},
        refuse_delete=True,
    )
    service = _deletion_service(ledger_path, cache)
    plan = service.plan_deletion(action="scope", scope_id=scope)
    confirmation = service.confirm_deletion(
        plan, challenge=plan.challenge, scope_id=scope
    )
    result = service.verify_deletion(
        plan, service.execute_deletion(plan, confirmation)
    )
    assert result.status is DeletionStatus.INCOMPLETE
    assert result.failed_targets == ()
    assert result.retry_targets == ("provider-cache",)


@pytest.mark.parametrize("action", ("archive", "backup_generation"))
def test_archive_and_backup_purge_require_independent_restore_proof(
    ledger_path: Path,
    action: str,
) -> None:
    """Normal lifecycle code never infers permission to purge retained copies."""
    service = _deletion_service(ledger_path)
    with pytest.raises(StorageFailure, match="deletion_restore_proof_required"):
        service.plan_deletion(
            action=action,
            scope_id="scope-synthetic",
            target_id="synthetic-generation",
        )
