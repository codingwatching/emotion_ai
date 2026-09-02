"""Synthetic-only validation for Phase 3 preservation evidence gates."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sqlite3
from pathlib import Path
from typing import Any

import pytest

from aura_backend.preservation import cli as preservation_cli


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PROPOSAL_PATH = (
    REPOSITORY_ROOT / ".planning/evidence/phase-03/inventory-proposal.json"
)
PHASE1_ARTIFACTS = {
    REPOSITORY_ROOT / ".planning/evidence/phase-01/inventory-summary.json": (
        "debf25853a2301d4372ec8df210128cf31b8d0adb62d3509d1d3ccaabfbd43cc"
    ),
    REPOSITORY_ROOT / ".planning/evidence/phase-01/restore-drill-summary.json": (
        "2eb1e0c4cb4903f5434b5a4a8f7d79bd5e438a29e15b5b6f2c3278c88eeee400"
    ),
}
EXPECTED_ROOTS = (
    ("active-01", "active", "aura_chroma_db", True),
    ("active-02", "active", "aura_backend/aura_chroma_db", True),
    ("active-03", "active", "aura_data", True),
    ("active-04", "active", "aura_backend/aura_data", True),
    ("active-05", "active", "memvid_data", True),
    ("active-06", "active", "aura_backend/memvid_data", True),
    ("backup-01", "backup", "auto_backups", True),
    ("backup-02", "backup", "aura_backend/auto_backups", True),
    ("backup-03", "backup", "aura_backend/chromadb_backups", True),
    ("test-01", "test", "aura_backend/test_chroma_db", True),
    ("test-02", "test", "aura_backend/tests/test_aura_chroma_db", True),
    ("archive-01", "archive", "archive", True),
    ("archive-02", "archive", "aura_backend/archive_unused", True),
    ("archive-03", "archive", "aura_backend/aura_archives", True),
    ("archive-04", "archive", "aura_backend/memvid_videos", True),
    ("active-07", "active", "aura_data_v2", False),
)
FORBIDDEN_OBSERVATION_FIELDS = {
    "aggregate_sha256",
    "byte_total",
    "checks",
    "created_at_utc",
    "database_checks",
    "exists",
    "file_count",
    "filename",
    "free_bytes",
    "hash",
    "mtime",
    "observed",
    "open_handles",
    "processes",
    "sha256",
    "size",
    "source_set_sha256",
    "status",
    "timestamp",
}


def _proposal() -> dict[str, Any]:
    return json.loads(PROPOSAL_PATH.read_text(encoding="utf-8"))


def _assert_exact_proposal(value: dict[str, Any]) -> None:
    """Reject incomplete, widened, observed, or path-drifting proposals."""
    assert set(value) == {
        "schema_version",
        "proposal_id",
        "run_id",
        "operation",
        "repository_root",
        "backup_target",
        "roots",
        "conditional_rule",
        "evidence_paths",
        "phase1_evidence",
        "allowed_operations",
        "prohibited_operations",
        "approval_required",
    }
    assert value["schema_version"] == 1
    assert value["proposal_id"] == "phase-03-inventory-preflight-proposal-01"
    assert value["run_id"] == "phase-03-storage-gate-01"
    assert value["operation"] == "metadata_inventory_preflight_only"
    assert value["repository_root"] == "/home/ty/Repositories/ai_workspace/emotion_ai"
    assert value["backup_target"] == "/backup/aura-phase-03"
    roots = tuple(
        (root["alias"], root["role"], root["repository_relative_path"], root["required"])
        for root in value["roots"]
    )
    assert roots == EXPECTED_ROOTS
    assert value["conditional_rule"] == {
        "alias": "active-07",
        "repository_relative_path": "aura_data_v2",
        "rule": "include_only_if_exists_when_approved_inventory_runs",
        "absence_result": "record_not_run_without_widening_scope",
    }
    assert value["evidence_paths"] == {
        "public_inventory": ".planning/evidence/phase-03/inventory-summary.json",
        "public_quiescence": ".planning/evidence/phase-03/quiescence-summary.json",
        "private_inventory": (
            "/backup/aura-phase-03/phase-03-storage-gate-01/"
            "inventory.private.json"
        ),
        "private_quiescence_pattern": (
            "/backup/aura-phase-03/phase-03-storage-gate-01/"
            "quiescence.<ticket-id>.private.json"
        ),
    }
    assert value["phase1_evidence"] == [
        ".planning/evidence/phase-01/inventory-summary.json",
        ".planning/evidence/phase-01/restore-drill-summary.json",
    ]
    assert value["allowed_operations"] == [
        "filesystem_metadata_and_sha256_inventory",
        "read_only_sqlite_integrity_and_foreign_key_checks",
        "writer_process_and_open_handle_scan",
        "destination_capacity_and_path_containment_preflight",
        "public_private_evidence_publication",
    ]
    assert set(value["prohibited_operations"]) == {
        "backup_copy",
        "chroma_or_memvid_instantiation",
        "content_or_metadata_value_read",
        "deletion_or_cleanup",
        "historical_root_repair",
        "import_or_migration",
        "read_owner_switch",
        "restore",
        "source_mutation",
    }
    assert value["approval_required"] == {
        "before_real_path_observation": True,
        "approval_phrase": "approved",
        "authority_granted": "exact_metadata_inventory_preflight_only",
    }
    _assert_observation_free(value)


def _assert_observation_free(value: Any) -> None:
    if isinstance(value, dict):
        assert not (set(value) & FORBIDDEN_OBSERVATION_FIELDS)
        for child in value.values():
            _assert_observation_free(child)
    elif isinstance(value, list):
        for child in value:
            _assert_observation_free(child)


def _create_synthetic_root(root: Path, relative: str) -> None:
    target = root / relative
    target.mkdir(parents=True)
    (target / "opaque.bin").write_bytes(b"invented fixture bytes")


def _run_synthetic_inventory(tmp_path: Path) -> tuple[Path, Path, Path]:
    repository = tmp_path / "synthetic-repository"
    backup = tmp_path / "synthetic-outside-git"
    repository.mkdir()
    backup.mkdir()
    _create_synthetic_root(repository, "active")
    database = repository / "active" / "fixture.sqlite3"
    with sqlite3.connect(database) as connection:
        connection.execute("CREATE TABLE fixture(id INTEGER PRIMARY KEY)")
    _create_synthetic_root(repository, "backup")
    _create_synthetic_root(repository, "test")
    _create_synthetic_root(repository, "archive")
    private = backup / "synthetic-run" / "inventory.private.json"
    public = repository / "evidence" / "inventory-summary.json"
    result = preservation_cli.main(
        [
            "inventory",
            "--repository-root",
            str(repository),
            "--backup-root",
            str(backup),
            "--run-id",
            "synthetic-run",
            "--private-manifest",
            str(private),
            "--public-summary",
            str(public),
            "--root",
            "active=active",
            "--root",
            "backup=backup",
            "--root",
            "test=test",
            "--root",
            "archive=archive",
            "--require-role",
            "active",
            "--require-role",
            "backup",
            "--require-role",
            "test",
            "--require-role",
            "archive",
        ]
    )
    assert result == 0
    return repository, backup, public


def test_inventory_proposal_is_exact_and_observation_free() -> None:
    _assert_exact_proposal(_proposal())


@pytest.mark.parametrize("mutation", ["absent", "incomplete", "widened", "observed"])
def test_inventory_proposal_rejects_absent_incomplete_widened_or_observed_scope(
    mutation: str,
) -> None:
    proposal = _proposal()
    if mutation == "absent":
        proposal.pop("run_id")
    elif mutation == "incomplete":
        proposal["roots"] = proposal["roots"][:-1]
    elif mutation == "widened":
        proposal["roots"].append(
            {
                "alias": "active-08",
                "role": "active",
                "repository_relative_path": "unapproved",
                "required": True,
            }
        )
    else:
        proposal["roots"][0]["exists"] = True
    with pytest.raises(AssertionError):
        _assert_exact_proposal(proposal)


def test_phase1_evidence_hashes_remain_invariant() -> None:
    for path, expected in PHASE1_ARTIFACTS.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected


def test_synthetic_readiness_inventory_and_privacy(tmp_path: Path) -> None:
    _, _, public_path = _run_synthetic_inventory(tmp_path)
    public_bytes = public_path.read_bytes()
    public = json.loads(public_bytes)
    assert public["status"] == "pass"
    assert public["totals"]["root_count"] == 4
    assert b"invented fixture bytes" not in public_bytes
    assert preservation_cli.main(
        [
            "validate-summary",
            "--summary",
            str(public_path),
            "--require-role",
            "active",
            "--require-role",
            "backup",
            "--require-role",
            "test",
            "--require-role",
            "archive",
        ]
    ) == 0


def test_synthetic_readiness_missing_required_root_is_non_pass(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    backup = tmp_path / "outside"
    repository.mkdir()
    backup.mkdir()
    result = preservation_cli.main(
        [
            "inventory",
            "--repository-root",
            str(repository),
            "--backup-root",
            str(backup),
            "--run-id",
            "missing-root",
            "--private-manifest",
            str(backup / "missing-root" / "inventory.private.json"),
            "--public-summary",
            str(repository / "inventory-summary.json"),
            "--root",
            "active=missing",
            "--require-role",
            "active",
        ]
    )
    assert result == 1


@pytest.mark.parametrize("failure", ["writer", "handle", "capacity"])
def test_synthetic_readiness_quiescence_failures_are_non_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    _, backup, inventory = _run_synthetic_inventory(tmp_path)
    monkeypatch.setattr(preservation_cli, "_find_writer_processes", lambda: ())
    monkeypatch.setattr(preservation_cli, "_find_open_handles", lambda _sources: ())
    if failure == "writer":
        monkeypatch.setattr(
            preservation_cli, "_find_writer_processes", lambda: ("writer-category-1",)
        )
    elif failure == "handle":
        monkeypatch.setattr(
            preservation_cli, "_find_open_handles", lambda _sources: ("active-01:open",)
        )
    else:
        disk_usage = shutil.disk_usage(backup)
        monkeypatch.setattr(
            shutil,
            "disk_usage",
            lambda _path: type(disk_usage)(disk_usage.total, disk_usage.used, 0),
        )
    public = inventory.parent / f"quiescence-{failure}.json"
    result = preservation_cli.main(
        [
            "preflight",
            "--inventory-summary",
            str(inventory),
            "--backup-root",
            str(backup),
            "--public-summary",
            str(public),
            "--ticket-ttl-seconds",
            "900",
        ]
    )
    assert result == 3
    assert json.loads(public.read_bytes())["status"] == "blocked"


def test_synthetic_readiness_path_and_binding_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    repository, backup, inventory = _run_synthetic_inventory(tmp_path)
    monkeypatch.setattr(preservation_cli, "_find_writer_processes", lambda: ())
    monkeypatch.setattr(preservation_cli, "_find_open_handles", lambda _sources: ())
    escaped = preservation_cli.main(
        [
            "inventory",
            "--repository-root",
            str(repository),
            "--backup-root",
            str(backup),
            "--run-id",
            "escaped",
            "--private-manifest",
            str(repository / "private.json"),
            "--public-summary",
            str(repository / "escaped-summary.json"),
            "--root",
            "active=active",
        ]
    )
    assert escaped == 2
    quiescence = repository / "evidence" / "quiescence-summary.json"
    assert preservation_cli.main(
        [
            "preflight",
            "--inventory-summary",
            str(inventory),
            "--backup-root",
            str(backup),
            "--public-summary",
            str(quiescence),
            "--ticket-ttl-seconds",
            "900",
        ]
    ) == 0
    forged = repository / "evidence" / "forged-inventory.json"
    forged_value = copy.deepcopy(json.loads(inventory.read_bytes()))
    forged_value["source_set_sha256"] = "0" * 64
    forged.write_text(json.dumps(forged_value), encoding="utf-8")
    assert preservation_cli.main(
        [
            "validate-quiescence",
            "--summary",
            str(quiescence),
            "--inventory",
            str(forged),
            "--require-pass",
        ]
    ) == 4
