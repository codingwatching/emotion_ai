"""Static ownership gates for the Phase 3 storage boundary."""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path

import pytest

from aura_backend.aura_internal_tools import AuraInternalTools
from aura_backend.storage import cli
from aura_backend.storage.models import RetrievalPage


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_projection_uses_repository_and_public_chroma_only() -> None:
    path = ROOT / "aura_backend" / "storage" / "projection.py"
    imports = _imports(path)
    source = path.read_text(encoding="utf-8")

    assert "aura_backend.storage.repository" in imports
    assert "sqlite3" not in imports
    assert "aura_backend.robust_vector_db" not in imports
    assert not any(name.startswith("chromadb.db") for name in imports)
    assert "chroma.sqlite3" not in source
    assert "PRAGMA" not in source
    assert "REINDEX" not in source
    assert "PersistentClient(" not in source.split("class ProjectionAdapter", 1)[0]


def test_projection_does_not_capture_singleton_or_default_data_path() -> None:
    source = (
        ROOT / "aura_backend" / "storage" / "projection.py"
    ).read_text(encoding="utf-8")

    assert "get_embedding_service" not in source
    assert "./aura_chroma_db" not in source
    assert "aura_data_v2" not in source


def test_migration_uses_repository_and_public_chroma_only() -> None:
    path = ROOT / "aura_backend" / "storage" / "migration.py"
    imports = _imports(path)
    source = path.read_text(encoding="utf-8")

    assert "aura_backend.storage.repository" in imports
    assert "sqlite3" not in imports
    assert "aura_backend.robust_vector_db" not in imports
    assert not any(name.startswith("chromadb.db") for name in imports)
    assert "chroma.sqlite3" not in source
    assert "PRAGMA" not in source
    assert "REINDEX" not in source
    assert "PersistentClient(" not in source.split("class LegacyImporter", 1)[0]
    assert "def import_path" not in source


class _InjectedRetriever:
    def __init__(self) -> None:
        self.calls: list[dict[str, object]] = []

    def retrieve(self, **kwargs: object) -> RetrievalPage:
        self.calls.append(kwargs)
        return RetrievalPage(
            items=(), next_cursor=None, has_more=False, trace_id="trace-a", traces=()
        )


@pytest.mark.asyncio
async def test_internal_search_uses_injected_owner_not_memvid_primary() -> None:
    retriever = _InjectedRetriever()
    tools = AuraInternalTools(
        None,
        None,
        retriever=retriever,
        read_owner="sqlite",
    )

    result = await tools.search_memories("scope-a", "synthetic", n_results=5000)

    assert retriever.calls == [
        {
            "scope_id": "scope-a",
            "query": "synthetic",
            "page_size": 100,
            "cursor": None,
        }
    ]
    assert result == {
        "status": "success",
        "query": "synthetic",
        "user_id": "scope-a",
        "results_count": 0,
        "memories": [],
        "next_cursor": None,
        "has_more": False,
        "trace_id": "trace-a",
    }


def test_read_owner_marker_is_explicit_and_rollback_is_atomic(tmp_path: Path) -> None:
    marker = tmp_path / "read-owner.json"

    assert cli.select_read_owner(marker, clean_install=False) == "legacy"
    assert cli.select_read_owner(marker, clean_install=True) == "sqlite"

    cli.write_read_owner_marker(marker, owner="sqlite", evidence_sha256="a" * 64)
    assert cli.select_read_owner(marker, clean_install=False) == "sqlite"

    cli.write_read_owner_marker(marker, owner="legacy", evidence_sha256="b" * 64)
    assert cli.select_read_owner(marker, clean_install=False) == "legacy"


def test_storage_cli_exposes_only_declared_fail_closed_commands() -> None:
    parser = cli.build_parser()
    subparsers = next(
        action for action in parser._actions if action.__class__.__name__ == "_SubParsersAction"
    )
    assert set(subparsers.choices) == {
        "benchmark",
        "snapshot-verify",
        "record-authorization",
        "migrate-authorized-copy",
        "switch-reads",
        "rollback-reads",
        "prepare-ci-publication",
        "verify-ci-publication",
        "collect-ci-evidence",
        "validate-evidence",
    }


def test_authorization_receipt_is_exact_private_and_public_safe(tmp_path: Path) -> None:
    restore = tmp_path / "restore.json"
    migration = tmp_path / "migration.json"
    private = tmp_path / "outside-git" / "authorization.private.json"
    public = tmp_path / "authorization.json"
    restore.write_text(json.dumps({"status": "pass", "source_aliases": ["a", "b"]}))
    migration.write_text(json.dumps({"status": "pass", "scope": "import-and-switch"}))

    receipt = cli.record_authorization(
        restore_path=restore,
        migration_plan_path=migration,
        checkpoint_result="approved",
        scope="import-and-switch",
        private_path=private,
        public_path=public,
        expires_at="2026-09-02T00:00:00Z",
        issued_at="2026-09-01T00:00:00Z",
    )

    assert receipt["status"] == "approved"
    assert private.stat().st_mode & 0o777 == 0o600
    assert str(tmp_path) not in public.read_text()
    assert "source_aliases" not in public.read_text()
    with pytest.raises(FileExistsError):
        cli.record_authorization(
            restore_path=restore,
            migration_plan_path=migration,
            checkpoint_result="approved",
            scope="import-and-switch",
            private_path=private,
            public_path=public,
            expires_at="2026-09-02T00:00:00Z",
            issued_at="2026-09-01T00:00:00Z",
        )


def test_ci_evidence_requires_current_sha_workflow_and_five_green_jobs() -> None:
    workflow_digest = hashlib.sha256(b"workflow").hexdigest()
    jobs = [
        {"name": name, "status": "completed", "conclusion": "success"}
        for name in cli.REQUIRED_CI_JOBS
    ]
    payload = {
        "head_sha": "1" * 40,
        "workflow_digest": workflow_digest,
        "jobs": jobs,
    }

    assert cli.evaluate_ci_evidence(
        payload,
        expected_sha="1" * 40,
        expected_workflow_digest=workflow_digest,
    )["status"] == "pass"
    payload["jobs"][0]["conclusion"] = "skipped"
    assert cli.evaluate_ci_evidence(
        payload,
        expected_sha="1" * 40,
        expected_workflow_digest=workflow_digest,
    )["status"] == "non_pass"
