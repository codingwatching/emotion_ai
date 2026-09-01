"""Contracts for immutable, authorized-copy legacy Chroma import."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import chromadb
import pytest

from aura_backend.storage.migration import (
    LegacyImportAuthorization,
    LegacyImporter,
    legacy_tree_sha256,
)
from aura_backend.storage.models import StorageFailure
from aura_backend.storage.repository import StorageRepository


def _create_legacy_root(path: Path, records: list[dict[str, Any]]) -> None:
    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.create_collection("legacy_conversations")
        collection.add(
            ids=[str(record["id"]) for record in records],
            documents=[str(record["document"]) for record in records],
            embeddings=[record["embedding"] for record in records],
            metadatas=[record["metadata"] for record in records],
        )
    finally:
        client.close()


def _authorization(
    *,
    source_alias: str,
    disposable_root: Path,
    workspace: Path,
    forbidden_roots: tuple[Path, ...],
) -> LegacyImportAuthorization:
    fingerprint = legacy_tree_sha256(disposable_root)
    return LegacyImportAuthorization(
        source_alias=source_alias,
        root_fingerprint=fingerprint,
        disposable_root=disposable_root,
        restore_workspace=workspace,
        allowlisted_sources=((source_alias, fingerprint),),
        evidence_status="pass",
        evidence_sha256="a" * 64,
        forbidden_roots=forbidden_roots,
        anomaly_codes=("preserved_legacy_fk_anomaly",),
    )


@pytest.fixture
def synthetic_restores(tmp_path: Path) -> tuple[Path, Path, Path, tuple[Path, ...]]:
    """Create divergent sources, then copy them into a disposable workspace."""
    originals = tmp_path / "synthetic-originals"
    durable = tmp_path / "synthetic-durable-backup"
    workspace = tmp_path / "authorized-restore-workspace"
    originals.mkdir()
    durable.mkdir()
    workspace.mkdir()

    root_a = originals / "active-a"
    root_b = originals / "active-b"
    shared = "Invented duplicate legacy content"
    _create_legacy_root(
        root_a,
        [
            {
                "id": "duplicate-a",
                "document": shared,
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {
                    "user_id": "synthetic-user",
                    "sender": "user",
                    "timestamp": "2024-01-01T00:00:00Z",
                    "unmapped_field": "preserve-a",
                },
            },
            {
                "id": "missing-time",
                "document": "Invented fragment without time",
                "embedding": [0.0, 1.0, 0.0],
                "metadata": {"user_id": "synthetic-user", "sender": "aura"},
            },
            {
                "id": "malformed-meta",
                "document": "Invented fragment with malformed timestamp",
                "embedding": [0.0, 0.0, 1.0],
                "metadata": {
                    "user_id": "synthetic-user",
                    "timestamp": "not-a-timestamp",
                    "confidence": "NaN",
                },
            },
        ],
    )
    _create_legacy_root(
        root_b,
        [
            {
                "id": "duplicate-b",
                "document": shared,
                "embedding": [1.0, 0.0, 0.0],
                "metadata": {
                    "user_id": "synthetic-user",
                    "sender": "user",
                    "timestamp": "2024-01-02T00:00:00Z",
                    "unmapped_field": "preserve-b",
                },
            },
            {
                "id": "root-b-only",
                "document": "Invented divergent second-root evidence",
                "embedding": [0.5, 0.5, 0.0],
                "metadata": {"sender": "user"},
            },
        ],
    )
    shutil.copytree(root_a, durable / "active-a")
    shutil.copytree(root_b, durable / "active-b")
    restored_a = workspace / "attempt-001" / "active-a"
    restored_b = workspace / "attempt-001" / "active-b"
    restored_a.parent.mkdir()
    shutil.copytree(durable / "active-a", restored_a)
    shutil.copytree(durable / "active-b", restored_b)
    return restored_a, restored_b, workspace, (root_a, root_b, durable)


def test_two_roots_import_independently_with_exact_rerun_safety(
    tmp_path: Path,
    ledger_path: Path,
    synthetic_restores: tuple[Path, Path, Path, tuple[Path, ...]],
) -> None:
    restored_a, restored_b, workspace, forbidden = synthetic_restores
    repository = StorageRepository(ledger_path)
    importer = LegacyImporter(repository=repository)
    auth_a = _authorization(
        source_alias="active-a",
        disposable_root=restored_a,
        workspace=workspace,
        forbidden_roots=forbidden,
    )
    auth_b = _authorization(
        source_alias="active-b",
        disposable_root=restored_b,
        workspace=workspace,
        forbidden_roots=forbidden,
    )
    before_a = legacy_tree_sha256(restored_a)
    before_b = legacy_tree_sha256(restored_b)

    first_a = importer.import_authorized(auth_a, page_size=2)
    first_b = importer.import_authorized(auth_b, page_size=1)
    rerun_a = importer.import_authorized(auth_a, page_size=1)
    rerun_b = importer.import_authorized(auth_b, page_size=2)

    assert first_a.imported_count == 3
    assert first_b.imported_count == 2
    assert rerun_a.imported_count == rerun_b.imported_count == 0
    assert rerun_a.existing_count == 3
    assert rerun_b.existing_count == 2
    assert before_a == legacy_tree_sha256(restored_a) == first_a.source_after_sha256
    assert before_b == legacy_tree_sha256(restored_b) == first_b.source_after_sha256

    sources = repository.legacy_source_records()
    assert len(sources) == 5
    assert {
        (row.root_fingerprint, row.collection_name, row.legacy_id)
        for row in sources
    } == {
        (auth_a.root_fingerprint, "legacy_conversations", "duplicate-a"),
        (auth_a.root_fingerprint, "legacy_conversations", "missing-time"),
        (auth_a.root_fingerprint, "legacy_conversations", "malformed-meta"),
        (auth_b.root_fingerprint, "legacy_conversations", "duplicate-b"),
        (auth_b.root_fingerprint, "legacy_conversations", "root-b-only"),
    }
    assert len({row.imported_origin_id for row in sources}) == 5


def test_duplicates_fragments_raw_metadata_and_anomalies_remain_explicit(
    ledger_path: Path,
    synthetic_restores: tuple[Path, Path, Path, tuple[Path, ...]],
) -> None:
    restored_a, restored_b, workspace, forbidden = synthetic_restores
    repository = StorageRepository(ledger_path)
    importer = LegacyImporter(repository=repository)
    auth_a = _authorization(
        source_alias="active-a",
        disposable_root=restored_a,
        workspace=workspace,
        forbidden_roots=forbidden,
    )
    auth_b = _authorization(
        source_alias="active-b",
        disposable_root=restored_b,
        workspace=workspace,
        forbidden_roots=forbidden,
    )

    result_a = importer.import_authorized(auth_a, page_size=2)
    result_b = importer.import_authorized(auth_b, page_size=2)
    fragments = repository.legacy_fragment_records()

    duplicate_fragments = [
        row for row in fragments if row.content == "Invented duplicate legacy content"
    ]
    assert len(duplicate_fragments) == 2
    assert len({row.fragment_id for row in duplicate_fragments}) == 2
    assert {row.source_alias for row in duplicate_fragments} == {
        "active-a",
        "active-b",
    }
    missing = next(row for row in fragments if row.legacy_id == "missing-time")
    malformed = next(row for row in fragments if row.legacy_id == "malformed-meta")
    assert missing.observed_at is None
    assert "timestamp_missing" in missing.status_codes
    assert "pair_unproven" in missing.status_codes
    assert "timestamp_malformed" in malformed.status_codes
    assert "metadata_nonfinite" in malformed.status_codes
    assert "unmapped_field" in duplicate_fragments[0].raw_metadata_json
    assert result_a.anomaly_counts == {"preserved_legacy_fk_anomaly": 1}
    assert result_b.anomaly_counts == {"preserved_legacy_fk_anomaly": 1}
    assert all("Invented" not in code for code in result_a.mapping_codes)


@pytest.mark.parametrize(
    "case",
    [
        "original",
        "durable",
        "outside_workspace",
        "symlink",
        "inner_symlink",
        "missing_evidence",
        "wrong_alias",
        "wrong_fingerprint",
    ],
)
def test_unauthorized_paths_fail_before_chroma_opens(
    case: str,
    tmp_path: Path,
    ledger_path: Path,
    synthetic_restores: tuple[Path, Path, Path, tuple[Path, ...]],
) -> None:
    restored_a, _, workspace, forbidden = synthetic_restores
    original, _, durable = forbidden
    opened: list[Path] = []

    def spy(path: str) -> Any:
        opened.append(Path(path))
        raise AssertionError("Chroma must not open an unauthorized path")

    importer = LegacyImporter(
        repository=StorageRepository(ledger_path),
        client_factory=spy,
    )
    authorization = _authorization(
        source_alias="active-a",
        disposable_root=restored_a,
        workspace=workspace,
        forbidden_roots=forbidden,
    )
    if case == "original":
        authorization = authorization.replace(disposable_root=original)
    elif case == "durable":
        authorization = authorization.replace(disposable_root=durable / "active-a")
    elif case == "outside_workspace":
        outside = tmp_path / "outside"
        shutil.copytree(restored_a, outside)
        authorization = authorization.replace(
            disposable_root=outside,
            root_fingerprint=legacy_tree_sha256(outside),
            allowlisted_sources=(("active-a", legacy_tree_sha256(outside)),),
        )
    elif case == "symlink":
        linked = workspace / "linked-root"
        linked.symlink_to(restored_a, target_is_directory=True)
        authorization = authorization.replace(disposable_root=linked)
    elif case == "inner_symlink":
        (restored_a / "external-link").symlink_to(
            tmp_path / "outside-sentinel",
            target_is_directory=False,
        )
    elif case == "missing_evidence":
        authorization = authorization.replace(evidence_status="not_run")
    elif case == "wrong_alias":
        authorization = authorization.replace(source_alias="active-other")
    elif case == "wrong_fingerprint":
        authorization = authorization.replace(root_fingerprint="0" * 64)

    with pytest.raises(StorageFailure):
        importer.import_authorized(authorization)
    assert opened == []


def test_importer_opens_only_an_operation_copy_and_keeps_source_immutable(
    ledger_path: Path,
    synthetic_restores: tuple[Path, Path, Path, tuple[Path, ...]],
) -> None:
    restored_a, _, workspace, forbidden = synthetic_restores
    authorization = _authorization(
        source_alias="active-a",
        disposable_root=restored_a,
        workspace=workspace,
        forbidden_roots=forbidden,
    )
    source_before = legacy_tree_sha256(restored_a)

    class MutatingClient:
        def __init__(self, path: str) -> None:
            self.path = Path(path)
            self.real = chromadb.PersistentClient(path=path)

        def list_collections(self, **kwargs: Any) -> Any:
            (self.path / "mutation-marker").write_bytes(b"mutated")
            return self.real.list_collections(**kwargs)

        def get_collection(self, name: str) -> Any:
            return self.real.get_collection(name)

        def close(self) -> None:
            self.real.close()

    importer = LegacyImporter(
        repository=StorageRepository(ledger_path),
        client_factory=MutatingClient,
    )
    result = importer.import_authorized(authorization)
    assert result.imported_count == 3
    assert result.source_before_sha256 == result.source_after_sha256 == source_before
    assert legacy_tree_sha256(restored_a) == source_before
