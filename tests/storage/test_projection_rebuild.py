"""Contracts for a disposable Chroma projection of SQLite truth."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import chromadb
import pytest

from aura_backend.storage.models import StorageFailure, TurnCommand
from aura_backend.storage.projection import ProjectionAdapter
from aura_backend.storage.repository import StorageRepository, canonical_request_hash


class FakeEmbeddingService:
    """Small deterministic embedding seam with no model or network access."""

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vectors.append(
                [
                    (digest[0] + 1) / 256,
                    (digest[1] + 1) / 256,
                    (digest[2] + 1) / 256,
                ]
            )
        return vectors

    def get_model_info(self) -> dict[str, Any]:
        return {
            "model_name": "all-MiniLM-L6-v2",
            "embedding_dimension": 3,
            "normalize_embeddings": False,
        }


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _second_turn(command: TurnCommand) -> TurnCommand:
    user = replace(
        command.user_event,
        event_id="event-user-002",
        content="Synthetic second preference",
        content_sha256=_digest("Synthetic second preference"),
    )
    aura = replace(
        command.aura_event,
        event_id="event-aura-002",
        content="Synthetic second response",
        content_sha256=_digest("Synthetic second response"),
    )
    memory = replace(
        command.derived_memories[0],
        memory_id="memory-002",
        canonical_text="The synthetic user prefers explicit evidence",
        primary_source_event_id=user.event_id,
        source_event_ids=(user.event_id, aura.event_id),
    )
    return replace(
        command,
        turn_id="turn-002",
        idempotency_key="request-002",
        request_hash=canonical_request_hash(
            command.scope_id,
            command.session_id,
            user.content,
            version=command.request_hash_version,
        ),
        response_hash=_digest(aura.content),
        user_event=user,
        aura_event=aura,
        derived_memories=(memory,),
    )


def _adapter(tmp_path: Path, repository: StorageRepository) -> ProjectionAdapter:
    return ProjectionAdapter(
        projection_root=tmp_path / "projection-generations",
        repository=repository,
        embedding_service=FakeEmbeddingService(),
    )


def test_locked_chroma_api_creates_explicit_cosine_collection(tmp_path: Path) -> None:
    """The installed API must persist cosine rather than Chroma's L2 default."""
    client = chromadb.PersistentClient(path=str(tmp_path / "locked-api"))
    try:
        collection = client.create_collection(
            "projection_contract",
            configuration={"hnsw": {"space": "cosine"}},
        )
        assert collection.configuration["hnsw"]["space"] == "cosine"
        collection.upsert(
            ids=["left", "right"],
            documents=["left", "right"],
            embeddings=[[1.0, 0.0], [-1.0, 0.0]],
        )
        result = collection.query(
            query_embeddings=[[1.0, 0.0]],
            n_results=2,
            include=["distances"],
        )
        assert result["distances"] is not None
        assert result["distances"][0] == pytest.approx([0.0, 2.0])
    finally:
        client.close()


def test_stable_ids_metadata_and_cosine_query_are_bound_to_sqlite(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    adapter = _adapter(tmp_path, repository)

    generation = adapter.rebuild(
        generation_id="generation-001",
        created_at="2026-09-01T00:00:00Z",
        page_size=2,
    )

    assert generation.generation_id == "generation-001"
    assert generation.metric == "cosine"
    assert generation.origin_count == 3
    assert adapter.current_generation().generation_id == generation.generation_id

    client = chromadb.PersistentClient(path=str(adapter.generation_path("generation-001")))
    try:
        collection = client.get_collection(adapter.collection_name)
        result = collection.get(include=["documents", "metadatas"])
        assert set(result["ids"]) == {
            "event:event-user-001",
            "event:event-aura-001",
            "memory:memory-001",
        }
        metadatas = {
            identity: metadata
            for identity, metadata in zip(
                result["ids"], result["metadatas"] or [], strict=True
            )
        }
        memory_metadata = metadatas["memory:memory-001"]
        assert memory_metadata == {
            "active": True,
            "content_sha256": _digest(
                "The synthetic user prefers concise answers"
            ),
            "embedding_config_sha256": generation.embedding_config_sha256,
            "embedding_model": "all-MiniLM-L6-v2",
            "generation": "generation-001",
            "origin_id": "memory-001",
            "origin_kind": "memory",
            "scope_id": "scope-alpha",
            "sqlite_schema": 1,
        }
    finally:
        client.close()

    candidates = adapter.query_candidates(
        scope_id="scope-alpha",
        query="The synthetic user prefers concise answers",
        n_results=3,
    )
    assert candidates
    assert all(0.0 <= item.cosine_similarity <= 1.0 for item in candidates)
    assert all(item.raw_distance >= 0.0 for item in candidates)
    assert candidates[0].generation_id == "generation-001"


def test_post_commit_projection_failures_reconcile_idempotently(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    repository = StorageRepository(ledger_path)
    adapter = _adapter(tmp_path, repository)
    adapter.rebuild(
        generation_id="generation-001",
        created_at="2026-09-01T00:00:00Z",
    )

    with pytest.raises(StorageFailure, match="after_commit"):
        repository.append_turn(
            turn_command.scope_id,
            turn_command,
            fault_hook=lambda stage: (
                (_ for _ in ()).throw(StorageFailure("after_commit"))
                if stage == "after_commit"
                else None
            ),
        )
    assert repository.pending_projection_turn_ids() == (turn_command.turn_id,)

    calls = 0

    def fail_mid_upsert(stage: str) -> None:
        nonlocal calls
        if stage.startswith("after_upsert:"):
            calls += 1
            if calls == 1:
                raise StorageFailure("synthetic_projection_fault")

    with pytest.raises(StorageFailure, match="synthetic_projection_fault"):
        adapter.reconcile(fault_hook=fail_mid_upsert)
    assert repository.pending_projection_turn_ids() == (turn_command.turn_id,)

    with pytest.raises(StorageFailure, match="synthetic_before_retry"):
        adapter.reconcile(
            fault_hook=lambda stage: (
                (_ for _ in ()).throw(StorageFailure("synthetic_before_retry"))
                if stage == "before_retry"
                else None
            )
        )
    assert adapter.reconcile() == 1
    assert adapter.reconcile() == 0
    assert repository.pending_projection_turn_ids() == ()

    client = chromadb.PersistentClient(path=str(adapter.generation_path("generation-001")))
    try:
        collection = client.get_collection(adapter.collection_name)
        assert collection.count() == 3
        assert len(set(collection.get(include=[])["ids"])) == 3
    finally:
        client.close()


def test_query_rejects_tampered_hash_generation_and_orphan_origins(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    adapter = _adapter(tmp_path, repository)
    adapter.rebuild(
        generation_id="generation-001",
        created_at="2026-09-01T00:00:00Z",
    )
    path = adapter.generation_path("generation-001")
    client = chromadb.PersistentClient(path=str(path))
    try:
        collection = client.get_collection(adapter.collection_name)
        record = collection.get(
            ids=["memory:memory-001"],
            include=["documents", "embeddings", "metadatas"],
        )
        metadata = dict((record["metadatas"] or [])[0])
        metadata["content_sha256"] = "0" * 64
        collection.upsert(
            ids=["memory:memory-001"],
            documents=[(record["documents"] or [])[0]],
            embeddings=[(record["embeddings"] or [])[0]],
            metadatas=[metadata],
        )
        collection.upsert(
            ids=["event:orphan"],
            documents=["Synthetic orphan"],
            embeddings=[[1.0, 0.0, 0.0]],
            metadatas=[
                {
                    **metadata,
                    "content_sha256": _digest("Synthetic orphan"),
                    "origin_id": "orphan",
                    "origin_kind": "event",
                }
            ],
        )
    finally:
        client.close()

    candidates = adapter.query_candidates(
        scope_id="scope-alpha",
        query="The synthetic user prefers concise answers",
        n_results=10,
    )
    assert "memory-001" not in {item.origin_id for item in candidates}
    assert "orphan" not in {item.origin_id for item in candidates}


def test_failed_rebuild_never_switches_and_fresh_rebuild_has_exact_parity(
    tmp_path: Path,
    ledger_path: Path,
    turn_command: TurnCommand,
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    repository.append_turn(turn_command.scope_id, _second_turn(turn_command))
    adapter = _adapter(tmp_path, repository)
    first = adapter.rebuild(
        generation_id="generation-001",
        created_at="2026-09-01T00:00:00Z",
        page_size=2,
    )

    with pytest.raises(StorageFailure, match="synthetic_before_switch"):
        adapter.rebuild(
            generation_id="generation-bad",
            created_at="2026-09-01T00:01:00Z",
            page_size=1,
            fault_hook=lambda stage: (
                (_ for _ in ()).throw(StorageFailure("synthetic_before_switch"))
                if stage == "before_switch"
                else None
            ),
        )
    assert adapter.current_generation().generation_id == first.generation_id
    assert repository.projection_generation("generation-bad").build_status == "failed"

    second = adapter.rebuild(
        generation_id="generation-002",
        created_at="2026-09-01T00:02:00Z",
        page_size=1,
    )
    assert second.origin_count == 6
    assert adapter.current_generation().generation_id == "generation-002"
    assert repository.projection_origin_ids() == (
        "event:event-aura-001",
        "event:event-aura-002",
        "event:event-user-001",
        "event:event-user-002",
        "memory:memory-001",
        "memory:memory-002",
    )
    assert second.origin_ids_sha256 == hashlib.sha256(
        json.dumps(
            repository.projection_origin_ids(),
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

