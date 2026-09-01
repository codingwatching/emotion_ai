"""Disposable cosine Chroma projection derived solely from SQLite truth."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import chromadb

from aura_backend.storage.models import StorageFailure
from aura_backend.storage.repository import (
    ProjectionGenerationRecord,
    ProjectionOrigin,
    StorageRepository,
)

ProjectionFaultHook = Callable[[str], None]


class EmbeddingService(Protocol):
    """Injected subset of the existing embedding service used by projections."""

    def encode_batch(self, texts: list[str]) -> list[list[float]]: ...

    def get_model_info(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class ProjectionGeneration:
    """Verified parity result for one fresh projection generation."""

    generation_id: str
    metric: str
    embedding_model: str
    embedding_config_sha256: str
    sqlite_watermark: int
    origin_count: int
    origin_ids_sha256: str


@dataclass(frozen=True, slots=True)
class ProjectionCandidate:
    """One SQLite-validated vector candidate with distinct distance measures."""

    projection_id: str
    origin_kind: str
    origin_id: str
    scope_id: str
    content: str
    content_sha256: str
    generation_id: str
    raw_distance: float
    cosine_similarity: float


class ProjectionAdapter:
    """Ordinary injected adapter for rebuildable local Chroma generations."""

    collection_name = "aura_projection"
    metric = "cosine"

    def __init__(
        self,
        *,
        projection_root: Path,
        repository: StorageRepository,
        embedding_service: EmbeddingService,
    ) -> None:
        if not projection_root.is_absolute():
            raise StorageFailure("absolute_projection_path_required")
        self.projection_root = projection_root
        self.repository = repository
        self.embedding_service = embedding_service
        self._embedding_info = self._canonical_embedding_info(
            embedding_service.get_model_info()
        )
        self.embedding_model = str(self._embedding_info["model_name"])
        self.embedding_config_sha256 = hashlib.sha256(
            json.dumps(
                self._embedding_info,
                ensure_ascii=True,
                separators=(",", ":"),
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()

    @staticmethod
    def _canonical_embedding_info(info: Mapping[str, Any]) -> dict[str, Any]:
        """Accept only scalar model facts suitable for stable Chroma metadata."""
        allowed = (str, int, float, bool)
        canonical = {
            str(key): value
            for key, value in info.items()
            if isinstance(value, allowed)
            and (not isinstance(value, float) or math.isfinite(value))
        }
        model_name = canonical.get("model_name")
        if not isinstance(model_name, str) or not model_name:
            raise StorageFailure("embedding_model_identity_missing")
        return canonical

    def generation_path(self, generation_id: str) -> Path:
        """Return a contained path for one opaque generation identifier."""
        if (
            not generation_id
            or generation_id in {".", ".."}
            or "/" in generation_id
            or "\\" in generation_id
        ):
            raise StorageFailure(
                "invalid_projection_generation_id", identifier=generation_id
            )
        return self.projection_root / generation_id

    def current_generation(self) -> ProjectionGenerationRecord:
        """Return the SQLite-selected verified generation."""
        generation = self.repository.current_projection_generation()
        if generation is None:
            raise StorageFailure("projection_generation_unavailable")
        self._validate_generation_config(generation)
        return generation

    def upsert_committed(
        self,
        turn_id: str,
        *,
        fault_hook: ProjectionFaultHook | None = None,
    ) -> int:
        """Project every active origin tied to a committed SQLite turn."""
        generation = self.current_generation()
        origins = self.repository.projection_origins(turn_id=turn_id)
        if not origins:
            raise StorageFailure("committed_turn_not_found", identifier=turn_id)
        client, collection = self._open_generation(generation)
        try:
            self._upsert_origins(
                collection,
                origins,
                generation_id=generation.generation_id,
                fault_hook=fault_hook,
            )
        finally:
            client.close()
        self.repository.mark_turn_projection_complete(turn_id)
        return len(origins)

    def delete_origins(self, projection_ids: Iterable[str]) -> int:
        """Delete derived vector rows without changing canonical SQLite origins."""
        identities = tuple(sorted(set(projection_ids)))
        if not identities:
            return 0
        generation = self.current_generation()
        client, collection = self._open_generation(generation)
        try:
            collection.delete(ids=list(identities))
        finally:
            client.close()
        return len(identities)

    def query_candidates(
        self,
        *,
        scope_id: str,
        query: str,
        n_results: int = 50,
    ) -> tuple[ProjectionCandidate, ...]:
        """Return only generation/hash/scope-valid candidates resolved in SQLite."""
        if not query or n_results <= 0 or n_results > 100:
            raise StorageFailure("invalid_projection_query")
        generation = self.current_generation()
        client, collection = self._open_generation(generation)
        try:
            count = collection.count()
            if count == 0:
                return ()
            result = collection.query(
                query_embeddings=self._embeddings([query]),
                n_results=min(n_results, count),
                where={"scope_id": {"$eq": scope_id}},
                include=["documents", "metadatas", "distances"],
            )
        finally:
            client.close()
        identities = self._first_query_row(result.get("ids"))
        documents = self._first_query_row(result.get("documents"))
        metadatas = self._first_query_row(result.get("metadatas"))
        distances = self._first_query_row(result.get("distances"))
        if not (
            len(identities)
            == len(documents)
            == len(metadatas)
            == len(distances)
        ):
            raise StorageFailure("projection_result_shape_invalid")
        candidates: list[ProjectionCandidate] = []
        for identity, document, metadata, distance in zip(
            identities, documents, metadatas, distances, strict=True
        ):
            if not isinstance(metadata, dict):
                continue
            origin = self.repository.projection_origin(str(identity))
            if origin is None or not self._metadata_matches(
                metadata,
                origin,
                generation.generation_id,
            ):
                continue
            if origin.scope_id != scope_id or str(document) != origin.content:
                continue
            raw_distance = float(distance)
            if not math.isfinite(raw_distance):
                continue
            candidates.append(
                ProjectionCandidate(
                    projection_id=origin.projection_id,
                    origin_kind=origin.origin_kind,
                    origin_id=origin.origin_id,
                    scope_id=origin.scope_id,
                    content=origin.content,
                    content_sha256=origin.content_sha256,
                    generation_id=generation.generation_id,
                    raw_distance=raw_distance,
                    cosine_similarity=min(1.0, max(0.0, 1.0 - raw_distance)),
                )
            )
        return tuple(candidates)

    def reconcile(
        self,
        *,
        fault_hook: ProjectionFaultHook | None = None,
    ) -> int:
        """Converge pending and stale projection rows to current SQLite truth."""
        self._fault(fault_hook, "before_retry")
        pending = self.repository.pending_projection_turn_ids()
        for turn_id in pending:
            self.upsert_committed(turn_id, fault_hook=fault_hook)

        generation = self.current_generation()
        expected = self._all_origins()
        expected_by_id = {item.projection_id: item for item in expected}
        client, collection = self._open_generation(generation)
        try:
            actual = collection.get(include=["documents", "metadatas"])
            actual_ids = tuple(str(value) for value in actual.get("ids", []))
            documents = actual.get("documents") or []
            metadatas = actual.get("metadatas") or []
            remove: list[str] = []
            repair: list[ProjectionOrigin] = []
            for identity, document, metadata in zip(
                actual_ids, documents, metadatas, strict=True
            ):
                origin = expected_by_id.get(identity)
                if (
                    origin is None
                    or not isinstance(metadata, dict)
                    or str(document) != origin.content
                    or not self._metadata_matches(
                        metadata, origin, generation.generation_id
                    )
                ):
                    remove.append(identity)
                    if origin is not None:
                        repair.append(origin)
            actual_set = set(actual_ids)
            repair.extend(
                origin
                for identity, origin in expected_by_id.items()
                if identity not in actual_set
            )
            if remove:
                collection.delete(ids=remove)
            if repair:
                unique = {item.projection_id: item for item in repair}
                self._upsert_origins(
                    collection,
                    tuple(unique[key] for key in sorted(unique)),
                    generation_id=generation.generation_id,
                    fault_hook=fault_hook,
                )
        finally:
            client.close()
        return len(pending)

    def rebuild(
        self,
        *,
        generation_id: str,
        created_at: str,
        page_size: int = 100,
        fault_hook: ProjectionFaultHook | None = None,
    ) -> ProjectionGeneration:
        """Build and verify a fresh directory before switching SQLite state."""
        path = self.generation_path(generation_id)
        if path.exists():
            raise StorageFailure(
                "projection_generation_path_exists", identifier=generation_id
            )
        self.repository.begin_projection_generation(
            generation_id=generation_id,
            embedding_model=self.embedding_model,
            config_sha256=self.embedding_config_sha256,
            created_at=created_at,
        )
        completed_at = created_at
        client: Any | None = None
        try:
            self.projection_root.mkdir(parents=True, exist_ok=True)
            client = chromadb.PersistentClient(path=str(path))
            if client is None:
                raise StorageFailure(
                    "projection_client_unavailable", identifier=generation_id
                )
            collection = client.create_collection(
                self.collection_name,
                configuration={"hnsw": {"space": self.metric}},
            )
            self._assert_cosine(collection)
            after = ""
            origins: list[ProjectionOrigin] = []
            while True:
                page = self.repository.projection_origins(
                    after_projection_id=after,
                    limit=page_size,
                )
                if not page:
                    break
                self._upsert_origins(
                    collection,
                    page,
                    generation_id=generation_id,
                    fault_hook=fault_hook,
                )
                origins.extend(page)
                after = page[-1].projection_id
            self._verify_parity(collection, origins, generation_id)
            self._fault(fault_hook, "before_switch")
            completed_at = datetime.now(UTC).isoformat().replace("+00:00", "Z")
            self.repository.finish_projection_generation(
                generation_id,
                status="ready",
                completed_at=completed_at,
            )
            for turn_id in self.repository.pending_projection_turn_ids():
                try:
                    self.repository.mark_turn_projection_complete(turn_id)
                except StorageFailure:
                    # The verified generation remains safe; reconciliation can
                    # retry the acknowledgement without rewriting vector truth.
                    pass
            identities = tuple(item.projection_id for item in origins)
            return ProjectionGeneration(
                generation_id=generation_id,
                metric=self.metric,
                embedding_model=self.embedding_model,
                embedding_config_sha256=self.embedding_config_sha256,
                sqlite_watermark=self.repository.sqlite_projection_watermark(),
                origin_count=len(identities),
                origin_ids_sha256=self._ids_sha256(identities),
            )
        except Exception:
            try:
                self.repository.finish_projection_generation(
                    generation_id,
                    status="failed",
                    completed_at=completed_at,
                )
            except StorageFailure:
                pass
            raise
        finally:
            if client is not None:
                client.close()

    def _all_origins(self) -> tuple[ProjectionOrigin, ...]:
        origins: list[ProjectionOrigin] = []
        after = ""
        while True:
            page = self.repository.projection_origins(
                after_projection_id=after,
                limit=1_000,
            )
            if not page:
                return tuple(origins)
            origins.extend(page)
            after = page[-1].projection_id

    def _open_generation(self, generation: ProjectionGenerationRecord) -> tuple[Any, Any]:
        path = self.generation_path(generation.generation_id)
        if not path.is_dir():
            raise StorageFailure(
                "projection_generation_directory_missing",
                identifier=generation.generation_id,
            )
        client = chromadb.PersistentClient(path=str(path))
        try:
            collection = client.get_collection(self.collection_name)
            self._assert_cosine(collection)
            return client, collection
        except Exception:
            client.close()
            raise

    def _upsert_origins(
        self,
        collection: Any,
        origins: Sequence[ProjectionOrigin],
        *,
        generation_id: str,
        fault_hook: ProjectionFaultHook | None,
    ) -> None:
        embeddings = self._embeddings([item.content for item in origins])
        for origin, embedding in zip(origins, embeddings, strict=True):
            self._fault(fault_hook, f"before_upsert:{origin.projection_id}")
            collection.upsert(
                ids=[origin.projection_id],
                documents=[origin.content],
                embeddings=[embedding],
                metadatas=[self._metadata(origin, generation_id)],
            )
            self._fault(fault_hook, f"after_upsert:{origin.projection_id}")

    def _embeddings(self, texts: list[str]) -> list[list[float]]:
        embeddings = self.embedding_service.encode_batch(texts)
        if len(embeddings) != len(texts) or any(not item for item in embeddings):
            raise StorageFailure("embedding_result_shape_invalid")
        converted = [[float(value) for value in embedding] for embedding in embeddings]
        if any(
            not all(math.isfinite(value) for value in embedding)
            for embedding in converted
        ):
            raise StorageFailure("embedding_result_invalid")
        return converted

    def _metadata(
        self, origin: ProjectionOrigin, generation_id: str
    ) -> dict[str, str | int | bool]:
        return {
            "active": True,
            "content_sha256": origin.content_sha256,
            "embedding_config_sha256": self.embedding_config_sha256,
            "embedding_model": self.embedding_model,
            "generation": generation_id,
            "origin_id": origin.origin_id,
            "origin_kind": origin.origin_kind,
            "scope_id": origin.scope_id,
            "sqlite_schema": self.repository.schema_version,
        }

    def _metadata_matches(
        self,
        metadata: Mapping[str, Any],
        origin: ProjectionOrigin,
        generation_id: str,
    ) -> bool:
        return dict(metadata) == self._metadata(origin, generation_id)

    def _verify_parity(
        self,
        collection: Any,
        origins: Sequence[ProjectionOrigin],
        generation_id: str,
    ) -> None:
        self._assert_cosine(collection)
        result = collection.get(include=["documents", "metadatas", "embeddings"])
        identities = tuple(str(value) for value in result.get("ids", []))
        expected = {item.projection_id: item for item in origins}
        if collection.count() != len(expected) or set(identities) != set(expected):
            raise StorageFailure("projection_rebuild_count_mismatch")
        documents = result.get("documents") or []
        metadatas = result.get("metadatas") or []
        embeddings = result.get("embeddings")
        if embeddings is None or not (
            len(identities) == len(documents) == len(metadatas) == len(embeddings)
        ):
            raise StorageFailure("projection_rebuild_shape_mismatch")
        for identity, document, metadata in zip(
            identities, documents, metadatas, strict=True
        ):
            origin = expected[identity]
            if (
                str(document) != origin.content
                or not isinstance(metadata, dict)
                or not self._metadata_matches(metadata, origin, generation_id)
            ):
                raise StorageFailure(
                    "projection_rebuild_hash_mismatch", identifier=identity
                )
        if identities:
            selected = min(identities)
            index = identities.index(selected)
            query = collection.query(
                query_embeddings=[cast(list[float], embeddings[index])],
                n_results=1,
                include=["distances"],
            )
            query_ids = self._first_query_row(query.get("ids"))
            distances = self._first_query_row(query.get("distances"))
            if (
                not query_ids
                or query_ids[0] not in expected
                or not distances
                or not math.isclose(float(distances[0]), 0.0, abs_tol=1e-6)
            ):
                raise StorageFailure("projection_rebuild_fixture_mismatch")

    def _validate_generation_config(
        self, generation: ProjectionGenerationRecord
    ) -> None:
        if (
            generation.metric != self.metric
            or generation.embedding_model != self.embedding_model
            or generation.config_sha256 != self.embedding_config_sha256
            or generation.build_status != "ready"
        ):
            raise StorageFailure(
                "projection_generation_config_mismatch",
                identifier=generation.generation_id,
            )

    @staticmethod
    def _assert_cosine(collection: Any) -> None:
        configuration = getattr(collection, "configuration", None)
        hnsw = configuration.get("hnsw") if isinstance(configuration, dict) else None
        if not isinstance(hnsw, dict) or hnsw.get("space") != "cosine":
            raise StorageFailure("projection_metric_mismatch")

    @staticmethod
    def _first_query_row(value: Any) -> list[Any]:
        if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], list):
            return []
        return value[0]

    @staticmethod
    def _fault(fault_hook: ProjectionFaultHook | None, stage: str) -> None:
        if fault_hook is not None:
            fault_hook(stage)

    @staticmethod
    def _ids_sha256(identities: tuple[str, ...]) -> str:
        return hashlib.sha256(
            json.dumps(identities, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
