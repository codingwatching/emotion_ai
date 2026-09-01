"""Deterministic affect-neutral hybrid retrieval over SQLite and Chroma."""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol

from aura_backend.storage.connection import open_database
from aura_backend.storage.models import (
    RetrievalGate,
    RetrievalItem,
    RetrievalResult,
    RetrievalTrace,
    StorageFailure,
)
from aura_backend.storage.projection import ProjectionCandidate
from aura_backend.storage.repository import StorageRepository


_TOKEN = re.compile(r"\w+", flags=re.UNICODE)


class ProjectionSource(Protocol):
    """Injected vector-candidate surface used by the neutral retriever."""

    def current_generation(self) -> Any: ...

    def query_candidates(
        self,
        *,
        scope_id: str,
        query: str,
        n_results: int = 50,
    ) -> tuple[ProjectionCandidate, ...]: ...


@dataclass(frozen=True, slots=True)
class RetrievalConfig:
    """Versioned fixed bounds for Phase 3 neutral retrieval."""

    version: int = 1
    lexical_candidate_cap: int = 50
    vector_candidate_cap: int = 50
    rrf_k: int = 60
    vector_similarity_floor: float = 0.2

    def __post_init__(self) -> None:
        if (
            self.version <= 0
            or not 1 <= self.lexical_candidate_cap <= 50
            or not 1 <= self.vector_candidate_cap <= 50
            or self.rrf_k != 60
            or not 0.0 <= self.vector_similarity_floor <= 1.0
        ):
            raise StorageFailure("invalid_retrieval_config")

    @property
    def sha256(self) -> str:
        """Return a canonical identifier without caller-controlled data."""
        payload = json.dumps(
            {
                "lexical_candidate_cap": self.lexical_candidate_cap,
                "rrf_k": self.rrf_k,
                "vector_candidate_cap": self.vector_candidate_cap,
                "vector_similarity_floor": self.vector_similarity_floor,
                "version": self.version,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True, slots=True)
class _AuthoritativeOrigin:
    origin_id: str
    origin_kind: str
    scope_id: str
    content: str
    content_sha256: str
    observed_at: str
    provenance_event_ids: tuple[str, ...]
    superseded: bool
    retracted: bool


@dataclass(slots=True)
class _EligibleCandidate:
    origin: _AuthoritativeOrigin
    lexical_rank: int | None = None
    lexical_raw_value: float | None = None
    vector_rank: int | None = None
    vector_raw_distance: float | None = None
    vector_similarity: float | None = None
    exact_match: bool = False


class HybridRetriever:
    """Materialize, revalidate, fuse, and explain neutral candidates."""

    def __init__(
        self,
        *,
        repository: StorageRepository,
        projection: ProjectionSource,
        config: RetrievalConfig | None = None,
    ) -> None:
        self.repository = repository
        self.projection = projection
        self.config = config or RetrievalConfig()

    def retrieve(
        self,
        *,
        scope_id: str,
        query: str,
        caller_filter: Mapping[str, Any] | None = None,
    ) -> RetrievalResult:
        """Return one complete fixed neutral run for an exact mandatory scope.

        ``caller_filter`` is accepted only for compatibility. It is deliberately
        never merged into the mandatory scope or configuration-owned bounds.
        """
        del caller_filter
        if not scope_id or not query or not query.strip():
            raise StorageFailure("invalid_retrieval_query")
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        generation = self.projection.current_generation()
        generation_id = str(generation.generation_id)
        run_id = hashlib.sha256(
            json.dumps(
                (scope_id, query_sha256, self.config.sha256, generation_id),
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()[:32]

        eligible: dict[str, _EligibleCandidate] = {}
        rejected: list[RetrievalTrace] = []
        lexical_rows = self._query_fts(scope_id=scope_id, query=query)
        for lexical_rank, (origin_kind, origin_id, raw_value) in enumerate(
            lexical_rows,
            start=1,
        ):
            origin = self._load_origin(origin_kind, origin_id)
            rejection = self._hard_gate(
                origin=origin,
                required_scope=scope_id,
                supplied_scope=scope_id,
                supplied_hash=None,
                supplied_content=None,
                supplied_generation=generation_id,
                current_generation=generation_id,
                vector_similarity=None,
            )
            if rejection is not None:
                rejected.append(
                    self._rejected_trace(
                        run_id=run_id,
                        query_sha256=query_sha256,
                        generation_id=generation_id,
                        origin_kind=origin_kind,
                        origin_id=origin_id,
                        origin=origin,
                        rejection_code=rejection,
                        lexical_rank=lexical_rank,
                        lexical_raw_value=raw_value,
                    )
                )
                continue
            assert origin is not None
            candidate = eligible.setdefault(origin_id, _EligibleCandidate(origin))
            if candidate.lexical_rank is None:
                candidate.lexical_rank = lexical_rank
                candidate.lexical_raw_value = raw_value
            candidate.exact_match = self._normalize(origin.content) == self._normalize(
                query
            )

        vector_rows = self.projection.query_candidates(
            scope_id=scope_id,
            query=query,
            n_results=self.config.vector_candidate_cap,
        )[: self.config.vector_candidate_cap]
        for vector_rank, vector in enumerate(vector_rows, start=1):
            identity_matches = vector.projection_id == (
                f"{vector.origin_kind}:{vector.origin_id}"
            )
            origin = (
                self._load_origin(vector.origin_kind, vector.origin_id)
                if identity_matches
                else None
            )
            rejection = self._hard_gate(
                origin=origin,
                required_scope=scope_id,
                supplied_scope=vector.scope_id,
                supplied_hash=vector.content_sha256,
                supplied_content=vector.content,
                supplied_generation=vector.generation_id,
                current_generation=generation_id,
                vector_similarity=vector.cosine_similarity,
            )
            if rejection is not None:
                rejected.append(
                    self._rejected_trace(
                        run_id=run_id,
                        query_sha256=query_sha256,
                        generation_id=generation_id,
                        origin_kind=vector.origin_kind,
                        origin_id=vector.origin_id,
                        origin=origin,
                        rejection_code=rejection,
                        vector_rank=vector_rank,
                        vector_raw_distance=vector.raw_distance,
                        vector_similarity=vector.cosine_similarity,
                    )
                )
                continue
            assert origin is not None
            candidate = eligible.setdefault(
                origin.origin_id,
                _EligibleCandidate(origin),
            )
            if candidate.vector_rank is None:
                candidate.vector_rank = vector_rank
                candidate.vector_raw_distance = vector.raw_distance
                candidate.vector_similarity = vector.cosine_similarity
            candidate.exact_match = candidate.exact_match or (
                self._normalize(origin.content) == self._normalize(query)
            )

        ranked, collapsed = self._rank_and_deduplicate(eligible.values())
        items: list[RetrievalItem] = []
        selected_traces: list[RetrievalTrace] = []
        for selected_rank, (candidate, contributors, provenance) in enumerate(
            ranked,
            start=1,
        ):
            neutral_score = self._neutral_score(candidate)
            item = RetrievalItem(
                origin_id=candidate.origin.origin_id,
                origin_kind=candidate.origin.origin_kind,
                scope_id=candidate.origin.scope_id,
                content=candidate.origin.content,
                content_sha256=candidate.origin.content_sha256,
                observed_at=candidate.origin.observed_at,
                provenance_event_ids=provenance,
                contributor_ids=contributors,
                neutral_score=neutral_score,
                exact_match=candidate.exact_match,
                selected_rank=selected_rank,
            )
            items.append(item)
            selected_traces.append(
                self._eligible_trace(
                    run_id=run_id,
                    query_sha256=query_sha256,
                    generation_id=generation_id,
                    candidate=candidate,
                    contributor_ids=contributors,
                    provenance_event_ids=provenance,
                    selected_rank=selected_rank,
                )
            )

        collapsed_traces = [
            self._eligible_trace(
                run_id=run_id,
                query_sha256=query_sha256,
                generation_id=generation_id,
                candidate=candidate,
                contributor_ids=contributors,
                provenance_event_ids=provenance,
                selected_rank=None,
                rejection_code="duplicate_collapsed",
            )
            for candidate, contributors, provenance in collapsed
        ]
        traces = tuple(
            selected_traces
            + sorted(
                collapsed_traces + rejected,
                key=lambda trace: (
                    trace.rejection_code or "",
                    trace.origin_kind,
                    trace.origin_id,
                ),
            )
        )
        return RetrievalResult(run_id=run_id, items=tuple(items), traces=traces)

    def _query_fts(
        self,
        *,
        scope_id: str,
        query: str,
    ) -> tuple[tuple[str, str, float], ...]:
        tokens = tuple(dict.fromkeys(token.casefold() for token in _TOKEN.findall(query)))
        if not tokens:
            return ()
        expression = " AND ".join(f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens)
        connection = open_database(self.repository.database_path)
        try:
            rows = connection.execute(
                """
                SELECT origin_kind, origin_id, raw_value
                FROM (
                    SELECT 'event' AS origin_kind, event.event_id AS origin_id,
                           bm25(event_fts) AS raw_value
                    FROM event_fts
                    JOIN events AS event ON event.event_pk = event_fts.rowid
                    WHERE event_fts MATCH ? AND event.scope_id = ?
                    UNION ALL
                    SELECT 'memory' AS origin_kind, memory.memory_id AS origin_id,
                           bm25(memory_fts) AS raw_value
                    FROM memory_fts
                    JOIN derived_memories AS memory
                      ON memory.memory_pk = memory_fts.rowid
                    WHERE memory_fts MATCH ? AND memory.scope_id = ?
                )
                ORDER BY raw_value ASC, origin_kind ASC, origin_id ASC
                LIMIT ?
                """,
                (
                    expression,
                    scope_id,
                    expression,
                    scope_id,
                    self.config.lexical_candidate_cap,
                ),
            ).fetchall()
            return tuple((str(row[0]), str(row[1]), float(row[2])) for row in rows)
        finally:
            connection.close()

    def _load_origin(
        self,
        origin_kind: str,
        origin_id: str,
    ) -> _AuthoritativeOrigin | None:
        connection = open_database(self.repository.database_path)
        try:
            if origin_kind == "event":
                row = connection.execute(
                    "SELECT scope_id, content, content_sha256, observed_at "
                    "FROM events WHERE event_id = ?",
                    (origin_id,),
                ).fetchone()
                if row is None:
                    return None
                return _AuthoritativeOrigin(
                    origin_id=origin_id,
                    origin_kind="event",
                    scope_id=str(row[0]),
                    content=str(row[1]),
                    content_sha256=str(row[2]),
                    observed_at=str(row[3]),
                    provenance_event_ids=(),
                    superseded=False,
                    retracted=False,
                )
            if origin_kind != "memory":
                return None
            row = connection.execute(
                "SELECT scope_id, canonical_text, content_sha256, created_at "
                "FROM derived_memories WHERE memory_id = ?",
                (origin_id,),
            ).fetchone()
            if row is None:
                return None
            provenance = tuple(
                str(source[0])
                for source in connection.execute(
                    "SELECT event_id FROM memory_sources WHERE memory_id = ? "
                    "ORDER BY event_id",
                    (origin_id,),
                ).fetchall()
            )
            superseded = connection.execute(
                "SELECT 1 FROM memory_supersessions WHERE old_memory_id = ? LIMIT 1",
                (origin_id,),
            ).fetchone()
            retracted = connection.execute(
                "SELECT 1 FROM memory_retractions WHERE memory_id = ? LIMIT 1",
                (origin_id,),
            ).fetchone()
            return _AuthoritativeOrigin(
                origin_id=origin_id,
                origin_kind="memory",
                scope_id=str(row[0]),
                content=str(row[1]),
                content_sha256=str(row[2]),
                observed_at=str(row[3]),
                provenance_event_ids=provenance,
                superseded=superseded is not None,
                retracted=retracted is not None,
            )
        finally:
            connection.close()

    def _hard_gate(
        self,
        *,
        origin: _AuthoritativeOrigin | None,
        required_scope: str,
        supplied_scope: str,
        supplied_hash: str | None,
        supplied_content: str | None,
        supplied_generation: str,
        current_generation: str,
        vector_similarity: float | None,
    ) -> str | None:
        if origin is None:
            return "origin_missing"
        if supplied_scope != required_scope or origin.scope_id != required_scope:
            return "scope_mismatch"
        if origin.origin_kind == "memory" and not origin.provenance_event_ids:
            return "provenance_missing"
        if origin.superseded:
            return "superseded"
        if origin.retracted:
            return "retracted"
        calculated_hash = hashlib.sha256(origin.content.encode("utf-8")).hexdigest()
        if not hmac.compare_digest(origin.content_sha256, calculated_hash):
            return "content_hash_mismatch"
        if supplied_hash is not None and (
            supplied_content != origin.content
            or not hmac.compare_digest(supplied_hash, origin.content_sha256)
        ):
            return "content_hash_mismatch"
        if supplied_generation != current_generation:
            return "projection_generation_mismatch"
        if vector_similarity is not None and (
            not math.isfinite(vector_similarity)
            or vector_similarity < self.config.vector_similarity_floor
        ):
            return "neutral_relevance_failed"
        return None

    def _rank_and_deduplicate(
        self,
        candidates: Any,
    ) -> tuple[
        list[tuple[_EligibleCandidate, tuple[str, ...], tuple[str, ...]]],
        list[tuple[_EligibleCandidate, tuple[str, ...], tuple[str, ...]]],
    ]:
        groups: dict[str, list[_EligibleCandidate]] = {}
        for candidate in candidates:
            groups.setdefault(candidate.origin.content_sha256, []).append(candidate)
        selected: list[tuple[_EligibleCandidate, tuple[str, ...], tuple[str, ...]]] = []
        collapsed: list[tuple[_EligibleCandidate, tuple[str, ...], tuple[str, ...]]] = []
        for group in groups.values():
            ordered = sorted(group, key=self._sort_key)
            contributors = tuple(sorted(item.origin.origin_id for item in group))
            provenance = tuple(
                sorted(
                    {
                        event_id
                        for item in group
                        for event_id in item.origin.provenance_event_ids
                    }
                )
            )
            selected.append((ordered[0], contributors, provenance))
            collapsed.extend(
                (candidate, contributors, provenance) for candidate in ordered[1:]
            )
        selected.sort(key=lambda entry: self._sort_key(entry[0]))
        return selected, collapsed

    def _sort_key(self, candidate: _EligibleCandidate) -> tuple[Any, ...]:
        timestamp = datetime.fromisoformat(
            candidate.origin.observed_at.replace("Z", "+00:00")
        ).timestamp()
        return (
            -self._neutral_score(candidate),
            not candidate.exact_match,
            -timestamp,
            candidate.origin.origin_id,
        )

    def _rrf(self, rank: int | None) -> float:
        if rank is None:
            return 0.0
        return (1.0 / (self.config.rrf_k + rank)) / (
            1.0 / (self.config.rrf_k + 1)
        )

    def _neutral_score(self, candidate: _EligibleCandidate) -> float:
        return (self._rrf(candidate.lexical_rank) + self._rrf(candidate.vector_rank)) / 2

    def _eligible_trace(
        self,
        *,
        run_id: str,
        query_sha256: str,
        generation_id: str,
        candidate: _EligibleCandidate,
        contributor_ids: tuple[str, ...],
        provenance_event_ids: tuple[str, ...],
        selected_rank: int | None,
        rejection_code: str | None = None,
    ) -> RetrievalTrace:
        origin = candidate.origin
        return RetrievalTrace(
            run_id=run_id,
            query_sha256=query_sha256,
            config_sha256=self.config.sha256,
            sqlite_schema=self.repository.schema_version,
            projection_generation=generation_id,
            origin_id=origin.origin_id,
            origin_kind=origin.origin_kind,
            scope_id=origin.scope_id,
            content_sha256=origin.content_sha256,
            observed_at=origin.observed_at,
            lexical_rank=candidate.lexical_rank,
            lexical_raw_value=candidate.lexical_raw_value,
            vector_rank=candidate.vector_rank,
            vector_raw_distance=candidate.vector_raw_distance,
            vector_similarity=candidate.vector_similarity,
            lexical_rrf=self._rrf(candidate.lexical_rank),
            vector_rrf=self._rrf(candidate.vector_rank),
            neutral_score=self._neutral_score(candidate),
            exact_match=candidate.exact_match,
            gates=(RetrievalGate("eligible", rejection_code is None, rejection_code),),
            rejection_code=rejection_code,
            contributor_ids=contributor_ids,
            provenance_event_ids=provenance_event_ids,
            selected_rank=selected_rank,
        )

    def _rejected_trace(
        self,
        *,
        run_id: str,
        query_sha256: str,
        generation_id: str,
        origin_kind: str,
        origin_id: str,
        origin: _AuthoritativeOrigin | None,
        rejection_code: str,
        lexical_rank: int | None = None,
        lexical_raw_value: float | None = None,
        vector_rank: int | None = None,
        vector_raw_distance: float | None = None,
        vector_similarity: float | None = None,
    ) -> RetrievalTrace:
        return RetrievalTrace(
            run_id=run_id,
            query_sha256=query_sha256,
            config_sha256=self.config.sha256,
            sqlite_schema=self.repository.schema_version,
            projection_generation=generation_id,
            origin_id=origin_id,
            origin_kind=origin_kind,
            scope_id=None if origin is None else origin.scope_id,
            content_sha256=None if origin is None else origin.content_sha256,
            observed_at=None if origin is None else origin.observed_at,
            lexical_rank=lexical_rank,
            lexical_raw_value=lexical_raw_value,
            vector_rank=vector_rank,
            vector_raw_distance=vector_raw_distance,
            vector_similarity=vector_similarity,
            lexical_rrf=self._rrf(lexical_rank),
            vector_rrf=self._rrf(vector_rank),
            neutral_score=0.0,
            exact_match=False,
            gates=(RetrievalGate(rejection_code, False, rejection_code),),
            rejection_code=rejection_code,
            contributor_ids=(origin_id,),
            provenance_event_ids=(
                () if origin is None else origin.provenance_event_ids
            ),
            selected_rank=None,
        )

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())
