"""Deterministic affect-neutral hybrid retrieval over SQLite and Chroma."""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import math
import re
import secrets
from collections import OrderedDict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any, Protocol

from aura_backend.storage.connection import open_database
from aura_backend.storage.models import (
    HistoryItem,
    HistoryPage,
    RetrievalGate,
    RetrievalItem,
    RetrievalPage,
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


class SalienceScorer(Protocol):
    """Phase 4 seam, constrained to exact zero throughout Phase 3."""

    def score(self, item: RetrievalItem) -> float: ...


class ZeroSalienceScorer:
    """Shipped Phase 3 scorer that cannot modify eligibility or order."""

    def score(self, item: RetrievalItem) -> float:
        """Return the exact constant-zero control for every eligible item."""
        del item
        return 0.0


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


@dataclass(frozen=True, slots=True)
class _StoredRun:
    run_id: str
    scope_id: str
    query_sha256: str
    config_sha256: str
    projection_generation: str
    expires_at: datetime
    items: tuple[RetrievalItem, ...]
    traces: tuple[RetrievalTrace, ...]


@dataclass(frozen=True, slots=True)
class _CursorBinding:
    kind: str
    expires_at: datetime
    scope_id: str
    run_id: str | None = None
    query_sha256: str | None = None
    config_sha256: str | None = None
    projection_generation: str | None = None
    offset: int = 0
    retrieval_last: tuple[float, bool, float, str] | None = None
    history_last: tuple[str, str] | None = None
    history_horizon: tuple[str, str] | None = None
    history_watermark: int | None = None


class HybridRetriever:
    """Materialize, revalidate, fuse, and explain neutral candidates."""

    def __init__(
        self,
        *,
        repository: StorageRepository,
        projection: ProjectionSource,
        config: RetrievalConfig | None = None,
        cursor_secret: bytes | None = None,
        clock: Callable[[], datetime] | None = None,
        cursor_ttl: timedelta = timedelta(minutes=5),
        max_retained_runs: int = 100,
        salience_scorer: SalienceScorer | None = None,
    ) -> None:
        self.repository = repository
        self.projection = projection
        self.config = config or RetrievalConfig()
        self._cursor_secret = cursor_secret or secrets.token_bytes(32)
        if len(self._cursor_secret) < 32:
            raise StorageFailure("cursor_secret_too_short")
        if cursor_ttl <= timedelta(0) or max_retained_runs <= 0:
            raise StorageFailure("invalid_retrieval_retention")
        self._clock = clock or (lambda: datetime.now(UTC))
        self._cursor_ttl = cursor_ttl
        self._max_retained_runs = max_retained_runs
        self._salience_scorer = salience_scorer or ZeroSalienceScorer()
        self._runs: OrderedDict[str, _StoredRun] = OrderedDict()
        self._cursors: dict[str, _CursorBinding] = {}

    def retrieve(
        self,
        *,
        scope_id: str,
        query: str,
        caller_filter: Mapping[str, Any] | None = None,
        page_size: int = 20,
        cursor: str | None = None,
    ) -> RetrievalPage:
        """Return one complete fixed neutral run for an exact mandatory scope.

        ``caller_filter`` is accepted only for compatibility. It is deliberately
        never merged into the mandatory scope or configuration-owned bounds.
        """
        del caller_filter
        if not scope_id or not query or not query.strip():
            raise StorageFailure("invalid_retrieval_query")
        self._validate_page_size(page_size)
        query_sha256 = hashlib.sha256(query.encode("utf-8")).hexdigest()
        if cursor is not None:
            return self._retrieval_page_from_cursor(
                cursor,
                scope_id=scope_id,
                query_sha256=query_sha256,
                page_size=page_size,
            )
        generation = self.projection.current_generation()
        generation_id = str(generation.generation_id)
        run_id = secrets.token_hex(16)

        eligible: dict[str, _EligibleCandidate] = {}
        rejected: list[RetrievalTrace] = []
        lexical_rows = self._query_fts(scope_id=scope_id, query=query)
        for lexical_rank, (origin_kind, origin_id, raw_value) in enumerate(
            lexical_rows,
            start=1,
        ):
            origin = self._load_origin(origin_kind, origin_id)
            rejection, gates = self._hard_gate(
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
                        gates=gates,
                    )
                )
                continue
            if origin is None:
                continue
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
            rejection, gates = self._hard_gate(
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
                        gates=gates,
                    )
                )
                continue
            if origin is None:
                continue
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
        for item in items:
            salience = self._salience_scorer.score(item)
            if not isinstance(salience, (int, float)) or float(salience) != 0.0:
                raise StorageFailure("nonzero_salience_forbidden")
        expires_at = self._now() + self._cursor_ttl
        run = _StoredRun(
            run_id=run_id,
            scope_id=scope_id,
            query_sha256=query_sha256,
            config_sha256=self.config.sha256,
            projection_generation=generation_id,
            expires_at=expires_at,
            items=tuple(items),
            traces=traces,
        )
        self._retain_run(run)
        self._persist_run(run)
        return self._retrieval_page(run, offset=0, page_size=page_size)

    def _query_fts(
        self,
        *,
        scope_id: str,
        query: str,
    ) -> tuple[tuple[str, str, float], ...]:
        tokens = tuple(
            dict.fromkeys(token.casefold() for token in _TOKEN.findall(query))
        )
        if not tokens:
            return ()
        expression = " AND ".join(
            f'"{token.replace(chr(34), chr(34) * 2)}"' for token in tokens
        )
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
                    provenance_event_ids=(origin_id,),
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
    ) -> tuple[str | None, tuple[RetrievalGate, ...]]:
        origin_exists = origin is not None
        scope_pass = bool(
            origin is not None
            and supplied_scope == required_scope
            and origin.scope_id == required_scope
        )
        provenance_pass = bool(origin is not None and origin.provenance_event_ids)
        supersession_pass = bool(
            origin is not None and not origin.superseded and not origin.retracted
        )
        content_hash_pass = False
        if origin is not None:
            calculated_hash = hashlib.sha256(origin.content.encode("utf-8")).hexdigest()
            content_hash_pass = hmac.compare_digest(
                origin.content_sha256,
                calculated_hash,
            ) and (
                supplied_hash is None
                or (
                    supplied_content == origin.content
                    and hmac.compare_digest(supplied_hash, origin.content_sha256)
                )
            )
        generation_pass = supplied_generation == current_generation
        relevance_pass = bool(
            vector_similarity is None
            or (
                math.isfinite(vector_similarity)
                and vector_similarity >= self.config.vector_similarity_floor
            )
        )
        gates = (
            RetrievalGate(
                "origin", origin_exists, None if origin_exists else "origin_missing"
            ),
            RetrievalGate(
                "scope", scope_pass, None if scope_pass else "scope_mismatch"
            ),
            RetrievalGate(
                "provenance",
                provenance_pass,
                None if provenance_pass else "provenance_missing",
            ),
            RetrievalGate(
                "supersession",
                supersession_pass,
                None if supersession_pass else "superseded",
            ),
            RetrievalGate(
                "content_hash",
                content_hash_pass,
                None if content_hash_pass else "content_hash_mismatch",
            ),
            RetrievalGate(
                "generation",
                generation_pass,
                None if generation_pass else "projection_generation_mismatch",
            ),
            RetrievalGate(
                "neutral_relevance",
                relevance_pass,
                None if relevance_pass else "neutral_relevance_failed",
            ),
        )
        if not origin_exists:
            return "origin_missing", gates
        if not scope_pass:
            return "scope_mismatch", gates
        if not provenance_pass:
            return "provenance_missing", gates
        if origin.superseded:
            return "superseded", gates
        if origin.retracted:
            return "retracted", gates
        if not content_hash_pass:
            return "content_hash_mismatch", gates
        if not generation_pass:
            return "projection_generation_mismatch", gates
        if not relevance_pass:
            return "neutral_relevance_failed", gates
        return None, gates

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
        collapsed: list[tuple[_EligibleCandidate, tuple[str, ...], tuple[str, ...]]] = (
            []
        )
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
        return (1.0 / (self.config.rrf_k + rank)) / (1.0 / (self.config.rrf_k + 1))

    def _neutral_score(self, candidate: _EligibleCandidate) -> float:
        return (
            self._rrf(candidate.lexical_rank) + self._rrf(candidate.vector_rank)
        ) / 2

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
            gates=(
                RetrievalGate("origin", True),
                RetrievalGate("scope", True),
                RetrievalGate("provenance", True),
                RetrievalGate("supersession", True),
                RetrievalGate("content_hash", True),
                RetrievalGate("generation", True),
                RetrievalGate("neutral_relevance", True),
            ),
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
        gates: tuple[RetrievalGate, ...],
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
            gates=gates,
            rejection_code=rejection_code,
            contributor_ids=(origin_id,),
            provenance_event_ids=(
                () if origin is None else origin.provenance_event_ids
            ),
            selected_rank=None,
        )

    def history(
        self,
        *,
        scope_id: str,
        page_size: int = 20,
        cursor: str | None = None,
    ) -> HistoryPage:
        """Return bounded immutable events using a frozen keyset watermark."""
        if not scope_id:
            raise StorageFailure("invalid_history_scope")
        self._validate_page_size(page_size)
        if cursor is not None:
            return self._history_page_from_cursor(
                cursor,
                scope_id=scope_id,
                page_size=page_size,
            )
        connection = open_database(self.repository.database_path)
        try:
            horizon_row = connection.execute(
                "SELECT observed_at, event_id FROM events WHERE scope_id = ? "
                "ORDER BY observed_at DESC, event_id DESC LIMIT 1",
                (scope_id,),
            ).fetchone()
            watermark = int(
                connection.execute(
                    "SELECT COALESCE(MAX(event_pk), 0) FROM events WHERE scope_id = ?",
                    (scope_id,),
                ).fetchone()[0]
            )
        finally:
            connection.close()
        if horizon_row is None:
            return HistoryPage(items=(), next_cursor=None, has_more=False)
        horizon = (str(horizon_row[0]), str(horizon_row[1]))
        return self._query_history_page(
            scope_id=scope_id,
            page_size=page_size,
            last=None,
            horizon=horizon,
            watermark=watermark,
            expires_at=self._now() + self._cursor_ttl,
        )

    def _retrieval_page(
        self,
        run: _StoredRun,
        *,
        offset: int,
        page_size: int,
    ) -> RetrievalPage:
        end = min(offset + page_size, len(run.items))
        has_more = end < len(run.items)
        next_cursor = None
        if has_more:
            next_cursor = self._issue_cursor(
                _CursorBinding(
                    kind="retrieval",
                    expires_at=run.expires_at,
                    scope_id=run.scope_id,
                    run_id=run.run_id,
                    query_sha256=run.query_sha256,
                    config_sha256=run.config_sha256,
                    projection_generation=run.projection_generation,
                    offset=end,
                    retrieval_last=self._retrieval_sort_tuple(run.items[end - 1]),
                )
            )
        return RetrievalPage(
            items=run.items[offset:end],
            next_cursor=next_cursor,
            has_more=has_more,
            trace_id=run.run_id,
            traces=run.traces,
        )

    def _retrieval_page_from_cursor(
        self,
        cursor: str,
        *,
        scope_id: str,
        query_sha256: str,
        page_size: int,
    ) -> RetrievalPage:
        binding = self._cursor_binding(cursor)
        if binding.kind != "retrieval":
            raise StorageFailure("cursor_binding_mismatch")
        if (
            binding.scope_id != scope_id
            or binding.query_sha256 != query_sha256
            or binding.config_sha256 != self.config.sha256
            or binding.run_id is None
            or binding.retrieval_last is None
        ):
            raise StorageFailure("cursor_binding_mismatch")
        run = self._runs.get(binding.run_id)
        if run is None:
            raise StorageFailure("cursor_run_unavailable")
        if (
            run.scope_id != binding.scope_id
            or run.query_sha256 != binding.query_sha256
            or run.config_sha256 != binding.config_sha256
            or run.projection_generation != binding.projection_generation
            or binding.offset <= 0
            or binding.offset > len(run.items)
            or self._retrieval_sort_tuple(run.items[binding.offset - 1])
            != binding.retrieval_last
        ):
            raise StorageFailure("cursor_binding_mismatch")
        return self._retrieval_page(run, offset=binding.offset, page_size=page_size)

    def _query_history_page(
        self,
        *,
        scope_id: str,
        page_size: int,
        last: tuple[str, str] | None,
        horizon: tuple[str, str],
        watermark: int,
        expires_at: datetime,
    ) -> HistoryPage:
        connection = open_database(self.repository.database_path)
        try:
            if last is None:
                query_sql = (
                    "SELECT event_id, scope_id, turn_id, actor, content, "
                    "content_sha256, observed_at FROM events "
                    "WHERE scope_id = ? AND event_pk <= ? "
                    "AND (observed_at < ? OR (observed_at = ? AND event_id <= ?)) "
                    "ORDER BY observed_at, event_id LIMIT ?"
                )
                parameters: list[object] = [
                    scope_id,
                    watermark,
                    horizon[0],
                    horizon[0],
                    horizon[1],
                    page_size + 1,
                ]
            else:
                query_sql = (
                    "SELECT event_id, scope_id, turn_id, actor, content, "
                    "content_sha256, observed_at FROM events "
                    "WHERE scope_id = ? AND event_pk <= ? "
                    "AND (observed_at < ? OR (observed_at = ? AND event_id <= ?)) "
                    "AND (observed_at > ? OR (observed_at = ? AND event_id > ?)) "
                    "ORDER BY observed_at, event_id LIMIT ?"
                )
                parameters = [
                    scope_id,
                    watermark,
                    horizon[0],
                    horizon[0],
                    horizon[1],
                    last[0],
                    last[0],
                    last[1],
                    page_size + 1,
                ]
            rows = connection.execute(query_sql, parameters).fetchall()
        finally:
            connection.close()
        has_more = len(rows) > page_size
        selected = rows[:page_size]
        items = tuple(
            HistoryItem(
                event_id=str(row[0]),
                scope_id=str(row[1]),
                turn_id=str(row[2]),
                actor=str(row[3]),
                content=str(row[4]),
                content_sha256=str(row[5]),
                observed_at=str(row[6]),
            )
            for row in selected
        )
        next_cursor = None
        if has_more and items:
            final = items[-1]
            next_cursor = self._issue_cursor(
                _CursorBinding(
                    kind="history",
                    expires_at=expires_at,
                    scope_id=scope_id,
                    history_last=(final.observed_at, final.event_id),
                    history_horizon=horizon,
                    history_watermark=watermark,
                )
            )
        return HistoryPage(items=items, next_cursor=next_cursor, has_more=has_more)

    def _history_page_from_cursor(
        self,
        cursor: str,
        *,
        scope_id: str,
        page_size: int,
    ) -> HistoryPage:
        binding = self._cursor_binding(cursor)
        if (
            binding.kind != "history"
            or binding.scope_id != scope_id
            or binding.history_last is None
            or binding.history_horizon is None
            or binding.history_watermark is None
        ):
            raise StorageFailure("cursor_binding_mismatch")
        return self._query_history_page(
            scope_id=scope_id,
            page_size=page_size,
            last=binding.history_last,
            horizon=binding.history_horizon,
            watermark=binding.history_watermark,
            expires_at=binding.expires_at,
        )

    def _retain_run(self, run: _StoredRun) -> None:
        self._prune_expired()
        self._runs[run.run_id] = run
        while len(self._runs) > self._max_retained_runs:
            old_run_id, _ = self._runs.popitem(last=False)
            self._cursors = {
                key: value
                for key, value in self._cursors.items()
                if value.run_id != old_run_id
            }
            self._delete_persisted_run(old_run_id)

    def _persist_run(self, run: _StoredRun) -> None:
        connection = open_database(self.repository.database_path)
        try:
            generation_exists = connection.execute(
                "SELECT 1 FROM projection_generations WHERE generation_id = ?",
                (run.projection_generation,),
            ).fetchone()
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                "INSERT INTO retrieval_runs(run_id, scope_id, query_sha256, "
                "config_version, projection_generation_id, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (
                    run.run_id,
                    run.scope_id,
                    run.query_sha256,
                    self.config.version,
                    run.projection_generation if generation_exists else None,
                    self._now().isoformat().replace("+00:00", "Z"),
                ),
            )
            for trace in run.traces:
                if trace.origin_kind not in {"event", "memory"}:
                    continue
                connection.execute(
                    "INSERT OR REPLACE INTO retrieval_candidates("
                    "run_id, origin_id, origin_kind, lexical_rank, vector_rank, "
                    "neutral_score, gate_status, rejection_code, selected_rank) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run.run_id,
                        trace.origin_id,
                        trace.origin_kind,
                        trace.lexical_rank,
                        trace.vector_rank,
                        trace.neutral_score,
                        "eligible" if trace.rejection_code is None else "rejected",
                        trace.rejection_code,
                        trace.selected_rank,
                    ),
                )
            connection.commit()
        except Exception as error:
            connection.rollback()
            raise StorageFailure(
                "retrieval_run_store_failed", identifier=run.run_id
            ) from error
        finally:
            connection.close()

    def _delete_persisted_run(self, run_id: str) -> None:
        connection = open_database(self.repository.database_path)
        try:
            connection.execute("DELETE FROM retrieval_runs WHERE run_id = ?", (run_id,))
            connection.commit()
        finally:
            connection.close()

    def _issue_cursor(self, binding: _CursorBinding) -> str:
        token_id = secrets.token_bytes(24)
        signature = hmac.new(self._cursor_secret, token_id, hashlib.sha256).digest()
        raw = token_id + signature
        token = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        self._cursors[token_id.hex()] = binding
        return token

    def _cursor_binding(self, token: str) -> _CursorBinding:
        try:
            padding = "=" * (-len(token) % 4)
            raw = base64.b64decode(
                token + padding,
                altchars=b"-_",
                validate=True,
            )
            canonical = base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")
        except (ValueError, UnicodeEncodeError) as error:
            raise StorageFailure("cursor_invalid") from error
        if canonical != token or len(raw) != 56:
            raise StorageFailure("cursor_invalid")
        token_id, supplied_signature = raw[:24], raw[24:]
        expected_signature = hmac.new(
            self._cursor_secret,
            token_id,
            hashlib.sha256,
        ).digest()
        if not hmac.compare_digest(supplied_signature, expected_signature):
            raise StorageFailure("cursor_invalid")
        binding = self._cursors.get(token_id.hex())
        if binding is None:
            raise StorageFailure("cursor_invalid")
        if self._now() > binding.expires_at:
            self._cursors.pop(token_id.hex(), None)
            raise StorageFailure("cursor_expired")
        return binding

    def _prune_expired(self) -> None:
        now = self._now()
        expired_runs = {
            run_id for run_id, run in self._runs.items() if now > run.expires_at
        }
        for run_id in expired_runs:
            self._runs.pop(run_id, None)
            self._delete_persisted_run(run_id)
        self._cursors = {
            key: binding
            for key, binding in self._cursors.items()
            if now <= binding.expires_at and binding.run_id not in expired_runs
        }

    def _now(self) -> datetime:
        value = self._clock()
        if value.tzinfo is None:
            raise StorageFailure("clock_timezone_required")
        return value.astimezone(UTC)

    @staticmethod
    def _validate_page_size(page_size: int) -> None:
        if isinstance(page_size, bool) or not 1 <= page_size <= 100:
            raise StorageFailure("invalid_page_size")

    @staticmethod
    def _retrieval_sort_tuple(item: RetrievalItem) -> tuple[float, bool, float, str]:
        """Return the immutable neutral ordering boundary stored by a cursor."""
        timestamp = datetime.fromisoformat(
            item.observed_at.replace("Z", "+00:00")
        ).timestamp()
        return (-item.neutral_score, not item.exact_match, -timestamp, item.origin_id)

    @staticmethod
    def _normalize(value: str) -> str:
        return " ".join(value.casefold().split())
