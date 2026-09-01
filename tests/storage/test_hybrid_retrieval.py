"""Adversarial contracts for affect-neutral hybrid retrieval."""

from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from aura_backend.storage.models import (
    DerivedMemoryInput,
    EpistemicStatus,
    EventInput,
    MemoryKind,
    TurnCommand,
)
from aura_backend.storage.projection import ProjectionCandidate
from aura_backend.storage.repository import StorageRepository, canonical_request_hash
from aura_backend.storage.retrieval import HybridRetriever, RetrievalConfig


def _digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _command(
    *,
    scope: str,
    suffix: str,
    user_text: str,
    aura_text: str,
    memory_text: str,
    observed_at: str,
) -> TurnCommand:
    user = EventInput(
        event_id=f"event-user-{suffix}",
        actor="user",
        content=user_text,
        observed_at=observed_at,
        content_sha256=_digest(user_text),
    )
    aura = EventInput(
        event_id=f"event-aura-{suffix}",
        actor="aura",
        content=aura_text,
        observed_at=observed_at,
        content_sha256=_digest(aura_text),
    )
    memory = DerivedMemoryInput(
        memory_id=f"memory-{suffix}",
        kind=MemoryKind.FACT,
        canonical_text=memory_text,
        confidence=0.9,
        epistemic_status=EpistemicStatus.INFERRED,
        primary_source_event_id=user.event_id,
        source_event_ids=(user.event_id, aura.event_id),
        created_at=observed_at,
    )
    return TurnCommand(
        scope_id=scope,
        session_id=f"session-{scope}",
        turn_id=f"turn-{suffix}",
        idempotency_key=f"request-{suffix}",
        request_hash_version=1,
        request_hash=canonical_request_hash(
            scope,
            f"session-{scope}",
            user_text,
            version=1,
        ),
        response_hash=_digest(aura_text),
        occurred_at=observed_at,
        user_event=user,
        aura_event=aura,
        derived_memories=(memory,),
    )


class FakeProjection:
    """Return deliberately untrusted candidates and record enforced bounds."""

    def __init__(
        self,
        candidates: tuple[ProjectionCandidate, ...],
        *,
        generation_id: str = "generation-current",
    ) -> None:
        self.candidates = candidates
        self.generation_id = generation_id
        self.calls: list[tuple[str, str, int]] = []

    def current_generation(self) -> SimpleNamespace:
        return SimpleNamespace(generation_id=self.generation_id)

    def query_candidates(
        self,
        *,
        scope_id: str,
        query: str,
        n_results: int = 50,
    ) -> tuple[ProjectionCandidate, ...]:
        self.calls.append((scope_id, query, n_results))
        return self.candidates


def _candidate(
    command: TurnCommand,
    *,
    target: str = "user",
    generation: str = "generation-current",
    raw_distance: float = 0.1,
    scope_id: str | None = None,
    content_sha256: str | None = None,
) -> ProjectionCandidate:
    if target == "memory":
        memory = command.derived_memories[0]
        projection_id = f"memory:{memory.memory_id}"
        origin_kind = "memory"
        origin_id = memory.memory_id
        content = memory.canonical_text
        digest = _digest(content)
    else:
        event = command.user_event if target == "user" else command.aura_event
        projection_id = f"event:{event.event_id}"
        origin_kind = "event"
        origin_id = event.event_id
        content = event.content
        digest = event.content_sha256
    return ProjectionCandidate(
        projection_id=projection_id,
        origin_kind=origin_kind,
        origin_id=origin_id,
        scope_id=command.scope_id if scope_id is None else scope_id,
        content=content,
        content_sha256=digest if content_sha256 is None else content_sha256,
        generation_id=generation,
        raw_distance=raw_distance,
        cosine_similarity=max(0.0, min(1.0, 1.0 - raw_distance)),
    )


def test_gate_reasons_cover_authoritative_vector_admission(tmp_path: Path) -> None:
    """Every untrusted vector failure must produce one stable content-free code."""
    repository = StorageRepository(tmp_path / "ledger.sqlite3")
    active = _command(
        scope="scope-alpha",
        suffix="active",
        user_text="Alpha valid vector",
        aura_text="Alpha stale hash",
        memory_text="Alpha active memory",
        observed_at="2026-09-01T00:00:00Z",
    )
    other = _command(
        scope="scope-alpha",
        suffix="other",
        user_text="Alpha stale generation",
        aura_text="Alpha irrelevant vector",
        memory_text="Alpha orphan memory",
        observed_at="2026-09-01T00:01:00Z",
    )
    stale = _command(
        scope="scope-alpha",
        suffix="stale",
        user_text="Alpha stale evidence",
        aura_text="Alpha stale response",
        memory_text="Alpha superseded memory",
        observed_at="2026-09-01T00:02:00Z",
    )
    replacement = _command(
        scope="scope-alpha",
        suffix="replacement",
        user_text="Alpha correction evidence",
        aura_text="Alpha correction response",
        memory_text="Alpha current memory",
        observed_at="2026-09-01T00:03:00Z",
    )
    beta = _command(
        scope="scope-beta",
        suffix="beta",
        user_text="Beta collision",
        aura_text="Beta response",
        memory_text="Beta memory",
        observed_at="2026-09-01T00:04:00Z",
    )
    for command in (active, other, stale, replacement, beta):
        repository.append_turn(command.scope_id, command)
    repository.supersede_memory(
        "scope-alpha",
        old_memory_id="memory-stale",
        new_memory_id="memory-replacement",
        basis_event_id="event-user-replacement",
        reason="synthetic correction",
        created_at="2026-09-01T00:05:00Z",
    )
    with sqlite3.connect(repository.database_path) as connection:
        connection.execute(
            "DELETE FROM memory_sources WHERE memory_id = ?",
            ("memory-other",),
        )

    orphan = replace(
        _candidate(active),
        projection_id="event:missing",
        origin_id="missing",
    )
    projection = FakeProjection(
        (
            _candidate(active),
            _candidate(active, target="aura", content_sha256="0" * 64),
            _candidate(other, generation="generation-stale"),
            _candidate(other, target="aura", raw_distance=0.99),
            _candidate(stale, target="memory"),
            _candidate(other, target="memory"),
            _candidate(beta, scope_id="scope-alpha"),
            orphan,
        )
    )
    result = HybridRetriever(repository=repository, projection=projection).retrieve(
        scope_id="scope-alpha",
        query="needle-target",
    )

    assert [item.origin_id for item in result.items] == ["event-user-active"]
    reasons = {
        trace.origin_id: trace.rejection_code
        for trace in result.traces
        if trace.rejection_code is not None
    }
    assert reasons == {
        "event-aura-active": "content_hash_mismatch",
        "event-user-other": "projection_generation_mismatch",
        "event-aura-other": "neutral_relevance_failed",
        "memory-stale": "superseded",
        "memory-other": "provenance_missing",
        "event-user-beta": "scope_mismatch",
        "missing": "origin_missing",
    }
    assert all(trace.remembered_text is None for trace in result.traces)


def test_scope_filter_and_candidate_caps_are_configuration_owned(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "ledger.sqlite3")
    command = _command(
        scope="scope-alpha",
        suffix="scope",
        user_text="Bounded scope target",
        aura_text="Bounded scope response",
        memory_text="Bounded scope memory",
        observed_at="2026-09-01T00:00:00Z",
    )
    repository.append_turn(command.scope_id, command)
    projection = FakeProjection((_candidate(command),))
    retriever = HybridRetriever(repository=repository, projection=projection)

    page = retriever.retrieve(
        scope_id="scope-alpha",
        query="Bounded scope target",
        caller_filter={"scope_id": "scope-beta", "n_results": 50_000},
    )

    assert projection.calls == [("scope-alpha", "Bounded scope target", 50)]
    assert page.items
    assert all(item.scope_id == "scope-alpha" for item in page.items)


def test_rrf_is_fixed_normalized_bounded_and_raw_score_independent(
    tmp_path: Path,
) -> None:
    repository = StorageRepository(tmp_path / "ledger.sqlite3")
    first = _command(
        scope="scope-alpha",
        suffix="rrf-a",
        user_text="RRF lexical target",
        aura_text="RRF response one",
        memory_text="RRF memory one",
        observed_at="2026-09-01T00:00:00Z",
    )
    second = _command(
        scope="scope-alpha",
        suffix="rrf-b",
        user_text="RRF lexical target variant",
        aura_text="RRF response two",
        memory_text="RRF memory two",
        observed_at="2026-09-01T00:01:00Z",
    )
    repository.append_turn(first.scope_id, first)
    repository.append_turn(second.scope_id, second)
    low_raw = HybridRetriever(
        repository=repository,
        projection=FakeProjection(
            (_candidate(first, raw_distance=0.01), _candidate(second, raw_distance=0.2))
        ),
    ).retrieve(scope_id="scope-alpha", query="RRF lexical target")
    high_raw = HybridRetriever(
        repository=repository,
        projection=FakeProjection(
            (_candidate(first, raw_distance=0.2), _candidate(second, raw_distance=0.79))
        ),
    ).retrieve(scope_id="scope-alpha", query="RRF lexical target")

    low_scores = {trace.origin_id: trace.neutral_score for trace in low_raw.traces}
    high_scores = {trace.origin_id: trace.neutral_score for trace in high_raw.traces}
    assert low_scores == high_scores
    trace = next(item for item in low_raw.traces if item.origin_id == "event-user-rrf-a")
    assert trace.lexical_rank is not None
    assert trace.vector_rank == 1
    expected_lexical = (1 / (60 + trace.lexical_rank)) / (1 / 61)
    expected_vector = (1 / (60 + 1)) / (1 / 61)
    assert trace.lexical_rrf == pytest.approx(expected_lexical)
    assert trace.vector_rrf == pytest.approx(expected_vector)
    assert trace.neutral_score == pytest.approx((expected_lexical + expected_vector) / 2)
    assert all(0.0 <= item.neutral_score <= 1.0 for item in low_raw.traces)


def test_duplicate_content_collapses_without_losing_contributors_or_provenance(
    tmp_path: Path,
) -> None:
    repository = StorageRepository(tmp_path / "ledger.sqlite3")
    older = _command(
        scope="scope-alpha",
        suffix="duplicate-a",
        user_text="First duplicate evidence",
        aura_text="First duplicate response",
        memory_text="Canonical duplicate memory",
        observed_at="2026-09-01T00:00:00Z",
    )
    newer = _command(
        scope="scope-alpha",
        suffix="duplicate-b",
        user_text="Second duplicate evidence",
        aura_text="Second duplicate response",
        memory_text="Canonical duplicate memory",
        observed_at="2026-09-01T00:01:00Z",
    )
    repository.append_turn(older.scope_id, older)
    repository.append_turn(newer.scope_id, newer)

    result = HybridRetriever(
        repository=repository,
        projection=FakeProjection(
            (_candidate(newer, target="memory"), _candidate(older, target="memory"))
        ),
    ).retrieve(scope_id="scope-alpha", query="Canonical duplicate memory")

    memories = [item for item in result.items if item.origin_kind == "memory"]
    assert len(memories) == 1
    assert memories[0].contributor_ids == ("memory-duplicate-a", "memory-duplicate-b")
    assert memories[0].provenance_event_ids == (
        "event-aura-duplicate-a",
        "event-aura-duplicate-b",
        "event-user-duplicate-a",
        "event-user-duplicate-b",
    )
    trace = next(item for item in result.traces if item.selected_rank == 1)
    assert trace.contributor_ids == memories[0].contributor_ids


def test_order_uses_relevance_before_newer_timestamp(tmp_path: Path) -> None:
    repository = StorageRepository(tmp_path / "ledger.sqlite3")
    exact = _command(
        scope="scope-alpha",
        suffix="order-a",
        user_text="Exact order target",
        aura_text="Older response",
        memory_text="Older memory",
        observed_at="2026-09-01T00:00:00Z",
    )
    newer = _command(
        scope="scope-alpha",
        suffix="order-b",
        user_text="Exact order target with distractor",
        aura_text="Newer response",
        memory_text="Newer memory",
        observed_at="2026-09-01T01:00:00Z",
    )
    repository.append_turn(exact.scope_id, exact)
    repository.append_turn(newer.scope_id, newer)

    result = HybridRetriever(
        repository=repository,
        projection=FakeProjection(()),
        config=RetrievalConfig(vector_similarity_floor=0.2),
    ).retrieve(scope_id="scope-alpha", query="Exact order target")

    assert [item.origin_id for item in result.items[:2]] == [
        "event-user-order-a",
        "event-user-order-b",
    ]
    assert result.items[0].exact_match is True
    assert result.items[0].observed_at < result.items[1].observed_at
