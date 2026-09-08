"""Frozen domain records for Aura's durable event ledger."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class MemoryKind(str, Enum):
    """Supported lossy derived-memory categories."""

    FACT = "fact"
    PREFERENCE = "preference"
    EPISODE = "episode"
    GOAL = "goal"
    RELATIONSHIP = "relationship"


class EpistemicStatus(str, Enum):
    """Explicit uncertainty state for a derived memory."""

    OBSERVED = "observed"
    INFERRED = "inferred"
    UNCERTAIN = "uncertain"
    DISPUTED = "disputed"


class TurnWriteStatus(str, Enum):
    """Observable disposition of an idempotent turn request."""

    STORED = "stored"
    REPLAYED = "replayed"


class StorageFailure(RuntimeError):
    """Content-free storage error identified by a stable code."""

    def __init__(self, code: str, *, identifier: str | None = None) -> None:
        self.code = code
        self.identifier = identifier
        detail = f" identifier={identifier}" if identifier is not None else ""
        super().__init__(f"storage failure: code={code}{detail}")


@dataclass(frozen=True, slots=True)
class EventInput:
    """One immutable source event supplied to a turn transaction."""

    event_id: str
    actor: str
    content: str
    observed_at: str
    content_sha256: str
    source_kind: str = "conversation"
    payload_json: str = "{}"


@dataclass(frozen=True, slots=True)
class DerivedMemoryInput:
    """One typed interpretation plus complete source identity."""

    memory_id: str
    kind: MemoryKind
    canonical_text: str
    confidence: float
    epistemic_status: EpistemicStatus
    primary_source_event_id: str
    source_event_ids: tuple[str, ...]
    created_at: str


@dataclass(frozen=True, slots=True)
class TurnCommand:
    """All durable inputs for one complete user/Aura turn."""

    scope_id: str
    session_id: str
    turn_id: str
    idempotency_key: str
    request_hash_version: int
    request_hash: str
    response_hash: str
    occurred_at: str
    user_event: EventInput
    aura_event: EventInput
    derived_memories: tuple[DerivedMemoryInput, ...] = ()
    affect_transition: Any | None = None
    expected_state_revision: int | None = None


@dataclass(frozen=True, slots=True)
class PersistedTurn:
    """Stable identifiers and hashes returned for a durable turn."""

    scope_id: str
    session_id: str
    turn_id: str
    idempotency_key: str
    request_hash_version: int
    request_hash: str
    response_hash: str
    user_event_id: str
    aura_event_id: str
    status: TurnWriteStatus
    projection_reconciliation_required: bool


@dataclass(frozen=True, slots=True)
class IdempotencyConflict:
    """Content-free conflict preserving the original durable identity."""

    scope_id: str
    idempotency_key: str
    existing_turn_id: str
    code: str = "idempotency_conflict"


@dataclass(frozen=True, slots=True)
class DerivedMemory:
    """Persisted, provenance-bearing interpretation returned to callers."""

    memory_id: str
    scope_id: str
    kind: MemoryKind
    canonical_text: str
    confidence: float
    epistemic_status: EpistemicStatus
    primary_source_event_id: str
    source_event_ids: tuple[str, ...]
    created_at: str
    content_sha256: str


@dataclass(frozen=True, slots=True)
class RetrievalGate:
    """One content-free admission decision for a considered candidate."""

    name: str
    passed: bool
    reason_code: str | None = None


@dataclass(frozen=True, slots=True)
class RetrievalTrace:
    """Content-free evidence explaining one neutral retrieval decision."""

    run_id: str
    query_sha256: str
    config_sha256: str
    sqlite_schema: int
    projection_generation: str
    origin_id: str
    origin_kind: str
    scope_id: str | None
    content_sha256: str | None
    observed_at: str | None
    lexical_rank: int | None
    lexical_raw_value: float | None
    vector_rank: int | None
    vector_raw_distance: float | None
    vector_similarity: float | None
    lexical_rrf: float
    vector_rrf: float
    neutral_score: float
    exact_match: bool
    gates: tuple[RetrievalGate, ...]
    rejection_code: str | None
    contributor_ids: tuple[str, ...]
    provenance_event_ids: tuple[str, ...]
    selected_rank: int | None
    salience_score: float = 0.0
    remembered_text: None = None


@dataclass(frozen=True, slots=True)
class RetrievalItem:
    """One eligible memory item returned through the public retrieval boundary."""

    origin_id: str
    origin_kind: str
    scope_id: str
    content: str
    content_sha256: str
    observed_at: str
    provenance_event_ids: tuple[str, ...]
    contributor_ids: tuple[str, ...]
    neutral_score: float
    exact_match: bool
    selected_rank: int
    salience_score: float = 0.0


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A complete affect-neutral result before cursor pagination is applied."""

    run_id: str
    items: tuple[RetrievalItem, ...]
    traces: tuple[RetrievalTrace, ...]


@dataclass(frozen=True, slots=True)
class RetrievalPage:
    """One bounded page from a frozen neutral retrieval run."""

    items: tuple[RetrievalItem, ...]
    next_cursor: str | None
    has_more: bool
    trace_id: str
    traces: tuple[RetrievalTrace, ...]


@dataclass(frozen=True, slots=True)
class HistoryItem:
    """One immutable source event from bounded ledger history."""

    event_id: str
    scope_id: str
    turn_id: str
    actor: str
    content: str
    content_sha256: str
    observed_at: str


@dataclass(frozen=True, slots=True)
class HistoryPage:
    """One keyset page frozen to an initial ledger watermark."""

    items: tuple[HistoryItem, ...]
    next_cursor: str | None
    has_more: bool
