"""Frozen domain records for Aura's durable event ledger."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


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
