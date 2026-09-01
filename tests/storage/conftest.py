"""Deterministic, temporary-only fixtures for storage contracts."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from aura_backend.storage.models import (
    DerivedMemoryInput,
    EpistemicStatus,
    EventInput,
    MemoryKind,
    TurnCommand,
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@pytest.fixture
def ledger_path(tmp_path: Path) -> Path:
    """Return an absolute path below pytest's isolated directory."""
    return tmp_path / "ledger.sqlite3"


@pytest.fixture
def turn_command() -> TurnCommand:
    """Return a complete deterministic command with one typed derivation."""
    user = EventInput(
        event_id="event-user-001",
        actor="user",
        content="Synthetic user preference",
        observed_at="2026-08-31T12:00:00Z",
        content_sha256=_digest("Synthetic user preference"),
    )
    aura = EventInput(
        event_id="event-aura-001",
        actor="aura",
        content="Synthetic Aura response",
        observed_at="2026-08-31T12:00:01Z",
        content_sha256=_digest("Synthetic Aura response"),
    )
    memory = DerivedMemoryInput(
        memory_id="memory-001",
        kind=MemoryKind.PREFERENCE,
        canonical_text="The synthetic user prefers concise answers",
        confidence=0.8,
        epistemic_status=EpistemicStatus.INFERRED,
        primary_source_event_id=user.event_id,
        source_event_ids=(user.event_id, aura.event_id),
        created_at="2026-08-31T12:00:02Z",
    )
    return TurnCommand(
        scope_id="scope-alpha",
        session_id="session-001",
        turn_id="turn-001",
        idempotency_key="request-001",
        request_hash_version=1,
        request_hash=_digest("canonical request"),
        response_hash=_digest("canonical response"),
        occurred_at="2026-08-31T12:00:00Z",
        user_event=user,
        aura_event=aura,
        derived_memories=(memory,),
    )
