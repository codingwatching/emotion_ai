"""Phase 3 runtime/storage compatibility tests using disposable fixtures only."""

from __future__ import annotations

import inspect
import sqlite3
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient

import aura_backend.main as main
from aura_backend.conversation_persistence_service import (
    ConversationExchange,
    ConversationPersistenceService,
)
from aura_backend.runtime import RuntimeConfigurationError, RuntimeSettings
from aura_backend.storage.repository import StorageRepository
from aura_backend.storage.models import RetrievalItem, RetrievalPage
from tests.api.test_provider_compatibility import (
    ANSWER_SENTINEL,
    EXPECTED_RESPONSE_KEYS,
    _FakePersistence,
    _install_route_collaborators,
    _payload,
    _runtime,
    _success_provider,
)


class _ProjectionFake:
    """Post-commit projection fake that can fail exactly once."""

    def __init__(self, repository: StorageRepository, *, fail_once: bool = False) -> None:
        self.repository = repository
        self.fail_once = fail_once
        self.calls: list[str] = []

    def upsert_committed(self, turn_id: str) -> int:
        self.calls.append(turn_id)
        if self.fail_once:
            self.fail_once = False
            raise RuntimeError("synthetic projection interruption")
        self.repository.mark_turn_projection_complete(turn_id)
        return 2


class _KeyRecordingPersistence(_FakePersistence):
    """Record the route's retry identity while preserving the Phase 2 fake."""

    def __init__(self) -> None:
        super().__init__(fail=True)
        self.keys: list[str] = []

    async def persist_conversation_exchange_immediate(
        self,
        exchange: Any,
        *,
        update_profile: bool,
        timeout: float,
    ) -> dict[str, Any]:
        self.keys.append(exchange.idempotency_key)
        return await super().persist_conversation_exchange_immediate(
            exchange,
            update_profile=update_profile,
            timeout=timeout,
        )

    async def persist_conversation_exchange(
        self,
        exchange: Any,
        update_profile: bool = True,
    ) -> dict[str, Any]:
        self.keys.append(exchange.idempotency_key)
        return await super().persist_conversation_exchange(
            exchange,
            update_profile=update_profile,
        )


def _exchange(*, key: str, user_text: str = "synthetic question") -> ConversationExchange:
    return ConversationExchange(
        user_memory=SimpleNamespace(
            user_id="scope-a",
            message=user_text,
            sender="user",
            session_id="session-a",
            emotional_state=None,
            cognitive_state=None,
        ),
        ai_memory=SimpleNamespace(
            user_id="scope-a",
            message="synthetic answer",
            sender="aura",
            session_id="session-a",
            emotional_state=None,
            cognitive_state=None,
        ),
        session_id="session-a",
        idempotency_key=key,
    )


def _row_counts(database_path: Path) -> tuple[int, int]:
    with sqlite3.connect(database_path) as connection:
        return (
            int(connection.execute("SELECT COUNT(*) FROM turns").fetchone()[0]),
            int(connection.execute("SELECT COUNT(*) FROM events").fetchone()[0]),
        )


def test_runtime_settings_resolve_a_disjoint_absolute_ledger_root(
    tmp_path: Path,
) -> None:
    legacy_root = tmp_path / "legacy"
    ledger_root = tmp_path / "ledger-v2"

    settings = RuntimeSettings.from_mapping(
        {
            "AURA_DATA_DIRECTORY": str(legacy_root),
            "AURA_LEDGER_DIRECTORY": str(ledger_root),
        }
    )

    assert settings.storage_root == legacy_root
    assert settings.ledger_root == ledger_root
    assert settings.ledger_database_path == ledger_root / "aura.sqlite3"
    assert settings.projection_root == ledger_root / "projections"
    assert settings.ledger_root.is_absolute()


@pytest.mark.parametrize("ledger_value", ("relative-v2", "", "\x00bad"))
def test_runtime_settings_reject_malformed_ledger_overrides(ledger_value: str) -> None:
    with pytest.raises(RuntimeConfigurationError) as captured:
        RuntimeSettings.from_mapping({"AURA_LEDGER_DIRECTORY": ledger_value})

    assert captured.value.setting_name == "AURA_LEDGER_DIRECTORY"


def test_runtime_settings_reject_ledger_overlap_with_historical_roots(
    tmp_path: Path,
) -> None:
    historical = tmp_path / "historical"
    with pytest.raises(RuntimeConfigurationError) as captured:
        RuntimeSettings.from_mapping(
            {
                "AURA_DATA_DIRECTORY": str(historical),
                "CHROMA_PERSIST_DIRECTORY": str(historical / "chroma"),
                "AURA_LEDGER_DIRECTORY": str(historical / "v2"),
            }
        )

    assert captured.value.setting_name == "AURA_LEDGER_DIRECTORY"


@pytest.mark.asyncio
async def test_sqlite_first_persistence_replays_and_rejects_changed_hash(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "ledger" / "aura.sqlite3"
    repository = StorageRepository(database_path)
    projection = _ProjectionFake(repository)
    service = ConversationPersistenceService(repository, projection)

    stored = await service.persist_conversation_exchange_immediate(
        _exchange(key="stable-key"), timeout=1.0
    )
    replayed = await service.persist_conversation_exchange(_exchange(key="stable-key"))
    conflict = await service.persist_conversation_exchange(
        _exchange(key="stable-key", user_text="changed request")
    )

    assert stored["success"] is True
    assert stored["status"] == "stored"
    assert replayed["success"] is True
    assert replayed["status"] == "replayed"
    assert replayed["turn_id"] == stored["turn_id"]
    assert conflict["success"] is False
    assert conflict["status"] == "idempotency_conflict"
    assert conflict["turn_id"] == stored["turn_id"]
    assert _row_counts(database_path) == (1, 2)
    assert projection.calls == [stored["turn_id"]]


@pytest.mark.asyncio
async def test_projection_failure_commits_once_then_reconciles_same_turn(
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "ledger" / "aura.sqlite3"
    repository = StorageRepository(database_path)
    projection = _ProjectionFake(repository, fail_once=True)
    service = ConversationPersistenceService(repository, projection)
    exchange = _exchange(key="retry-key")

    degraded = await service.persist_conversation_exchange_immediate(
        exchange, timeout=1.0
    )
    reconciled = await service.persist_conversation_exchange(exchange)

    assert degraded["success"] is False
    assert degraded["durable_status"] == "stored"
    assert degraded["projection_status"] == "pending"
    assert degraded["retry_identity"] == "retry-key"
    assert reconciled["success"] is True
    assert reconciled["status"] == "replayed"
    assert reconciled["projection_status"] == "complete"
    assert reconciled["turn_id"] == degraded["turn_id"]
    assert _row_counts(database_path) == (1, 2)
    assert projection.calls == [degraded["turn_id"], degraded["turn_id"]]


def test_route_reuses_one_optional_idempotency_key_for_background_retry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    persistence = _KeyRecordingPersistence()
    _install_route_collaborators(monkeypatch, persistence)
    provider = _success_provider()
    application_runtime, _selected_runtime = _runtime(provider)
    app = main.create_app(runtime_builder=lambda: application_runtime)
    payload = {**_payload(), "idempotency_key": "caller-stable-key"}

    with TestClient(app) as client:
        response = client.post("/conversation", json=payload)

    assert response.status_code == 200
    assert set(response.json()) == EXPECTED_RESPONSE_KEYS
    assert response.json()["response"] == ANSWER_SENTINEL
    assert persistence.keys == ["caller-stable-key", "caller-stable-key"]


def test_base_storage_source_excludes_legacy_writer_and_live_backup() -> None:
    source = inspect.getsource(main._start_base_resources)

    assert "DatabaseProtection" not in source
    assert "get_protection_service" not in source
    assert "RobustAuraVectorDB" not in source
    assert "StorageRepository" in source
    assert "ProjectionAdapter" in source


class _RetrieverPageFake:
    """Return one fixed neutral page while recording the hard API bound."""

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def retrieve(self, **kwargs: Any) -> RetrievalPage:
        self.calls.append(kwargs)
        item = RetrievalItem(
            origin_id="event-synthetic",
            origin_kind="event",
            scope_id="scope-a",
            content="synthetic memory",
            content_sha256="a" * 64,
            observed_at="2026-09-01T00:00:00Z",
            provenance_event_ids=("event-synthetic",),
            contributor_ids=("event-synthetic",),
            neutral_score=0.0,
            exact_match=True,
            selected_rank=1,
        )
        return RetrievalPage(
            items=(item,),
            next_cursor="opaque-next",
            has_more=True,
            trace_id="trace-synthetic",
            traces=(),
        )


@pytest.mark.asyncio
async def test_search_boundary_is_bounded_traceable_and_compatibility_shaped() -> None:
    retriever = _RetrieverPageFake()

    result = await main._search_storage_boundary(
        retriever,
        scope_id="scope-a",
        query="synthetic",
        page_size=100,
        cursor=None,
    )

    assert retriever.calls == [
        {
            "scope_id": "scope-a",
            "query": "synthetic",
            "page_size": 100,
            "cursor": None,
        }
    ]
    assert result["results"][0]["content"] == "synthetic memory"
    assert result["next_cursor"] == "opaque-next"
    assert result["has_more"] is True
    assert result["trace_id"] == "trace-synthetic"
    assert result["includes_video_archives"] is False


def test_search_request_rejects_unbounded_pages() -> None:
    with pytest.raises(Exception):
        main.SearchRequest(user_id="scope-a", query="synthetic", n_results=101)
