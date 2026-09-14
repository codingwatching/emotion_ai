"""Real SQLite/Chroma maintenance recovery, with deterministic offline embeddings."""

from __future__ import annotations

from pathlib import Path

import pytest

from aura_backend.aura_autonomic_system import TaskStatus
from aura_backend.runtime.memory import maintain_memory, memory_inventory
from aura_backend.storage.models import TurnCommand
from aura_backend.storage.repository import StorageRepository
from tests.runtime.test_autonomic_queue import system
from tests.storage.test_projection_rebuild import _adapter, _second_turn


@pytest.mark.asyncio
async def test_background_maintenance_repairs_missing_vectors_and_pending_turns(
    tmp_path: Path, ledger_path: Path, turn_command: TurnCommand,
) -> None:
    repository = StorageRepository(ledger_path)
    repository.append_turn(turn_command.scope_id, turn_command)
    adapter = _adapter(tmp_path, repository)
    adapter.rebuild(generation_id="initial", created_at="2026-09-01T00:00:00Z")
    original_ids = repository.projection_origin_ids()
    assert adapter.delete_origins(original_ids[:1]) == 1
    second = _second_turn(turn_command)
    repository.append_turn(second.scope_id, second)
    assert memory_inventory(ledger_path)["pending_index_turns"] == 1
    worker = system()
    worker.processor.memory_maintenance = lambda: maintain_memory(repository, adapter)
    await worker.start()
    try:
        accepted, task_id = await worker.request_memory_maintenance()
        assert accepted
        result = await worker.get_task_result(task_id, 5)
        assert result and result.status is TaskStatus.COMPLETED
        assert result.result["reconciled_turns"] == 1
        assert result.result["pending_index_turns"] == 0
        assert result.result["source_events"] == 4
        generation = adapter.current_generation()
        client, collection = adapter._open_generation(generation)
        try:
            assert set(collection.get()["ids"]) == set(repository.projection_origin_ids())
        finally:
            client.close()
    finally:
        await worker.stop()
