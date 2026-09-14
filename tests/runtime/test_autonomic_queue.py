"""Behavioral tests for admission, ownership, limits and truthful task results."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from aura_backend.aura_autonomic_system import (
    AutonomicNervousSystem,
    TaskClassifier,
    TaskStatus,
    TaskType,
)
from aura_backend.providers.runtime import ProviderRuntime
from aura_backend.runtime.autonomic_config import AutonomicSettings
from aura_backend.runtime.config import RuntimeConfigurationError, RuntimeSettings
from tests.providers.fakes import ScriptedComplete, ScriptedProvider


def system(**kwargs: object) -> AutonomicNervousSystem:
    """Use an offline provider; tests inject blocking tools when needed."""
    runtime = ProviderRuntime(
        ScriptedProvider((ScriptedComplete("done"),)), timeout_seconds=1
    )
    return AutonomicNervousSystem(provider_runtime=runtime, **kwargs)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_classifier_accepts_real_analysis_at_default_threshold() -> None:
    accepted, kind, _ = await TaskClassifier().should_offload_task(
        "Analyze trends", {"data": [1, 2]}
    )
    assert accepted and kind is TaskType.DATA_ANALYSIS
    accepted, _, _ = await TaskClassifier().should_offload_task("Hello", {})
    assert not accepted


@pytest.mark.asyncio
async def test_bounded_workers_queue_unique_ids_and_owned_shutdown() -> None:
    worker = system(max_concurrent_tasks=2, queue_max_size=3)
    entered = asyncio.Event()
    gate = asyncio.Event()
    active = 0
    peak = 0

    async def execute_tool(name: str, arguments: dict) -> dict:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        if active == 2:
            entered.set()
        try:
            await gate.wait()
            return {"status": "success"}
        finally:
            active -= 1

    worker.internal_tools = SimpleNamespace(execute_tool=execute_tool)  # type: ignore[assignment]
    assert await worker.submit_task("x", {}, "u", force_offload=True) == (False, None)
    await worker.start()
    ids = []
    try:
        for _ in range(2):
            accepted, task_id = await worker.submit_task(
                "tool", {"tool_name": "aura.test"}, "u", force_offload=True
            )
            assert accepted
            ids.append(task_id)
        await asyncio.wait_for(entered.wait(), 1)
        for _ in range(3):
            accepted, task_id = await worker.submit_task(
                "tool", {"tool_name": "aura.test"}, "u", force_offload=True
            )
            assert accepted
            ids.append(task_id)
        assert len(set(ids)) == 5
        queued = await worker.get_task_result(ids[-1])
        assert queued and queued.status is TaskStatus.PENDING
        assert await worker.submit_task("overflow", {}, "u", force_offload=True) == (
            False,
            None,
        )
        assert peak == 2 and active == 2
    finally:
        await worker.stop()
    assert active == 0 and not worker.active_tasks and not worker.queued_tasks
    assert all(
        task.status is TaskStatus.FAILED and task.error == "cancelled"
        for task in worker.completed_tasks.values()
    )
    await asyncio.wait_for(worker.task_queue.join(), 1)
    assert await worker.submit_task("stopped", {}, "u", force_offload=True) == (
        False,
        None,
    )
    await worker.start()
    try:
        _, task_id = await worker.submit_task("hello", {}, "u", force_offload=True)
        result = await worker.get_task_result(task_id, 1)
        assert result and result.status is TaskStatus.COMPLETED
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_forced_tool_error_is_failed_without_model_call() -> None:
    worker = system()
    calls = []

    async def execute_tool(name: str, arguments: dict) -> dict:
        calls.append(name)
        return {"status": "error", "error": "unavailable"}

    worker.internal_tools = SimpleNamespace(
        execute_tool=execute_tool, tools={"aura.search_memories": {}}
    )  # type: ignore[assignment]
    await worker.start()
    try:
        _, task_id = await worker.submit_task(
            "search", {"tool_name": "search_memories"}, "u", force_offload=True
        )
        result = await worker.get_task_result(task_id, 1)
        assert result and result.task_type is TaskType.MCP_TOOL_CALL
        assert result.status is TaskStatus.FAILED and result.result is None
        assert calls == ["search_memories"]
        assert worker.processor.rate_limiter.total_requests == 0
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_tool_timeout_covers_the_whole_task() -> None:
    worker = system(timeout_seconds=0.01)

    async def execute_tool(name: str, arguments: dict) -> dict:
        await asyncio.Event().wait()
        return {}

    worker.internal_tools = SimpleNamespace(execute_tool=execute_tool)  # type: ignore[assignment]
    await worker.start()
    try:
        _, task_id = await worker.submit_task(
            "tool", {"tool_name": "aura.test"}, "u", force_offload=True
        )
        result = await worker.get_task_result(task_id, 1)
        assert result and result.status is TaskStatus.TIMEOUT
    finally:
        await worker.stop()


@pytest.mark.asyncio
async def test_maintenance_triggers_coalesce_without_model_generation() -> None:
    worker = system()
    entered = asyncio.Event()
    gate = asyncio.Event()
    calls = 0

    async def maintain() -> dict:
        nonlocal calls
        calls += 1
        entered.set()
        await gate.wait()
        return {"pending_index_turns": 0}

    worker.processor.memory_maintenance = maintain
    await worker.start()
    try:
        await asyncio.wait_for(entered.wait(), 1)
        ids = [await worker.request_memory_maintenance() for _ in range(10)]
        assert len(set(ids)) == 1 and calls == 1
        gate.set()
        result = await worker.get_task_result(ids[0][1], 1)
        assert result and result.status is TaskStatus.COMPLETED
        assert (
            worker.get_system_status()["last_memory_maintenance"]["result"][
                "pending_index_turns"
            ]
            == 0
        )
        assert worker.processor.rate_limiter.total_requests == 0
    finally:
        await worker.stop()


@pytest.mark.parametrize(
    "mapping",
    [
        {"AUTONOMIC_MAX_CONCURRENT_TASKS": "0"},
        {"AUTONOMIC_QUEUE_MAX_SIZE": "-1"},
        {"AUTONOMIC_QUEUE_PRIORITY_ENABLED": "maybe"},
        {"AUTONOMIC_TASK_THRESHOLD": "bogus"},
        {"AURA_AUTONOMIC_MAX_OUTPUT_TOKENS": "1000000"},
    ],
)
def test_invalid_resource_configuration_rejected_before_startup(mapping: dict) -> None:
    with pytest.raises(RuntimeConfigurationError):
        RuntimeSettings.from_mapping({"AUTONOMIC_ENABLED": "true", **mapping})


def test_defaults_share_provider_and_keep_local_work_bounded() -> None:
    settings = AutonomicSettings.from_mapping({})
    assert settings.concurrency == 1 and settings.max_tokens == 2048
    assert settings.priority_enabled


@pytest.mark.asyncio
@pytest.mark.parametrize("priority_enabled", [True, False])
async def test_priority_queue_and_fifo_setting_have_distinct_execution_order(
    priority_enabled: bool,
) -> None:
    worker = system(priority_enabled=priority_enabled)
    seen = []

    async def execute(task: object, *_args: object) -> object:
        seen.append(task.description)
        task.status = TaskStatus.COMPLETED
        return task

    worker.processor.execute_task = execute
    await worker.start()
    try:
        # Admission has no suspension: both items are queued before workers run.
        await worker.submit_task("simple", {}, "u", force_offload=True)
        await worker.submit_task("Analyze trends", {}, "u", force_offload=True)
        await asyncio.wait_for(worker.task_queue.join(), 1)
        assert seen == (["Analyze trends", "simple"] if priority_enabled else ["simple", "Analyze trends"])
    finally:
        await worker.stop()
