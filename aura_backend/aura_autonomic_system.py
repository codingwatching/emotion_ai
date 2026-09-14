"""
Aura Autonomic Nervous System
============================

Bounded background maintenance and explicit task execution. Model tasks use the
same provider runtime as conversation; memory maintenance uses committed SQLite
records and does not generate or promote speculative personal facts.

This system acts as Aura's "autonomic nervous system" - handling background
processing while the main consciousness focuses on user interaction.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from collections.abc import Awaitable, Callable
from itertools import count
from uuid import uuid4
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from aura_backend.providers.base import ProviderMessage, ProviderRequest
from aura_backend.providers.errors import ProviderErrorCode, ProviderFailure
from aura_backend.providers.runtime import ProviderRuntime

if TYPE_CHECKING:
    from aura_backend.aura_internal_tools import AuraInternalTools
    from aura_backend.mcp_to_gemini_bridge import MCPGeminiBridge

logger = logging.getLogger(__name__)


class RateLimiter:
    """
    Sophisticated rate limiting implementation for API request management

    Tracks both requests per minute (RPM) and requests per day (RPD) with
    sliding window algorithms for accurate rate limit enforcement.
    """

    def __init__(self, rpm_limit: int, rpd_limit: int):
        self.rpm_limit = rpm_limit
        self.rpd_limit = rpd_limit

        # Sliding window for RPM tracking (60-second windows)
        self.minute_requests: deque = deque()

        # Daily request tracking
        self.daily_requests = 0
        self.daily_reset_time = datetime.now().replace(
            hour=0, minute=0, second=0, microsecond=0
        ) + timedelta(days=1)

        # Statistics tracking
        self.total_requests = 0
        self.total_rejected = 0
        self.last_request_time: Optional[datetime] = None

        logger.info("🚦 Rate limiter initialized: %s RPM, %s RPD", rpm_limit, rpd_limit)

    async def can_make_request(self) -> bool:
        """Check if a request can be made without exceeding rate limits"""
        current_time = datetime.now()

        # Clean up old minute requests (sliding window)
        cutoff_time = current_time - timedelta(minutes=1)
        while self.minute_requests and self.minute_requests[0] < cutoff_time:
            self.minute_requests.popleft()

        # Reset daily counter if needed
        if current_time >= self.daily_reset_time:
            self.daily_requests = 0
            self.daily_reset_time = current_time.replace(
                hour=0, minute=0, second=0, microsecond=0
            ) + timedelta(days=1)
            logger.info("🔄 Daily rate limit counter reset")

        # Check both RPM and RPD limits
        rpm_available = len(self.minute_requests) < self.rpm_limit
        rpd_available = self.daily_requests < self.rpd_limit

        return rpm_available and rpd_available

    async def acquire(self) -> bool:
        """
        Attempt to acquire a request slot

        Returns:
            True if request can proceed, False if rate limited
        """
        if not await self.can_make_request():
            self.total_rejected += 1
            return False

        current_time = datetime.now()

        # Record the request
        self.minute_requests.append(current_time)
        self.daily_requests += 1
        self.total_requests += 1
        self.last_request_time = current_time

        return True

    async def wait_for_availability(self, max_wait_seconds: float = 60.0) -> bool:
        """
        Wait until a request slot becomes available

        Args:
            max_wait_seconds: Maximum time to wait in seconds

        Returns:
            True if slot became available, False if timeout
        """
        start_time = time.time()

        while time.time() - start_time < max_wait_seconds:
            if await self.can_make_request():
                return await self.acquire()

            # Calculate optimal wait time
            if self.minute_requests:
                # Wait until the oldest request in the current minute expires
                oldest_request = self.minute_requests[0]
                wait_time = (
                    oldest_request + timedelta(minutes=1) - datetime.now()
                ).total_seconds()
                wait_time = max(0.1, min(wait_time, 5.0))  # Wait 0.1-5 seconds
            else:
                wait_time = 0.1  # Brief wait if no current requests

            await asyncio.sleep(wait_time)

        logger.warning("⏰ Rate limiter timeout after %ss", max_wait_seconds)
        return False

    def get_status(self) -> Dict[str, Any]:
        """Get current rate limiter status and statistics"""
        current_time = datetime.now()

        # Clean up old requests for accurate counting
        cutoff_time = current_time - timedelta(minutes=1)
        current_minute_requests = sum(
            1 for req_time in self.minute_requests if req_time >= cutoff_time
        )

        return {
            "rpm_limit": self.rpm_limit,
            "rpm_current": current_minute_requests,
            "rpm_available": self.rpm_limit - current_minute_requests,
            "rpd_limit": self.rpd_limit,
            "rpd_current": self.daily_requests,
            "rpd_available": self.rpd_limit - self.daily_requests,
            "total_requests": self.total_requests,
            "total_rejected": self.total_rejected,
            "rejection_rate": (
                self.total_rejected / max(1, self.total_requests + self.total_rejected)
            )
            * 100,
            "last_request": (
                self.last_request_time.isoformat() if self.last_request_time else None
            ),
            "daily_reset_time": self.daily_reset_time.isoformat(),
        }


class TaskPriority(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class TaskType(str, Enum):
    MCP_TOOL_CALL = "mcp_tool_call"
    DATA_ANALYSIS = "data_analysis"
    CODE_GENERATION = "code_generation"
    MEMORY_SEARCH = "memory_search"
    MEMORY_MAINTENANCE = "memory_maintenance"
    PATTERN_ANALYSIS = "pattern_analysis"
    COMPLEX_REASONING = "complex_reasoning"
    BACKGROUND_PROCESSING = "background_processing"


class TaskStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    TIMEOUT = "timeout"


class AutonomicState(str, Enum):
    """Stable availability states for the optional background subsystem."""

    DISABLED = "disabled"
    STOPPED = "stopped"
    RUNNING = "running"


@dataclass(frozen=True, slots=True)
class _NeutralFunctionCall:
    """Minimal structural input accepted by the legacy MCP execution bridge."""

    name: str
    args: Dict[str, Any]


@dataclass
class AutonomicTask:
    """Represents a task that can be offloaded to the autonomic system"""

    task_id: str
    task_type: TaskType
    priority: TaskPriority
    description: str
    payload: Dict[str, Any]
    user_id: str
    session_id: Optional[str] = None
    created_at: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: TaskStatus = TaskStatus.PENDING
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time_ms: Optional[float] = None

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()


class TaskClassifier:
    """Classify explicit requests using achievable, bounded complexity scores."""

    def __init__(self, threshold: str = "medium") -> None:
        if threshold not in {"low", "medium", "high"}:
            raise ValueError("Invalid AUTONOMIC_TASK_THRESHOLD")
        self.threshold = threshold

    async def should_offload_task(
        self,
        task_description: str,
        task_payload: Dict[str, Any],
        user_context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[bool, TaskType, TaskPriority]:
        task_type = self._classify_task_type(task_description, task_payload)
        score = 0.2 if task_type is TaskType.BACKGROUND_PROCESSING else 0.6
        if any(
            word in task_description.lower()
            for word in ("deep", "complex", "comprehensive")
        ):
            score += 0.2
        if len(json.dumps(task_payload)) > 1000:
            score += 0.1
        urgent = bool(user_context and user_context.get("is_urgent"))
        priority = (
            TaskPriority.HIGH
            if urgent
            else TaskPriority.MEDIUM
            if score >= 0.5
            else TaskPriority.LOW
        )
        return (
            score >= {"low": 0.3, "medium": 0.5, "high": 0.7}[self.threshold],
            task_type,
            priority,
        )

    def _classify_task_type(
        self, description: str, payload: Dict[str, Any]
    ) -> TaskType:
        if payload.get("operation") == "maintain_memory":
            return TaskType.MEMORY_MAINTENANCE
        if "tool_name" in payload:
            return TaskType.MCP_TOOL_CALL
        text = description.lower()
        if any(word in text for word in ("search", "memory", "recall")):
            return TaskType.MEMORY_SEARCH
        if any(word in text for word in ("analyze", "analysis", "pattern")):
            return TaskType.DATA_ANALYSIS
        if any(word in text for word in ("code", "generate", "script")):
            return TaskType.CODE_GENERATION
        if any(word in text for word in ("reason", "think", "complex")):
            return TaskType.COMPLEX_REASONING
        return TaskType.BACKGROUND_PROCESSING


class AutonomicProcessor:
    """Core processor for autonomic task execution with rate limiting"""

    def __init__(
        self,
        autonomic_model: str = "selected_provider",
        max_output_tokens: int = 2048,
        timeout_seconds: int = 30,
        rpm_limit: int = 25,
        rpd_limit: int = 1200,
        provider_runtime: ProviderRuntime | None = None,
    ):
        # Keep the legacy argument callable without treating its unvalidated text
        # as runtime ownership or safe diagnostic metadata.
        del autonomic_model
        self.autonomic_model = (
            "selected_provider" if provider_runtime is not None else "not_configured"
        )
        self.max_output_tokens = max_output_tokens
        self.timeout_seconds = timeout_seconds
        self._provider_runtime = provider_runtime
        self.memory_maintenance: Callable[[], Awaitable[Dict[str, Any]]] | None = None

        # Initialize rate limiter
        self.rate_limiter = RateLimiter(rpm_limit, rpd_limit)

        self.execution_stats = {
            "tasks_processed": 0,
            "tasks_successful": 0,
            "tasks_failed": 0,
            "tasks_rate_limited": 0,
            "average_execution_time": 0.0,
            "total_execution_time": 0.0,
        }

    async def execute_task(
        self,
        task: AutonomicTask,
        mcp_bridge: Optional[MCPGeminiBridge] = None,
        internal_tools: Optional[AuraInternalTools] = None,
    ) -> AutonomicTask:
        """Execute a task using the autonomic processor"""

        task.status = TaskStatus.PROCESSING
        task.started_at = datetime.now()
        start_time = time.time()

        try:
            logger.info(
                "Autonomic processor executing task type=%s",
                task.task_type.value,
            )

            # Cover tools and rate-limit waiting as well as model generation.
            async with asyncio.timeout(self.timeout_seconds):
                result = await self._dispatch(task, mcp_bridge, internal_tools)

            # Update task with results
            task.status = TaskStatus.COMPLETED
            task.result = result
            task.completed_at = datetime.now()

            # Update stats
            execution_time = (time.time() - start_time) * 1000
            task.execution_time_ms = execution_time
            self._update_execution_stats(execution_time, True)

            logger.info("Autonomic task completed duration_ms=%.1f", execution_time)
            return task

        except asyncio.CancelledError:
            task.status = TaskStatus.FAILED
            task.error = ProviderErrorCode.CANCELLED.value
            task.completed_at = datetime.now()
            self._update_execution_stats((time.time() - start_time) * 1000, False)
            logger.warning("Autonomic task ended code=cancelled")
            raise

        except TimeoutError:
            task.status = TaskStatus.TIMEOUT
            task.error = ProviderErrorCode.TIMEOUT.value
            task.completed_at = datetime.now()
            self._update_execution_stats((time.time() - start_time) * 1000, False)
            logger.warning("Autonomic task ended code=timeout")
            return task

        except ProviderFailure as failure:
            task.status = (
                TaskStatus.TIMEOUT
                if failure.code is ProviderErrorCode.TIMEOUT
                else TaskStatus.FAILED
            )
            task.error = failure.code.value
            task.completed_at = datetime.now()
            self._update_execution_stats((time.time() - start_time) * 1000, False)
            logger.warning("Autonomic task ended code=%s", failure.code.value)
            return task

        except Exception:
            task.status = TaskStatus.FAILED
            task.error = ProviderErrorCode.MALFORMED_RESPONSE.value
            task.completed_at = datetime.now()
            self._update_execution_stats((time.time() - start_time) * 1000, False)
            logger.error("Autonomic task ended code=malformed_response")
            return task

    async def _dispatch(
        self,
        task: AutonomicTask,
        mcp_bridge: Optional[MCPGeminiBridge],
        internal_tools: Optional[AuraInternalTools],
    ) -> Dict[str, Any]:
        """Route by the classified type even when admission was forced."""
        if task.task_type is TaskType.MEMORY_MAINTENANCE:
            if self.memory_maintenance is None:
                raise ProviderFailure(code=ProviderErrorCode.UNAVAILABLE)
            return await self.memory_maintenance()
        if task.task_type is TaskType.MCP_TOOL_CALL:
            return await self._execute_mcp_tool_task(task, mcp_bridge, internal_tools)
        if task.task_type is TaskType.DATA_ANALYSIS:
            return await self._execute_analysis_task(task)
        if task.task_type is TaskType.CODE_GENERATION:
            return await self._execute_code_generation_task(task)
        if task.task_type is TaskType.MEMORY_SEARCH:
            return await self._execute_memory_search_task(task, internal_tools)
        return await self._execute_general_task(task)

    @staticmethod
    def _require_tool_success(result: Any) -> None:
        """A returned error envelope is a failure, never a completed task."""
        if isinstance(result, dict) and (
            result.get("status") in {"error", "failed"}
            or result.get("success") is False
            or result.get("error")
        ):
            raise ProviderFailure(code=ProviderErrorCode.UNAVAILABLE)

    async def _execute_mcp_tool_task(
        self,
        task: AutonomicTask,
        mcp_bridge: Optional[MCPGeminiBridge],
        internal_tools: Optional[AuraInternalTools],
    ) -> Dict[str, Any]:
        """Execute MCP tool calls through autonomic processor"""

        tool_name = task.payload.get("tool_name")
        arguments = task.payload.get("arguments", {})

        # Try internal tools first
        if (
            internal_tools
            and tool_name
            and (
                tool_name.startswith("aura.")
                or f"aura.{tool_name}" in getattr(internal_tools, "tools", {})
            )
        ):
            result = await internal_tools.execute_tool(tool_name, arguments)
            self._require_tool_success(result)
            return {"tool_result": result, "execution_method": "internal_tools"}

        # Use MCP bridge for external tools
        elif mcp_bridge and tool_name:
            # The legacy bridge consumes only ``name`` and ``args`` attributes;
            # keep that boundary structural so this module never imports Google.
            function_call = _NeutralFunctionCall(name=tool_name, args=arguments)

            execution_result = await mcp_bridge.execute_function_call(
                function_call, task.user_id
            )

            if not execution_result.success:
                raise ProviderFailure(code=ProviderErrorCode.UNAVAILABLE)
            self._require_tool_success(execution_result.result)
            return {
                "tool_result": execution_result.result,
                "success": execution_result.success,
                "error": execution_result.error,
                "execution_method": "mcp_bridge",
            }

        else:
            raise ValueError(f"No suitable tool executor found for: {tool_name}")

    async def _execute_analysis_task(self, task: AutonomicTask) -> Dict[str, Any]:
        """Execute data analysis tasks using autonomic model"""

        analysis_prompt = f"""
        Analyze the following data/information and provide insights:

        Task: {task.description}
        Data: {json.dumps(task.payload, indent=2)}

        Provide a comprehensive analysis including:
        1. Key patterns and trends
        2. Notable insights
        3. Recommendations
        4. Summary of findings

        Format your response as JSON with clear sections.
        """

        result = await self._call_autonomic_model(analysis_prompt)
        return {"analysis_result": result, "task_type": "data_analysis"}

    async def _execute_code_generation_task(
        self, task: AutonomicTask
    ) -> Dict[str, Any]:
        """Execute code generation tasks using autonomic model"""

        code_prompt = f"""
        Generate code based on the following requirements:

        Task: {task.description}
        Requirements: {json.dumps(task.payload, indent=2)}

        Provide:
        1. Complete, working code
        2. Documentation and comments
        3. Usage examples
        4. Error handling

        Use best practices and include proper structure.
        """

        result = await self._call_autonomic_model(code_prompt)
        return {"code_result": result, "task_type": "code_generation"}

    async def _execute_memory_search_task(
        self, task: AutonomicTask, internal_tools: Optional[AuraInternalTools]
    ) -> Dict[str, Any]:
        """Execute memory search tasks using internal tools"""

        if not internal_tools:
            raise ValueError("Internal tools not available for memory search")

        search_query = task.payload.get("query", task.description)
        user_id = task.user_id
        max_results = task.payload.get("max_results", 10)

        # Use comprehensive memory search
        result = await internal_tools.execute_tool(
            "aura.search_memories",
            {"query": search_query, "user_id": user_id, "n_results": max_results},
        )

        self._require_tool_success(result)
        return {"memory_search_result": result, "task_type": "memory_search"}

    async def _execute_general_task(self, task: AutonomicTask) -> Dict[str, Any]:
        """Execute general processing tasks using autonomic model"""

        general_prompt = f"""
        Process the following task:

        Description: {task.description}
        Details: {json.dumps(task.payload, indent=2)}
        Priority: {task.priority.value}

        Provide a thorough response that addresses the task requirements.
        Include reasoning, methodology, and actionable results.
        """

        result = await self._call_autonomic_model(general_prompt)
        return {"general_result": result, "task_type": "general_processing"}

    async def _call_autonomic_model(self, prompt: str) -> str:
        """Generate through the injected provider-neutral runtime."""
        if self._provider_runtime is None:
            raise ProviderFailure(
                code=ProviderErrorCode.CONFIGURATION,
                setting_name="AUTONOMIC_PROVIDER",
                retryable=False,
            )

        # Rate limiting: Wait for availability with timeout
        rate_limit_acquired = await self.rate_limiter.wait_for_availability(
            max_wait_seconds=60.0
        )

        if not rate_limit_acquired:
            self.execution_stats["tasks_rate_limited"] += 1
            raise ProviderFailure(
                code=ProviderErrorCode.RATE_LIMITED,
                retryable=True,
            )

        try:
            async with asyncio.timeout(self.timeout_seconds):
                result = await self._provider_runtime.generate(
                    ProviderRequest(
                        messages=(ProviderMessage(role="user", content=prompt),),
                        temperature=0.3,
                        max_tokens=self.max_output_tokens,
                    )
                )
        except TimeoutError as error:
            raise ProviderFailure(
                code=ProviderErrorCode.TIMEOUT,
                retryable=True,
            ) from error
        return result.content

    def _update_execution_stats(self, execution_time: float, success: bool):
        """Update execution statistics"""
        self.execution_stats["tasks_processed"] += 1

        if success:
            self.execution_stats["tasks_successful"] += 1
        else:
            self.execution_stats["tasks_failed"] += 1

        # Update timing stats
        total_time = self.execution_stats["total_execution_time"] + execution_time
        self.execution_stats["total_execution_time"] = total_time
        self.execution_stats["average_execution_time"] = (
            total_time / self.execution_stats["tasks_processed"]
        )

    def get_stats(self) -> Dict[str, Any]:
        """Get execution statistics including rate limiting metrics"""
        stats = self.execution_stats.copy()
        stats["rate_limiter"] = self.rate_limiter.get_status()
        return stats


class AutonomicNervousSystem:
    """Own a bounded priority queue, execution workers and maintenance timer."""

    def __init__(
        self,
        autonomic_model: str = "selected_provider",
        max_concurrent_tasks: int = 1,
        task_threshold: str = "medium",
        max_output_tokens: int = 2048,
        timeout_seconds: int = 120,
        rpm_limit: int = 10,
        rpd_limit: int = 500,
        queue_max_size: int = 32,
        provider_runtime: ProviderRuntime | None = None,
        priority_enabled: bool = True,
        maintenance_interval_seconds: int = 300,
    ) -> None:
        if (
            min(
                max_concurrent_tasks,
                queue_max_size,
                timeout_seconds,
                rpm_limit,
                rpd_limit,
                max_output_tokens,
                maintenance_interval_seconds,
            )
            <= 0
        ):
            raise ValueError("Autonomic resource limits must be positive")
        self.classifier = TaskClassifier(task_threshold)
        self.processor = AutonomicProcessor(
            autonomic_model,
            max_output_tokens,
            timeout_seconds,
            rpm_limit,
            rpd_limit,
            provider_runtime,
        )
        self._provider_runtime = provider_runtime
        self._disabled_reason = "not_configured" if provider_runtime is None else None
        self.max_concurrent_tasks = max_concurrent_tasks
        self.queue_max_size = queue_max_size
        self.priority_enabled = priority_enabled
        self.maintenance_interval_seconds = maintenance_interval_seconds
        self.task_queue: asyncio.PriorityQueue = asyncio.PriorityQueue(
            maxsize=queue_max_size
        )
        self.queued_tasks: Dict[str, AutonomicTask] = {}
        self.active_tasks: Dict[str, AutonomicTask] = {}
        self.completed_tasks: Dict[str, AutonomicTask] = {}
        self._sequence = count()
        self._running = False
        self._worker_task: asyncio.Task | None = None
        self._workers: list[asyncio.Task] = []
        self._maintenance_task: asyncio.Task | None = None
        self.mcp_bridge: Optional[MCPGeminiBridge] = None
        self.internal_tools: Optional[AuraInternalTools] = None
        self._maintenance_id: str | None = None
        self.last_maintenance: Dict[str, Any] | None = None

    def set_external_systems(
        self,
        mcp_bridge: Optional[MCPGeminiBridge] = None,
        internal_tools: Optional[AuraInternalTools] = None,
    ) -> None:
        self.mcp_bridge = mcp_bridge
        self.internal_tools = internal_tools

    async def start(self) -> None:
        """Start exactly the configured number of owned execution workers."""
        if self._provider_runtime is None or self._running:
            return
        self._running = True
        self._workers = [
            asyncio.create_task(self._task_worker())
            for _ in range(self.max_concurrent_tasks)
        ]
        self._worker_task = self._workers[0]
        if self.processor.memory_maintenance is not None:
            self._maintenance_task = asyncio.create_task(self._maintenance_loop())

    async def stop(self) -> None:
        """Cancel and await owned execution, then terminate every queued task."""
        self._running = False
        owned = self._workers + (
            [self._maintenance_task] if self._maintenance_task else []
        )
        for worker in owned:
            worker.cancel()
        await asyncio.gather(*owned, return_exceptions=True)
        self._workers = []
        self._worker_task = None
        self._maintenance_task = None
        while not self.task_queue.empty():
            _, _, task = self.task_queue.get_nowait()
            task.status = TaskStatus.FAILED
            task.error = ProviderErrorCode.CANCELLED.value
            task.completed_at = datetime.now()
            self._finish(task)
            self.task_queue.task_done()

    async def submit_task(
        self,
        description: str,
        payload: Dict[str, Any],
        user_id: str,
        session_id: Optional[str] = None,
        force_offload: bool = False,
    ) -> Tuple[bool, Optional[str]]:
        """Admit once; forced admission preserves tool routing and queue bounds."""
        if not self._running:
            return False, None
        should_offload, task_type, priority = await self.classifier.should_offload_task(
            description,
            payload,
            {"user_id": user_id, "user_waiting": False},
        )
        if not force_offload and not should_offload:
            return False, None
        task = AutonomicTask(
            task_id=f"task_{uuid4().hex}",
            task_type=task_type,
            priority=priority,
            description=description,
            payload=payload,
            user_id=user_id,
            session_id=session_id,
        )
        order = (
            {
                TaskPriority.CRITICAL: 0,
                TaskPriority.HIGH: 1,
                TaskPriority.MEDIUM: 2,
                TaskPriority.LOW: 3,
            }[priority]
            if self.priority_enabled
            else 0
        )
        try:
            self.task_queue.put_nowait((order, next(self._sequence), task))
        except asyncio.QueueFull:
            return False, None
        self.queued_tasks[task.task_id] = task
        return True, task.task_id

    async def request_memory_maintenance(self) -> Tuple[bool, Optional[str]]:
        """Coalesce startup, timer and conversation triggers into one operation."""
        if self.processor.memory_maintenance is None:
            return False, None
        if (
            self._maintenance_id in self.queued_tasks
            or self._maintenance_id in self.active_tasks
        ):
            return True, self._maintenance_id
        accepted, task_id = await self.submit_task(
            "Maintain searchable memory",
            {"operation": "maintain_memory"},
            user_id="system",
            force_offload=True,
        )
        if accepted:
            self._maintenance_id = task_id
        return accepted, task_id

    async def _maintenance_loop(self) -> None:
        while self._running:
            await self.request_memory_maintenance()
            await asyncio.sleep(self.maintenance_interval_seconds)

    async def get_task_result(
        self, task_id: str, timeout: Optional[float] = None
    ) -> Optional[AutonomicTask]:
        deadline = time.monotonic() + max(0.0, min(timeout or 0.0, 60.0))
        while task_id not in self.completed_tasks and time.monotonic() < deadline:
            if task_id not in self.queued_tasks and task_id not in self.active_tasks:
                return None
            await asyncio.sleep(0.05)
        return (
            self.completed_tasks.get(task_id)
            or self.active_tasks.get(task_id)
            or self.queued_tasks.get(task_id)
        )

    async def _task_worker(self) -> None:
        while self._running:
            _, _, task = await self.task_queue.get()
            self.queued_tasks.pop(task.task_id, None)
            self.active_tasks[task.task_id] = task
            try:
                await self.processor.execute_task(
                    task, self.mcp_bridge, self.internal_tools
                )
            finally:
                self._finish(task)
                self.task_queue.task_done()

    def _finish(self, task: AutonomicTask) -> None:
        self.active_tasks.pop(task.task_id, None)
        self.queued_tasks.pop(task.task_id, None)
        self.completed_tasks[task.task_id] = task
        if task.task_type is TaskType.MEMORY_MAINTENANCE:
            self.last_maintenance = {
                "status": task.status.value,
                "error": task.error,
                "completed_at": task.completed_at.isoformat()
                if task.completed_at
                else None,
                "result": task.result,
            }
        while len(self.completed_tasks) > 1000:
            self.completed_tasks.pop(next(iter(self.completed_tasks)))

    def get_system_status(self) -> Dict[str, Any]:
        stats = self.processor.get_stats()
        state = (
            AutonomicState.DISABLED
            if self._provider_runtime is None
            else AutonomicState.RUNNING
            if self._running
            else AutonomicState.STOPPED
        )
        return {
            "status": state.value,
            "disabled_reason": self._disabled_reason,
            "running": self._running,
            "queued_tasks": len(self.queued_tasks),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "max_concurrent_tasks": self.max_concurrent_tasks,
            "queue_max_size": self.queue_max_size,
            "queue_utilization": self.task_queue.qsize() / self.queue_max_size * 100,
            "processor_stats": stats,
            "rate_limiting": stats["rate_limiter"],
            "task_threshold": self.classifier.threshold,
            "autonomic_model": self.processor.autonomic_model,
            "model_selection": "shared_conversation_provider",
            "priority_enabled": self.priority_enabled,
            "maintenance_interval_seconds": self.maintenance_interval_seconds,
            "last_memory_maintenance": self.last_maintenance,
            "task_history": "in_process_only",
        }


# Global autonomic system instance
_autonomic_system: Optional[AutonomicNervousSystem] = None


async def initialize_autonomic_system(
    mcp_bridge: Optional[MCPGeminiBridge] = None,
    internal_tools: Optional[AuraInternalTools] = None,
    provider_runtime: ProviderRuntime | None = None,
    memory_maintenance: Callable[[], Awaitable[Dict[str, Any]]] | None = None,
) -> AutonomicNervousSystem:
    """Initialize with an explicit runtime or return a disabled subsystem."""
    global _autonomic_system

    import os

    from aura_backend.runtime.autonomic_config import AutonomicSettings

    settings = AutonomicSettings.from_mapping(os.environ)
    _autonomic_system = AutonomicNervousSystem(
        max_concurrent_tasks=settings.concurrency,
        task_threshold=settings.threshold,
        max_output_tokens=settings.max_tokens,
        timeout_seconds=settings.timeout_seconds,
        rpm_limit=settings.rpm,
        rpd_limit=settings.rpd,
        queue_max_size=settings.queue_size,
        provider_runtime=provider_runtime,
        priority_enabled=settings.priority_enabled,
        maintenance_interval_seconds=settings.maintenance_interval_seconds,
    )
    _autonomic_system.processor.memory_maintenance = memory_maintenance
    # Set external system references
    _autonomic_system.set_external_systems(mcp_bridge, internal_tools)

    # Start the system
    await _autonomic_system.start()

    logger.info(
        "Global autonomic nervous system initialized status=%s",
        _autonomic_system.get_system_status()["status"],
    )
    return _autonomic_system


def get_autonomic_system() -> Optional[AutonomicNervousSystem]:
    """Get the global autonomic system instance"""
    return _autonomic_system


async def shutdown_autonomic_system():
    """Shutdown the global autonomic system"""
    global _autonomic_system

    if _autonomic_system:
        await _autonomic_system.stop()
        _autonomic_system = None
        logger.info("🛑 Global autonomic nervous system shutdown complete")
