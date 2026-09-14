"""Validated resource budgets for Aura's optional background worker."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from aura_backend.runtime.config import (
    RuntimeConfigurationError,
    _bounded_integer,
    _strict_boolean,
)


@dataclass(frozen=True, slots=True)
class AutonomicSettings:
    """Small local defaults; background generation shares the selected provider."""

    concurrency: int = 1
    queue_size: int = 32
    max_tokens: int = 2048
    timeout_seconds: int = 120
    rpm: int = 10
    rpd: int = 500
    threshold: str = "medium"
    priority_enabled: bool = True
    maintenance_interval_seconds: int = 300

    @classmethod
    def from_mapping(cls, mapping: Mapping[str, str | None]) -> AutonomicSettings:
        """Reject invalid limits before a queue or worker can be created."""

        def number(key: str, default: int, maximum: int) -> int:
            return _bounded_integer(mapping, key, default, minimum=1, maximum=maximum)

        threshold = mapping.get("AUTONOMIC_TASK_THRESHOLD", "medium")
        if threshold not in {"low", "medium", "high"}:
            raise RuntimeConfigurationError("AUTONOMIC_TASK_THRESHOLD")
        return cls(
            concurrency=number("AUTONOMIC_MAX_CONCURRENT_TASKS", 1, 32),
            queue_size=number("AUTONOMIC_QUEUE_MAX_SIZE", 32, 1000),
            max_tokens=number("AURA_AUTONOMIC_MAX_OUTPUT_TOKENS", 2048, 32768),
            timeout_seconds=number("AUTONOMIC_TIMEOUT_SECONDS", 120, 3600),
            rpm=number("AUTONOMIC_RATE_LIMIT_RPM", 10, 1000),
            rpd=number("AUTONOMIC_RATE_LIMIT_RPD", 500, 100000),
            threshold=threshold,
            priority_enabled=_strict_boolean(
                {"AUTONOMIC_QUEUE_PRIORITY_ENABLED": "true", **mapping},
                "AUTONOMIC_QUEUE_PRIORITY_ENABLED",
            ),
            maintenance_interval_seconds=number(
                "AUTONOMIC_MAINTENANCE_INTERVAL_SECONDS",
                300,
                86400,
            ),
        )
