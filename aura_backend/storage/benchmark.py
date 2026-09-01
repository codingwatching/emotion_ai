"""Fail-closed contracts for Aura's sanitized memory benchmark."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


class InstrumentValidationError(ValueError):
    """Content-free benchmark validation error with a stable code."""

    def __init__(self, code: str, *, identifier: str | None = None) -> None:
        self.code = code
        self.identifier = identifier
        detail = f" identifier={identifier}" if identifier is not None else ""
        super().__init__(f"benchmark instrument invalid: code={code}{detail}")


@dataclass(frozen=True, slots=True)
class LoadedInstrument:
    """Validated immutable benchmark inputs."""

    manifest: dict[str, Any]
    records: tuple[dict[str, Any], ...]
    case_ids: tuple[str, ...]
    load_event_count: int


def load_instrument(corpus_path: Path, manifest_path: Path) -> LoadedInstrument:
    """Load and validate a bounded corpus plus its frozen manifest."""
    raise InstrumentValidationError("not_implemented")
