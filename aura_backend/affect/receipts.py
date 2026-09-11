"""Durability evidence, deliberately separate from simulated task outcomes."""

from dataclasses import asdict, dataclass
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class DurableReceipt:
    """A scope coordinator's acknowledgement of one exact turn."""

    status: Literal["committed", "replayed", "rejected", "pending", "unknown", "ephemeral"]
    idempotency_key: str
    turn_id: str | None = None
    transition_id: str | None = None
    projection_status: str = "unknown"

    @property
    def committed(self) -> bool:
        return self.status in ("committed", "replayed")

    def to_dict(self) -> dict[str, Any]:
        """Return the public persistence receipt without task-outcome claims."""
        return asdict(self)
