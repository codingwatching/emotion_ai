"""Observable maintenance and bounded, source-labelled conversation recall."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from aura_backend.storage.connection import open_database


def memory_inventory(database_path: Path) -> dict[str, int]:
    """Count authoritative records without including stored conversation text."""
    connection = open_database(database_path)
    try:
        connection.execute("BEGIN")
        return {
            "committed_turns": connection.execute(
                "SELECT COUNT(*) FROM turns"
            ).fetchone()[0],
            "source_events": connection.execute(
                "SELECT COUNT(*) FROM events"
            ).fetchone()[0],
            "derived_memories": connection.execute(
                "SELECT COUNT(*) FROM derived_memories"
            ).fetchone()[0],
            "pending_index_turns": connection.execute(
                "SELECT COUNT(*) FROM turns WHERE projection_status = 'pending'"
            ).fetchone()[0],
        }
    finally:
        connection.close()


async def maintain_memory(repository: Any, projection: Any) -> dict[str, Any]:
    """Repair the derived index from committed sources, keeping shutdown owned.

    Cancelling an asyncio thread await does not stop its SQLite/Chroma writes.
    Await that worker before releasing runtime storage, including on timeout.
    """

    def repair() -> dict[str, Any]:
        reconciled = projection.reconcile()
        return {
            "reconciled_turns": reconciled,
            **memory_inventory(repository.database_path),
        }

    operation = asyncio.create_task(asyncio.to_thread(repair))
    try:
        return await asyncio.shield(operation)
    except asyncio.CancelledError:
        try:
            await operation
        finally:
            raise


def format_memory_context(
    memories: list[dict[str, Any]], max_chars: int = 12000
) -> str:
    """Fit historical evidence in a fixed prompt budget with source labels."""
    parts: list[str] = []
    remaining = max_chars
    for memory in memories:
        content = memory.get("content")
        if not isinstance(content, str) or not content.strip():
            continue
        source = memory.get("origin_id", "legacy")
        observed = memory.get("observed_at", "date unknown")
        prefix = f"Previous context [{observed}; source={source}]: "
        if remaining <= len(prefix) + 32:
            break
        available = remaining - len(prefix) - 1
        excerpt = (
            content
            if len(content) <= available
            else content[: available - 14] + " [excerpt ends]"
        )
        part = prefix + excerpt
        parts.append(part)
        remaining -= len(part) + 1
    return "\n".join(parts)
