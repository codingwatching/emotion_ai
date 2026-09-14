"""Real SDK disk round trips, with only embeddings replaced for offline tests."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from aura_backend.runtime.memvid import MemvidArchiveService


class Embeddings:
    def get_model_info(self):
        return {"model_name": "synthetic-test"}

    def encode_batch(self, texts):
        return [[1.0, 0.2, 0.3] for _ in texts]

    def encode_single(self, text):
        return self.encode_batch([text])[0]


def service(tmp_path: Path) -> MemvidArchiveService:
    pytest.importorskip("memvid_sdk", reason="Run uv sync --locked --extra memvid for real archive tests")
    messages = [{"id": "event-1", "role": "user", "content": "The vault code is AMBER-7429."}]
    repository = SimpleNamespace(session_messages=lambda user, session, limit: messages if user == "test" else [])
    return MemvidArchiveService(repository, tmp_path, Embeddings())


@pytest.mark.asyncio
async def test_real_sdk_archive_reopen_search_and_scope(tmp_path: Path):
    first = service(tmp_path)
    result = await first.archive_session("test", "session")
    assert result["messages_archived"] == 1
    assert result["source_records_retained"] is True
    assert await first.archive_session("test", "session") == result
    await first.close()
    reopened = service(tmp_path)
    assert len(await reopened.list_archives("test")) == 1
    assert await reopened.list_archives("other") == []
    found = await reopened.search_archives("vault code", "test")
    assert found[0]["content"] == "The vault code is AMBER-7429."
    assert await reopened.search_archives("vault code", "other") == []
    await reopened.close()


@pytest.mark.asyncio
async def test_archive_failure_never_publishes_manifest(tmp_path: Path):
    archive = service(tmp_path)
    def fail(texts):
        raise RuntimeError("embedding unavailable")
    archive.embeddings.encode_batch = fail
    with pytest.raises(RuntimeError, match="embedding unavailable"):
        await archive.archive_session("test", "session")
    assert await archive.list_archives("test") == []


@pytest.mark.asyncio
async def test_archive_embedding_mismatch_fails_explicitly(tmp_path: Path):
    archive = service(tmp_path)
    await archive.archive_session("test", "session")
    archive.embeddings.get_model_info = lambda: {"model_name": "different"}
    with pytest.raises(RuntimeError, match="configuration changed"):
        await archive.search_archives("vault", "test")
