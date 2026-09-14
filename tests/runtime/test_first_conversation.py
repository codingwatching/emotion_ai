"""Exercise actual startup, storage, and first request rather than readiness alone."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import aura_backend.main as main
from tests.providers.fakes import ScriptedComplete, ScriptedProvider
from aura_backend.providers.base import ProviderHealth, ProviderHealthStatus


@pytest.mark.parametrize("memvid_enabled", [False, True])
def test_fresh_start_can_commit_first_conversation_and_read_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, memvid_enabled: bool,
) -> None:
    if memvid_enabled:
        pytest.importorskip("memvid_sdk")
    root = tmp_path / "not-created-yet"
    monkeypatch.setenv("AURA_CLEAN_INSTALL", "true")
    monkeypatch.setattr(main, "affect_service", None)
    class SyntheticEmbeddings:
        def get_model_info(self) -> dict[str, str]:
            return {"model_name": "synthetic"}

        def encode_batch(self, texts: list[str]) -> list[list[float]]:
            return [[1.0, float(len(text))] for text in texts]

        def encode_single(self, text: str) -> list[float]:
            return self.encode_batch([text])[0]

    monkeypatch.setattr(main, "_RuntimeEmbeddingService", SyntheticEmbeddings)
    monkeypatch.setattr("aura_backend.runtime.memvid.RuntimeEmbeddingService", lambda env: SyntheticEmbeddings())
    monkeypatch.setattr(main, "_composition_environment", lambda: {
        "AURA_LEDGER_DIRECTORY": str(root), "AURA_DEFAULT_PROVIDER": "ollama",
        "OLLAMA_MODEL": "synthetic", "AURA_CLEAN_INSTALL": "true",
        "AURA_MEMVID_ENABLED": str(memvid_enabled).lower(),
    })
    class LocalProvider(ScriptedProvider):
        requests: list = []

        async def generate(self, request):
            self.requests.append(request)
            return await super().generate(request)

        async def health(self) -> ProviderHealth:
            return ProviderHealth(provider="ollama", model="synthetic", status=ProviderHealthStatus.READY)

    provider = LocalProvider((ScriptedComplete("Synthetic reply"),))
    monkeypatch.setattr("aura_backend.providers.factory.ModelProviderFactory.create_provider", lambda *_args, **_kwargs: provider)
    with TestClient(main.create_app()) as client:
        ready = client.get("/ready")
        assert ready.status_code == 200, ready.text
        response = client.post("/conversation", json={
            "user_id": "fresh", "session_id": "first-session", "message": "Hello",
            "idempotency_key": "first-send",
        })
        assert response.status_code == 200, response.text
        assert response.json()["emotional_state"]["simulation"]["disposition"] == "committed"
        history = client.get("/chat-history/fresh?limit=100")
        assert history.status_code == 200
        assert history.json()["sessions"]
        assert history.json()["sessions"][0]["session_id"] == "first-session"
        assert history.json()["sessions"][0]["last_message"] == "Hello"
        definitions = main.aura_internal_tools.get_tool_definitions()
        assert "aura.search_memories" in definitions
        assert "aura.get_user_profile" in definitions
        import asyncio
        profile = asyncio.run(main.aura_internal_tools.get_user_profile("fresh"))
        assert profile["status"] == "not_found"
        asyncio.run(main.aura_file_system.save_user_profile("fresh", {"name": "Synthetic User"}))
        profile = asyncio.run(main.aura_internal_tools.get_user_profile("fresh"))
        assert profile["status"] == "success"
        assert profile["profile"]["name"] == "Synthetic User"
        followup = client.post("/conversation", json={
            "user_id": "fresh", "session_id": "first-session", "message": "Please",
            "idempotency_key": "second-send",
        })
        assert followup.status_code == 200
        request = next(request for request in reversed(provider.requests) if request.messages[-1].content == "Please")
        assert [(message.role, message.content) for message in request.messages] == [
            ("user", "Hello"), ("assistant", "Synthetic reply"), ("user", "Please"),
        ]
        messages = client.get("/chat-history/fresh/first-session?limit=100")
        assert messages.status_code == 200
        assert [item["content"] for item in messages.json()] == ["Hello", "Synthetic reply", "Please", "Synthetic reply"]
        assert client.get("/chat-history/fresh?limit=100").json()["sessions"][0]["last_message"] == "Hello"
        assert client.get("/chat-history/other/first-session?limit=100").json() == []
        assert client.get("/chat-history/fresh/other-session?limit=100").json() == []
        if memvid_enabled:
            archived = client.post("/memvid/archive-session", json={"user_id": "fresh", "session_id": "first-session"})
            assert archived.status_code == 200, archived.text
            assert archived.json()["messages_archived"] == 4
            status = client.get("/memvid/status?user_id=fresh").json()
            assert status["status"] == "operational"
            assert status["archives_count"] == 1
            found = client.post("/search", json={"user_id": "fresh", "query": "Hello", "include_active": False, "include_archives": True})
            assert found.status_code == 200, found.text
            assert found.json()["includes_video_archives"] is True
            assert any(item["content"] == "Hello" for item in found.json()["results"])
            tools = main.aura_internal_tools.get_tool_definitions()
            assert "aura.search_archives" in tools
            assert "aura.archive_session" in tools
            assert "aura_search_archives" in {definition.name for definition in request.tools}
            tool_result = asyncio.run(main.aura_internal_tools.execute_tool("aura.search_archives", {"user_id": "fresh", "query": "Hello"}))
            assert tool_result["status"] == "success"
            assert tool_result["memories"]
            assert len(client.get("/chat-history/fresh/first-session?limit=100").json()) == 4
            assert client.get("/memvid/status?user_id=other").json()["archives_count"] == 0


def test_invalid_history_limit_preserves_client_error() -> None:
    import asyncio
    from fastapi import HTTPException

    with pytest.raises(HTTPException) as error:
        asyncio.run(main.get_chat_history("synthetic", limit=200000))
    assert error.value.status_code == 400
