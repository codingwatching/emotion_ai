"""Configured runtime embeddings without real models or personal storage."""

import json

import httpx
import pytest

from aura_backend.runtime.embeddings import RuntimeEmbeddingService


def test_ollama_embedding_settings_reach_native_batch_endpoint() -> None:
    requests = []

    def respond(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return httpx.Response(200, json={"model": "embeddinggemma:latest", "embeddings": [[1, 0], [0, 1]]})

    service = RuntimeEmbeddingService({
        "AURA_EMBEDDING_PROVIDER": "ollama",
        "AURA_EMBEDDING_MODEL": "embeddinggemma:latest",
        "OLLAMA_BASE_URL": "http://localhost:11434/v1",
    }, transport=httpx.MockTransport(respond))
    assert service.encode_batch(["first", "second"]) == [[1.0, 0.0], [0.0, 1.0]]
    assert str(requests[0].url) == "http://localhost:11434/api/embed"
    assert json.loads(requests[0].content) == {
        "model": "embeddinggemma:latest", "input": ["first", "second"], "truncate": False,
    }
    assert service.get_model_info()["model_name"] == "embeddinggemma:latest"


@pytest.mark.parametrize("payload", [
    {"embeddings": []}, {"embeddings": [[True]]},
    {"embeddings": [[0, 0]]}, {"embeddings": [["1"]]},
    {"model": "wrong-model", "embeddings": [[1, 0]]},
])
def test_malformed_or_wrong_model_embeddings_are_rejected(payload: dict) -> None:
    payload = {"model": "embeddinggemma:latest", **payload}
    service = RuntimeEmbeddingService({"AURA_EMBEDDING_PROVIDER": "ollama"},
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=payload)))
    with pytest.raises(ValueError):
        service.encode_batch(["synthetic input"])


def test_unconfigured_embeddings_preserve_existing_index_identity() -> None:
    assert RuntimeEmbeddingService({}).get_model_info() == {
        "model_name": "all-MiniLM-L6-v2", "runtime_contract": "phase-03",
    }
