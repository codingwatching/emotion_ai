"""Explicit, lazy embedding selection for the SQLite-derived search index."""

from __future__ import annotations

import math
import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

import httpx


class RuntimeEmbeddingService:
    """Keep legacy index identity stable unless an operator selects a new model."""

    def __init__(
        self, environment: Mapping[str, str] | None = None,
        *, transport: httpx.BaseTransport | None = None,
    ) -> None:
        env = os.environ if environment is None else environment
        self.provider = env.get("AURA_EMBEDDING_PROVIDER", "sentence_transformers").strip()
        if self.provider not in {"sentence_transformers", "ollama"}:
            raise ValueError("Unsupported AURA_EMBEDDING_PROVIDER")
        default_model = "all-MiniLM-L6-v2" if self.provider == "sentence_transformers" else "embeddinggemma:latest"
        self.model = env.get("AURA_EMBEDDING_MODEL", default_model).strip()
        if not self.model or len(self.model) > 200:
            raise ValueError("Invalid AURA_EMBEDDING_MODEL")
        if self.provider == "sentence_transformers" and self.model != "all-MiniLM-L6-v2":
            raise ValueError("The sentence_transformers adapter supports all-MiniLM-L6-v2 only")
        self.base_url = env.get("AURA_EMBEDDING_BASE_URL", env.get("OLLAMA_BASE_URL", "http://127.0.0.1:11434")).rstrip("/")
        if self.base_url.endswith("/v1"):
            self.base_url = self.base_url[:-3]
        if self.provider == "ollama":
            url = urlsplit(self.base_url)
            if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
                raise ValueError("Invalid AURA_EMBEDDING_BASE_URL")
        self._transport = transport

    def get_model_info(self) -> dict[str, Any]:
        """Return stable identity used to prevent mixing incompatible indexes."""
        if self.provider == "sentence_transformers":
            # Preserve the existing persisted generation hash exactly.
            return {"model_name": self.model, "runtime_contract": "phase-03"}
        return {
            "model_name": self.model, "provider": self.provider,
            "base_url": self.base_url, "runtime_contract": "ollama-embed-v1",
            "truncate": False,
        }

    def encode_single(self, text: str) -> list[float]:
        """Embed one query with the same configuration as indexed documents."""
        return self.encode_batch([text])[0]

    def encode_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed bounded batches; reject failed, truncated, or malformed results."""
        if not texts:
            return []
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("Embedding inputs must be nonempty text")
        if self.provider == "sentence_transformers":
            from aura_backend.shared_embedding_service import get_embedding_service

            return get_embedding_service().encode_batch(texts)
        vectors: list[list[float]] = []
        width: int | None = None
        with httpx.Client(transport=self._transport, timeout=60.0) as client:
            for offset in range(0, len(texts), 64):
                batch = texts[offset:offset + 64]
                response = client.post(f"{self.base_url}/api/embed", json={
                    "model": self.model, "input": batch, "truncate": False,
                })
                response.raise_for_status()
                data = response.json()
                if isinstance(data, dict) and data.get("model") != self.model:
                    raise ValueError("Embedding response model mismatch")
                rows = data.get("embeddings") if isinstance(data, dict) else None
                if not isinstance(rows, list) or len(rows) != len(batch):
                    raise ValueError("Embedding response count mismatch")
                for row in rows:
                    if not isinstance(row, list) or not row or any(
                        isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)
                        for value in row
                    ):
                        raise ValueError("Malformed embedding vector")
                    if width is None:
                        width = len(row)
                    if len(row) != width or not any(value != 0 for value in row):
                        raise ValueError("Inconsistent or zero embedding vector")
                    vectors.append([float(value) for value in row])
        return vectors
