"""Real, copy-only Memvid snapshots of committed Aura conversation sessions.

SQLite remains the source of truth. Archives use their own precomputed vectors,
never the SDK's implicit cloud provider or an unrelated Chroma collection.
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import json
import os
import threading
import uuid
from pathlib import Path
from typing import Any

from aura_backend.runtime.embeddings import RuntimeEmbeddingService


class MemvidArchiveService:
    """Publish immutable .mv2 snapshots only after a successful close/reopen."""

    def __init__(self, repository: Any, archive_root: Path, embeddings: Any = None) -> None:
        self._lock = threading.RLock()
        self.repository = repository
        self.root = archive_root
        self.root.mkdir(parents=True, exist_ok=True)
        self.sdk = importlib.import_module("memvid_sdk")
        env = dict(os.environ)
        env["AURA_EMBEDDING_PROVIDER"] = env.get("MEMVID_EMBEDDING_PROVIDER", "ollama")
        env["AURA_EMBEDDING_MODEL"] = env.get("MEMVID_EMBEDDING_MODEL", "embeddinggemma:latest")
        self.embeddings = embeddings or RuntimeEmbeddingService(env)

    @staticmethod
    def _scope(user_id: str) -> str:
        if not user_id.strip():
            raise ValueError("A user scope is required")
        return hashlib.sha256(user_id.encode()).hexdigest()

    async def archive_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        """Copy up to the latest 100 exchanges; never delete the source."""
        return await asyncio.to_thread(self._archive_session, user_id, session_id)

    def _archive_session(self, user_id: str, session_id: str) -> dict[str, Any]:
        with self._lock:
            directory = self.root / self._scope(user_id)
            messages = self.repository.session_messages(user_id, session_id, 100)
            if not messages:
                raise ValueError("No committed messages in this session")
            identity = self.embeddings.get_model_info()
            payload = json.dumps([session_id, messages, identity], sort_keys=True)
            archive_id = hashlib.sha256(payload.encode()).hexdigest()
            directory.mkdir(exist_ok=True)
            target = directory / f"{archive_id}.mv2"
            manifest = directory / f"{archive_id}.json"
            if manifest.exists():
                if not target.is_file():
                    raise RuntimeError("Memvid archive file is missing")
                return json.loads(manifest.read_text())
            # Chunk before embedding; no implicit model truncation is allowed.
            docs = []
            for message in messages:
                content = message["content"]
                for offset in range(0, len(content), 2000):
                    docs.append({
                        "uri": f"aura://{archive_id}/{len(docs)}",
                        "title": f"{message['role']} conversation",
                        "label": "conversation",
                        "text": content[offset:offset + 2000],
                        "metadata": {"user_id": user_id, "session_id": session_id,
                                     "event_id": message["id"], "offset": offset,
                                     "original_text": content[offset:offset + 2000]},
                    })
            vectors = self.embeddings.encode_batch([doc["text"] for doc in docs])
            temporary = directory / f".pending-{uuid.uuid4().hex}.mv2"
            archive = self.sdk.create(str(temporary), enable_vec=True, enable_lex=True)
            try:
                frame_ids = archive.put_many(docs, embeddings=vectors, opts={"enable_embedding": False})
                if len(frame_ids) != len(docs):
                    raise RuntimeError("Memvid archive frame count mismatch")
                archive.commit()
            finally:
                archive.close()
            # Opening and reading every frame proves stored bytes, not just a receipt.
            archive = self.sdk.use("basic", str(temporary))
            try:
                for doc in docs:
                    if self._metadata(archive.frame(doc["uri"])) != doc["metadata"]:
                        raise RuntimeError("Memvid archive read-back mismatch")
            finally:
                archive.close()
            os.replace(temporary, target)
            result = {
                "status": "success", "archive_id": archive_id,
                "name": f"Session {session_id}", "session_id": session_id,
                "messages_archived": len(messages), "chunks": len(docs),
                "selection": "latest_100_exchanges", "source_records_retained": True,
                "embedding_identity": identity, "size_bytes": target.stat().st_size,
            }
            pending_manifest = directory / f".pending-{uuid.uuid4().hex}.json"
            pending_manifest.write_text(json.dumps(result, indent=2))
            os.replace(pending_manifest, manifest)
            return result

    async def list_archives(self, user_id: str | None = None) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._list_archives, user_id)

    @staticmethod
    def _metadata(frame: dict[str, Any]) -> dict[str, Any]:
        # SDK 2.0.160 frame() returns metadata only and JSON-encoded values.
        # Keep exact source text there: search snippets include appended fields
        # and are not a faithful serialization of the original message.
        return {key: json.loads(value) for key, value in frame["extra_metadata"].items()}

    def _list_archives(self, user_id: str | None) -> list[dict[str, Any]]:
        with self._lock:
            directory = self.root / self._scope(user_id) if user_id else self.root
            pattern = "*.json" if user_id else "*/*.json"
            results = []
            for path in sorted(directory.glob(pattern)):
                if path.name.startswith("."):
                    continue
                if not path.with_suffix(".mv2").is_file():
                    raise RuntimeError("Memvid archive file is missing")
                results.append(json.loads(path.read_text()))
            return results

    async def search_archives(self, query: str, user_id: str, max_results: int = 10) -> list[dict[str, Any]]:
        return await asyncio.to_thread(self._search, query, user_id, max_results)

    def _search(self, query: str, user_id: str, max_results: int) -> list[dict[str, Any]]:
        if not query.strip() or not 1 <= max_results <= 100:
            raise ValueError("A query and a result limit between 1 and 100 are required")
        with self._lock:
            manifests = self._list_archives(user_id)
            if not manifests:
                return []
            identity = self.embeddings.get_model_info()
            if any(item["embedding_identity"] != identity for item in manifests):
                raise RuntimeError("Archive embedding configuration changed; recreate the session snapshot")
            vector = self.embeddings.encode_single(query)
            results: dict[str, dict[str, Any]] = {}
            for item in manifests:
                path = self.root / self._scope(user_id) / f"{item['archive_id']}.mv2"
                archive = self.sdk.use("basic", str(path))
                try:
                    found = archive.find(query, k=max_results, mode="sem", query_embedding=vector)
                    for hit in found["hits"]:
                        frame = archive.frame(hit["uri"])
                        metadata = self._metadata(frame)
                        if metadata.get("user_id") != user_id:
                            raise RuntimeError("Memvid archive scope mismatch")
                        key = f"{metadata['event_id']}:{metadata['offset']}"
                        results[key] = {
                            "content": metadata.pop("original_text"), "metadata": metadata,
                            "source": "memvid", "archive_id": item["archive_id"],
                            "score": hit.get("score", 0),
                        }
                finally:
                    archive.close()
            return sorted(results.values(), key=lambda result: result["score"], reverse=True)[:max_results]

    async def close(self) -> None:
        """Operations own and close their SDK handles; wait for in-flight work."""
        await asyncio.to_thread(self._close)

    def _close(self) -> None:
        with self._lock:
            pass
