"""Static ownership gates for the Phase 3 storage boundary."""

from __future__ import annotations

import ast
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _imports(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
    return names


def test_projection_uses_repository_and_public_chroma_only() -> None:
    path = ROOT / "aura_backend" / "storage" / "projection.py"
    imports = _imports(path)
    source = path.read_text(encoding="utf-8")

    assert "aura_backend.storage.repository" in imports
    assert "sqlite3" not in imports
    assert "aura_backend.robust_vector_db" not in imports
    assert not any(name.startswith("chromadb.db") for name in imports)
    assert "chroma.sqlite3" not in source
    assert "PRAGMA" not in source
    assert "REINDEX" not in source
    assert "PersistentClient(" not in source.split("class ProjectionAdapter", 1)[0]


def test_projection_does_not_capture_singleton_or_default_data_path() -> None:
    source = (
        ROOT / "aura_backend" / "storage" / "projection.py"
    ).read_text(encoding="utf-8")

    assert "get_embedding_service" not in source
    assert "./aura_chroma_db" not in source
    assert "aura_data_v2" not in source

