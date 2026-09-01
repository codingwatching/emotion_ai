"""Synthetic Phase 3 generated-data and staged-artifact Git gates."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path, PurePosixPath


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
PHASE3_CANDIDATES = (
    "aura_data_v2/aura.sqlite3",
    "aura_data_v2/projection-generations/gen-001/chroma.sqlite3",
    "aura_snapshots/snapshot-001.sqlite3",
    "aura_backups/generation-001/aura.sqlite3",
    "aura_exports/scope.export.json",
    "aura_profiles/scope.profile.json",
    "aura_traces/run.trace.json",
    "aura_logs/aura.log",
    "aura_secrets/provider.secret",
    "memvid_videos/synthetic.mv2",
)
_GENERATED_PARTS = {
    "aura_data_v2",
    "projection-generations",
    "aura_snapshots",
    "aura_backups",
    "aura_exports",
    "aura_profiles",
    "aura_traces",
    "aura_logs",
    "aura_secrets",
    "memvid_videos",
}


def _git(root: Path, *args: str, check: bool = True) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    ).stdout


def _phase3_generated(path: str) -> bool:
    candidate = PurePosixPath(path)
    return bool(_GENERATED_PARTS.intersection(candidate.parts)) or candidate.suffix in {
        ".mv2",
        ".secret",
    }


def _new_staged_generated(root: Path, baseline: set[str]) -> set[str]:
    staged = set(_git(root, "diff", "--cached", "--name-only").splitlines())
    return {path for path in staged - baseline if _phase3_generated(path)}


def test_phase3_generated_paths_are_ignored_without_reading_artifacts() -> None:
    """Every representative ledger/projection/backup/export/secret path is ignored."""
    ignored = subprocess.run(
        ["git", "check-ignore", "--no-index", "--stdin"],
        cwd=REPOSITORY_ROOT,
        check=False,
        input="\n".join(PHASE3_CANDIDATES) + "\n",
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    assert set(ignored) == set(PHASE3_CANDIDATES)


def test_synthetic_force_staged_phase3_artifact_is_rejected(tmp_path: Path) -> None:
    """Ignore bypass cannot make a newly staged generated artifact acceptable."""
    root = tmp_path / "synthetic-git"
    root.mkdir()
    shutil.copyfile(REPOSITORY_ROOT / ".gitignore", root / ".gitignore")
    _git(root, "init", "-q")
    _git(root, "config", "user.email", "synthetic@example.invalid")
    _git(root, "config", "user.name", "Synthetic Test")
    _git(root, "add", ".gitignore")
    _git(root, "commit", "-qm", "synthetic baseline")
    baseline = set(_git(root, "ls-files").splitlines())

    artifact = root / "aura_data_v2" / "projection-generations" / "gen-001" / "chroma.sqlite3"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"synthetic disposable bytes")
    _git(root, "add", "-f", artifact.relative_to(root).as_posix())

    assert _new_staged_generated(root, baseline) == {
        "aura_data_v2/projection-generations/gen-001/chroma.sqlite3"
    }


def test_current_index_keeps_exact_grandfathered_runtime_baseline() -> None:
    """Phase 3 adds no newly tracked generated artifact to the real index."""
    from tests.test_repository_hygiene import (
        BASELINE_PATH,
        tracked_runtime_records,
    )

    import json

    expected = json.loads(BASELINE_PATH.read_text(encoding="utf-8"))["artifacts"]
    assert tracked_runtime_records() == expected
