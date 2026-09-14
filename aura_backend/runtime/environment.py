"""Shared local configuration loading for CLI and application composition."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


def load_runtime_environment(repository_root: Path | None = None) -> dict[str, str]:
    """Load this project's .env, preserving explicit shell overrides.

    Populate the process environment so owned child processes inherit the same
    settings as preflight. Never search parent directories for unrelated files.
    """
    root = repository_root or Path(__file__).resolve().parents[2]
    load_dotenv(root / ".env", override=False)
    return dict(os.environ)
