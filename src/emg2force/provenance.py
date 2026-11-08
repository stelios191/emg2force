"""Repository root and the code commit recorded with every output."""

from __future__ import annotations

import subprocess
from pathlib import Path

_SOURCE_ROOT = Path(__file__).resolve().parents[2]  # src/emg2force/ -> the checkout
# The checkout (runs/, results/, assets/) when running from source. The current folder for an installed package.
REPO = _SOURCE_ROOT if (_SOURCE_ROOT / "pyproject.toml").is_file() else Path.cwd()


def git_state() -> dict:
    """Code commit and whether the working tree had uncommitted changes (None outside git)."""

    def git(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()

    try:
        return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain", "--untracked-files=no"))}
    except (OSError, subprocess.CalledProcessError):
        return {"git_commit": None, "git_dirty": None}
