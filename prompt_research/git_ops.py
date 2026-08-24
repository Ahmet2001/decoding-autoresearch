from __future__ import annotations

import subprocess
from pathlib import Path


def _git(root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=root,
        check=check,
        capture_output=True,
        text=True,
    )


def is_git_repository(root: Path) -> bool:
    result = _git(root, "rev-parse", "--is-inside-work-tree", check=False)
    return result.returncode == 0 and result.stdout.strip() == "true"


def commit_decoding_config(root: Path, candidate_id: str) -> str:
    """Commit only decoding_config.toml. Return a marker when Git is not initialized."""
    if not is_git_repository(root):
        return "no-git"
    _git(root, "add", "--", "decoding_config.toml")
    staged = _git(
        root, "diff", "--cached", "--quiet", "--", "decoding_config.toml", check=False
    )
    if staged.returncode == 0:
        return _git(root, "rev-parse", "--short=7", "HEAD").stdout.strip()
    result = _git(root, "commit", "-m", f"autoresearch: accept {candidate_id}", check=False)
    if result.returncode != 0:
        # A missing Git identity should not destroy an accepted result; make the issue explicit.
        return "uncommitted"
    return _git(root, "rev-parse", "--short=7", "HEAD").stdout.strip()
