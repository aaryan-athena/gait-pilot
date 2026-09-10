"""Identify which code is actually running.

This exists because of a specific failure that cost three deploy cycles.
Streamlit Community Cloud hot-reloads the entry script when source changes, but
it does not re-import modules already in ``sys.modules`` -- it only restarts the
process when *dependencies* change. So a fix pushed to an imported package sits
on disk, unloaded, while the old code keeps running and the old error keeps
appearing. The logs show it as a "Pulling code changes / Updated app!" pair with
no "Stopping..." and no "Uvicorn server started" in between.

That is indistinguishable from "the fix does not work" unless the running code
can be identified. So the app displays this, and failures quote it: if the
revision shown is not the revision you pushed, the process needs rebooting
rather than the code needing another change.
"""
from __future__ import annotations

import subprocess
from functools import lru_cache
from pathlib import Path

from .version import ALGO_VERSION

#: Bump when the video encoder's process handling changes, so a stale module is
#: obvious from the UI. Kept separate from ALGO_VERSION, which identifies the
#: *measurement* and must not churn for reasons unrelated to it.
ENCODER_REVISION = 3


@lru_cache(maxsize=1)
def git_revision(project_root: str | Path = ".") -> str | None:
    """Short commit hash of the checkout, if this is one."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(project_root), capture_output=True, text=True, timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else None


def describe(project_root: str | Path = ".") -> str:
    """A one-line identifier for the code currently loaded in this process."""
    parts = [f"algo {ALGO_VERSION}", f"encoder rev {ENCODER_REVISION}"]
    revision = git_revision(project_root)
    if revision:
        parts.insert(0, f"build {revision}")
    return " · ".join(parts)
