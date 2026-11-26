# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess
from pathlib import Path

from .paths import REPO_ROOT

GIT_TIMEOUT = 30
COMMIT_INFO_FIELDS = ("sha", "datetime", "author", "description", "branch")


class CommitInfoError(Exception):
    """The checked-out commit cannot be identified."""


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True, text=True, timeout=GIT_TIMEOUT, check=False,
    )


def read_commit_info(repo: Path = REPO_ROOT) -> dict[str, str]:
    """Return the committed HEAD's sha, committer datetime, author, subject and branch."""
    if shutil.which("git") is None:
        raise CommitInfoError("git is not installed, so the checked-out commit cannot be read.")

    log = _git(repo, "log", "-1", "--format=%H%x00%cI%x00%an%x00%s")
    if log.returncode != 0:
        raise CommitInfoError(
            f"Cannot read the checked-out commit of {repo}: {log.stderr.strip() or log.stdout.strip()}. "
            "pysweep must run from a git checkout (a synced copy needs a readable .git)."
        )
    sha, committed_at, author, description = log.stdout.rstrip("\n").split("\x00")

    branch = _git(repo, "symbolic-ref", "--short", "-q", "HEAD")
    if branch.returncode != 0 or not branch.stdout.strip():
        raise CommitInfoError(
            f"HEAD of {repo} is detached ({sha[:12]}); check out a branch so the results "
            "can be attributed to one."
        )

    return {
        "sha": sha,
        "datetime": committed_at,
        "author": author,
        "description": description,
        "branch": branch.stdout.strip(),
    }
