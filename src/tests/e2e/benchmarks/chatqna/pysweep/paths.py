# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
"""Where pysweep looks for the installed environment's files.

The ai-solutions installer keeps this repo at ``<deploy-root>/ext/enterprise.ai-erag``
and the environment it built at ``<deploy-root>/env/local``. So when pysweep runs from
that checkout, the deploy root follows from this file's own location, whoever the
installing user was. ``/root/ai-solutions`` (the default for a root install) is tried
after it.

Probing is done with :func:`is_file` / :func:`is_dir` rather than the ``Path``
methods: on Python < 3.14 those raise ``PermissionError`` for a path under an
unreadable directory such as ``/root``, which would turn "not here, try the next
candidate" into a crash for any non-root user.
"""

from pathlib import Path

ROOT_DEPLOY_ROOT = Path("/root/ai-solutions")

# parents[6] is the repo root: pysweep/chatqna/benchmarks/e2e/tests/src/<repo>.
REPO_ROOT = Path(__file__).resolve().parents[6]


def deploy_roots() -> tuple[Path, ...]:
    """Candidate ai-solutions deploy roots, most specific first."""
    roots = []
    if REPO_ROOT.parent.name == "ext":
        roots.append(REPO_ROOT.parent.parent)
    if ROOT_DEPLOY_ROOT not in roots:
        roots.append(ROOT_DEPLOY_ROOT)
    return tuple(roots)


def is_file(path: Path) -> bool:
    """``path.is_file()``, treating an unreadable location as absent."""
    try:
        return path.is_file()
    except OSError:
        return False


def is_dir(path: Path) -> bool:
    """``path.is_dir()``, treating an unreadable location as absent."""
    try:
        return path.is_dir()
    except OSError:
        return False
