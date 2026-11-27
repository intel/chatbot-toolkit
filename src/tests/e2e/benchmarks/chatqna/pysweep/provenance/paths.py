# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

ROOT_DEPLOY_ROOT = Path("/root/ai-solutions")

# parents[7] is the repo root: provenance/pysweep/chatqna/benchmarks/e2e/tests/src/<repo>.
REPO_ROOT = Path(__file__).resolve().parents[7]


def deploy_roots() -> tuple[Path, ...]:
    """Candidate ai-solutions deploy roots, most specific first."""
    roots = []
    if REPO_ROOT.parent.name == "ext":
        roots.append(REPO_ROOT.parent.parent)
    if ROOT_DEPLOY_ROOT not in roots:
        roots.append(ROOT_DEPLOY_ROOT)
    return tuple(roots)


# Path.is_file/is_dir raise PermissionError under /root on Python < 3.14.
def is_file(path: Path) -> bool:
    try:
        return path.is_file()
    except OSError:
        return False


def is_dir(path: Path) -> bool:
    try:
        return path.is_dir()
    except OSError:
        return False
