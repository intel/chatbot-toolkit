# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

POLL_INTERVAL_S = 30.0
INGEST_TIMEOUT_S = 4 * 3600.0


class VectorDbError(Exception):
    """The vector DB's state cannot be read from EDP."""


@dataclass
class VectorDbState:
    vectors: int = 0
    in_progress: int = 0
    failed: int = 0
    files: set[str] = field(default_factory=set)


def read_state(edp) -> VectorDbState:
    """Ingested vectors (chunks) and files, as EDP reports them."""
    response = edp.list_files()
    if response.status_code != 200:
        raise VectorDbError(f"EDP /files returned HTTP {response.status_code}: {response.text[:300]}")
    state = VectorDbState()
    for item in response.json():
        status = item.get("status")
        state.files.add(Path(item.get("object_name", "")).name)
        if status == "ingested":
            state.vectors += int(item.get("chunks_processed") or 0)
        elif status == "error":
            state.failed += 1
        else:
            state.in_progress += 1
    return state


def wait_until_idle(edp, timeout=INGEST_TIMEOUT_S, poll=POLL_INTERVAL_S, sleep=time.sleep, clock=time.monotonic):
    deadline = clock() + timeout
    while True:
        state = read_state(edp)
        if not state.in_progress:
            return state
        if clock() >= deadline:
            raise VectorDbError(
                f"EDP still processes {state.in_progress} file(s) after {timeout:.0f}s "
                f"({state.vectors} vectors ingested so far)."
            )
        logger.info("Waiting for EDP: %d file(s) in progress, %d vectors ingested.", state.in_progress, state.vectors)
        sleep(poll)


def edp_client():
    from tests.e2e.helpers.edp_helper import EdpHelper

    from .auth import keycloak_helper

    return EdpHelper(keycloak_helper())


def edp_enabled() -> bool:
    from .auth import load_build_config

    return bool(load_build_config().get("edp_enabled"))


def observed_settings(edp, **wait) -> dict[str, int]:
    """The vector DB as each run records it, under ``settings.edp``."""
    return {"vector_db.num_vectors": wait_until_idle(edp, **wait).vectors}
