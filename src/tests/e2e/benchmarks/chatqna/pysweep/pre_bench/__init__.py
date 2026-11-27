# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
from typing import Any

from ..cluster.vector_db import edp_client
from . import seed
from .seed import SeedError

logger = logging.getLogger(__name__)

NUM_VECTORS = "edp.vector_db.num_vectors"

__all__ = ["NUM_VECTORS", "SeedError", "edp_client", "run_pre_bench", "seed"]


def run_pre_bench(settings: dict[str, Any], edp=None) -> None:
    """Prepare the deployment state every run of the sweep shares; ``settings`` are dotted pre_bench paths."""
    if not settings:
        logger.info("No pre_bench settings declared.")
        return
    logger.info("=== pre_bench: %s ===", settings)
    num_vectors = settings.get(NUM_VECTORS)
    if num_vectors is not None:
        seed.seed(num_vectors, edp or edp_client())
