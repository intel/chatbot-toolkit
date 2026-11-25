# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

from ..base import Plane, Property, PropertyReadError
from ..runtime_api import FingerprintClient

logger = logging.getLogger(__name__)

# The fingerprint component and parameter key this property drives.
COMPONENT = "llm"
PARAMETER = "max_new_tokens"


class MaxNewTokens(Property):
    """Number of new tokens the LLM is allowed to generate per request."""

    plane = Plane.RUNTIME_API

    def __init__(self, client: FingerprintClient):
        self.client = client

    def read(self) -> int:
        """Return the observed ``max_new_tokens`` from the live fingerprint API."""
        values = self.client.read_values(COMPONENT)
        if PARAMETER not in values:
            raise PropertyReadError(
                f"'{PARAMETER}' is absent from the fingerprint '{COMPONENT}' "
                f"arguments; got keys: {sorted(values)}"
            )
        raw = values[PARAMETER]
        try:
            return int(raw)
        except (TypeError, ValueError) as exc:
            raise PropertyReadError(
                f"'{PARAMETER}' is not an integer (got {raw!r})."
            ) from exc

    def is_satisfied(self, desired: int) -> bool:
        return self.read() == int(desired)

    def apply(self, desired: int) -> None:
        """Set ``max_new_tokens`` to ``desired`` (idempotent)."""
        desired = int(desired)
        if self.is_satisfied(desired):
            logger.debug("max_new_tokens already %s; skipping change_arguments.", desired)
            return
        logger.info("Setting %s.%s = %s via change_arguments.", COMPONENT, PARAMETER, desired)
        self.client.change_arguments(
            [{"name": COMPONENT, "data": {PARAMETER: desired}}]
        )

    # Runtime-API plane is ~synchronous: the value is live once change_arguments
    # returns, so the inherited no-op wait_until_ready is correct.
