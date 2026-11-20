# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``RetrieverK`` — the retriever's ``k`` knob (Runtime API plane).

Set with:

    POST /v1/system_fingerprint/change_arguments
        [{"name": "retriever", "data": {"k": <N>}}]

and read back from ``GET config?params_key=retriever`` (the ``values`` object).

``k`` is how many documents the retriever fetches per query, so it sets the
context the LLM has to process — the reason it is a benchmark dimension. Note it
is *not* the number the LLM finally sees: the reranker's ``top_n`` narrows the
retrieved set afterwards, so ``k`` bounds retrieval work while ``top_n`` bounds
generation work. Sweeping the two together is what separates the costs.

The value the app validates against is ``PositiveInt`` (see
``comps.cores.proto.docarray``, where ``k`` defaults to 10); ``k=0`` would
retrieve nothing, so the config schema rejects it rather than letting the
deployment do so an hour into a sweep.
"""

import logging

from ..base import Plane, Property, PropertyReadError
from ..runtime_api import FingerprintClient

logger = logging.getLogger(__name__)

# The fingerprint component and parameter key this property drives. The component
# is both what change_arguments addresses and the params_key the reader asks for;
# the parameter is the key the value comes back under in that component's 'values'.
COMPONENT = "retriever"
PARAMETER = "k"


class RetrieverK(Property):
    """Number of documents the retriever fetches per query."""

    plane = Plane.RUNTIME_API

    def __init__(self, client: FingerprintClient):
        self.client = client

    def read(self) -> int:
        """Return the observed ``k`` from the live fingerprint API.

        Raises :class:`PropertyReadError` if the parameter is absent from the
        response or is not an integer. A missing value is not an acceptable
        state: it means we cannot know what configuration is live, so it fails
        loudly rather than degrading to a value that silently never satisfies a
        desired one.
        """
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
        """Set the retriever's ``k`` to ``desired`` (idempotent)."""
        desired = int(desired)
        if self.is_satisfied(desired):
            logger.debug("retriever k already %s; skipping change_arguments.", desired)
            return
        logger.info("Setting %s.%s = %s via change_arguments.", COMPONENT, PARAMETER, desired)
        self.client.change_arguments(
            [{"name": COMPONENT, "data": {PARAMETER: desired}}]
        )

    # Runtime-API plane is ~synchronous: the value is live once change_arguments
    # returns, so the inherited no-op wait_until_ready is correct.
