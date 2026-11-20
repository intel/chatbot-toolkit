# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``RerankerTopN`` — the reranker's ``top_n`` knob (Runtime API plane).

Set with:

    POST /v1/system_fingerprint/change_arguments
        [{"name": "reranker", "data": {"top_n": <N>}}]

and read back from ``GET config?params_key=reranker`` (the ``values`` object).
The same call ``prepare_change_reranker.sh $TOP_N`` made, minus the token
plumbing.

``top_n`` is how many of the retrieved documents survive reranking and are
appended to the prompt, so it is the knob that sets the LLM's *input* length:
prefill cost, KV-cache footprint, and therefore TTFT at any concurrency. Its
counterpart ``retriever.k`` bounds the retrieval and reranking work instead
(``k`` documents are fetched and scored, ``top_n`` of them are kept), which is
why the two are swept together — see :mod:`.retriever_k` and the README's
canonical pairings (``k=10, top_n=7``; ``k=4, top_n=2``; ``k=20, top_n=15``).

Note that ``top_n`` is a ceiling, not a guarantee: the reranker can only keep
what retrieval found and what ``rerank_score_threshold`` let through, so a
``top_n`` above the live ``k`` produces the same prompt as ``top_n == k``. That
is not an error the deployment reports — it is a sweep that spends runs on
duplicate configurations — so pair the two knobs deliberately rather than
sweeping ``top_n`` past ``k``.

The value the app validates against is ``PositiveInt`` (``top_n: PositiveInt = 3``
in both ``comps.cores.proto.docarray.SearchedDoc`` and the fingerprint store's
``RerankerParams``); ``top_n=0`` would rerank away the whole context and leave
the LLM answering from nothing, so the config schema rejects it rather than
letting the deployment do so an hour into a sweep.
"""

import logging

from ..base import Plane, Property, PropertyReadError
from ..runtime_api import FingerprintClient

logger = logging.getLogger(__name__)

# The fingerprint component and parameter key this property drives. The component
# is both what change_arguments addresses and the params_key the reader asks for;
# the parameter is the key the value comes back under in that component's 'values'.
COMPONENT = "reranker"
PARAMETER = "top_n"


class RerankerTopN(Property):
    """Number of reranked documents appended to the prompt."""

    plane = Plane.RUNTIME_API

    def __init__(self, client: FingerprintClient):
        self.client = client

    def read(self) -> int:
        """Return the observed ``top_n`` from the live fingerprint API.

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
        """Set the reranker's ``top_n`` to ``desired`` (idempotent)."""
        desired = int(desired)
        if self.is_satisfied(desired):
            logger.debug("reranker top_n already %s; skipping change_arguments.", desired)
            return
        logger.info("Setting %s.%s = %s via change_arguments.", COMPONENT, PARAMETER, desired)
        self.client.change_arguments(
            [{"name": COMPONENT, "data": {PARAMETER: desired}}]
        )

    # Runtime-API plane is ~synchronous: the value is live once change_arguments
    # returns, so the inherited no-op wait_until_ready is correct.
