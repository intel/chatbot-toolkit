# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``RerankerTopN`` property.

These cover read, is_satisfied, and apply in isolation by stubbing the
fingerprint client — no cluster needed. The live contract (that the deployment
really exposes ``reranker``/``top_n`` at this shape) is covered by
``test_reranker_top_n.py``.

The payload assertions are the API contract: the fingerprint microservice takes
``[{"name": "reranker", "data": {"top_n": N}}]`` — the body
``prepare_change_reranker.sh`` POSTed — and a property that sent anything else
would silently change nothing.
"""

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.base import Plane, PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.properties.reranker_top_n import RerankerTopN


class FakeClient:
    """Records what was written and read, and serves a settable values dict.

    ``read_values`` mirrors the real client's per-component reader, including
    recording the component asked for: the reader is scoped to one component, so
    asking for the wrong one would return another component's values — and both
    the reranker and the retriever carry document counts, so a mis-scoped read
    here would not even look wrong.
    """

    def __init__(self, params=None):
        self.params = {"top_n": 3} if params is None else params
        self.bodies = []
        self.read_components = []

    def read_values(self, component):
        self.read_components.append(component)
        return dict(self.params)

    def change_arguments(self, body):
        self.bodies.append(body)
        for entry in body:
            self.params.update(entry["data"])


@allure.feature("pysweep")
def test_is_runtime_api_plane():
    """Runtime-API plane: varies as an inner dimension, no rollout to wait out."""
    assert RerankerTopN.plane == Plane.RUNTIME_API


@allure.feature("pysweep")
def test_read_returns_the_observed_top_n():
    assert RerankerTopN(FakeClient({"top_n": 7})).read() == 7


@allure.feature("pysweep")
def test_read_asks_for_the_reranker_component():
    """The reader is per-component, so it must name the one this property drives."""
    client = FakeClient({"top_n": 7})
    RerankerTopN(client).read()
    assert client.read_components == ["reranker"]


@allure.feature("pysweep")
def test_read_coerces_a_numeric_string():
    """The API has returned stringly-typed numbers before; top_n is used as an int."""
    assert RerankerTopN(FakeClient({"top_n": "7"})).read() == 7


@allure.feature("pysweep")
def test_read_raises_when_top_n_is_absent():
    """An absent value must fail loudly, not degrade to a None that never satisfies."""
    with pytest.raises(PropertyReadError, match="absent"):
        RerankerTopN(FakeClient({"rerank_score_threshold": 0.02})).read()


@allure.feature("pysweep")
def test_read_error_lists_the_keys_it_did_get():
    """The message has to be enough to spot a renamed key without a debugger."""
    with pytest.raises(PropertyReadError, match="rerank_score_threshold"):
        RerankerTopN(FakeClient({"rerank_score_threshold": 0.02})).read()


@allure.feature("pysweep")
def test_read_raises_when_top_n_is_not_an_integer():
    with pytest.raises(PropertyReadError, match="not an integer"):
        RerankerTopN(FakeClient({"top_n": "all of them"})).read()


@allure.feature("pysweep")
def test_read_does_not_fall_back_to_the_retrievers_k():
    """``k`` is a document count too, and it is not this knob.

    A reranker component that answered with only ``k`` would mean the fingerprint
    schema changed under us; reading it as ``top_n`` would benchmark a
    configuration nobody declared.
    """
    with pytest.raises(PropertyReadError, match="absent"):
        RerankerTopN(FakeClient({"k": 10})).read()


@allure.feature("pysweep")
def test_is_satisfied_compares_observed_value():
    prop = RerankerTopN(FakeClient({"top_n": 5}))
    assert prop.is_satisfied(5)
    assert not prop.is_satisfied(6)


@allure.feature("pysweep")
def test_apply_posts_the_reranker_component_and_top_n():
    """The exact payload prepare_change_reranker.sh sent."""
    client = FakeClient({"top_n": 3})
    RerankerTopN(client).apply(7)

    assert client.bodies == [[{"name": "reranker", "data": {"top_n": 7}}]]


@allure.feature("pysweep")
def test_apply_writes_nothing_when_already_satisfied():
    """Idempotence: a re-run must not re-POST a value that already holds."""
    client = FakeClient({"top_n": 5})
    RerankerTopN(client).apply(5)
    assert client.bodies == []


@allure.feature("pysweep")
def test_apply_leaves_the_other_reranker_arguments_alone():
    """change_arguments merges per component, so top_n must be the only key sent.

    ``rerank_score_threshold`` lives in the same component and also decides how
    much context survives; a property that rewrote the whole component would
    reset it to a default mid-sweep and quietly change what was being measured.
    """
    client = FakeClient({"top_n": 3, "rerank_score_threshold": 0.5})
    RerankerTopN(client).apply(7)

    assert client.bodies == [[{"name": "reranker", "data": {"top_n": 7}}]]
    assert client.params["rerank_score_threshold"] == 0.5


@allure.feature("pysweep")
def test_apply_then_read_reflects_desired():
    client = FakeClient({"top_n": 3})
    prop = RerankerTopN(client)
    prop.apply(4)
    assert prop.read() == 4
    assert prop.is_satisfied(4)
