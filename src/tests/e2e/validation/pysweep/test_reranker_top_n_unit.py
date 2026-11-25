# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.base import Plane, PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.properties.reranker_top_n import RerankerTopN


class FakeClient:
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
    assert RerankerTopN.plane == Plane.RUNTIME_API


@allure.feature("pysweep")
def test_read_returns_the_observed_top_n():
    assert RerankerTopN(FakeClient({"top_n": 7})).read() == 7


@allure.feature("pysweep")
def test_read_asks_for_the_reranker_component():
    client = FakeClient({"top_n": 7})
    RerankerTopN(client).read()
    assert client.read_components == ["reranker"]


@allure.feature("pysweep")
def test_read_coerces_a_numeric_string():
    assert RerankerTopN(FakeClient({"top_n": "7"})).read() == 7


@allure.feature("pysweep")
def test_read_raises_when_top_n_is_absent():
    with pytest.raises(PropertyReadError, match="absent"):
        RerankerTopN(FakeClient({"rerank_score_threshold": 0.02})).read()


@allure.feature("pysweep")
def test_read_error_lists_the_keys_it_did_get():
    with pytest.raises(PropertyReadError, match="rerank_score_threshold"):
        RerankerTopN(FakeClient({"rerank_score_threshold": 0.02})).read()


@allure.feature("pysweep")
def test_read_raises_when_top_n_is_not_an_integer():
    with pytest.raises(PropertyReadError, match="not an integer"):
        RerankerTopN(FakeClient({"top_n": "all of them"})).read()


@allure.feature("pysweep")
def test_read_does_not_fall_back_to_the_retrievers_k():
    with pytest.raises(PropertyReadError, match="absent"):
        RerankerTopN(FakeClient({"k": 10})).read()


@allure.feature("pysweep")
def test_is_satisfied_compares_observed_value():
    prop = RerankerTopN(FakeClient({"top_n": 5}))
    assert prop.is_satisfied(5)
    assert not prop.is_satisfied(6)


@allure.feature("pysweep")
def test_apply_posts_the_reranker_component_and_top_n():
    client = FakeClient({"top_n": 3})
    RerankerTopN(client).apply(7)

    assert client.bodies == [[{"name": "reranker", "data": {"top_n": 7}}]]


@allure.feature("pysweep")
def test_apply_writes_nothing_when_already_satisfied():
    client = FakeClient({"top_n": 5})
    RerankerTopN(client).apply(5)
    assert client.bodies == []


@allure.feature("pysweep")
def test_apply_leaves_the_other_reranker_arguments_alone():
    """change_arguments merges per component, so only top_n may be sent."""
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
