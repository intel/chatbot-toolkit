# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.base import Plane, PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.properties.retriever_k import RetrieverK


class FakeClient:
    def __init__(self, params=None):
        self.params = {"k": 10} if params is None else params
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
    assert RetrieverK.plane == Plane.RUNTIME_API


@allure.feature("pysweep")
def test_read_returns_the_observed_k():
    assert RetrieverK(FakeClient({"k": 7})).read() == 7


@allure.feature("pysweep")
def test_read_asks_for_the_retriever_component():
    client = FakeClient({"k": 7})
    RetrieverK(client).read()
    assert client.read_components == ["retriever"]


@allure.feature("pysweep")
def test_read_coerces_a_numeric_string():
    assert RetrieverK(FakeClient({"k": "7"})).read() == 7


@allure.feature("pysweep")
def test_read_raises_when_k_is_absent():
    with pytest.raises(PropertyReadError, match="absent"):
        RetrieverK(FakeClient({"top_n": 3})).read()


@allure.feature("pysweep")
def test_read_error_lists_the_keys_it_did_get():
    with pytest.raises(PropertyReadError, match="top_n"):
        RetrieverK(FakeClient({"top_n": 3})).read()


@allure.feature("pysweep")
def test_read_raises_when_k_is_not_an_integer():
    with pytest.raises(PropertyReadError, match="not an integer"):
        RetrieverK(FakeClient({"k": "many"})).read()


@allure.feature("pysweep")
def test_is_satisfied_compares_observed_value():
    prop = RetrieverK(FakeClient({"k": 5}))
    assert prop.is_satisfied(5)
    assert not prop.is_satisfied(6)


@allure.feature("pysweep")
def test_apply_posts_the_retriever_component_and_k():
    client = FakeClient({"k": 10})
    RetrieverK(client).apply(5)

    assert client.bodies == [[{"name": "retriever", "data": {"k": 5}}]]


@allure.feature("pysweep")
def test_apply_writes_nothing_when_already_satisfied():
    client = FakeClient({"k": 5})
    RetrieverK(client).apply(5)
    assert client.bodies == []


@allure.feature("pysweep")
def test_apply_then_read_reflects_desired():
    client = FakeClient({"k": 10})
    prop = RetrieverK(client)
    prop.apply(4)
    assert prop.read() == 4
    assert prop.is_satisfied(4)
