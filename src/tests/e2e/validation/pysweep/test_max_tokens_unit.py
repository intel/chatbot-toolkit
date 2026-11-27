# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.base import Plane, PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens


class FakeClient:
    def __init__(self, values=None):
        self.values = {"max_new_tokens": 128} if values is None else values
        self.bodies = []
        self.read_components = []

    def read_values(self, component):
        self.read_components.append(component)
        return dict(self.values)

    def change_arguments(self, body):
        self.bodies.append(body)
        for entry in body:
            self.values.update(entry["data"])


@allure.feature("pysweep")
def test_is_runtime_api_plane():
    assert MaxNewTokens.plane == Plane.RUNTIME_API


@allure.feature("pysweep")
def test_read_returns_the_observed_value():
    assert MaxNewTokens(FakeClient({"max_new_tokens": 256})).read() == 256


@allure.feature("pysweep")
def test_read_asks_for_the_llm_component():
    client = FakeClient()
    MaxNewTokens(client).read()
    assert client.read_components == ["llm"]


@allure.feature("pysweep")
def test_read_coerces_a_string_value():
    assert MaxNewTokens(FakeClient({"max_new_tokens": "512"})).read() == 512


@allure.feature("pysweep")
def test_read_raises_when_the_parameter_is_absent():
    """The error lists the keys that were offered."""
    with pytest.raises(PropertyReadError) as excinfo:
        MaxNewTokens(FakeClient({"streaming": True})).read()
    assert "streaming" in str(excinfo.value)


@allure.feature("pysweep")
def test_read_raises_on_an_unparseable_value():
    with pytest.raises(PropertyReadError, match="not an integer"):
        MaxNewTokens(FakeClient({"max_new_tokens": "unlimited"})).read()


@allure.feature("pysweep")
def test_is_satisfied_compares_as_integers():
    prop = MaxNewTokens(FakeClient({"max_new_tokens": 128}))
    assert prop.is_satisfied("128")
    assert not prop.is_satisfied(256)


@allure.feature("pysweep")
def test_apply_posts_the_documented_body():
    client = FakeClient({"max_new_tokens": 128})
    MaxNewTokens(client).apply(256)
    assert client.bodies == [[{"name": "llm", "data": {"max_new_tokens": 256}}]]


@allure.feature("pysweep")
def test_apply_then_read_reflects_the_desired_value():
    client = FakeClient({"max_new_tokens": 128})
    prop = MaxNewTokens(client)
    prop.apply(256)
    assert prop.read() == 256
    assert prop.is_satisfied(256)


@allure.feature("pysweep")
def test_apply_skips_the_write_when_already_satisfied():
    client = FakeClient({"max_new_tokens": 128})
    MaxNewTokens(client).apply(128)
    assert client.bodies == []


@allure.feature("pysweep")
def test_apply_coerces_the_desired_value():
    client = FakeClient({"max_new_tokens": 128})
    MaxNewTokens(client).apply("256")
    (body,) = client.bodies
    assert body[0]["data"]["max_new_tokens"] == 256
