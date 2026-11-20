# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``MaxNewTokens`` property.

The fingerprint client is stubbed, so these cover the property's own logic with no
deployment: the request body it sends, the component it reads back from, and its
refusal to guess when the observed value is missing or unparseable. The live
contract (that the deployment really exposes ``llm``/``max_new_tokens`` at this
shape) is ``test_max_tokens.py``.

The payload assertions are the API contract: ``change_arguments`` takes
``[{"name": "llm", "data": {"max_new_tokens": N}}]``, and a property that posted
anything else would be accepted and change nothing — the worst outcome for a
sweep, since every row would then be a rerun of the previous configuration.
"""

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.base import Plane, PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens


class FakeClient:
    """Records what was written and read, and serves a settable values dict."""

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
    """Runtime-API plane: varies as an inner dimension, no rollout to wait out."""
    assert MaxNewTokens.plane == Plane.RUNTIME_API


@allure.feature("pysweep")
def test_read_returns_the_observed_value():
    assert MaxNewTokens(FakeClient({"max_new_tokens": 256})).read() == 256


@allure.feature("pysweep")
def test_read_asks_for_the_llm_component():
    """The reader is per-component; another component could carry a same-named key."""
    client = FakeClient()
    MaxNewTokens(client).read()
    assert client.read_components == ["llm"]


@allure.feature("pysweep")
def test_read_coerces_a_string_value():
    """The API may return the value as a string; the sweep compares ints."""
    assert MaxNewTokens(FakeClient({"max_new_tokens": "512"})).read() == 512


@allure.feature("pysweep")
def test_read_raises_when_the_parameter_is_absent():
    """Unknown is not a value.

    Returning ``None`` would make ``is_satisfied`` permanently false and the sweep
    would re-apply forever; failing says which keys were actually offered.
    """
    with pytest.raises(PropertyReadError) as excinfo:
        MaxNewTokens(FakeClient({"streaming": True})).read()
    assert "streaming" in str(excinfo.value)


@allure.feature("pysweep")
def test_read_raises_on_an_unparseable_value():
    with pytest.raises(PropertyReadError, match="not an integer"):
        MaxNewTokens(FakeClient({"max_new_tokens": "unlimited"})).read()


@allure.feature("pysweep")
def test_is_satisfied_compares_as_integers():
    """YAML can hand over a string; a type mismatch must not force a needless write."""
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
    """Idempotence: re-running a combination must not touch the deployment."""
    client = FakeClient({"max_new_tokens": 128})
    MaxNewTokens(client).apply(128)
    assert client.bodies == []


@allure.feature("pysweep")
def test_apply_coerces_the_desired_value():
    """A value read from YAML arrives as a string; the API expects a number."""
    client = FakeClient({"max_new_tokens": 128})
    MaxNewTokens(client).apply("256")
    (body,) = client.bodies
    assert body[0]["data"]["max_new_tokens"] == 256
