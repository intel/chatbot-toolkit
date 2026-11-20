# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""e2e test for the ``MaxNewTokens`` property.

This is the DESIGN.md objective-6 contract: it exercises ``read`` / ``apply`` /
``is_satisfied`` against the live ``system_fingerprint`` API, so if the
deployment's runtime-API contract drifts (endpoint path, payload shape, or the
``values`` response key) this test fails *before* a benchmark silently runs
against the wrong configuration.

The test restores the original value on teardown so it leaves no side effect.
"""

import logging

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens

logger = logging.getLogger(__name__)


@pytest.fixture
def max_tokens_property(fingerprint_client):
    return MaxNewTokens(fingerprint_client)


@pytest.fixture
def restore_max_tokens(max_tokens_property):
    """Snapshot the observed value and restore it after the test."""
    original = max_tokens_property.read()
    yield original
    max_tokens_property.apply(original)


@allure.feature("pysweep")
def test_read_returns_int(max_tokens_property):
    """read() must return an int parsed from observed API state.

    A missing value is not acceptable — read() raises rather than returning
    None (see MaxNewTokens.read / PropertyReadError), so any deployment where
    the value is absent fails here.
    """
    value = max_tokens_property.read()
    assert isinstance(value, int)


@allure.feature("pysweep")
def test_apply_then_read_reflects_desired(max_tokens_property, restore_max_tokens):
    """apply(desired) makes the observed value satisfy desired.

    Uses a value different from the current one so the write path is exercised.
    """
    current = restore_max_tokens
    desired = 256 if current != 256 else 128

    max_tokens_property.apply(desired)

    assert max_tokens_property.read() == desired
    assert max_tokens_property.is_satisfied(desired)


@allure.feature("pysweep")
def test_apply_is_idempotent(max_tokens_property, restore_max_tokens):
    """A second apply of an already-satisfied value is a no-op that still holds."""
    desired = 200
    max_tokens_property.apply(desired)
    assert max_tokens_property.is_satisfied(desired)

    # Applying again must not change or break the observed state.
    max_tokens_property.apply(desired)
    assert max_tokens_property.read() == desired
