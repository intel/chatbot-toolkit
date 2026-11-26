# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    original = max_tokens_property.read()
    yield original
    max_tokens_property.apply(original)


@allure.feature("pysweep")
def test_read_returns_int(max_tokens_property):
    value = max_tokens_property.read()
    assert isinstance(value, int)


@allure.feature("pysweep")
def test_apply_then_read_reflects_desired(max_tokens_property, restore_max_tokens):
    current = restore_max_tokens
    desired = 256 if current != 256 else 128

    max_tokens_property.apply(desired)

    assert max_tokens_property.read() == desired
    assert max_tokens_property.is_satisfied(desired)


@allure.feature("pysweep")
def test_apply_is_idempotent(max_tokens_property, restore_max_tokens):
    desired = 200
    max_tokens_property.apply(desired)
    assert max_tokens_property.is_satisfied(desired)

    max_tokens_property.apply(desired)
    assert max_tokens_property.read() == desired
