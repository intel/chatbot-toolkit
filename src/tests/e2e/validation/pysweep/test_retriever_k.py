# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""e2e test for the ``RetrieverK`` property.

This is the DESIGN.md objective-6 contract: it exercises ``read`` / ``apply`` /
``is_satisfied`` against the live ``system_fingerprint`` API, so if the
deployment's runtime-API contract drifts (endpoint path, the ``retriever``
component name, or the flat ``k`` key in the ``values`` response) this test
fails *before* a benchmark silently runs against the wrong configuration.

The test restores the original value on teardown so it leaves no side effect.
"""

import logging

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.retriever_k import RetrieverK

logger = logging.getLogger(__name__)


@pytest.fixture
def retriever_k_property(fingerprint_client):
    return RetrieverK(fingerprint_client)


@pytest.fixture
def restore_retriever_k(retriever_k_property):
    """Snapshot the observed value and restore it after the test."""
    original = retriever_k_property.read()
    yield original
    retriever_k_property.apply(original)


@allure.feature("pysweep")
def test_read_returns_int(retriever_k_property):
    """read() must return an int parsed from observed API state.

    A missing value is not acceptable — read() raises rather than returning
    None (see RetrieverK.read / PropertyReadError), so any deployment where the
    value is absent fails here.
    """
    value = retriever_k_property.read()
    assert isinstance(value, int)


@allure.feature("pysweep")
def test_read_returns_a_usable_k(retriever_k_property):
    """A retriever fetching 0 documents would make the whole RAG pipeline moot.

    The app's own schema types k as PositiveInt, so an observed non-positive
    value means the deployment is in a state pysweep should not benchmark.
    """
    assert retriever_k_property.read() > 0


@allure.feature("pysweep")
def test_apply_then_read_reflects_desired(retriever_k_property, restore_retriever_k):
    """apply(desired) makes the observed value satisfy desired.

    Uses a value different from the current one so the write path is exercised.
    """
    current = restore_retriever_k
    desired = 5 if current != 5 else 8

    retriever_k_property.apply(desired)

    assert retriever_k_property.read() == desired
    assert retriever_k_property.is_satisfied(desired)


@allure.feature("pysweep")
def test_apply_is_idempotent(retriever_k_property, restore_retriever_k):
    """A second apply of an already-satisfied value is a no-op that still holds."""
    desired = 6
    retriever_k_property.apply(desired)
    assert retriever_k_property.is_satisfied(desired)

    # Applying again must not change or break the observed state.
    retriever_k_property.apply(desired)
    assert retriever_k_property.read() == desired
