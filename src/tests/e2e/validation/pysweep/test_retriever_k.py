# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
    original = retriever_k_property.read()
    yield original
    retriever_k_property.apply(original)


@allure.feature("pysweep")
def test_read_returns_int(retriever_k_property):
    value = retriever_k_property.read()
    assert isinstance(value, int)


@allure.feature("pysweep")
def test_read_returns_a_usable_k(retriever_k_property):
    assert retriever_k_property.read() > 0


@allure.feature("pysweep")
def test_apply_then_read_reflects_desired(retriever_k_property, restore_retriever_k):
    current = restore_retriever_k
    desired = 5 if current != 5 else 8

    retriever_k_property.apply(desired)

    assert retriever_k_property.read() == desired
    assert retriever_k_property.is_satisfied(desired)


@allure.feature("pysweep")
def test_apply_is_idempotent(retriever_k_property, restore_retriever_k):
    desired = 6
    retriever_k_property.apply(desired)
    assert retriever_k_property.is_satisfied(desired)

    retriever_k_property.apply(desired)
    assert retriever_k_property.read() == desired
