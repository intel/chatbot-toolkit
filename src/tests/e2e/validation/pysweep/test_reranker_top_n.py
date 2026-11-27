# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

# test_read_* only GET and test_apply_* write; -k "read and not apply" selects the read-only ones.

import logging

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.reranker_top_n import RerankerTopN

logger = logging.getLogger(__name__)


@pytest.fixture
def reranker_top_n_property(fingerprint_client):
    return RerankerTopN(fingerprint_client)


@pytest.fixture
def restore_reranker_top_n(reranker_top_n_property):
    original = reranker_top_n_property.read()
    yield original
    reranker_top_n_property.apply(original)


@allure.feature("pysweep")
def test_read_returns_int(reranker_top_n_property):
    value = reranker_top_n_property.read()
    logger.info("Observed reranker top_n: %s", value)
    assert isinstance(value, int)


@allure.feature("pysweep")
def test_read_returns_a_usable_top_n(reranker_top_n_property):
    assert reranker_top_n_property.read() > 0


@allure.feature("pysweep")
def test_read_agrees_with_is_satisfied(reranker_top_n_property):
    observed = reranker_top_n_property.read()

    assert reranker_top_n_property.is_satisfied(observed)
    assert not reranker_top_n_property.is_satisfied(observed + 1)


@allure.feature("pysweep")
def test_apply_makes_observed_satisfy_desired(reranker_top_n_property, restore_reranker_top_n):
    current = restore_reranker_top_n
    desired = 7 if current != 7 else 2

    reranker_top_n_property.apply(desired)

    assert reranker_top_n_property.read() == desired
    assert reranker_top_n_property.is_satisfied(desired)


@allure.feature("pysweep")
def test_apply_is_idempotent(reranker_top_n_property, restore_reranker_top_n):
    desired = 2
    reranker_top_n_property.apply(desired)
    assert reranker_top_n_property.is_satisfied(desired)

    reranker_top_n_property.apply(desired)
    assert reranker_top_n_property.read() == desired
