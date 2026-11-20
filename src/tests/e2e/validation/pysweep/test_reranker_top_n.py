# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""e2e test for the ``RerankerTopN`` property.

This is the DESIGN.md objective-6 contract: it exercises ``read`` / ``apply`` /
``is_satisfied`` against the live ``system_fingerprint`` API, so if the
deployment's runtime-API contract drifts (endpoint path, the ``reranker``
component name, or the flat ``top_n`` key in the ``values`` response) this test
fails *before* a benchmark silently runs against the wrong configuration.

The mutating tests restore the original value on teardown so they leave no side
effect.

**Running this against a deployment that is mid-benchmark.** ``top_n`` decides
how much context each request carries, so applying it changes what a running
load test is measuring — and the restore only happens at teardown, well after
those requests are gone. The read-only checks are therefore selectable on their
own:

    pytest e2e/validation/pysweep/test_reranker_top_n.py -k "read and not apply"

Every test named ``test_read_*`` in this module only GETs ``config``; the
``test_apply_*`` ones write. Keep that naming — and keep ``read`` out of the
mutating names — if you add to the file, since it is what the selector above
relies on. ``-k`` matches substrings, so ``and not apply`` is not redundant: it
is what stops a name like ``test_apply_then_read_…`` from being selected as a
read.

Note also that the shared ``e2e/validation`` conftest attaches an autouse
fixture that harvests every pod's logs in 13 namespaces after **each** test. On
a host under measurement that is itself a perturbation, so prefer running these
checks after the benchmark set finishes, and treat the read-only selection above
as the "I need the value now" path rather than the routine one.
"""

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
    """Snapshot the observed value and restore it after the test."""
    original = reranker_top_n_property.read()
    yield original
    reranker_top_n_property.apply(original)


@allure.feature("pysweep")
def test_read_returns_int(reranker_top_n_property):
    """read() must return an int parsed from observed API state.

    A missing value is not acceptable — read() raises rather than returning
    None (see RerankerTopN.read / PropertyReadError), so any deployment where the
    value is absent fails here.
    """
    value = reranker_top_n_property.read()
    logger.info("Observed reranker top_n: %s", value)
    assert isinstance(value, int)


@allure.feature("pysweep")
def test_read_returns_a_usable_top_n(reranker_top_n_property):
    """A reranker keeping 0 documents would leave the LLM with no context at all.

    The app's own schema types top_n as PositiveInt (``RerankerParams``), so an
    observed non-positive value means the deployment is in a state pysweep should
    not benchmark.
    """
    assert reranker_top_n_property.read() > 0


@allure.feature("pysweep")
def test_read_agrees_with_is_satisfied(reranker_top_n_property):
    """is_satisfied must be decided by the observed value, and nothing else.

    Read-only on purpose: it pins the read/compare half of the contract on a
    deployment that must not be reconfigured — including one that is mid-sweep,
    where this is the only check that can run.
    """
    observed = reranker_top_n_property.read()

    assert reranker_top_n_property.is_satisfied(observed)
    assert not reranker_top_n_property.is_satisfied(observed + 1)


@allure.feature("pysweep")
def test_apply_makes_observed_satisfy_desired(reranker_top_n_property, restore_reranker_top_n):
    """apply(desired) makes the observed value satisfy desired.

    Named without "read" on purpose: see the module docstring — the read-only
    selection is by name, and this test writes.

    Uses a value different from the current one so the write path is exercised.
    """
    current = restore_reranker_top_n
    desired = 7 if current != 7 else 2

    reranker_top_n_property.apply(desired)

    assert reranker_top_n_property.read() == desired
    assert reranker_top_n_property.is_satisfied(desired)


@allure.feature("pysweep")
def test_apply_is_idempotent(reranker_top_n_property, restore_reranker_top_n):
    """A second apply of an already-satisfied value is a no-op that still holds."""
    desired = 2
    reranker_top_n_property.apply(desired)
    assert reranker_top_n_property.is_satisfied(desired)

    # Applying again must not change or break the observed state.
    reranker_top_n_property.apply(desired)
    assert reranker_top_n_property.read() == desired
