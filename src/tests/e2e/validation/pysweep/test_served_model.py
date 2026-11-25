# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""e2e test for :mod:`served_model` — its bindings to the live deployment.

The runner tokenizes every load test with the model this module reads, and the
read rests on assumptions about objects this repo does not own: the ``llm-usvc``
label selector and container name, ``LLM_MODEL_NAME`` in the pods' env, and
KServe's ``LLMInferenceService`` at ``serving.kserve.io/v1alpha2``. Any of them can
be renamed by a chart bump without a single unit test noticing, and the symptom
would be a sweep that cannot pick a tokenizer. This module is where that drift
fails.

Read-only: nothing here changes the deployment.
"""

import logging

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import model_catalog, served_model
from tests.e2e.benchmarks.chatqna.pysweep.served_model import LLMInferenceService, ServedModel

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def observed_model(pysweep_kubernetes_access):
    """The model the deployment is observed to serve, gated on the RBAC preflight.

    Every test needs it, and it comes from ``read()`` — so a deployment that is
    mid-rollout or pointed at an unserved model fails here once, with the reason,
    rather than in each test.
    """
    model = ServedModel().read()
    logger.info("Deployment observed serving model %r.", model)
    return model


@allure.feature("pysweep")
def test_read_returns_the_served_model_name(observed_model):
    """The selector selects, the exec works, and the env var is populated.

    ``read`` must observe the pods, so if the label selector or the container name
    stopped matching, this is where it surfaces.
    """
    assert observed_model
    assert observed_model == observed_model.strip()


@allure.feature("pysweep")
def test_the_observed_model_is_a_catalog_name(observed_model):
    """Not a HuggingFace repo id.

    ``LLM_MODEL_NAME`` carries model-manager's short name, which is what the
    gateway routes on. If a deployment ever put a repo id here, the tokenizer
    lookup in :mod:`.model_catalog` would have nothing to map.
    """
    known = model_catalog.known_models()
    assert observed_model in known, (
        f"The deployment serves {observed_model!r}, which is not a name in the "
        f"model catalog ({known}), so no tokenizer can be derived for it."
    )


@allure.feature("pysweep")
def test_the_observed_model_has_a_tokenizer(observed_model):
    """The benchmark's ``-m`` is derived, never configured.

    Every token count in every result row depends on this mapping resolving for the
    model actually being served.
    """
    tokenizer = model_catalog.tokenizer_for(observed_model)
    assert "/" in tokenizer, (
        f"{tokenizer!r} does not look like a HuggingFace repo id; the benchmark "
        f"would fail to load a tokenizer for it."
    )


@allure.feature("pysweep")
def test_the_served_model_is_a_ready_llminferenceservice(observed_model):
    """Binds the hand-written kr8s class to the live CRD.

    The apiVersion is pinned in one place on purpose (see
    :class:`LLMInferenceService`); when KServe moves past ``v1alpha2`` this test is
    the notice.
    """
    llmisvc = LLMInferenceService.get(observed_model, namespace=served_model.INFERENCE_NAMESPACE)
    conditions = (llmisvc.raw.get("status", {}) or {}).get("conditions", []) or []
    assert any(
        cond.get("type") == "Ready" and cond.get("status") == "True" for cond in conditions
    ), f"{observed_model} is not Ready; conditions: {conditions}"
