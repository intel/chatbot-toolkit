# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import model_catalog
from tests.e2e.benchmarks.chatqna.pysweep.served_model import LLMInferenceService, ServedModel
from tests.e2e.validation.constants import LLM_INFERENCE_NAMESPACE

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def observed_model(pysweep_kubernetes_access):
    model = ServedModel().read()
    logger.info("Deployment observed serving model %r.", model)
    return model


@allure.feature("pysweep")
def test_read_returns_the_served_model_name(observed_model):
    assert observed_model
    assert observed_model == observed_model.strip()


@allure.feature("pysweep")
def test_the_observed_model_is_a_catalog_name(observed_model):
    known = model_catalog.known_models()
    assert observed_model in known, (
        f"The deployment serves {observed_model!r}, which is not a name in the "
        f"model catalog ({known}), so no tokenizer can be derived for it."
    )


@allure.feature("pysweep")
def test_the_observed_model_has_a_tokenizer(observed_model):
    tokenizer = model_catalog.tokenizer_for(observed_model)
    assert "/" in tokenizer, (
        f"{tokenizer!r} does not look like a HuggingFace repo id; the benchmark "
        f"would fail to load a tokenizer for it."
    )


@allure.feature("pysweep")
def test_the_served_model_is_a_ready_llminferenceservice(observed_model):
    llmisvc = LLMInferenceService.get(observed_model, namespace=LLM_INFERENCE_NAMESPACE)
    conditions = (llmisvc.raw.get("status", {}) or {}).get("conditions", []) or []
    assert any(
        cond.get("type") == "Ready" and cond.get("status") == "True" for cond in conditions
    ), f"{observed_model} is not Ready; conditions: {conditions}"
