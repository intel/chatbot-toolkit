# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""e2e test for the ``LlmModel`` property — the bindings, not a model swap.

``LlmModel`` is held together by five assumptions about objects this repo does not
own: the ``llm-usvc`` label selector and container name, ``LLM_MODEL_NAME`` in the
pods' env, ``llm-usvc.yaml`` inside ``gmc-config``, the rendered
``llm-usvc-config`` ConfigMap, and KServe's ``LLMInferenceService`` at
``serving.kserve.io/v1alpha2``. Any of them can be renamed by a chart bump without
a single unit test noticing, and the symptom would be a sweep that reports numbers
for a model it never established. This module is where that drift fails.

**These tests do not swap the model.** A swap redeploys weights and rolls the
pipeline — tens of minutes at best, and it reconciles the live deployment toward
the catalog. Instead the write path is exercised without side effects: the deploy
half through ``model-manager deploy --dry-run`` (which renders and runs the NUMA
preflight, touching nothing), and the ConfigMap half by rewriting the *live*
manifest in memory and asserting the diff is the one line. Whether an actual swap
converges is what the sweep itself demonstrates.

They also assert the deployment is *converged* — declared, rendered and observed
all agreeing. That is a precondition for benchmarking, so if a rollout is in
flight these fail, which is the honest outcome.
"""

import logging

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties import llm_model as llm_model_module
from tests.e2e.benchmarks.chatqna.pysweep.properties.llm_model import (
    LLMInferenceService,
    LlmModel,
)
from tests.e2e.benchmarks.chatqna.pysweep.properties.sub import model_catalog, model_manager
from tests.e2e.benchmarks.chatqna.pysweep.properties.sub.llm_usvc_manifest import (
    read_model,
    rewrite_for_model,
)

logger = logging.getLogger(__name__)


@pytest.fixture(scope="module")
def llm_model_property(pysweep_kubernetes_access):
    """The property under test, gated on the RBAC preflight.

    Module-scoped: nothing here mutates, so one instance is shared.
    """
    return LlmModel()


@pytest.fixture(scope="module")
def observed_model(llm_model_property):
    """The model the deployment is observed to serve.

    Every test needs it, and it comes from ``read()`` — so a deployment that is
    mid-rollout or pointed at an unserved model fails here once, with the reason,
    rather than in each test.
    """
    model = llm_model_property.read()
    logger.info("Deployment observed serving model %r.", model)
    return model


@allure.feature("pysweep")
def test_read_returns_the_served_model_name(observed_model):
    """The selector selects, the exec works, and the env var is populated.

    Four bindings at once, and the one that matters most: ``read`` must observe the
    pods, so if the label selector or the container name stopped matching, this is
    where it surfaces.
    """
    assert observed_model
    assert observed_model == observed_model.strip()


@allure.feature("pysweep")
def test_the_observed_model_is_a_catalog_name(observed_model):
    """Not a HuggingFace repo id.

    ``LLM_MODEL_NAME`` carries model-manager's short name, which is what the
    gateway routes on and what a sweep declares. If a deployment ever put a repo id
    here, ``llms.model`` values would have to be repo ids too, and the tokenizer
    lookup in :mod:`.sub.model_catalog` would have nothing to map.
    """
    known = model_catalog.known_models()
    assert observed_model in known, (
        f"The deployment serves {observed_model!r}, which is not a name in the "
        f"model-manager catalog ({known}). A sweep declares catalog names, so it "
        f"could not express the configuration currently deployed."
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
def test_is_satisfied_agrees_with_the_observed_model(llm_model_property, observed_model):
    assert llm_model_property.is_satisfied(observed_model)
    assert not llm_model_property.is_satisfied(f"{observed_model}-not-deployed")


@allure.feature("pysweep")
def test_the_served_model_is_a_ready_llminferenceservice(observed_model):
    """Binds the hand-written kr8s class to the live CRD.

    The apiVersion is pinned in one place on purpose (see
    :class:`LLMInferenceService`); when KServe moves past ``v1alpha2`` this test is
    the notice.
    """
    llmisvc = LLMInferenceService.get(
        observed_model, namespace=llm_model_module.INFERENCE_NAMESPACE
    )
    conditions = (llmisvc.raw.get("status", {}) or {}).get("conditions", []) or []
    assert any(
        cond.get("type") == "Ready" and cond.get("status") == "True" for cond in conditions
    ), f"{observed_model} is not Ready; conditions: {conditions}"


@allure.feature("pysweep")
def test_the_replica_count_is_readable(llm_model_property, observed_model):
    """``apply`` carries this over so two models are benchmarked at the same scale.

    Unreadable would silently mean "let the catalog decide", i.e. drop to 1 replica
    on the first swap and invalidate the comparison.
    """
    replicas = llm_model_property._served_replicas(observed_model)
    assert replicas is not None and replicas >= 1, (
        f"Cannot read spec.replicas for {observed_model}; a model swap would fall "
        f"back to the catalog default and change the scale being benchmarked."
    )
    logger.info("Model %r is served at %d replica(s).", observed_model, replicas)


@allure.feature("pysweep")
def test_gmc_config_declares_the_model(llm_model_property, observed_model):
    """The write target exists, and the deployment is converged.

    ``declared`` reads ``llm-usvc.yaml`` out of ``gmc-config`` in ns/system — the
    only object pysweep writes for this property. It agreeing with the observed
    model is what "no rollout in flight" means.
    """
    assert llm_model_property.declared() == observed_model


@allure.feature("pysweep")
def test_the_operator_rendered_the_declared_model(llm_model_property, observed_model):
    """The middle state: ``llm-usvc-config``, written by the GMC operator.

    ``apply`` waits on this before restarting pods. If the operator stopped
    rendering ``LLM_MODEL_NAME`` into this ConfigMap, every swap would time out
    there — with a clear message, thanks to this being a distinct step.
    """
    assert llm_model_property._rendered_model() == observed_model


@allure.feature("pysweep")
def test_rewriting_the_live_manifest_changes_exactly_one_line(llm_model_property):
    """The write path, without writing.

    ``test_llm_usvc_manifest_unit.py`` proves this against a captured manifest;
    here it runs against whatever the deployment currently holds, so a chart that
    reshaped ``llm-usvc.yaml`` (or added a second ``LLM_MODEL_NAME`` line) fails
    before a sweep patches gmc-config.
    """
    manifest = llm_model_property._llm_usvc_manifest(llm_model_property._gmc_config())
    rewritten = rewrite_for_model(manifest, "pysweep-live-rewrite-check")

    assert read_model(rewritten) == "pysweep-live-rewrite-check"
    before, after = manifest.splitlines(), rewritten.splitlines()
    assert len(before) == len(after)
    assert sum(1 for b, a in zip(before, after) if b != a) == 1

    # Nothing was written: the property only patches inside apply().
    assert read_model(
        llm_model_property._llm_usvc_manifest(llm_model_property._gmc_config())
    ) == read_model(manifest)


@allure.feature("pysweep")
def test_model_manager_dry_run_accepts_the_deployed_model(observed_model):
    """The CLI exists, is locatable, and accepts pysweep's arguments.

    This is the whole model-manager coupling checked end to end: path resolution,
    ``deploy <name> --dry-run``, and a clean exit. Dry-running the model that is
    *already* deployed keeps it side-effect free while still exercising the catalog
    lookup and the NUMA capacity gate the sweep preflight relies on.
    """
    logger.info("Using model-manager at %s.", model_manager.model_manager_path())
    output = model_manager.dry_run(observed_model)
    logger.info("model-manager dry-run output:\n%s", output)
