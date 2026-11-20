# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``LlmModel`` property.

``LlmModel`` is the only property that drives two owners (model-manager for
serving, the GMC operator for the pipeline's ``LLM_MODEL_NAME``) across three
states — **declared** in ``gmc-config``, **rendered** into ``llm-usvc-config``,
**observed** in the pods' env. Almost every way this can be wrong is an ordering
or a state-confusion bug that still leaves a green-looking cluster and a
benchmark of the *previous* model, so that is what these tests pin:

* ``read`` observes the pods, never the ConfigMap we wrote;
* ``apply`` serves the model before repointing the pipeline, waits for the render
  before restarting pods, and does neither when nothing changed;
* a failed deploy leaves ``gmc-config`` alone, so the pipeline is never pointed at
  a model nothing serves.

The cluster and the CLI are both stubbed: ``kr8s``, ``LLMInferenceService`` and
:mod:`.sub.model_manager` are replaced module-locally, and ``time`` with a fake
clock so waits do not sleep. That the real objects have this shape — the label
selector selects, ``exec`` works, ``gmc-config`` carries ``llm-usvc.yaml`` — is
covered by ``test_llm_model.py``, which needs a deployment.
"""

from types import SimpleNamespace

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.base import (
    Plane,
    PropertyApplyError,
    PropertyReadError,
)
from tests.e2e.benchmarks.chatqna.pysweep.properties import llm_model as llm_model_module
from tests.e2e.benchmarks.chatqna.pysweep.properties.llm_model import LlmModel

MODEL = "llama3-8b-awq"
OTHER = "qwen3-5-2b-ovms"


def manifest_for(model: str) -> str:
    """The shape of the ``llm-usvc.yaml`` entry that matters here.

    Only the one line the rewrite touches; the real manifest's full shape is
    pinned by ``test_llm_usvc_manifest_unit.py`` against a captured copy.
    """
    return f'data:\n  LLM_MODEL_NAME: "{model}"\n  LLM_MODEL_SERVER: vllm\n'


class FakePod:
    def __init__(self, name, model, terminating=False, exec_error=None):
        self.name = name
        self.model = model
        self.metadata = {"deletionTimestamp": "2026-01-01T00:00:00Z"} if terminating else {}
        self.exec_error = exec_error
        self.exec_calls = []
        self.deleted = False

    def exec(self, command, container=None):
        self.exec_calls.append((command, container))
        if self.exec_error is not None:
            raise self.exec_error
        return SimpleNamespace(stdout=self.model)

    def delete(self):
        self.deleted = True


class FakeConfigMap:
    def __init__(self, data):
        self.data = dict(data)
        self.patches = []

    def patch(self, body):
        self.patches.append(body)
        self.data.update(body.get("data", {}))


class FakeCluster:
    """A deployment's worth of state, plus a log of the mutations performed on it.

    ``events`` is what makes ordering assertable: ``apply`` has to serve the model
    before it repoints the pipeline and repoint before it restarts pods, and any
    other order produces pods that cannot become Ready.
    """

    def __init__(self, declared=MODEL, rendered=MODEL, pods=None, served=None, replicas=4):
        self.pods = [FakePod("llm-usvc-a", MODEL)] if pods is None else list(pods)
        self.gmc = FakeConfigMap({llm_model_module.LLM_USVC_MANIFEST_KEY: manifest_for(declared)})
        self.rendered = FakeConfigMap({llm_model_module.MODEL_NAME_ENV: rendered})
        # model name -> replica count; membership means "a Ready LLMInferenceService".
        self.served = {MODEL: replicas} if served is None else dict(served)
        self.not_ready = set()
        self.events = []
        self.deploy_calls = []

    # -- what the property is allowed to see ----------------------------------

    def get(self, resource, namespace=None, label_selector=None):
        assert resource == "pods"
        assert label_selector == llm_model_module.LLM_USVC_LABEL_SELECTOR
        return list(self.pods)

    def configmap_get(self, name, namespace=None):
        if name == llm_model_module.GMC_CONFIG_NAME:
            return self.gmc
        if name == llm_model_module.LLM_USVC_CONFIGMAP:
            if self.rendered is None:
                raise RuntimeError(f"configmaps '{name}' not found")
            return self.rendered
        raise AssertionError(f"unexpected ConfigMap {name!r}")

    def llmisvc_get(self, name, namespace=None):
        if name not in self.served:
            raise RuntimeError(f"llminferenceservices '{name}' not found")
        conditions = []
        if name not in self.not_ready:
            conditions = [{"type": "Ready", "status": "True"}]
        return SimpleNamespace(
            raw={"spec": {"replicas": self.served[name]}, "status": {"conditions": conditions}}
        )

    # -- what it does ---------------------------------------------------------

    def deploy(self, model, wait=True, replicas=None, timeout=None):
        self.deploy_calls.append({"model": model, "wait": wait, "replicas": replicas})
        self.events.append(("deploy", model))
        self.served[model] = replicas if replicas is not None else 1
        return "deployed"


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster()
    install(monkeypatch, fake)
    return fake


def install(monkeypatch, fake, deploy=None):
    """Point the property at ``fake`` instead of a cluster, a CLI and a clock."""
    monkeypatch.setattr(
        llm_model_module,
        "kr8s",
        SimpleNamespace(
            get=fake.get,
            objects=SimpleNamespace(ConfigMap=SimpleNamespace(get=fake.configmap_get)),
        ),
    )
    monkeypatch.setattr(
        llm_model_module,
        "LLMInferenceService",
        SimpleNamespace(get=fake.llmisvc_get),
    )
    monkeypatch.setattr(
        llm_model_module,
        "model_manager",
        SimpleNamespace(deploy=deploy or fake.deploy),
    )
    # A fake clock: waits advance time instead of spending it, so timeout paths are
    # testable and the happy paths do not sleep.
    now = {"t": 0.0}
    monkeypatch.setattr(
        llm_model_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: now["t"],
            sleep=lambda seconds: now.update(t=now["t"] + seconds),
        ),
    )
    return fake


# -- plane --------------------------------------------------------------------


@allure.feature("pysweep")
def test_is_k8s_spec_plane():
    """Expensive: a swap redeploys a model and rolls the pipeline, so it sweeps outermost."""
    assert LlmModel.plane == Plane.K8S_SPEC


# -- read ---------------------------------------------------------------------


@allure.feature("pysweep")
def test_read_returns_the_model_the_pods_actually_run(cluster):
    assert LlmModel().read() == MODEL


@allure.feature("pysweep")
def test_read_execs_into_the_application_container(cluster):
    """Not the init container: it has no ``LLM_MODEL_NAME`` and is gone by then."""
    LlmModel().read()
    (_, container), = cluster.pods[0].exec_calls
    assert container == llm_model_module.LLM_USVC_CONTAINER


@allure.feature("pysweep")
def test_read_ignores_what_we_declared(monkeypatch):
    """The invariant the whole design rests on.

    Mid-swap the ConfigMap already holds the new model while the pods still run the
    old one. Reading the declaration would report converged immediately and every
    sweep row would be labelled with a model that was not yet serving.
    """
    fake = FakeCluster(declared=OTHER, rendered=OTHER, served={MODEL: 4, OTHER: 4})
    install(monkeypatch, fake)

    assert LlmModel().read() == MODEL
    assert LlmModel().declared() == OTHER


@allure.feature("pysweep")
def test_read_ignores_terminating_pods(monkeypatch):
    """A pod being replaced is not part of the observed configuration.

    Counting it would make ``read`` report a disagreement for the whole grace
    period after a restart that has otherwise already converged.
    """
    fake = FakeCluster(
        pods=[FakePod("old", OTHER, terminating=True), FakePod("new", MODEL)],
        served={MODEL: 4, OTHER: 4},
    )
    install(monkeypatch, fake)

    assert LlmModel().read() == MODEL


@allure.feature("pysweep")
def test_read_raises_when_replicas_disagree(monkeypatch):
    """Half the fleet on each model is not a configuration a benchmark may run against."""
    fake = FakeCluster(
        pods=[FakePod("a", MODEL), FakePod("b", OTHER)],
        served={MODEL: 4, OTHER: 4},
    )
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="do not agree"):
        LlmModel().read()


@allure.feature("pysweep")
def test_read_raises_when_no_pods_are_running(monkeypatch):
    fake = FakeCluster(pods=[])
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="No live llm-usvc pods"):
        LlmModel().read()


@allure.feature("pysweep")
def test_read_raises_when_the_named_model_is_not_served(monkeypatch):
    """A pipeline pointed at an absent model answers nothing; that is not 'configured'."""
    fake = FakeCluster(served={})
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="nothing is serving it"):
        LlmModel().read()


@allure.feature("pysweep")
def test_read_raises_when_the_model_is_served_but_not_ready(monkeypatch):
    """Weights still loading is exactly the window a sweep must not benchmark in."""
    fake = FakeCluster()
    fake.not_ready = {MODEL}
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="no Ready"):
        LlmModel().read()


@allure.feature("pysweep")
def test_read_raises_on_an_empty_env_var(monkeypatch):
    fake = FakeCluster(pods=[FakePod("a", "")])
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="empty"):
        LlmModel().read()


@allure.feature("pysweep")
def test_read_raises_when_exec_fails(monkeypatch):
    """A starting container cannot be exec'd into; unreadable is not 'unchanged'."""
    fake = FakeCluster(pods=[FakePod("a", MODEL, exec_error=RuntimeError("container not running"))])
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="container not running"):
        LlmModel().read()


# -- is_satisfied -------------------------------------------------------------


@allure.feature("pysweep")
def test_is_satisfied_compares_against_the_observed_model(cluster):
    prop = LlmModel()
    assert prop.is_satisfied(MODEL)
    assert not prop.is_satisfied(OTHER)


@allure.feature("pysweep")
def test_is_satisfied_tolerates_surrounding_whitespace(cluster):
    """YAML values arrive as written; a stray space must not force a needless swap."""
    assert LlmModel().is_satisfied(f"  {MODEL} ")


# -- declared -----------------------------------------------------------------


@allure.feature("pysweep")
def test_declared_reads_gmc_config(cluster):
    assert LlmModel().declared() == MODEL


@allure.feature("pysweep")
def test_declared_raises_when_the_manifest_entry_is_missing(monkeypatch):
    fake = FakeCluster()
    fake.gmc.data[llm_model_module.LLM_USVC_MANIFEST_KEY] = "data:\n  LLM_MODEL_SERVER: vllm\n"
    install(monkeypatch, fake)

    with pytest.raises(PropertyApplyError, match="LLM_MODEL_NAME"):
        LlmModel().declared()


@allure.feature("pysweep")
def test_declared_raises_when_gmc_config_does_not_manage_the_microservice(monkeypatch):
    fake = FakeCluster()
    fake.gmc.data = {}
    install(monkeypatch, fake)

    with pytest.raises(PropertyApplyError, match="cannot repoint"):
        LlmModel().declared()


# -- apply --------------------------------------------------------------------


def converging(fake):
    """Make the fake behave like a cluster that reconciles and rolls out.

    The operator renders whatever ``gmc-config`` declares, and replacement pods come
    up on it — so ``apply``'s waits see progress instead of a frozen cluster.
    """
    original_patch = fake.gmc.patch

    def patch(body):
        original_patch(body)
        fake.events.append(("patch-gmc", llm_model_module.read_model(
            body["data"][llm_model_module.LLM_USVC_MANIFEST_KEY]
        )))
        # The operator reconciles on the write.
        fake.rendered.data[llm_model_module.MODEL_NAME_ENV] = llm_model_module.read_model(
            fake.gmc.data[llm_model_module.LLM_USVC_MANIFEST_KEY]
        )

    fake.gmc.patch = patch

    for pod in fake.pods:
        original_delete = pod.delete

        def delete(pod=pod, original_delete=original_delete):
            original_delete()
            fake.events.append(("delete-pod", pod.name))
            # The Deployment replaces it on the rendered model.
            pod.model = fake.rendered.data[llm_model_module.MODEL_NAME_ENV]

        pod.delete = delete
    return fake


@allure.feature("pysweep")
def test_apply_serves_the_model_before_repointing_the_pipeline(monkeypatch):
    """The only order that works.

    The replacement pods' init container blocks until the model answers on the
    gateway, so repointing first would leave the microservice unable to become
    Ready until the deploy happened to finish.
    """
    fake = converging(install(monkeypatch, FakeCluster()))
    LlmModel().apply(OTHER)

    kinds = [kind for kind, _ in fake.events]
    assert kinds.index("deploy") < kinds.index("patch-gmc") < kinds.index("delete-pod")


@allure.feature("pysweep")
def test_apply_carries_the_outgoing_replica_count_over(monkeypatch):
    """Two models benchmarked at different replica counts are not comparable.

    The catalog's default is 1 no matter what the cluster was scaled to, so the
    count has to come from the model being replaced.
    """
    fake = converging(install(monkeypatch, FakeCluster(replicas=4)))
    LlmModel().apply(OTHER)

    assert fake.deploy_calls[0]["replicas"] == 4
    assert fake.deploy_calls[0]["wait"] is True


@allure.feature("pysweep")
def test_apply_leaves_replicas_to_the_catalog_when_nothing_is_served(monkeypatch):
    """First deploy on a fresh box: there is no outgoing count to carry over."""
    fake = converging(install(monkeypatch, FakeCluster(served={})))
    LlmModel().apply(OTHER)

    assert fake.deploy_calls[0]["replicas"] is None


@allure.feature("pysweep")
def test_apply_patches_only_the_manifest_key(monkeypatch):
    """gmc-config holds every microservice's manifest; the patch must touch one."""
    fake = converging(install(monkeypatch, FakeCluster()))
    LlmModel().apply(OTHER)

    (body,) = fake.gmc.patches
    assert list(body["data"]) == [llm_model_module.LLM_USVC_MANIFEST_KEY]
    assert llm_model_module.read_model(
        body["data"][llm_model_module.LLM_USVC_MANIFEST_KEY]
    ) == OTHER


@allure.feature("pysweep")
def test_apply_is_a_no_op_when_already_converged(monkeypatch):
    """Re-running a combination must not bounce a healthy microservice.

    The deploy still runs — it is idempotent, and it is what repairs a declared
    model that is not actually being served — but nothing is written or restarted.
    """
    fake = converging(install(monkeypatch, FakeCluster()))
    LlmModel().apply(MODEL)

    assert fake.gmc.patches == []
    assert not any(pod.deleted for pod in fake.pods)
    assert fake.deploy_calls[0]["model"] == MODEL


@allure.feature("pysweep")
def test_apply_restarts_pods_when_only_they_are_behind(monkeypatch):
    """Mid-swap re-entry: declared and rendered are current, the pods are not.

    Nothing needs writing, but without the delete the pods would keep the env they
    started with forever — there is no checksum annotation to roll them.
    """
    fake = converging(install(monkeypatch, FakeCluster(
        declared=OTHER, rendered=OTHER, served={MODEL: 4, OTHER: 4},
    )))
    LlmModel().apply(OTHER)

    assert fake.gmc.patches == []
    assert all(pod.deleted for pod in fake.pods)


@allure.feature("pysweep")
def test_apply_does_not_restart_pods_before_the_render_lands(monkeypatch):
    """Deleting them early would just have the replacements read the old value.

    Here the operator never renders, so ``apply`` must fail rather than restart
    into the previous model and report success.
    """
    fake = install(monkeypatch, FakeCluster())  # no reconcile: rendered stays stale
    with pytest.raises(PropertyApplyError, match="did not render"):
        LlmModel().apply(OTHER)

    assert fake.gmc.patches, "the declaration should still have been written"
    assert not any(pod.deleted for pod in fake.pods)


@allure.feature("pysweep")
def test_a_failed_deploy_leaves_the_pipeline_alone(monkeypatch):
    """Otherwise the pipeline points at a model nothing serves, and stops answering."""

    def failing_deploy(model, wait=True, replicas=None, timeout=None):
        raise llm_model_module.ModelManagerError("no NUMA capacity for 32 CPUs")

    fake = FakeCluster()
    install(monkeypatch, fake, deploy=failing_deploy)

    with pytest.raises(PropertyApplyError, match="no NUMA capacity"):
        LlmModel().apply(OTHER)

    assert fake.gmc.patches == []
    assert not any(pod.deleted for pod in fake.pods)


# -- wait_until_ready ---------------------------------------------------------


@allure.feature("pysweep")
def test_wait_until_ready_returns_once_the_pods_carry_the_model(monkeypatch):
    fake = install(monkeypatch, FakeCluster(rendered=OTHER, served={MODEL: 4, OTHER: 4}))
    pod = fake.pods[0]

    reads = {"n": 0}
    original_exec = pod.exec

    def exec_(command, container=None):
        reads["n"] += 1
        if reads["n"] < 3:
            raise RuntimeError("container not running")  # still starting
        pod.model = OTHER
        return original_exec(command, container=container)

    pod.exec = exec_
    LlmModel().wait_until_ready(OTHER)
    assert reads["n"] >= 3


@allure.feature("pysweep")
def test_wait_until_ready_times_out_naming_the_last_observed_state(monkeypatch):
    """A sweep must never benchmark a configuration that never took effect."""
    fake = install(monkeypatch, FakeCluster(served={MODEL: 4, OTHER: 4}))
    assert fake.pods[0].model == MODEL

    with pytest.raises(PropertyApplyError) as excinfo:
        LlmModel().wait_until_ready(OTHER, timeout=60.0)

    message = str(excinfo.value)
    assert OTHER in message
    assert MODEL in message
