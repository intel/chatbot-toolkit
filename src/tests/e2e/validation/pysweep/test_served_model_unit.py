# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.cluster import served_model
from tests.e2e.benchmarks.chatqna.pysweep.properties.base import PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.cluster.served_model import ServedModel

MODEL = "llama3-8b-awq"
OTHER = "qwen3-5-2b-ovms"


class FakePod:
    def __init__(self, name, model, terminating=False, exec_error=None):
        self.name = name
        self.model = model
        self.metadata = {"deletionTimestamp": "2026-01-01T00:00:00Z"} if terminating else {}
        self.exec_error = exec_error
        self.exec_calls = []

    def exec(self, command, container=None):
        self.exec_calls.append((command, container))
        if self.exec_error is not None:
            raise self.exec_error
        return SimpleNamespace(stdout=self.model)


class FakeCluster:
    def __init__(self, pods=None, served=None):
        self.pods = [FakePod("llm-usvc-a", MODEL)] if pods is None else list(pods)
        # model name -> replicas; a key means an LLMInferenceService exists.
        self.served = {MODEL: 4} if served is None else dict(served)
        self.not_ready = set()

    def get(self, resource, namespace=None, label_selector=None):
        assert resource == "pods"
        assert label_selector == served_model.LLM_USVC_LABEL_SELECTOR
        return list(self.pods)

    def llmisvc_get(self, name, namespace=None):
        if name not in self.served:
            raise RuntimeError(f"llminferenceservices '{name}' not found")
        conditions = []
        if name not in self.not_ready:
            conditions = [{"type": "Ready", "status": "True"}]
        return SimpleNamespace(raw={"status": {"conditions": conditions}})


@pytest.fixture
def cluster(monkeypatch):
    fake = FakeCluster()
    install(monkeypatch, fake)
    return fake


def install(monkeypatch, fake):
    monkeypatch.setattr(served_model, "kr8s", SimpleNamespace(get=fake.get))
    monkeypatch.setattr(
        served_model, "LLMInferenceService", SimpleNamespace(get=fake.llmisvc_get)
    )
    return fake


@allure.feature("pysweep")
def test_read_returns_the_model_the_pods_actually_run(cluster):
    assert ServedModel().read() == MODEL


@allure.feature("pysweep")
def test_read_execs_into_the_application_container(cluster):
    ServedModel().read()
    (_, container), = cluster.pods[0].exec_calls
    assert container == served_model.LLM_USVC_CONTAINER


@allure.feature("pysweep")
def test_read_reports_the_pods_model_even_when_another_is_served(monkeypatch):
    fake = FakeCluster(served={MODEL: 4, OTHER: 4})
    install(monkeypatch, fake)

    assert ServedModel().read() == MODEL


@allure.feature("pysweep")
def test_read_ignores_terminating_pods(monkeypatch):
    fake = FakeCluster(
        pods=[FakePod("old", OTHER, terminating=True), FakePod("new", MODEL)],
        served={MODEL: 4, OTHER: 4},
    )
    install(monkeypatch, fake)

    assert ServedModel().read() == MODEL


@allure.feature("pysweep")
def test_read_raises_when_replicas_disagree(monkeypatch):
    fake = FakeCluster(
        pods=[FakePod("a", MODEL), FakePod("b", OTHER)],
        served={MODEL: 4, OTHER: 4},
    )
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="do not agree"):
        ServedModel().read()


@allure.feature("pysweep")
def test_read_raises_when_no_pods_are_running(monkeypatch):
    fake = FakeCluster(pods=[])
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="No live llm-usvc pods"):
        ServedModel().read()


@allure.feature("pysweep")
def test_read_raises_when_the_named_model_is_not_served(monkeypatch):
    fake = FakeCluster(served={})
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="nothing is serving it"):
        ServedModel().read()


@allure.feature("pysweep")
def test_read_raises_when_the_model_is_served_but_not_ready(monkeypatch):
    fake = FakeCluster()
    fake.not_ready = {MODEL}
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="no Ready"):
        ServedModel().read()


@allure.feature("pysweep")
def test_read_raises_on_an_empty_env_var(monkeypatch):
    fake = FakeCluster(pods=[FakePod("a", "")])
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="empty"):
        ServedModel().read()


@allure.feature("pysweep")
def test_read_raises_when_exec_fails(monkeypatch):
    fake = FakeCluster(pods=[FakePod("a", MODEL, exec_error=RuntimeError("container not running"))])
    install(monkeypatch, fake)

    with pytest.raises(PropertyReadError, match="container not running"):
        ServedModel().read()
