# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import provenance
from tests.e2e.benchmarks.chatqna.pysweep.cluster.served_model import ServedModel
from tests.e2e.benchmarks.chatqna.pysweep.properties.base import PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.provenance import MetadataError, collect_metadata


@pytest.fixture
def sources(monkeypatch):
    monkeypatch.setattr(provenance, "read_commit_info", lambda: {"sha": "abc"})
    monkeypatch.setattr(ServedModel, "read", lambda self: "llama3-8b-awq")
    monkeypatch.setattr(provenance, "read_hardware", lambda model: {"nodes": ["n"], "model": model})


@allure.feature("pysweep")
def test_metadata_groups_commit_info_and_the_served_models_hardware(sources):
    assert collect_metadata() == {
        "commit_info": {"sha": "abc"},
        "hardware": {"nodes": ["n"], "model": "llama3-8b-awq"},
    }


@allure.feature("pysweep")
def test_a_missing_commit_is_a_metadata_error(sources, monkeypatch):
    def unreadable():
        raise provenance.CommitInfoError("not a git repository")

    monkeypatch.setattr(provenance, "read_commit_info", unreadable)
    with pytest.raises(MetadataError, match="commit_info: not a git repository"):
        collect_metadata()


@allure.feature("pysweep")
def test_an_unrecognized_cpu_is_a_metadata_error(sources, monkeypatch):
    def unrecognized(model):
        raise provenance.HardwareError("Unrecognized CPU family on node n")

    monkeypatch.setattr(provenance, "read_hardware", unrecognized)
    with pytest.raises(MetadataError, match="hardware: Unrecognized CPU family"):
        collect_metadata()


@allure.feature("pysweep")
def test_an_unreadable_served_model_is_a_metadata_error(sources, monkeypatch):
    def unreadable(self):
        raise PropertyReadError("no llm-usvc pods")

    monkeypatch.setattr(ServedModel, "read", unreadable)
    with pytest.raises(MetadataError, match="hardware: no llm-usvc pods"):
        collect_metadata()
