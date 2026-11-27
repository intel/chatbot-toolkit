# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.cluster import model_catalog
from tests.e2e.benchmarks.chatqna.pysweep.cluster.model_catalog import (
    CatalogError,
    catalog_path,
    known_models,
    load_catalog,
    model_entry,
    tokenizer_for,
)

# Trimmed copy of the deployed env/local/models.yaml.
CATALOG_SAMPLE = """\
defaults:
  namespace: llm-inference
  runtime: vllm
  replicas: 1

storage:
  pvc_name: model-store

models:
  # LLM - quantized Llama-3 8B, chat/completion (shorthand form)
  - name: llama3-8b-awq
    model_id: casperhansen/llama-3-8b-instruct-awq
    category: llm
    runtime: vllm
    server_version: "0.24.0"
    cpu: 16
    memory: 32Gi

  - name: qwen3-5-2b-ovms
    model_id: OpenVINO/Qwen3.5-2B-int4-ov
    category: llm
    runtime: ovms
"""


@pytest.fixture
def catalog_file(tmp_path, monkeypatch):
    """A catalog on disk, selected via ``PYSWEEP_MODEL_CATALOG``."""
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG_SAMPLE)
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(path))
    monkeypatch.delenv("MM_CONFIG", raising=False)
    return path


@allure.feature("pysweep")
def test_loads_every_named_model(catalog_file):
    assert sorted(load_catalog()) == ["llama3-8b-awq", "qwen3-5-2b-ovms"]


@allure.feature("pysweep")
def test_entries_are_returned_verbatim(catalog_file):
    entry = model_entry("llama3-8b-awq")
    assert entry["model_id"] == "casperhansen/llama-3-8b-instruct-awq"
    assert entry["cpu"] == 16
    assert entry["memory"] == "32Gi"


@allure.feature("pysweep")
def test_known_models_is_sorted(catalog_file):
    assert known_models() == ["llama3-8b-awq", "qwen3-5-2b-ovms"]


@allure.feature("pysweep")
def test_tokenizer_is_the_huggingface_repo_id(catalog_file):
    assert tokenizer_for("llama3-8b-awq") == "casperhansen/llama-3-8b-instruct-awq"
    assert tokenizer_for("qwen3-5-2b-ovms") == "OpenVINO/Qwen3.5-2B-int4-ov"


@allure.feature("pysweep")
def test_unknown_model_error_lists_the_available_names(catalog_file):
    with pytest.raises(CatalogError) as excinfo:
        model_entry("casperhansen/llama-3-8b-instruct-awq")
    message = str(excinfo.value)
    assert "llama3-8b-awq" in message
    assert "qwen3-5-2b-ovms" in message


@allure.feature("pysweep")
def test_entry_without_a_model_id_has_no_tokenizer(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - name: mystery\n    category: llm\n")
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(path))

    with pytest.raises(CatalogError, match="model_id"):
        tokenizer_for("mystery")


@allure.feature("pysweep")
def test_env_var_selects_the_catalog(catalog_file, tmp_path, monkeypatch):
    assert catalog_path() == catalog_file

    other = tmp_path / "other.yaml"
    other.write_text("models:\n  - name: only-here\n    model_id: org/only\n")
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(other))
    assert known_models() == ["only-here"]


@allure.feature("pysweep")
def test_mm_config_is_honoured_as_a_fallback(tmp_path, monkeypatch):
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG_SAMPLE)
    monkeypatch.delenv("PYSWEEP_MODEL_CATALOG", raising=False)
    monkeypatch.setenv("MM_CONFIG", str(path))

    assert catalog_path() == path


@allure.feature("pysweep")
def test_pysweep_var_wins_over_mm_config(tmp_path, monkeypatch):
    preferred = tmp_path / "preferred.yaml"
    preferred.write_text(CATALOG_SAMPLE)
    other = tmp_path / "other.yaml"
    other.write_text(CATALOG_SAMPLE)
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(preferred))
    monkeypatch.setenv("MM_CONFIG", str(other))

    assert catalog_path() == preferred


@allure.feature("pysweep")
def test_a_configured_path_that_does_not_exist_is_an_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(tmp_path / "missing.yaml"))
    with pytest.raises(CatalogError, match="PYSWEEP_MODEL_CATALOG"):
        catalog_path()


@allure.feature("pysweep")
def test_no_catalog_anywhere_is_an_error(monkeypatch):
    monkeypatch.delenv("PYSWEEP_MODEL_CATALOG", raising=False)
    monkeypatch.delenv("MM_CONFIG", raising=False)
    monkeypatch.setattr(model_catalog, "CATALOG_FALLBACKS", ())

    with pytest.raises(CatalogError, match="Cannot find a model-manager catalog"):
        catalog_path()


@allure.feature("pysweep")
# Explicit ids: conftest names per-test log files after the test id.
@pytest.mark.parametrize(
    "content",
    [
        "models: [",
        "- just\n- a\n- list\n",
        "models:\n  llama3-8b-awq:\n    model_id: org/x\n",
        "defaults:\n  namespace: llm-inference\n",
    ],
    ids=["not-yaml", "top-level-list", "models-as-mapping", "no-models-section"],
)
def test_a_malformed_catalog_is_reported_as_such(tmp_path, monkeypatch, content):
    path = tmp_path / "models.yaml"
    path.write_text(content)
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(path))

    with pytest.raises(CatalogError):
        load_catalog()


@allure.feature("pysweep")
def test_the_repo_default_catalog_is_readable():
    """Also checks that ``paths.REPO_ROOT`` still points at the repo root."""
    repo_default = model_catalog.CATALOG_FALLBACKS[-1]
    models = known_models(repo_default)
    assert models, "the shipped catalog declares no models"
    for name in models:
        assert tokenizer_for(name, repo_default)


@allure.feature("pysweep")
def test_the_rag_catalog_is_preferred_over_the_platform_one():
    """Fallbacks pair models-rag.yaml then models.yaml per root, then the repo default."""
    fallbacks = [Path(p).name for p in model_catalog.CATALOG_FALLBACKS]
    for root_rag, root_platform in zip(fallbacks[0:-1:2], fallbacks[1:-1:2]):
        assert (root_rag, root_platform) == ("models-rag.yaml", "models.yaml")
