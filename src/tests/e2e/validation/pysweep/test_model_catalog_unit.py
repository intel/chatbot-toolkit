# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the model-manager catalog reader.

Two things are being pinned here, both of which pysweep depends on and neither of
which it owns:

* **The catalog's shape** — ``models:`` is a *list* of entries each carrying a
  ``name`` and a ``model_id``. ``CATALOG_SAMPLE`` below is copied from the
  deployed ``env/local/models.yaml``, so a format change in model-manager's
  catalog fails here rather than as an empty "available models" list in a
  preflight message.
* **Which file is read** — the deployment's own catalog, not this repo's shipped
  default, since the two drift and only the former lists the names the cluster was
  built from.

No cluster and no model-manager CLI are involved: this module is a YAML reader.
"""

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.sub import model_catalog
from tests.e2e.benchmarks.chatqna.pysweep.properties.sub.model_catalog import (
    CatalogError,
    catalog_path,
    known_models,
    load_catalog,
    model_entry,
    tokenizer_for,
)

# Trimmed from the deployed env/local/models.yaml: two model entries plus the
# surrounding sections the reader has to ignore.
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
    """Callers read whichever field they need, so nothing is dropped or renamed."""
    entry = model_entry("llama3-8b-awq")
    assert entry["model_id"] == "casperhansen/llama-3-8b-instruct-awq"
    assert entry["cpu"] == 16
    assert entry["memory"] == "32Gi"


@allure.feature("pysweep")
def test_known_models_is_sorted(catalog_file):
    """These names go into error messages, where a stable order is what makes them readable."""
    assert known_models() == ["llama3-8b-awq", "qwen3-5-2b-ovms"]


@allure.feature("pysweep")
def test_tokenizer_is_the_huggingface_repo_id(catalog_file):
    """The name/repo-id distinction the whole module exists for.

    ``llms.model`` and ``LLM_MODEL_NAME`` carry the catalog name; ``benchmark.py``'s
    ``-m`` needs the repo id. Conflating them would count every token with the
    wrong tokenizer.
    """
    assert tokenizer_for("llama3-8b-awq") == "casperhansen/llama-3-8b-instruct-awq"
    assert tokenizer_for("qwen3-5-2b-ovms") == "OpenVINO/Qwen3.5-2B-int4-ov"


@allure.feature("pysweep")
def test_unknown_model_error_lists_the_available_names(catalog_file):
    """The failure is nearly always a typo or a repo id in a name's place.

    Both are obvious the moment the real names are on screen, so the message
    carries them.
    """
    with pytest.raises(CatalogError) as excinfo:
        model_entry("casperhansen/llama-3-8b-instruct-awq")
    message = str(excinfo.value)
    assert "llama3-8b-awq" in message
    assert "qwen3-5-2b-ovms" in message


@allure.feature("pysweep")
def test_entry_without_a_model_id_has_no_tokenizer(tmp_path, monkeypatch):
    """Better to fail than to report throughput per token off a guessed tokenizer."""
    path = tmp_path / "models.yaml"
    path.write_text("models:\n  - name: mystery\n    category: llm\n")
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(path))

    with pytest.raises(CatalogError, match="model_id"):
        tokenizer_for("mystery")


@allure.feature("pysweep")
def test_env_var_selects_the_catalog(catalog_file, tmp_path, monkeypatch):
    """The deployment's catalog must win over this repo's shipped default."""
    assert catalog_path() == catalog_file

    other = tmp_path / "other.yaml"
    other.write_text("models:\n  - name: only-here\n    model_id: org/only\n")
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(other))
    assert known_models() == ["only-here"]


@allure.feature("pysweep")
def test_mm_config_is_honoured_as_a_fallback(tmp_path, monkeypatch):
    """``MM_CONFIG`` is what the model-manager wrapper exports.

    Honouring it means pysweep reads the same catalog the deploy will, with
    nothing extra to configure.
    """
    path = tmp_path / "models.yaml"
    path.write_text(CATALOG_SAMPLE)
    monkeypatch.delenv("PYSWEEP_MODEL_CATALOG", raising=False)
    monkeypatch.setenv("MM_CONFIG", str(path))

    assert catalog_path() == path


@allure.feature("pysweep")
def test_pysweep_var_wins_over_mm_config(tmp_path, monkeypatch):
    """An explicit pysweep setting is the one place to override a wrong environment."""
    preferred = tmp_path / "preferred.yaml"
    preferred.write_text(CATALOG_SAMPLE)
    other = tmp_path / "other.yaml"
    other.write_text(CATALOG_SAMPLE)
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(preferred))
    monkeypatch.setenv("MM_CONFIG", str(other))

    assert catalog_path() == preferred


@allure.feature("pysweep")
def test_a_configured_path_that_does_not_exist_is_an_error(tmp_path, monkeypatch):
    """Silently falling through would read a *different* catalog than asked for."""
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(tmp_path / "missing.yaml"))
    with pytest.raises(CatalogError, match="PYSWEEP_MODEL_CATALOG"):
        catalog_path()


@allure.feature("pysweep")
def test_no_catalog_anywhere_is_an_error(monkeypatch):
    """An empty catalog would make every model name 'unknown', which misreports the cause."""
    monkeypatch.delenv("PYSWEEP_MODEL_CATALOG", raising=False)
    monkeypatch.delenv("MM_CONFIG", raising=False)
    monkeypatch.setattr(model_catalog, "CATALOG_FALLBACKS", ())

    with pytest.raises(CatalogError, match="Cannot find a model-manager catalog"):
        catalog_path()


@allure.feature("pysweep")
# Explicit ids: these values are multi-line YAML, and the suite's conftest names
# per-test log files after the test id.
@pytest.mark.parametrize(
    "content",
    [
        "models: [",  # not YAML at all
        "- just\n- a\n- list\n",  # not a mapping at the top level
        "models:\n  llama3-8b-awq:\n    model_id: org/x\n",  # a mapping, not a list
        "defaults:\n  namespace: llm-inference\n",  # no models section
    ],
    ids=["not-yaml", "top-level-list", "models-as-mapping", "no-models-section"],
)
def test_a_malformed_catalog_is_reported_as_such(tmp_path, monkeypatch, content):
    """Every one of these would otherwise surface as 'model X is not in the catalog'."""
    path = tmp_path / "models.yaml"
    path.write_text(content)
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(path))

    with pytest.raises(CatalogError):
        load_catalog()


@allure.feature("pysweep")
def test_the_repo_default_catalog_is_readable():
    """The shipped ``deployment/models.yaml`` must parse with this reader.

    It is the fallback for anyone running pysweep from a checkout, and it is also
    the check that the fallback path (``paths.REPO_ROOT``) still points at the repo
    root after any move of this package.
    """
    repo_default = model_catalog.CATALOG_FALLBACKS[-1]
    models = known_models(repo_default)
    assert models, "the shipped catalog declares no models"
    for name in models:
        assert tokenizer_for(name, repo_default)
