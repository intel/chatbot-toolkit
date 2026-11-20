# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``llm-usvc.yaml`` model-name rewrite.

Every test here runs against ``data/llm-usvc.yaml.deployed`` — a verbatim copy of
the manifest read out of ``gmc-config`` on a real deployment
(``kubectl get cm gmc-config -n system -o go-template='{{index .data
"llm-usvc.yaml"}}'``). That is the point of the fixture: the rewrite is a text
edit, so its safety is a claim about *this file's* shape, not about YAML in
general. In particular the deployed manifest contains ``LLM_MODEL_NAME`` in two
roles — the ConfigMap entry to rewrite, and four shell references inside the init
container's probe script — and a regex that could not tell them apart would
corrupt the readiness gate of every LLM pod.

The functions are pure string transformations, so no cluster and no kr8s are
involved. Their live counterpart (that ``gmc-config`` really carries a manifest of
this shape) is covered by ``test_llm_model.py``.
"""

from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.sub.llm_usvc_manifest import (
    ManifestError,
    read_model,
    rewrite_for_model,
)

DEPLOYED_MANIFEST = Path(__file__).resolve().parent / "data" / "llm-usvc.yaml.deployed"

# The model the captured deployment declares.
DEPLOYED_MODEL = "llama3-8b-awq"


@pytest.fixture
def manifest() -> str:
    return DEPLOYED_MANIFEST.read_text()


@allure.feature("pysweep")
def test_reads_the_declared_model_from_the_deployed_manifest(manifest):
    assert read_model(manifest) == DEPLOYED_MODEL


@allure.feature("pysweep")
def test_rewrite_changes_the_declared_model(manifest):
    rewritten = rewrite_for_model(manifest, "qwen3-5-2b-ovms")
    assert read_model(rewritten) == "qwen3-5-2b-ovms"


@allure.feature("pysweep")
def test_rewrite_changes_exactly_one_line(manifest):
    """A one-field change must stay a one-line diff.

    The operator diffs this manifest, and a reviewer reads that diff; a rewrite
    that also reflowed comments or block scalars would make a model swap
    indistinguishable from an unrelated chart change.
    """
    rewritten = rewrite_for_model(manifest, "qwen3-5-2b-ovms")
    before = manifest.splitlines()
    after = rewritten.splitlines()

    assert len(before) == len(after)
    differing = [i for i, (b, a) in enumerate(zip(before, after)) if b != a]
    assert len(differing) == 1
    assert after[differing[0]].strip() == 'LLM_MODEL_NAME: "qwen3-5-2b-ovms"'


@allure.feature("pysweep")
def test_rewrite_preserves_the_shell_references(manifest):
    """``$LLM_MODEL_NAME`` in the init container's probe script must not be touched.

    Those references are how the init container waits for the model to actually
    answer; substituting a literal into them would break readiness gating for
    every llm-usvc pod. They are the reason the anchor requires a line start,
    indentation and a colon.
    """
    rewritten = rewrite_for_model(manifest, "qwen3-5-2b-ovms")

    for reference in ("$LLM_MODEL_NAME", "${LLM_MODEL_NAME}"):
        assert manifest.count(reference) == rewritten.count(reference)
        assert manifest.count(reference) > 0, "fixture no longer exercises this case"


@allure.feature("pysweep")
def test_rewrite_preserves_indentation(manifest):
    rewritten = rewrite_for_model(manifest, "qwen3-5-2b-ovms")
    assert '\n  LLM_MODEL_NAME: "qwen3-5-2b-ovms"\n' in rewritten


@allure.feature("pysweep")
def test_rewrite_is_byte_identical_when_already_correct(manifest):
    """Idempotence, and the basis of skipping the write.

    ``LlmModel.apply`` decides whether to patch the ConfigMap by comparing the
    rewritten manifest with the current one. If a no-op rewrite differed by even a
    byte, every combination would patch gmc-config and trigger a needless rollout.
    """
    assert rewrite_for_model(manifest, DEPLOYED_MODEL) == manifest


@allure.feature("pysweep")
def test_rewrite_quotes_the_value(manifest):
    """A model name YAML would read as a number or bool must stay a string."""
    rewritten = rewrite_for_model(manifest, "1.5")
    assert '\n  LLM_MODEL_NAME: "1.5"\n' in rewritten


@allure.feature("pysweep")
def test_rewrite_accepts_an_unquoted_declared_value():
    """The chart quotes the value today, but an unquoted one is still valid YAML."""
    manifest = "data:\n  LLM_MODEL_NAME: llama3-8b-awq\n  LLM_MODEL_SERVER: vllm\n"
    assert read_model(manifest) == "llama3-8b-awq"
    assert rewrite_for_model(manifest, "other") == (
        'data:\n  LLM_MODEL_NAME: "other"\n  LLM_MODEL_SERVER: vllm\n'
    )


@allure.feature("pysweep")
def test_missing_entry_raises_rather_than_appending():
    """If the chart stops declaring the model this way, pysweep must not guess.

    Inventing the key would produce a manifest the operator accepts and the
    pipeline ignores — a sweep that reports numbers for a configuration it never
    established.
    """
    with pytest.raises(ManifestError, match="No 'LLM_MODEL_NAME"):
        read_model("data:\n  LLM_MODEL_SERVER: vllm\n")


@allure.feature("pysweep")
def test_ambiguous_entry_raises_and_names_the_lines():
    """Two candidate entries mean the anchor can no longer identify the right one."""
    manifest = (
        "data:\n"
        '  LLM_MODEL_NAME: "a"\n'
        "---\n"
        "data:\n"
        '  LLM_MODEL_NAME: "b"\n'
    )
    with pytest.raises(ManifestError, match=r"expected exactly one"):
        rewrite_for_model(manifest, "c")

    # The message has to point at the lines, or the next person cannot tell which
    # two entries collided.
    with pytest.raises(ManifestError, match=r"lines \[2, 5\]"):
        read_model(manifest)


@allure.feature("pysweep")
# Explicit ids: the values contain quotes and newlines, and the suite's conftest
# names per-test log files after the test id.
@pytest.mark.parametrize(
    "bad",
    ["", "   ", 'has"quote', "has\nnewline"],
    ids=["empty", "blank", "embedded-quote", "embedded-newline"],
)
def test_unwritable_model_names_are_refused(manifest, bad):
    """Values that cannot be a one-line YAML scalar are rejected before the write."""
    with pytest.raises(ManifestError):
        rewrite_for_model(manifest, bad)
