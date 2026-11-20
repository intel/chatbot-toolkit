# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the ``model-manager`` CLI wrapper.

The wrapper is the entire coupling between pysweep and the deployment team's tool,
so what is worth pinning is the *argv* it builds and the *safety* of how it runs:

* the argument spelling (``deploy <name> --replicas N --wait --wait-timeout S``,
  ``--dry-run``), because a flag that the CLI silently ignores turns a model swap
  into a benchmark of the previous model;
* that ``--replicas`` is passed on, since a model deployed at the catalog default
  of 1 is not comparable with one benchmarked at the cluster's actual count;
* that ``--dry-run`` never carries ``--wait``, i.e. the preflight cannot block for
  hours;
* that nothing is run through a shell, so a model name from a YAML file cannot
  become a command;
* that a non-zero exit is raised with the CLI's own output, which is where the
  actionable reason (no NUMA capacity, HuggingFace unreachable) lives.

The real CLI is never executed here: ``subprocess.run`` is replaced. That the CLI
exists and accepts these arguments on a deployment is covered by
``test_llm_model.py``'s dry-run test.
"""

import subprocess

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.properties.sub import model_manager
from tests.e2e.benchmarks.chatqna.pysweep.properties.sub.model_manager import (
    ModelManagerError,
    deploy,
    dry_run,
    model_manager_path,
)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """An executable stand-in for the CLI, selected via ``PYSWEEP_MODEL_MANAGER``."""
    path = tmp_path / "model-manager"
    path.write_text("#!/bin/sh\nexit 0\n")
    path.chmod(0o755)
    monkeypatch.setenv("PYSWEEP_MODEL_MANAGER", str(path))
    return path


class RecordingRun:
    """Stand-in for ``subprocess.run`` that records the call and fakes a result."""

    def __init__(self, returncode=0, stdout="ok", stderr=""):
        self.calls = []
        self.kwargs = []
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr

    def __call__(self, argv, **kwargs):
        self.calls.append(argv)
        self.kwargs.append(kwargs)
        return subprocess.CompletedProcess(argv, self.returncode, self.stdout, self.stderr)


@pytest.fixture
def run(monkeypatch):
    recorder = RecordingRun()
    monkeypatch.setattr(model_manager.subprocess, "run", recorder)
    return recorder


@allure.feature("pysweep")
def test_deploy_builds_the_documented_argv(cli, run):
    deploy("llama3-8b-awq", timeout=1800)
    assert run.calls == [
        [str(cli), "deploy", "llama3-8b-awq", "--wait", "--wait-timeout", "1800"]
    ]


@allure.feature("pysweep")
def test_deploy_passes_the_replica_count(cli, run):
    """A model at the catalog default of 1 is not comparable with one at 4."""
    deploy("llama3-8b-awq", replicas=4, timeout=1800)
    assert run.calls[0][:5] == [str(cli), "deploy", "llama3-8b-awq", "--replicas", "4"]


@allure.feature("pysweep")
def test_deploy_omits_replicas_when_not_given(cli, run):
    """Omitted means "whatever the catalog says", not "1"."""
    deploy("llama3-8b-awq")
    assert "--replicas" not in run.calls[0]


@allure.feature("pysweep")
def test_deploy_without_wait_does_not_block(cli, run):
    deploy("llama3-8b-awq", wait=False)
    assert run.calls[0] == [str(cli), "deploy", "llama3-8b-awq"]


@allure.feature("pysweep")
def test_the_subprocess_ceiling_is_above_the_cli_wait_timeout(cli, run):
    """So the CLI reports its own timeout, with diagnostics, instead of being killed."""
    deploy("llama3-8b-awq", timeout=1800)
    assert run.kwargs[0]["timeout"] > 1800


@allure.feature("pysweep")
def test_dry_run_is_side_effect_free_and_never_waits(cli, run):
    """The preflight runs this per declared model, so it must not be able to block."""
    dry_run("llama3-8b-awq")
    assert run.calls == [[str(cli), "deploy", "llama3-8b-awq", "--dry-run"]]
    assert "--wait" not in run.calls[0]


@allure.feature("pysweep")
def test_nothing_is_run_through_a_shell(cli, run):
    """A model name comes from a YAML file; via a shell it could be a command."""
    deploy("llama3-8b-awq; rm -rf /", wait=False)
    argv = run.calls[0]
    assert argv[-1] == "llama3-8b-awq; rm -rf /"
    assert not run.kwargs[0].get("shell", False)


@allure.feature("pysweep")
def test_a_failed_deploy_raises_with_the_cli_output(cli, monkeypatch):
    """The reason is only actionable if the CLI's own message survives."""
    monkeypatch.setattr(
        model_manager.subprocess,
        "run",
        RecordingRun(returncode=1, stdout="rendering...", stderr="no NUMA capacity"),
    )
    with pytest.raises(ModelManagerError) as excinfo:
        deploy("llama3-8b-awq", wait=False)

    message = str(excinfo.value)
    assert "no NUMA capacity" in message
    assert "rendering..." in message


@allure.feature("pysweep")
def test_a_timeout_is_reported_as_such(cli, monkeypatch):
    def timeout(argv, **kwargs):
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))

    monkeypatch.setattr(model_manager.subprocess, "run", timeout)
    with pytest.raises(ModelManagerError, match="did not finish within"):
        dry_run("llama3-8b-awq")


@allure.feature("pysweep")
def test_the_configured_path_must_be_executable(tmp_path, monkeypatch):
    """A path that is not runnable is a configuration error, not a missing CLI."""
    not_executable = tmp_path / "model-manager"
    not_executable.write_text("#!/bin/sh\n")
    not_executable.chmod(0o644)
    monkeypatch.setenv("PYSWEEP_MODEL_MANAGER", str(not_executable))

    with pytest.raises(ModelManagerError, match="not an executable file"):
        model_manager_path()


@allure.feature("pysweep")
def test_a_missing_cli_is_an_error_not_a_fallback(monkeypatch):
    """pysweep must not fall back to patching the cluster itself.

    If the tool that owns model serving is unavailable, the honest outcome is that
    models cannot be swept here — not a second, unmaintained implementation of the
    deploy.
    """
    monkeypatch.delenv("PYSWEEP_MODEL_MANAGER", raising=False)
    monkeypatch.setattr(model_manager, "MODEL_MANAGER_FALLBACKS", ())
    monkeypatch.setattr(model_manager.shutil, "which", lambda _: None)

    with pytest.raises(ModelManagerError, match="Cannot find the 'model-manager' CLI"):
        model_manager_path()


@allure.feature("pysweep")
def test_the_wrapper_is_preferred_over_the_bare_script(tmp_path, monkeypatch):
    """The wrapper exports MM_CONFIG; the bare script would deploy from the wrong catalog."""
    wrapper = tmp_path / "wrapper"
    bare = tmp_path / "bare"
    for path in (wrapper, bare):
        path.write_text("#!/bin/sh\nexit 0\n")
        path.chmod(0o755)

    monkeypatch.delenv("PYSWEEP_MODEL_MANAGER", raising=False)
    monkeypatch.setattr(model_manager, "MODEL_MANAGER_FALLBACKS", (str(wrapper), str(bare)))

    assert model_manager_path() == wrapper
