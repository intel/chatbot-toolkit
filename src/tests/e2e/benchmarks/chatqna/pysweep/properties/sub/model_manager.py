# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Thin wrapper over the ``model-manager`` CLI (the model-serving half of a swap).

**Why delegate instead of writing the CR.** Serving a model is not "set a field":
it means downloading weights onto the ``model-store`` PVC, sizing CPU/memory for
the engine, checking the NRI balloons policy has NUMA capacity for the request,
rendering an ``LLMInferenceService``, and registering a route on the AI gateway.
``model-manager`` is the deployment team's tool for exactly that, it is
idempotent by construction (it applies via ``kubectl diff --server-side`` and
does nothing when the live object already matches), and it is actively
maintained. A pysweep that rendered its own CR would be a second, worse
implementation of all of the above that silently rots every time the real one
changes. So pysweep owns only the *decision* to serve a model; the CLI owns how.

This module is deliberately the whole extent of that coupling: one place that
knows the executable's location and its argument spelling.

**``--dry-run`` is what makes the preflight possible.** It renders and validates
— including the NUMA capacity gate — without touching the cluster, so a sweep can
check up front that every model it intends to visit *can* be deployed, instead of
discovering at combination 7 of 12 that the box has no room for the next one.

Locating the CLI: ``PYSWEEP_MODEL_MANAGER`` wins, otherwise the ai-solutions
wrapper, otherwise whatever is on ``PATH``. Prefer the wrapper over the real
script: it exports ``MM_CONFIG``/``MM_GLOBAL_CONFIG`` for the installed
environment, and the bare script without them falls back to its bundled defaults
and deploys from the wrong catalog.
"""

import logging
import os
import shutil
import subprocess  # nosec B404 - fixed argv, no shell; see _run()
from pathlib import Path

from ...paths import deploy_roots, is_file

logger = logging.getLogger(__name__)

MODEL_MANAGER_ENV_VAR = "PYSWEEP_MODEL_MANAGER"

# Preferred locations, in order. The ai-solutions wrapper comes first because it
# is the one that points the CLI at this environment's catalog.
MODEL_MANAGER_FALLBACKS = tuple(
    str(path)
    for root in deploy_roots()
    for path in (
        root / "model-manager",
        root / "ext" / "enterprise.ai-inference" / "model_manager" / "model-manager",
    )
)

# A cold deploy downloads weights (tens of GB over a proxy) and then waits for
# the engine to load them, so the ceiling has to be generous. It is a ceiling,
# not an expectation: --wait returns as soon as the model reports Ready.
DEPLOY_TIMEOUT = 3 * 60 * 60.0
# A dry run only renders and validates, so anything slow here is a hung kubectl.
DRY_RUN_TIMEOUT = 300.0


class ModelManagerError(Exception):
    """Raised when the CLI is missing, or a deploy/dry-run fails."""


def model_manager_path() -> Path:
    """Return the ``model-manager`` executable to use, or raise.

    Raises :class:`ModelManagerError` rather than falling back to patching the
    cluster directly: if the tool that owns model serving is not available, the
    honest outcome is that pysweep cannot sweep models here.
    """
    configured = os.getenv(MODEL_MANAGER_ENV_VAR)
    if configured:
        path = Path(configured)
        if not (is_file(path) and os.access(path, os.X_OK)):
            raise ModelManagerError(
                f"{MODEL_MANAGER_ENV_VAR}={configured!r} is not an executable file."
            )
        return path

    for candidate in MODEL_MANAGER_FALLBACKS:
        path = Path(candidate)
        if is_file(path) and os.access(path, os.X_OK):
            return path

    found = shutil.which("model-manager")
    if found:
        return Path(found)

    raise ModelManagerError(
        "Cannot find the 'model-manager' CLI. Set "
        f"{MODEL_MANAGER_ENV_VAR} to its path (prefer the ai-solutions wrapper, "
        "which points it at this environment's model catalog); tried "
        f"{list(MODEL_MANAGER_FALLBACKS)} and $PATH."
    )


def _run(argv: list[str], timeout: float) -> subprocess.CompletedProcess:
    """Run the CLI, capturing output; raise :class:`ModelManagerError` on failure.

    No shell: ``argv`` is a fixed list, so a model name can never be interpreted
    as a command. Output is captured and included in the error, because a
    model-manager failure (no NUMA capacity, HF unreachable, unknown model) is
    only actionable if the reason is in the message.
    """
    logger.debug("Running: %s", " ".join(argv))
    try:
        completed = subprocess.run(  # nosec B603 - fixed argv, shell=False
            argv,
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise ModelManagerError(
            f"'{' '.join(argv)}' did not finish within {timeout:.0f}s."
        ) from exc
    except OSError as exc:
        raise ModelManagerError(f"Cannot execute '{argv[0]}': {exc}") from exc

    if completed.returncode != 0:
        raise ModelManagerError(
            f"'{' '.join(argv)}' failed with exit status {completed.returncode}.\n"
            f"--- stdout ---\n{completed.stdout}\n--- stderr ---\n{completed.stderr}"
        )
    return completed


def deploy(
    model: str,
    wait: bool = True,
    replicas: int | None = None,
    timeout: float = DEPLOY_TIMEOUT,
) -> str:
    """Deploy (or reconcile) ``model`` from the catalog; return the CLI's stdout.

    Idempotent because the CLI is: it applies with ``kubectl diff --server-side``
    and skips the apply when the live ``LLMInferenceService`` already matches, so
    calling this for a model that is already serving does not restart it.

    Note the flip side, which callers should be aware of: it reconciles toward the
    *catalog*, so if the live deployment was hand-tuned away from catalog values
    (CPU, memory, engine version) this will pull it back and restart the model
    pods. That is the correct behaviour for a sweep — a benchmark should measure a
    reproducible configuration — but it does mean a sweep is not read-only with
    respect to models it visits.

    ``replicas`` is the one such value the caller is expected to pass rather than
    let the catalog decide, because it is the one that would invalidate the
    comparison the sweep exists to make: models benchmarked at different replica
    counts are not comparable, and the catalog's default is 1 regardless of what
    the cluster was scaled to. See :meth:`~..llm_model.LlmModel.apply`, which
    carries the outgoing model's count over to the incoming one.

    With ``wait`` the CLI blocks until the model reports Ready, so the caller can
    treat a clean return as "the model is being served". ``--wait-timeout`` is set
    from ``timeout`` because the CLI's own default (900s) is short for a cold
    weights download; the subprocess ceiling is left a minute higher so the CLI
    reports the timeout itself, with its diagnostics, instead of being killed.
    """
    argv = [str(model_manager_path()), "deploy", model]
    if replicas is not None:
        argv += ["--replicas", str(int(replicas))]
    if wait:
        argv += ["--wait", "--wait-timeout", str(int(timeout))]
    logger.info(
        "Deploying model %r via model-manager%s%s.",
        model,
        f" (replicas={replicas})" if replicas is not None else "",
        " (waiting for Ready)" if wait else "",
    )
    return _run(argv, timeout=timeout + 60.0).stdout


def dry_run(model: str, timeout: float = DRY_RUN_TIMEOUT) -> str:
    """Validate that ``model`` *could* be deployed, without touching the cluster.

    Side-effect free — it renders the manifest and runs model-manager's own
    preflight (catalog lookup, NUMA/balloons capacity for the requested CPU) and
    stops there. Used by the sweep preflight so an impossible model fails before
    the first benchmark rather than partway through the product.
    """
    argv = [str(model_manager_path()), "deploy", model, "--dry-run"]
    logger.debug("Dry-running model-manager deploy for %r.", model)
    return _run(argv, timeout=timeout).stdout
