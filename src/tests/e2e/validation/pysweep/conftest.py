# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Fixtures for pysweep's e2e tests, run *inside* the validation suite.

These tests live under ``validation`` (next to ``test_fingerprint.py`` et al.)
so they collect and report through the shared ``src/tests/e2e/conftest.py`` —
same invocation (``tox -e e2e``), same allure report, same session fixtures
(``k8s_helper``, ``keycloak_helper``, ``fingerprint_api_helper``, the
required-actions handling). The pysweep *package* itself lives at
``tests/e2e/benchmarks/chatqna/pysweep`` and is imported by absolute path.

Per DESIGN.md objective 6, these are the "stay in sync with the deployment"
contract for the property classes: they exercise ``read`` / ``apply`` /
``is_satisfied`` against the live deployment so a drift in the runtime-API
contract fails here rather than silently mis-configuring a benchmark.

Access is layered on the maintained suite helpers rather than duplicated:
- the fingerprint API is reached via the suite's ``fingerprint_api_helper``
  (ingress + Keycloak auth), falling back to a kr8s port-forward for
  deployments where the API has no ingress route;
- the pysweep-specific RBAC preflight (:class:`KubernetesAccess`) is layered on
  the suite ``k8s_helper`` to fail early and clearly if the kubeconfig identity
  lacks a permission a sweep will need.
"""

import logging

import pytest

from tests.e2e.benchmarks.chatqna.pysweep.authentication import (
    AccessError,
    KubernetesAccess,
)
from tests.e2e.benchmarks.chatqna.pysweep.runtime_api import (
    HelperFingerprintClient,
    PortForwardTransport,
    FingerprintClient,
)
from tests.e2e.validation.buildcfg import cfg

logger = logging.getLogger(__name__)

# Component asked for when probing whether the fingerprint API answers at all.
# Any deployed component would do; 'llm' is present in every ChatQNA pipeline.
PROBE_COMPONENT = "llm"


@pytest.fixture(scope="session")
def pysweep_kubernetes_access(k8s_helper):
    """Verify (without mutation) that pysweep has the cluster access it needs.

    Layered on the suite ``k8s_helper``. Skips if a required permission is
    missing — that is an environment/RBAC problem, not a property regression.
    """
    access = KubernetesAccess(k8s_helper=k8s_helper)
    try:
        access.verify()
    except AccessError as exc:
        pytest.skip(f"Kubernetes access prerequisites not met: {exc}")
    return access


@pytest.fixture(scope="session")
def fingerprint_client(fingerprint_api_helper, pysweep_kubernetes_access):
    """A pysweep FingerprintClient backed by the suite helper.

    Prefers the maintained ingress helper. If the fingerprint API is not routed
    on this deployment (no ingress route -> the probe fails), falls back to a
    kr8s port-forward to ``fingerprint-svc``. Skips if neither path reaches the
    API, so individual tests fail only on genuine property regressions.
    """
    if not cfg.get("fingerprint_enabled"):
        pytest.skip("Fingerprint is not deployed")

    # Reachability probe. The reader is per-component, so probing needs a
    # component name; 'llm' is the one every ChatQNA deployment has.
    helper_client = HelperFingerprintClient(fingerprint_api_helper)
    try:
        helper_client.read_values(PROBE_COMPONENT)
        return helper_client
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Fingerprint API not reachable via the suite helper (%s); "
            "falling back to a port-forward transport.",
            exc,
        )

    pf_client = FingerprintClient(PortForwardTransport())
    try:
        pf_client.read_values(PROBE_COMPONENT)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"system_fingerprint API not reachable (ingress or port-forward): {exc}")
    return pf_client
