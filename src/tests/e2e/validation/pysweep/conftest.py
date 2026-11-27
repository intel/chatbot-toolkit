# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import pytest

from tests.e2e.benchmarks.chatqna.pysweep.cluster.auth import (
    AccessError,
    KubernetesAccess,
)
from tests.e2e.benchmarks.chatqna.pysweep.cluster.fingerprint import (
    HelperFingerprintClient,
    PortForwardTransport,
    FingerprintClient,
)
from tests.e2e.validation.buildcfg import cfg

logger = logging.getLogger(__name__)

# 'llm' is present in every ChatQNA pipeline.
PROBE_COMPONENT = "llm"


@pytest.fixture(scope="session")
def pysweep_kubernetes_access(k8s_helper):
    """Skip unless the kubeconfig identity has the RBAC a sweep needs."""
    access = KubernetesAccess(k8s_helper=k8s_helper)
    try:
        access.verify()
    except AccessError as exc:
        pytest.skip(f"Kubernetes access prerequisites not met: {exc}")
    return access


@pytest.fixture(scope="session")
def fingerprint_client(fingerprint_api_helper, pysweep_kubernetes_access):
    """Prefer the ingress helper, fall back to a port-forward, skip if neither answers."""
    if not cfg.get("fingerprint_enabled"):
        pytest.skip("Fingerprint is not deployed")

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
