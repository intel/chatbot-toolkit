# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import kr8s
import requests
import urllib3
import yaml
from kr8s.objects import APIObject

from tests.e2e.helpers.k8s_helper import K8sHelper
from tests.e2e.helpers.keycloak_helper import KeycloakHelper
from tests.e2e.validation.buildcfg import cfg
from tests.e2e.validation.constants import (
    CHATQNA_NAMESPACE,
    FINGERPRINT_NAMESPACE,
    KEYCLOAK_NAMESPACE,
    LLM_INFERENCE_NAMESPACE,
    VITE_KEYCLOAK_CLIENT_ID,
    VITE_KEYCLOAK_REALM,
)

from ..provenance.paths import deploy_roots, is_dir, is_file

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

BUILD_CONFIG_DIR_ENV_VAR = "PYSWEEP_BUILD_CONFIG_DIR"
BUILD_CONFIG_FALLBACKS = tuple(str(root / "env" / "local") for root in deploy_roots())
# Same files and order as tests/e2e/conftest.py; later files win.
BUILD_CONFIG_FILES = ("global_config.yaml", "config.erag.yaml", "config.inference.yaml")


# Arbitrary; only has to outlast one run.
SWEEP_TOKEN_LIFESPAN_S = 3600
# An empty value means "inherit the realm setting".
CLIENT_TOKEN_LIFESPAN_ATTRIBUTE = "access.token.lifespan"
KEYCLOAK_ADMIN_TIMEOUT = 30


class AccessError(Exception):
    """Raised when a required cluster permission is missing or the API is unreachable."""


class BuildConfigError(Exception):
    """Raised when the installed environment's build configuration cannot be loaded."""


class KeycloakConfigError(Exception):
    """Raised when the ERAG UI client cannot be found or reconfigured in Keycloak."""


@dataclass(frozen=True)
class RequiredAccess:
    """One RBAC permission pysweep needs, expressed for SelfSubjectAccessReview."""

    verb: str
    resource: str
    namespace: str
    group: str = ""
    reason: str = ""

    def describe(self) -> str:
        grp = f"{self.group}/" if self.group else ""
        return f"{self.verb} {grp}{self.resource} in ns/{self.namespace}"


@dataclass
class AccessCheckResult:
    required: RequiredAccess
    allowed: bool
    reason: str = ""


REQUIRED_ACCESS = [
    RequiredAccess("list", "pods", CHATQNA_NAMESPACE, reason="locate the llm-usvc replicas"),
    RequiredAccess("create", "pods/exec", CHATQNA_NAMESPACE, reason="read the pods' effective LLM_MODEL_NAME"),
    RequiredAccess(
        "get", "llminferenceservices", LLM_INFERENCE_NAMESPACE,
        group="serving.kserve.io", reason="observed model readiness",
    ),
    RequiredAccess("list", "pods", FINGERPRINT_NAMESPACE, reason="locate fingerprint-svc endpoints"),
    RequiredAccess(
        "create", "pods/portforward", FINGERPRINT_NAMESPACE,
        reason="port-forward the fingerprint API",
    ),
    RequiredAccess("get", "secrets", KEYCLOAK_NAMESPACE, reason="Keycloak admin password for token minting"),
]


class _SelfSubjectAccessReview(APIObject):
    kind = "SelfSubjectAccessReview"
    version = "authorization.k8s.io/v1"
    endpoint = "selfsubjectaccessreviews"
    plural = "selfsubjectaccessreviews"
    singular = "selfsubjectaccessreview"
    namespaced = False


class KubernetesAccess:
    """Verifies the cluster is reachable and the identity has the needed RBAC."""

    def __init__(self, k8s_helper: K8sHelper | None = None):
        self.k8s = k8s_helper or K8sHelper()

    def ensure_reachable(self) -> None:
        try:
            api = kr8s.api()
            api.version()
        except Exception as exc:  # noqa: BLE001
            raise AccessError(
                "Cannot reach the Kubernetes API server. Check KUBECONFIG / the "
                "configured context points at a running cluster.\n"
                f"Underlying error: {exc}"
            ) from exc

    def check(self, required: RequiredAccess) -> AccessCheckResult:
        # The API wants subresources (pods/exec) split out, not slash-joined.
        resource, _, subresource = required.resource.partition("/")
        attributes = {
            "namespace": required.namespace,
            "verb": required.verb,
            "group": required.group,
            "resource": resource,
        }
        if subresource:
            attributes["subresource"] = subresource

        review = _SelfSubjectAccessReview({"spec": {"resourceAttributes": attributes}})
        review.create()
        # APIObject.status raises KeyError when the server returned no status.
        status = dict(review.raw.get("status", {}) or {})
        allowed = bool(status.get("allowed", False))
        reason = status.get("reason", "") or ("denied" if not allowed else "")
        return AccessCheckResult(required=required, allowed=allowed, reason=reason)

    def verify(self, raise_on_denied: bool = True) -> list[AccessCheckResult]:
        """Check every required permission; raise AccessError naming the denied ones."""
        self.ensure_reachable()

        results = [self.check(req) for req in REQUIRED_ACCESS]
        denied = [r for r in results if not r.allowed]
        if denied and raise_on_denied:
            lines = "\n".join(
                f"  - {r.required.describe()}  ({r.required.reason})"
                + (f" [{r.reason}]" if r.reason else "")
                for r in denied
            )
            raise AccessError(
                "The current kubeconfig identity is missing permissions pysweep "
                f"requires:\n{lines}"
            )
        return results


def build_config_dir() -> Path:
    configured = os.getenv(BUILD_CONFIG_DIR_ENV_VAR)
    if configured:
        path = Path(configured)
        if not is_dir(path):
            raise BuildConfigError(
                f"{BUILD_CONFIG_DIR_ENV_VAR}={configured!r} is not a directory."
            )
        return path

    for candidate in BUILD_CONFIG_FALLBACKS:
        path = Path(candidate)
        if is_dir(path):
            return path

    raise BuildConfigError(
        "Cannot find the deployment's build configuration directory. Set "
        f"{BUILD_CONFIG_DIR_ENV_VAR} to the env/<name>/ directory holding "
        f"{list(BUILD_CONFIG_FILES)}; tried {list(BUILD_CONFIG_FALLBACKS)}."
    )


def load_build_config(directory: str | Path | None = None) -> dict:
    """Load the deployment's config into the shared ``cfg`` dict the helpers read, and return it."""
    resolved = Path(directory) if directory is not None else build_config_dir()

    loaded: list[str] = []
    for filename in BUILD_CONFIG_FILES:
        path = resolved / filename
        if not is_file(path):
            continue
        try:
            content = yaml.safe_load(path.read_text())
        except OSError as exc:
            raise BuildConfigError(f"Cannot read '{path}': {exc}") from exc
        except yaml.YAMLError as exc:
            raise BuildConfigError(f"'{path}' is not valid YAML: {exc}") from exc
        if content:
            cfg.update(content)
            loaded.append(filename)

    if not loaded:
        raise BuildConfigError(
            f"No build configuration found in '{resolved}' (looked for "
            f"{list(BUILD_CONFIG_FILES)})."
        )
    if not cfg.get("base_domain_name"):
        raise BuildConfigError(
            f"Build configuration in '{resolved}' has no 'base_domain_name'; the "
            "deployment's domain is needed to reach its APIs."
        )
    logger.debug("Loaded build configuration from %s: %s", resolved, loaded)
    return cfg


def keycloak_helper(
    credentials_file: str | None = None, k8s_helper: K8sHelper | None = None
) -> KeycloakHelper:
    """Build the KeycloakHelper, loading ``cfg`` first since its constructor reads it."""
    load_build_config()
    return KeycloakHelper(credentials_file, k8s_helper or K8sHelper())


def _client_url(helper: KeycloakHelper) -> str:
    base = f"{helper.erag_auth_domain}/admin/realms/{VITE_KEYCLOAK_REALM}/clients"
    response = requests.get(
        base,
        params={"clientId": VITE_KEYCLOAK_CLIENT_ID},
        headers={"Authorization": f"Bearer {helper.admin_access_token}"},
        verify=False,  # self-signed deployment certs
        timeout=KEYCLOAK_ADMIN_TIMEOUT,
    )
    response.raise_for_status()
    clients = response.json()
    if not clients:
        raise KeycloakConfigError(
            f"Client '{VITE_KEYCLOAK_CLIENT_ID}' not found in realm '{VITE_KEYCLOAK_REALM}'."
        )
    return f"{base}/{clients[0]['id']}"


def _set_client_attribute(helper: KeycloakHelper, url: str, name: str, value: str) -> str:
    # Token fetched per call: the restore runs hours after the sweep started.
    headers = {"Authorization": f"Bearer {helper.admin_access_token}"}
    response = requests.get(
        url, headers=headers, verify=False, timeout=KEYCLOAK_ADMIN_TIMEOUT
    )
    response.raise_for_status()
    client = response.json()
    attributes = client.setdefault("attributes", {})
    previous = attributes.get(name, "")
    attributes[name] = value
    response = requests.put(
        url, json=client, headers=headers, verify=False, timeout=KEYCLOAK_ADMIN_TIMEOUT
    )
    response.raise_for_status()
    return previous


@contextmanager
def client_token_lifespan(helper: KeycloakHelper, seconds: int = SWEEP_TOKEN_LIFESPAN_S):
    """Make the ERAG UI client's access tokens last ``seconds`` inside the block, then restore."""
    url = _client_url(helper)
    previous = _set_client_attribute(
        helper, url, CLIENT_TOKEN_LIFESPAN_ATTRIBUTE, str(seconds)
    )
    logger.info(
        "Set %s access-token lifespan to %ss for the sweep (was %s).",
        VITE_KEYCLOAK_CLIENT_ID, seconds, previous or "the realm default",
    )
    try:
        yield
    finally:
        try:
            _set_client_attribute(helper, url, CLIENT_TOKEN_LIFESPAN_ATTRIBUTE, previous)
            logger.info(
                "Restored %s access-token lifespan to %s.",
                VITE_KEYCLOAK_CLIENT_ID, previous or "the realm default",
            )
        except Exception:  # pylint: disable=broad-except
            logger.exception(
                "Could not restore %s's '%s' attribute to %r; reset it by hand.",
                VITE_KEYCLOAK_CLIENT_ID, CLIENT_TOKEN_LIFESPAN_ATTRIBUTE, previous,
            )
