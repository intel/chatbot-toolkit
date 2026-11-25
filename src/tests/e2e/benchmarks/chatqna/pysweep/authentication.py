# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Authorization & access for pysweep.

Three concerns, each the smallest thing that lets pysweep run outside pytest:

- :class:`KubernetesAccess` — verifies, **up front and without mutating the
  cluster**, that the current kubeconfig identity can perform every operation a
  sweep needs. It uses ``SelfSubjectAccessReview``, which asks the API server
  "am I allowed to do X?" without doing X, so a missing permission fails loudly
  before the first rollout rather than halfway through a multi-hour product.

- :func:`load_build_config` — populates
  :data:`tests.e2e.validation.buildcfg.cfg` from the installed environment's
  YAML. The maintained helpers (``KeycloakHelper``, ``FingerprintApiHelper``)
  read the domain and routing mode from that dict, and under pytest it is
  ``conftest.pytest_configure`` that fills it. pysweep has no pytest, so it does
  the same load itself — deliberately reading the same three files in the same
  order, so a sweep and the e2e suite cannot end up disagreeing about which
  cluster they are talking to.

- :func:`keycloak_helper` — builds the maintained
  :class:`~tests.e2e.helpers.keycloak_helper.KeycloakHelper`. pysweep does
  **not** implement token minting: that helper already does the password grant,
  the required-actions dance, and the credentials-file lookup, and it is kept in
  step with the deployment. Duplicating it here is how the two would drift.

- :func:`client_token_lifespan` — for the duration of a sweep, lengthens the
  access tokens Keycloak issues to the ERAG UI client (see
  :data:`SWEEP_TOKEN_LIFESPAN_S`), then restores the client's previous setting.
"""

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

from .paths import deploy_roots, is_dir, is_file

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# The installed environment's config directory, mirroring the e2e suite's
# --build-config-dir default. Overridable so a sweep can be run against a
# different environment without editing anything.
BUILD_CONFIG_DIR_ENV_VAR = "PYSWEEP_BUILD_CONFIG_DIR"
BUILD_CONFIG_FALLBACKS = tuple(str(root / "env" / "local") for root in deploy_roots())
# Same files, same order as tests/e2e/conftest.py — later files win on conflict.
BUILD_CONFIG_FILES = ("global_config.yaml", "config.erag.yaml", "config.inference.yaml")


# Access-token lifespan pysweep gives the ERAG UI client while a sweep runs.
#
# ARBITRARY VALUE: 1h has no meaning beyond being comfortably longer than one load
# test. The tokens are re-minted before every run, so they only have to outlive a
# single run (client.duration plus warmup), not the whole sweep; the realm default
# (5 min) does not even cover one 15m run, and each benchmark worker reads its
# token once, when the run starts. Raise it if a run's duration gets near it.
SWEEP_TOKEN_LIFESPAN_S = 3600
# Keycloak's per-client override of the realm's access-token lifespan. An empty
# value means "inherit the realm setting", which is what an unset client has.
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
    """Outcome of a single access check."""

    required: RequiredAccess
    allowed: bool
    reason: str = ""


# The full set of permissions pysweep requires. Keeping this explicit means the
# access preflight names precisely which permission is missing, and reading the
# list is the fastest way to see everything a sweep can touch.
REQUIRED_ACCESS = [
    # The served model, read back to pick the tokenizer (served_model.py).
    RequiredAccess("list", "pods", CHATQNA_NAMESPACE, reason="locate the llm-usvc replicas"),
    RequiredAccess("create", "pods/exec", CHATQNA_NAMESPACE, reason="read the pods' effective LLM_MODEL_NAME"),
    RequiredAccess(
        "get", "llminferenceservices", LLM_INFERENCE_NAMESPACE,
        group="serving.kserve.io", reason="observed model readiness",
    ),
    # Runtime-API plane, reached by port-forwarding to fingerprint-svc.
    RequiredAccess("list", "pods", FINGERPRINT_NAMESPACE, reason="locate fingerprint-svc endpoints"),
    RequiredAccess(
        "create", "pods/portforward", FINGERPRINT_NAMESPACE,
        reason="port-forward the fingerprint API",
    ),
    # Keycloak admin password: token minting and the sweep-long token lifespan.
    RequiredAccess("get", "secrets", KEYCLOAK_NAMESPACE, reason="Keycloak admin password for token minting"),
]


class _SelfSubjectAccessReview(APIObject):
    """Minimal kr8s binding for authorization.k8s.io SelfSubjectAccessReview.

    Creating one is a read-only authorization probe: it asks the API server
    "am I allowed to do X?" without performing X, so the access preflight never
    mutates the cluster.
    """

    kind = "SelfSubjectAccessReview"
    version = "authorization.k8s.io/v1"
    endpoint = "selfsubjectaccessreviews"
    plural = "selfsubjectaccessreviews"
    singular = "selfsubjectaccessreview"
    namespaced = False


class KubernetesAccess:
    """Verifies the cluster is reachable and the identity has the needed RBAC."""

    def __init__(self, k8s_helper: K8sHelper | None = None):
        # Reuse the maintained k8s helper for all ordinary cluster access.
        self.k8s = k8s_helper or K8sHelper()

    def ensure_reachable(self) -> None:
        """Confirm the configured kubeconfig can actually reach the API server."""
        try:
            # Cheapest authenticated call that proves connectivity + credentials.
            api = kr8s.api()
            api.version()
        except Exception as exc:  # noqa: BLE001 - surfaced as a clear AccessError
            raise AccessError(
                "Cannot reach the Kubernetes API server. Check KUBECONFIG / the "
                "configured context points at a running cluster.\n"
                f"Underlying error: {exc}"
            ) from exc

    def check(self, required: RequiredAccess) -> AccessCheckResult:
        """Run a single SelfSubjectAccessReview for one required permission.

        ``pods/exec`` and ``pods/portforward`` are subresources; the API expects
        them split into resource + subresource rather than as one slashed name.
        """
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
        # APIObject.status raises KeyError if the server returned no status;
        # read from raw defensively.
        status = dict(review.raw.get("status", {}) or {})
        allowed = bool(status.get("allowed", False))
        reason = status.get("reason", "") or ("denied" if not allowed else "")
        return AccessCheckResult(required=required, allowed=allowed, reason=reason)

    def verify(self, raise_on_denied: bool = True) -> list[AccessCheckResult]:
        """Verify every required permission up front.

        Returns the per-permission results. With ``raise_on_denied`` (default)
        it raises :class:`AccessError` listing exactly which permissions are
        missing, so the sweep fails before any combination runs.
        """
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
    """Return the installed environment's config directory, or raise."""
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
    """Load the deployment's config into the shared ``cfg`` dict and return it.

    Mutating the module-level ``cfg`` rather than returning a private copy is not
    incidental — it is how the maintained helpers get configured, since they read
    that dict at construction time. Under pytest the same dict is filled by
    ``conftest.pytest_configure``; calling this when it is already populated is
    harmless and idempotent.

    Raises :class:`BuildConfigError` if none of the expected files is present, or
    if the loaded config carries no ``base_domain_name`` — without that the
    helpers would silently build URLs against ``https://None`` and every request
    would fail with a DNS error instead of a readable message.
    """
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
    """Build the maintained :class:`KeycloakHelper` for this deployment.

    ``credentials_file`` defaults to the helper's own
    ``default_credentials.txt`` lookup, which is where the installer writes the
    ERAG users' passwords — so on an installed box no argument is needed.

    :func:`load_build_config` is called first because the helper reads the domain
    and routing mode from ``cfg`` in its constructor; building it against an
    empty ``cfg`` would bake in an unusable auth URL.
    """
    load_build_config()
    return KeycloakHelper(credentials_file, k8s_helper or K8sHelper())


def _client_url(helper: KeycloakHelper) -> str:
    """Admin URL of the ERAG UI client (the one user tokens are minted for)."""
    base = f"{helper.erag_auth_domain}/admin/realms/{VITE_KEYCLOAK_REALM}/clients"
    response = requests.get(
        base,
        params={"clientId": VITE_KEYCLOAK_CLIENT_ID},
        headers={"Authorization": f"Bearer {helper.admin_access_token}"},
        verify=False,  # self-signed deployment certs, as in KeycloakHelper
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
    """Set one client attribute; return its previous value (``""`` if unset).

    The client is read back and PUT whole, because Keycloak's client PUT takes a
    full representation. An admin token is fetched per call: the restore runs
    after a sweep that lasts hours, long past any token fetched at the start.
    """
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
    """Make the ERAG UI client's access tokens last ``seconds`` inside the block.

    A per-client override, so nothing else in the realm changes. The previous
    value is restored on exit, including when the sweep fails; ``""`` restores
    "inherit from the realm". Restoring is best effort: a failure is logged, not
    raised, so it cannot hide the error that ended the sweep.
    """
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
