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
"""

import logging
import os
from dataclasses import dataclass
from pathlib import Path

import kr8s
import urllib3
import yaml
from kr8s.objects import APIObject

from tests.e2e.helpers.k8s_helper import K8sHelper
from tests.e2e.helpers.keycloak_helper import KeycloakHelper
from tests.e2e.validation.buildcfg import cfg

from .paths import deploy_roots, is_dir, is_file

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

# Kubernetes namespaces pysweep touches (see DESIGN.md).
GMC_NAMESPACE = "system"
PIPELINE_NAMESPACE = "chatqna"
INFERENCE_NAMESPACE = "llm-inference"
FINGERPRINT_NAMESPACE = "fingerprint"
KEYCLOAK_NAMESPACE = "keycloak"

# The installed environment's config directory, mirroring the e2e suite's
# --build-config-dir default. Overridable so a sweep can be run against a
# different environment without editing anything.
BUILD_CONFIG_DIR_ENV_VAR = "PYSWEEP_BUILD_CONFIG_DIR"
BUILD_CONFIG_FALLBACKS = tuple(str(root / "env" / "local") for root in deploy_roots())
# Same files, same order as tests/e2e/conftest.py — later files win on conflict.
BUILD_CONFIG_FILES = ("global_config.yaml", "config.erag.yaml", "config.inference.yaml")


class AccessError(Exception):
    """Raised when a required cluster permission is missing or the API is unreachable."""


class BuildConfigError(Exception):
    """Raised when the installed environment's build configuration cannot be loaded."""


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
#
# The llm-inference entries are not pysweep's own writes — model-manager makes
# them — but model-manager runs as this same identity, so a sweep that could not
# make them would fail inside a subprocess with a kubectl error instead of here
# with a sentence.
REQUIRED_ACCESS = [
    # K8s-spec plane: the operator's desired state for the LLM microservice.
    RequiredAccess("get", "configmaps", GMC_NAMESPACE, reason="read gmc-config (declared model)"),
    RequiredAccess("patch", "configmaps", GMC_NAMESPACE, reason="patch gmc-config (repoint the pipeline)"),
    RequiredAccess("get", "deployments", GMC_NAMESPACE, group="apps", reason="check gmc-controller is running"),
    # The operator's rendered output, and the workload that consumes it.
    RequiredAccess("get", "configmaps", PIPELINE_NAMESPACE, reason="read llm-usvc-config (rendered model)"),
    RequiredAccess("get", "deployments", PIPELINE_NAMESPACE, group="apps", reason="observed workload state"),
    RequiredAccess("list", "pods", PIPELINE_NAMESPACE, reason="wait_until_ready / observed pod state"),
    RequiredAccess("create", "pods/exec", PIPELINE_NAMESPACE, reason="read the pods' effective LLM_MODEL_NAME"),
    RequiredAccess("delete", "pods", PIPELINE_NAMESPACE, reason="restart llm-usvc to adopt a new model"),
    # Model serving, driven through model-manager under this same identity.
    RequiredAccess(
        "get", "llminferenceservices", INFERENCE_NAMESPACE,
        group="serving.kserve.io", reason="observed model readiness",
    ),
    RequiredAccess(
        "patch", "llminferenceservices", INFERENCE_NAMESPACE,
        group="serving.kserve.io", reason="model-manager deploy (server-side apply)",
    ),
    # Runtime-API plane, reached by port-forwarding to fingerprint-svc.
    RequiredAccess("list", "pods", FINGERPRINT_NAMESPACE, reason="locate fingerprint-svc endpoints"),
    RequiredAccess(
        "create", "pods/portforward", FINGERPRINT_NAMESPACE,
        reason="port-forward the fingerprint API",
    ),
    # Only needed on the ingress transport, but cheap to check and a clearer
    # failure here than a 401 an hour in.
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
