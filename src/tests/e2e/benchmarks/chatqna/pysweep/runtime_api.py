# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Client for the ChatQNA ``system_fingerprint`` runtime API (Runtime API plane).

Wraps the two endpoints the runtime-plane properties use:

- ``POST change_arguments`` — the **setter**. A list of
  ``{"name": <component>, "data": {...}}`` entries.
- ``GET config?params_key=<component>`` — the **reader**. Returns that one
  component's live arguments under ``values``.

The reader is per-component, which is why every property names the component it
drives: asking for ``retriever`` gets the retriever's arguments and nothing else,
so a property can never accidentally read a same-named key belonging to another
component.

**Scope.** Writes carry a ``pipeline``/``tenant`` pair, because the microservice
stores arguments per scope and the ChatQNA router reads the ``chatqna``/``_global``
one — a write with no scope lands somewhere the pipeline under test does not
read. Reads take the same pair for symmetry.

Two transports are supported (DESIGN.md "Reaching pod APIs"):

- :class:`PortForwardTransport` (default) — forwards to ``fingerprint-svc`` via
  ``kr8s``. Needs no token, because authentication is enforced at the ingress
  gateway and this path is pod-local. That makes it the robust choice for a
  multi-hour sweep: nothing to expire.
- :class:`IngressTransport` — the ERAG domain, authenticated with a Keycloak
  access token. Used when the cluster is only reachable through its ingress.
"""

import logging
import os
import secrets
import socket
import time
from contextlib import contextmanager

import kr8s
import requests
import urllib3

from tests.e2e.validation.buildcfg import cfg

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

FINGERPRINT_NAMESPACE = "fingerprint"
FINGERPRINT_SERVICE = "fingerprint-svc"
FINGERPRINT_PORT = 6012
API_PREFIX = "/v1/system_fingerprint"

# The scope the ChatQNA router reads its component arguments from. '_global' is
# the shared tenant; see FingerprintApiHelper.set_chatqa_component_parameters,
# which fixes the same pair for the e2e suite's ChatQNA tests.
PIPELINE = "chatqna"
TENANT = "_global"


def find_unused_port(start: int = 10000, end: int = 60000) -> int:
    """Return a currently-free local TCP port in ``[start, end)``."""
    while True:
        port = secrets.randbelow(end - start) + start
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port


def wait_until_listening(port: int, timeout: float = 10.0) -> None:
    """Block until a local port accepts connections.

    ``PortForward.start()`` returns before the tunnel is actually listening, so
    probe the socket to avoid a connection-refused race.
    """
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"port-forward to 127.0.0.1:{port} did not start within {timeout}s")


@contextmanager
def object_port_forward(obj, remote_port: int):
    """Port-forward ``remote_port`` on any kr8s object; yield the local port.

    Works for a ``Service`` (load-balanced across its endpoints) or a ``Pod``
    (one specific replica). The forward is scoped to the context and torn down
    afterwards. This is the single code path for reaching in-cluster HTTP APIs
    pod-locally — shared by the fingerprint transport and the benchmark's ChatQNA
    endpoint.
    """
    local_port = find_unused_port()
    pf = kr8s.portforward.PortForward(obj, remote_port=remote_port, local_port=local_port)
    pf.start()
    try:
        wait_until_listening(local_port)
        yield local_port
    finally:
        pf.stop()


@contextmanager
def service_port_forward(namespace: str, service: str, remote_port: int):
    """Port-forward ``service:remote_port`` via kr8s; yield the local port."""
    svc = kr8s.objects.Service.get(service, namespace=namespace)
    with object_port_forward(svc, remote_port) as local_port:
        yield local_port


class IngressTransport:
    """Reach the fingerprint API over the ERAG ingress with a Keycloak token.

    The token is fetched per session rather than cached, because a sweep runs for
    hours and a cached one would expire mid-product. ``keycloak_helper`` is the
    maintained helper (see :mod:`.authentication`), whose ``access_token``
    property already re-mints on each read.
    """

    def __init__(self, keycloak_helper, domain: str | None = None):
        self.keycloak_helper = keycloak_helper
        self.domain = domain or cfg.get("base_domain_name")
        if not self.domain:
            raise ValueError(
                "No ERAG domain available for the ingress transport; call "
                "authentication.load_build_config() first or pass domain=."
            )

    @contextmanager
    def session(self):
        base_url = f"https://{self.domain}{API_PREFIX}"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.keycloak_helper.access_token}",
        }
        yield base_url, headers


class PortForwardTransport:
    """Reach the fingerprint API by port-forwarding to the service via kr8s.

    The forward is scoped to the ``session()`` context and torn down afterwards,
    so a sweep holds no tunnel open between property reads.
    """

    def __init__(
        self,
        namespace: str = FINGERPRINT_NAMESPACE,
        service: str = FINGERPRINT_SERVICE,
        remote_port: int = FINGERPRINT_PORT,
    ):
        self.namespace = namespace
        self.service = service
        self.remote_port = remote_port

    @contextmanager
    def session(self):
        with service_port_forward(self.namespace, self.service, self.remote_port) as local_port:
            # No auth needed pod-locally; the gateway enforces it on ingress.
            yield f"http://127.0.0.1:{local_port}{API_PREFIX}", {"Content-Type": "application/json"}


class HelperFingerprintClient:
    """Adapter exposing the pysweep client interface over the suite helper.

    When pysweep runs as part of the e2e validation suite the maintained
    :class:`tests.e2e.helpers.fingerprint_api_helper.FingerprintApiHelper`
    already knows how to reach the fingerprint API over ingress (with the suite's
    Keycloak auth). Reusing it keeps the e2e property tests exercising the same
    transport the rest of the suite does, rather than a pysweep-only one. This
    adapter just re-expresses it as the ``change_arguments`` / ``read_values``
    interface the property classes expect.
    """

    def __init__(self, fingerprint_api_helper):
        self._helper = fingerprint_api_helper

    def change_arguments(self, body: list[dict]) -> None:
        resp = self._helper.change_arguments(body, pipeline=PIPELINE, tenant=TENANT)
        if resp.status_code != 200:
            raise RuntimeError(
                f"change_arguments failed (status {resp.status_code}): {resp.text}"
            )

    def read_values(self, component: str) -> dict:
        resp = self._helper.read_config(component, pipeline=PIPELINE, tenant=TENANT)
        if resp.status_code != 200:
            raise RuntimeError(
                f"config?params_key={component} failed (status {resp.status_code}): {resp.text}"
            )
        return resp.json().get("values", {}) or {}


class FingerprintClient:
    """Thin client over ``/v1/system_fingerprint/{change_arguments,config}``.

    Delegates connectivity to a transport so the same logic works whether the
    API is reached over ingress or a port-forward.
    """

    def __init__(self, transport, pipeline: str = PIPELINE, tenant: str = TENANT):
        self.transport = transport
        self.pipeline = pipeline
        self.tenant = tenant

    @property
    def _scope(self) -> dict:
        return {"pipeline": self.pipeline, "tenant": self.tenant}

    def change_arguments(self, body: list[dict]) -> None:
        """POST a change_arguments body to the configured scope; raises on non-2xx."""
        with self.transport.session() as (base_url, headers):
            resp = requests.post(
                f"{base_url}/change_arguments",
                headers=headers,
                json=body,
                params=self._scope,
                verify=False,
                timeout=30,
            )
            resp.raise_for_status()

    def read_values(self, component: str) -> dict:
        """Return one component's live arguments (the response's ``values`` object)."""
        with self.transport.session() as (base_url, headers):
            resp = requests.get(
                f"{base_url}/config",
                headers=headers,
                params={"params_key": component, **self._scope},
                verify=False,
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("values", {}) or {}


def default_transport(keycloak_helper=None):
    """Pick a transport from the environment.

    ``PYSWEEP_FINGERPRINT_TRANSPORT=portforward`` (default) forwards to the
    service; ``=ingress`` uses the ERAG domain and requires a Keycloak helper.
    """
    mode = os.getenv("PYSWEEP_FINGERPRINT_TRANSPORT", "portforward").lower()
    if mode == "ingress":
        if keycloak_helper is None:
            from .authentication import keycloak_helper as build_keycloak_helper

            keycloak_helper = build_keycloak_helper()
        return IngressTransport(keycloak_helper)
    return PortForwardTransport()
