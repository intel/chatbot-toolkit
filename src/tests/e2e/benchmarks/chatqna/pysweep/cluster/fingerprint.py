# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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
from tests.e2e.validation.constants import FINGERPRINT_NAMESPACE

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

logger = logging.getLogger(__name__)

FINGERPRINT_SERVICE = "fingerprint-svc"
FINGERPRINT_PORT = 6012
API_PREFIX = "/v1/system_fingerprint"

# The scope the ChatQNA router reads component arguments from.
PIPELINE = "chatqna"
TENANT = "_global"


def find_unused_port(start: int = 10000, end: int = 60000) -> int:
    while True:
        port = secrets.randbelow(end - start) + start
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port


def wait_until_listening(port: int, timeout: float = 10.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.1)
    raise TimeoutError(f"port-forward to 127.0.0.1:{port} did not start within {timeout}s")


@contextmanager
def object_port_forward(obj, remote_port: int):
    """Port-forward remote_port on a kr8s Service or Pod; yield the local port."""
    local_port = find_unused_port()
    pf = kr8s.portforward.PortForward(obj, remote_port=remote_port, local_port=local_port)
    pf.start()
    try:
        # PortForward.start() returns before the tunnel is listening.
        wait_until_listening(local_port)
        yield local_port
    finally:
        pf.stop()


@contextmanager
def service_port_forward(namespace: str, service: str, remote_port: int):
    svc = kr8s.objects.Service.get(service, namespace=namespace)
    with object_port_forward(svc, remote_port) as local_port:
        yield local_port


class IngressTransport:
    """Reach the fingerprint API over the ERAG ingress with a Keycloak token."""

    def __init__(self, keycloak_helper, domain: str | None = None):
        self.keycloak_helper = keycloak_helper
        self.domain = domain or cfg.get("base_domain_name")
        if not self.domain:
            raise ValueError(
                "No ERAG domain available for the ingress transport; call "
                "auth.load_build_config() first or pass domain=."
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
    """Reach the fingerprint API by port-forwarding to the service via kr8s."""

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
    """FingerprintClient interface over the suite's FingerprintApiHelper."""

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
    """Client for /v1/system_fingerprint/{change_arguments,config}."""

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
    """Pick a transport from PYSWEEP_FINGERPRINT_TRANSPORT: portforward (default) or ingress."""
    mode = os.getenv("PYSWEEP_FINGERPRINT_TRANSPORT", "portforward").lower()
    if mode == "ingress":
        if keycloak_helper is None:
            from .auth import keycloak_helper as build_keycloak_helper

            keycloak_helper = build_keycloak_helper()
        return IngressTransport(keycloak_helper)
    return PortForwardTransport()
