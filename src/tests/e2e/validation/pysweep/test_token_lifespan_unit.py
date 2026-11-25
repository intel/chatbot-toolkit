# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sweep-long access-token lifespan override.

Pinned: the override is per client (nothing else in the realm changes), the
client's previous value comes back afterwards, including after a failed sweep, and
a client with no override goes back to inheriting the realm setting.
No Keycloak is involved: ``requests`` is replaced with an in-memory client store.
"""

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import authentication
from tests.e2e.benchmarks.chatqna.pysweep.authentication import (
    CLIENT_TOKEN_LIFESPAN_ATTRIBUTE,
    KeycloakConfigError,
    client_token_lifespan,
)
from tests.e2e.validation.constants import VITE_KEYCLOAK_CLIENT_ID, VITE_KEYCLOAK_REALM

DOMAIN = "https://keycloak.box.example"
CLIENTS_URL = f"{DOMAIN}/admin/realms/{VITE_KEYCLOAK_REALM}/clients"


class _Response:
    def __init__(self, payload=None, status=200):
        self._payload = payload
        self.status_code = status

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class _FakeKeycloak:
    """Just enough of the admin API: list clients by clientId, GET/PUT one client."""

    def __init__(self, attributes=None, exists=True):
        self.client = {"id": "uuid-1", "clientId": VITE_KEYCLOAK_CLIENT_ID,
                       "attributes": dict(attributes or {})}
        self.exists = exists
        self.puts = []
        self.fail_puts = False

    def get(self, url, params=None, **kwargs):
        if url == CLIENTS_URL:
            assert params == {"clientId": VITE_KEYCLOAK_CLIENT_ID}
            return _Response([dict(self.client)] if self.exists else [])
        assert url == f"{CLIENTS_URL}/uuid-1"
        return _Response({**self.client, "attributes": dict(self.client["attributes"])})

    def put(self, url, json=None, **kwargs):
        assert url == f"{CLIENTS_URL}/uuid-1"
        if self.fail_puts:
            return _Response(status=500)
        self.puts.append(json["attributes"].get(CLIENT_TOKEN_LIFESPAN_ATTRIBUTE))
        self.client = json
        return _Response(status=204)

    @property
    def lifespan(self):
        return self.client["attributes"].get(CLIENT_TOKEN_LIFESPAN_ATTRIBUTE)


class _Helper:
    erag_auth_domain = DOMAIN
    admin_access_token = "admin-token"


@pytest.fixture
def keycloak(monkeypatch):
    def install(**kwargs):
        fake = _FakeKeycloak(**kwargs)
        monkeypatch.setattr(authentication.requests, "get", fake.get)
        monkeypatch.setattr(authentication.requests, "put", fake.put)
        return fake

    return install


@allure.feature("pysweep")
def test_the_sweep_value_is_arbitrary_but_outlasts_a_long_run():
    """Only has to cover one run; 15m runs are the longest used so far."""
    assert authentication.SWEEP_TOKEN_LIFESPAN_S == 3600


@allure.feature("pysweep")
def test_an_unset_client_goes_back_to_inheriting_the_realm(keycloak):
    fake = keycloak()
    with client_token_lifespan(_Helper(), seconds=3600):
        assert fake.lifespan == "3600"
    # "" is Keycloak's "inherit": the client is as it was before the sweep.
    assert fake.lifespan == ""
    assert fake.puts == ["3600", ""]


@allure.feature("pysweep")
def test_a_client_override_is_put_back(keycloak):
    fake = keycloak(attributes={CLIENT_TOKEN_LIFESPAN_ATTRIBUTE: "900", "other": "kept"})
    with client_token_lifespan(_Helper(), seconds=3600):
        assert fake.lifespan == "3600"
    assert fake.lifespan == "900"
    assert fake.client["attributes"]["other"] == "kept"


@allure.feature("pysweep")
def test_restored_even_when_the_sweep_fails(keycloak):
    fake = keycloak(attributes={CLIENT_TOKEN_LIFESPAN_ATTRIBUTE: "900"})
    with pytest.raises(ValueError):
        with client_token_lifespan(_Helper(), seconds=3600):
            raise ValueError("sweep failed")
    assert fake.lifespan == "900"


@allure.feature("pysweep")
def test_a_failed_restore_does_not_mask_the_sweeps_own_error(keycloak):
    fake = keycloak()
    with pytest.raises(ValueError, match="sweep failed"):
        with client_token_lifespan(_Helper(), seconds=3600):
            fake.fail_puts = True
            raise ValueError("sweep failed")


@allure.feature("pysweep")
def test_a_missing_client_fails_before_anything_is_changed(keycloak):
    fake = keycloak(exists=False)
    with pytest.raises(KeycloakConfigError, match=VITE_KEYCLOAK_CLIENT_ID):
        with client_token_lifespan(_Helper()):
            pass
    assert fake.puts == []
