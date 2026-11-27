#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the in-place rename of the Keycloak objects of a release before the Chatbot Toolkit
rename (app_keycloak_config/files/keycloak_rename_job.sh).

The script runs against an in-process fake of the parts of the Keycloak admin API it uses,
and a fake kubectl on PATH that keeps Secrets and ConfigMaps in a JSON file. It needs bash,
curl and jq; no cluster or network is used.
"""

import base64
import copy
import json
import os
import shutil
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

_SCRIPT = (Path(__file__).resolve().parents[4] / "deployment" / "roles" / "app_keycloak_config"
           / "files" / "keycloak_rename_job.sh")

pytestmark = pytest.mark.skipif(not all(shutil.which(t) for t in ("bash", "curl", "jq")),
                                reason="needs bash, curl and jq")

_FAKE_KUBECTL = r'''#!/usr/bin/env python3
import base64, json, os, sys
path = os.environ["FAKE_KUBE_STATE"]
state = json.load(open(path))
args = sys.argv[1:]
ns = args[args.index("-n") + 1] if "-n" in args else None

def save():
    json.dump(state, open(path, "w"))

verb = args[0]
if verb == "create" and args[1] == "-f":
    obj = json.load(sys.stdin)
    key = "secret/%s/%s" % (obj["metadata"]["namespace"], obj["metadata"]["name"])
    if key in state:
        sys.exit(1)
    state[key] = obj
    save()
    sys.exit(0)
kind, name = args[1], args[2]
key = "%s/%s/%s" % (kind, ns, name)
if verb == "get":
    if key not in state:
        print('Error from server (NotFound)', file=sys.stderr)
        sys.exit(1)
    if "-o" in args:
        out = args[args.index("-o") + 1]
        if out == "json":
            print(json.dumps(state[key]))
        else:
            print(state[key]["data"]["editUsernameAllowed"], end="")
elif verb == "create":
    if key in state:
        sys.exit(1)
    k, v = [a for a in args if a.startswith("--from-literal=")][0].split("=", 1)[1].split("=", 1)
    state[key] = {"data": {k: v}}
    save()
elif verb == "delete":
    state.pop(key, None)
    save()
'''


def _b64(s):
    return base64.b64encode(s.encode()).decode()


def _b64d(s):
    return base64.b64decode(s).decode()


def _old_realm():
    """A realm as a release before the rename leaves it, with SSO and a directory set up."""
    def mapper(i, name, **config):
        return {"id": i, "name": name, "protocol": "openid-connect", "config": config}
    return {
        "realm": {"id": "EnterpriseRAG", "realm": "EnterpriseRAG", "editUsernameAllowed": False,
                  "passwordPolicy": "length(12)"},
        "clients": [
            {"id": "c1", "clientId": "EnterpriseRAG-oidc", "publicClient": True, "redirectUris": ["https://ui/*"],
             "roles": [{"id": "r1", "name": "ERAG-admin"}, {"id": "r2", "name": "ERAG-user"}],
             "mappers": [mapper("m1", "erag-backend-audience",
                                **{"included.client.audience": "EnterpriseRAG-oidc-backend"}),
                         mapper("m2", "minio_roles",
                                **{"usermodel.clientRoleMapping.clientId": "EnterpriseRAG-oidc-minio"})]},
            {"id": "c2", "clientId": "EnterpriseRAG-oidc-minio", "publicClient": False,
             "roles": [{"id": "r3", "name": "consoleAdmin"}, {"id": "r4", "name": "erag-admin-group"}],
             "mappers": []},
            {"id": "c3", "clientId": "account", "roles": [{"id": "r5", "name": "manage-account"}],
             "mappers": [mapper("m3", "audience resolve")]},
        ],
        "roles": [{"id": "rr1", "name": "ERAG-SSO-Admin", "description": "kept"},
                  {"id": "rr2", "name": "offline_access"}],
        "groups": [{"id": "g1", "name": "erag-admin-group", "attributes": {"a": ["b"]}},
                   {"id": "g2", "name": "other-group"}],
        "users": [{"id": "u1", "username": "erag-admin", "email": "a@example.com"},
                  {"id": "u2", "username": "erag-user", "email": "u@example.com"},
                  {"id": "u3", "username": "erag-maintainer", "federationLink": "ldap1"},
                  {"id": "u5", "username": "erag-validation-user"},
                  {"id": "u4", "username": "alice"}],
        "idps": {"enterprise-sso": [{"id": "i1", "name": "SSO-Admin-Mapper",
                                     "config": {"claim.value": "EnterpriseRAG.AdminAccess", "role": "ERAG-SSO-Admin"}}]},
        "components": [{"id": "p1", "name": "Active Directory Federation", "parentId": "EnterpriseRAG",
                        "providerType": "org.keycloak.storage.UserStorageProvider", "config": {}},
                       {"id": "k1", "name": "ERAG-admin-oidc", "parentId": "p1",
                        "providerType": "org.keycloak.storage.ldap.mappers.LDAPStorageMapper",
                        "config": {"client.id": ["EnterpriseRAG-oidc"],
                                   "groups.ldap.filter": ["(cn=erag-admin-group)"],
                                   "roles.ldap.filter": ["(cn=erag-prod-admins)"],
                                   "roles.dn": ["OU=erag-groups,DC=corp"]}}],
    }


class _Keycloak:
    """Holds realms by name; fail names one (method, path, query parameter) request that fails once."""

    def __init__(self, realms):
        self.realms = {r["realm"]["realm"]: r for r in realms}
        self.puts = []
        self.created = 0
        self.fail = None
        self.tokens = 0
        self.token = None
        self.requests = 0
        self.expire_at = None
        self.not_ready = 0
        self.ready_checks = 0

    def handle(self, method, path, query, body, auth=None, form=None):  # noqa: C901 - one branch per endpoint
        if path == "/realms/master":
            self.ready_checks += 1
            return (200, {"realm": "master"}) if self.ready_checks > self.not_ready else (503, {})
        if path == "/realms/master/protocol/openid-connect/token":
            if (form or {}).get("password") != ["admin"]:
                return 401, {"error": "invalid_grant"}
            self.tokens += 1
            self.token = f"t{self.tokens}"
            return 200, {"access_token": self.token}
        self.requests += 1
        if self.requests == self.expire_at:
            self.token = None
        if auth != f"Bearer {self.token}":
            return 401, {"error": "HTTP 401 Unauthorized"}
        parts = path.split("/")[3:]  # after /admin/realms
        realm = self.realms.get(parts[0])
        if realm is None:
            return 404, {"error": "Realm not found."}
        rest = parts[1:]
        if method != "GET":
            self.puts.append(path)
        if self.fail and self.fail[:2] == (method, path) and (self.fail[2] is None or self.fail[2] in query):
            self.fail = None
            return 500, {}
        clients = {c["id"]: c for c in realm["clients"]}
        if not rest:
            if method == "GET":
                return 200, realm["realm"]
            realm["realm"].update(body)
            if "realm" in body and body["realm"] != parts[0]:
                self.realms[body["realm"]] = self.realms.pop(parts[0])
            return 204, None
        if rest == ["clients"]:
            return 200, [{k: v for k, v in c.items() if k not in ("roles", "mappers")} for c in realm["clients"]]
        if rest[0] == "clients":
            c = clients[rest[1]]
            if len(rest) == 2:
                if method == "GET":
                    return 200, {k: v for k, v in c.items() if k not in ("roles", "mappers")}
                c.update(body)
                return 204, None
            if rest[2] == "roles":
                return 200, c["roles"]
            return self._mappers(c["mappers"], rest[4:], method, body)
        if rest[0] == "roles":
            return 200, realm["roles"]
        if rest[0] == "roles-by-id":
            role = next(r for r in realm["roles"] + [r for c in realm["clients"] for r in c["roles"]]
                        if r["id"] == rest[1])
            if method == "GET":
                return 200, role
            role.update(body)
            return 204, None
        if rest[0] == "groups":
            if len(rest) == 1:
                return 200, realm["groups"]
            return self._get_or_put(realm["groups"], rest[1], method, body)
        if rest[0] == "users":
            if len(rest) == 1:
                name = query["username"][0]
                if query.get("exact") == ["true"]:
                    return 200, [u for u in realm["users"] if u["username"] == name]
                return 200, [u for u in realm["users"] if name in u["username"]]
            user = next(u for u in realm["users"] if u["id"] == rest[1])
            if method == "PUT" and body["username"] != user["username"]:
                if not realm["realm"]["editUsernameAllowed"] or user.get("federationLink"):
                    return 400, {"errorMessage": "error-user-attribute-read-only"}
            return self._get_or_put(realm["users"], rest[1], method, body)
        if rest[0] == "identity-provider":
            if len(rest) == 2:
                return 200, [{"alias": a} for a in realm["idps"]]
            return self._mappers(realm["idps"][rest[2]], rest[4:], method, body)
        if rest[0] == "components":
            if not rest[1:] and method == "GET":
                # As in Keycloak: the parent defaults to the realm.
                parent = query.get("parent", [realm["realm"]["id"]])[0]
                return 200, [c for c in realm["components"] if c["parentId"] == parent
                             and c["providerType"] == query["type"][0]]
            return self._mappers(realm["components"], rest[1:], method, body)
        return 404, {}

    @staticmethod
    def _get_or_put(items, item_id, method, body):
        item = next(i for i in items if i["id"] == item_id)
        if method == "GET":
            return 200, item
        item.clear()
        item.update(body)
        return 204, None

    def _mappers(self, items, rest, method, body):
        """A mapper list, where an update keeps the mapper's name, as in Keycloak."""
        if not rest:
            if method == "GET":
                return 200, items
            if any(i["name"] == body["name"] for i in items):
                return 409, {}
            self.created += 1
            items.append({**body, "id": f"new{self.created}"})
            return 201, None
        item = next(i for i in items if i["id"] == rest[0])
        if method == "DELETE":
            items.remove(item)
            return 204, None
        item.update({k: v for k, v in body.items() if k != "name"})
        return 204, None


@pytest.fixture
def env(tmp_path):
    keycloak = _Keycloak([_old_realm()])

    class Handler(BaseHTTPRequestHandler):
        def _serve(self):
            url = urlparse(self.path)
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            body = json.loads(raw) if raw and url.path.startswith("/admin/") else None
            form = parse_qs(raw.decode()) if raw and not url.path.startswith("/admin/") else None
            code, out = keycloak.handle(self.command, url.path, parse_qs(url.query), body,
                                        self.headers.get("Authorization"), form)
            data = json.dumps(out).encode() if out is not None else b""
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_PUT = do_POST = do_DELETE = _serve

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    kubectl = bin_dir / "kubectl"
    kubectl.write_text(_FAKE_KUBECTL.replace("/usr/bin/env python3", sys.executable, 1))
    kubectl.chmod(0o755)
    state = tmp_path / "kube.json"
    state.write_text(json.dumps({"secret/auth/erag-credentials": {
        "apiVersion": "v1", "kind": "Secret", "type": "Opaque",
        "metadata": {"name": "erag-credentials", "namespace": "auth",
                     "labels": {"app.kubernetes.io/name": "keycloak-configurator",
                                "velero.io/restore-name": "r1"}},
        "data": {"KEYCLOAK_ERAG_ADMIN_USERNAME": _b64("erag-admin"),
                 "KEYCLOAK_ERAG_ADMIN_PASSWORD": _b64("Secret1!xyzab"),
                 "MCP_CLIENT_SECRET": _b64("mcp")}}}))

    def run(**extra_env):
        return subprocess.run(
            ["bash", str(_SCRIPT)], capture_output=True, text=True, timeout=60,
            env={"PATH": f"{bin_dir}:{os.environ['PATH']}", "FAKE_KUBE_STATE": str(state),
                 "KEYCLOAK_URL": f"http://127.0.0.1:{server.server_port}",
                 "KEYCLOAK_ADMIN_USER": "admin", "KEYCLOAK_ADMIN_PASSWORD": "admin",
                 "KEYCLOAK_REALM": "ChatbotToolkit", "KEYCLOAK_PREVIOUS_REALM": "EnterpriseRAG", "SECRET_NAMESPACE": "auth",
                 "no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1", **extra_env})

    yield keycloak, run, lambda: json.loads(state.read_text())
    server.shutdown()


def test_renames_every_object_in_place(env):
    keycloak, run, kube = env
    before = copy.deepcopy(keycloak.realms["EnterpriseRAG"])
    result = run()
    assert result.returncode == 0, result.stderr

    assert set(keycloak.realms) == {"ChatbotToolkit"}
    realm = keycloak.realms["ChatbotToolkit"]
    assert realm["realm"]["editUsernameAllowed"] is False
    assert realm["realm"]["passwordPolicy"] == "length(12)"
    assert [(c["id"], c["clientId"]) for c in realm["clients"]] == [
        ("c1", "ChatbotToolkit-oidc"), ("c2", "ChatbotToolkit-oidc-minio"), ("c3", "account")]
    assert realm["clients"][0]["redirectUris"] == before["clients"][0]["redirectUris"]
    assert [r["name"] for c in realm["clients"] for r in c["roles"]] == [
        "CHATTOOL-admin", "CHATTOOL-user", "consoleAdmin", "chattool-admin-group", "manage-account"]
    assert realm["roles"] == [{"id": "rr1", "name": "CHATTOOL-SSO-Admin", "description": "kept"},
                              {"id": "rr2", "name": "offline_access"}]
    assert realm["groups"][0] == {"id": "g1", "name": "chattool-admin-group", "attributes": {"a": ["b"]}}
    assert [(u["id"], u["username"]) for u in realm["users"]] == [
        ("u1", "chattool-admin"), ("u2", "chattool-user"), ("u3", "erag-maintainer"),
        ("u5", "erag-validation-user"), ("u4", "alice")]
    assert realm["users"][0]["email"] == "a@example.com"

    minio_roles, audience = realm["clients"][0]["mappers"]
    assert (minio_roles["id"], audience["name"]) == ("m2", "chattool-backend-audience")
    assert audience["config"]["included.client.audience"] == "ChatbotToolkit-oidc-backend"
    assert minio_roles["config"]["usermodel.clientRoleMapping.clientId"] == "ChatbotToolkit-oidc-minio"
    assert realm["idps"]["enterprise-sso"][0]["config"] == {
        "claim.value": "ChatbotToolkit.AdminAccess", "role": "CHATTOOL-SSO-Admin"}
    [component] = [c for c in realm["components"] if c["parentId"] == "p1"]
    assert component["name"] == "CHATTOOL-admin-oidc"
    assert component["config"] == {"client.id": ["ChatbotToolkit-oidc"],
                                   "groups.ldap.filter": ["(cn=chattool-admin-group)"],
                                   "roles.ldap.filter": ["(cn=erag-prod-admins)"],  # the site's own
                                   "roles.dn": ["OU=erag-groups,DC=corp"]}  # the site's own

    state = kube()
    new = state["secret/auth/chattool-credentials"]
    assert {k: _b64d(v) for k, v in new["data"].items()} == {
        "KEYCLOAK_CHATTOOL_ADMIN_USERNAME": "chattool-admin",
        "KEYCLOAK_CHATTOOL_ADMIN_PASSWORD": "Secret1!xyzab", "MCP_CLIENT_SECRET": "mcp"}
    assert new["metadata"]["labels"] == {"app.kubernetes.io/name": "keycloak-configurator"}
    assert "secret/auth/erag-credentials" in state
    assert not any(k.startswith("configmap/") for k in state)


def test_second_run_changes_nothing(env):
    keycloak, run, kube = env
    assert run().returncode == 0
    after, state, keycloak.puts = copy.deepcopy(keycloak.realms), kube(), []
    result = run()
    assert result.returncode == 0, result.stderr
    assert "nothing to rename" in result.stdout
    assert keycloak.puts == [] and keycloak.realms == after and kube() == state


def test_interrupted_run_resumes_and_restores_username_edits(env):
    keycloak, run, kube = env
    keycloak.fail = ("PUT", "/admin/realms/EnterpriseRAG/users/u2", None)
    assert run().returncode != 0
    stopped = keycloak.realms["EnterpriseRAG"]
    assert stopped["realm"]["editUsernameAllowed"] is False  # switched back by the failed run
    assert stopped["users"][0]["username"] == "chattool-admin"

    result = run()
    assert result.returncode == 0, result.stderr
    realm = keycloak.realms["ChatbotToolkit"]
    assert realm["realm"]["editUsernameAllowed"] is False
    assert [u["username"] for u in realm["users"]][:2] == ["chattool-admin", "chattool-user"]
    assert not any(k.startswith("configmap/") for k in kube())


def test_keeps_username_edits_allowed_when_they_were(env):
    keycloak, run, _ = env
    keycloak.realms["EnterpriseRAG"]["realm"]["editUsernameAllowed"] = True
    assert run().returncode == 0
    assert keycloak.realms["ChatbotToolkit"]["realm"]["editUsernameAllowed"] is True


def test_refuses_when_both_realms_exist(env):
    keycloak, run, _ = env
    keycloak.realms["ChatbotToolkit"] = {"realm": {"realm": "ChatbotToolkit"}, "clients": []}
    keycloak.puts = []
    result = run()
    assert result.returncode == 1
    assert "Both realms EnterpriseRAG and ChatbotToolkit exist" in result.stderr
    assert keycloak.puts == []


def test_fresh_install_has_nothing_to_rename(env):
    keycloak, run, kube = env
    keycloak.realms = {}
    state = kube()
    result = run()
    assert result.returncode == 0, result.stderr
    assert kube() == state


def test_a_failed_user_lookup_stops_before_the_realm_is_renamed(env):
    keycloak, run, _ = env
    keycloak.fail = ("GET", "/admin/realms/EnterpriseRAG/users", "exact")  # the lookup of the new name
    keycloak.realms["EnterpriseRAG"]["users"] = keycloak.realms["EnterpriseRAG"]["users"][:1]
    result = run()
    assert result.returncode != 0
    assert set(keycloak.realms) == {"EnterpriseRAG"}
    assert keycloak.realms["EnterpriseRAG"]["users"][0]["username"] == "erag-admin"
    assert run().returncode == 0
    assert keycloak.realms["ChatbotToolkit"]["users"][0]["username"] == "chattool-admin"


def test_a_configured_realm_name_is_kept_and_renamed_inside(env):
    keycloak, run, _ = env
    realm = keycloak.realms.pop("EnterpriseRAG")
    realm["realm"]["realm"] = "Corp"
    keycloak.realms["Corp"] = realm
    result = run(KEYCLOAK_PREVIOUS_REALM="Corp", KEYCLOAK_REALM="Corp")
    assert result.returncode == 0, result.stderr
    assert set(keycloak.realms) == {"Corp"}
    assert [c["clientId"] for c in realm["clients"]][:2] == ["ChatbotToolkit-oidc", "ChatbotToolkit-oidc-minio"]
    assert realm["users"][0]["username"] == "chattool-admin"
    keycloak.puts = []
    second = run(KEYCLOAK_PREVIOUS_REALM="Corp", KEYCLOAK_REALM="Corp")
    assert second.returncode == 0 and "already renamed" in second.stdout
    assert keycloak.puts == []


def test_refuses_a_configured_realm_with_both_sets_of_objects(env, tmp_path):
    keycloak, run, kube = env
    realm = keycloak.realms.pop("EnterpriseRAG")
    realm["realm"]["realm"] = "Corp"
    keycloak.realms["Corp"] = realm
    state = kube()
    state["secret/auth/chattool-credentials"] = {"metadata": {"name": "chattool-credentials"}, "data": {}}
    (tmp_path / "kube.json").write_text(json.dumps(state))
    keycloak.puts = []
    result = run(KEYCLOAK_PREVIOUS_REALM="Corp", KEYCLOAK_REALM="Corp")
    assert result.returncode == 1
    assert "holds clients under both the old and the new names" in result.stderr
    assert keycloak.puts == []


@pytest.mark.parametrize("expire_at", range(2, 22, 2))
def test_an_expired_token_is_fetched_again_once(env, expire_at):
    keycloak, run, _ = env
    keycloak.expire_at = expire_at
    result = run()
    assert result.returncode == 0, result.stderr
    assert keycloak.tokens == 2
    assert set(keycloak.realms) == {"ChatbotToolkit"}


def test_refuses_a_configured_realm_name_the_earlier_release_did_not_use(env):
    keycloak, run, _ = env
    keycloak.puts = []
    result = run(KEYCLOAK_PREVIOUS_REALM="MyRealm", KEYCLOAK_REALM="MyRealm")
    assert result.returncode == 1
    assert "configuration names realm MyRealm" in result.stderr
    assert keycloak.puts == [] and set(keycloak.realms) == {"EnterpriseRAG"}


def test_a_mapper_already_under_the_new_name_replaces_the_old_one(env):
    keycloak, run, _ = env
    mappers = keycloak.realms["EnterpriseRAG"]["clients"][0]["mappers"]
    mappers.append({"id": "m9", "name": "chattool-backend-audience", "config": {"included.client.audience": "x"}})
    assert run().returncode == 0
    assert [(m["id"], m["name"]) for m in mappers] == [("m2", "minio_roles"), ("m9", "chattool-backend-audience")]


def test_waits_for_keycloak_before_the_first_request(env):
    keycloak, run, _ = env
    keycloak.not_ready = 1
    result = run()
    assert result.returncode == 0, result.stderr
    assert keycloak.ready_checks == 2
    assert set(keycloak.realms) == {"ChatbotToolkit"}


def test_stops_before_the_realm_rename_when_the_old_secret_cannot_be_read(env, tmp_path):
    keycloak, run, _ = env
    empty = tmp_path / "empty.json"
    empty.write_text("{}")
    result = run(FAKE_KUBE_STATE=str(empty))
    assert result.returncode != 0 and "Cannot read Secret erag-credentials" in result.stderr, result.stderr
    assert "EnterpriseRAG" in keycloak.realms and "ChatbotToolkit" not in keycloak.realms
