#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard which realm the identity count of app_data_consistency (files/query_identity.sh) reads.

Before an upgrade renames a previous release's realm in place, the users are still under the
previous name, so the count before the upgrade reads that one. The script runs against an
in-process fake of the two Keycloak endpoints it calls. It needs bash, curl and jq.
"""

import json
import os
import shutil
import subprocess
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

_SCRIPT = (Path(__file__).resolve().parents[4] / "deployment" / "roles" / "app_data_consistency"
           / "files" / "query_identity.sh")

pytestmark = pytest.mark.skipif(not all(shutil.which(t) for t in ("bash", "curl", "jq")),
                                reason="needs bash, curl and jq")


def _count(realms, previous=None):
    class Handler(BaseHTTPRequestHandler):
        def _reply(self, code, body):
            data = json.dumps(body).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            self._reply(200, {"access_token": "t"})

        def do_GET(self):
            realm = self.path.split("/")[3]
            self._reply(200, realms[realm]) if realm in realms else self._reply(404, {"error": "Realm not found."})

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    env = {"PATH": os.environ["PATH"], "KEYCLOAK_URL": f"http://127.0.0.1:{server.server_port}",
           "KEYCLOAK_REALM": "ChatbotToolkit", "KEYCLOAK_ADMIN_USER": "admin", "KEYCLOAK_ADMIN_PASSWORD": "pw",
           "no_proxy": "127.0.0.1", "NO_PROXY": "127.0.0.1"}
    if previous:
        env["KEYCLOAK_PREVIOUS_REALM"] = previous
    try:
        return subprocess.run(["bash", str(_SCRIPT)], capture_output=True, text=True, timeout=30, env=env)
    finally:
        server.shutdown()


def test_counts_the_realm():
    result = _count({"ChatbotToolkit": 8, "EnterpriseRAG": 3}, previous="EnterpriseRAG")
    assert result.returncode == 0 and result.stdout.splitlines()[-1] == "DATA_COUNT=8"


def test_counts_the_previous_realm_until_it_is_renamed():
    result = _count({"EnterpriseRAG": 3}, previous="EnterpriseRAG")
    assert result.returncode == 0 and result.stdout.splitlines()[-1] == "DATA_COUNT=3"


@pytest.mark.parametrize("previous", [None, "EnterpriseRAG", "ChatbotToolkit"])
def test_fails_when_no_realm_exists(previous):
    result = _count({}, previous=previous)
    assert result.returncode == 1
    assert "user count for realm ChatbotToolkit returned HTTP 404" in result.stderr
