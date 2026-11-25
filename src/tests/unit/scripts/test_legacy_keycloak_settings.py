#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the check that rejects the pre-rebrand erag_keycloak_* settings.

Nothing reads those keys any more, so a config that still sets one would deploy with SSO,
LDAP federation or SharePoint silently off. app_pre_install/tasks/load_settings.yaml fails
instead. These tests read that task from disk and render its condition and message with
Jinja2, with query('ansible.builtin.varnames', ...) emulated over a dict of variables. No
Ansible, cluster or network is used.
"""

import re
from pathlib import Path

import jinja2
import pytest
import yaml

_TASKS = (Path(__file__).resolve().parents[4] / "deployment" / "roles" / "app_pre_install"
          / "tasks" / "load_settings.yaml")
_TASK_NAME = "app_pre_install | Reject pre-rebrand erag_keycloak_* settings"


def _task():
    tasks = yaml.safe_load(_TASKS.read_text())
    return next(t for t in tasks if t.get("name") == _TASK_NAME)


def _render(template, variables):
    """Render template the way Ansible renders the task: task vars first, then the template."""
    def query(plugin, pattern):
        assert plugin == "ansible.builtin.varnames"
        return sorted(name for name in variables if re.search(pattern, name))

    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    task_vars = {}
    for name, expr in (_task().get("vars") or {}).items():
        # A template that is a single {{ ... }} keeps its native type, as in Ansible.
        task_vars[name] = env.compile_expression(expr.strip()[2:-2])(query=query, **variables)
    return env.from_string(template).render(query=query, **variables, **task_vars)


def _passes(variables):
    assert_args = _task()["ansible.builtin.assert"]
    return all(_render("{{ " + cond + " }}", variables) == "True" for cond in assert_args["that"])


@pytest.mark.parametrize("variables", [
    {},
    {"chattool_keycloak_oidc_endpoint": "https://login.example/v2.0", "chattool_keycloak_realm": "ChatbotToolkit"},
    {"keycloak_namespace": "keycloak", "erag_domain": "erag.example"},  # other erag_* names are fine
])
def test_current_settings_pass(variables):
    assert _passes(variables)


@pytest.mark.parametrize("variables", [
    {"erag_keycloak_oidc_endpoint": "https://login.example/v2.0"},
    {"chattool_keycloak_oidc_endpoint": "", "erag_keycloak_federation_endpoint": "ldaps://ad.example"},
])
def test_pre_rebrand_settings_fail(variables):
    assert not _passes(variables)


def test_message_names_every_old_key_and_the_replacement():
    msg = _render(_task()["ansible.builtin.assert"]["fail_msg"],
                  {"erag_keycloak_oidc_endpoint": "x", "erag_keycloak_oidc_tenant_id": "y"})
    assert "erag_keycloak_oidc_endpoint, erag_keycloak_oidc_tenant_id are no longer read" in msg
    assert "chattool_keycloak_*" in msg
