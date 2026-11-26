#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the two admin API Jobs of app_keycloak_config: the configurator and the in-place rename.

Both render from templates/_admin_api_job.j2. These tests render the manifests with Jinja2 the
way Ansible's template module does, and check that the rename Job passes every variable its
script requires and that its Role grants exactly the names the script uses. They also render
the condition of tasks/rename_realm.yaml. No Ansible, cluster or network is used.
"""

import base64
import json
import re
from pathlib import Path

import jinja2
import pytest
import yaml

_ROLE = Path(__file__).resolve().parents[4] / "deployment" / "roles" / "app_keycloak_config"
_SCRIPT = (_ROLE / "files" / "keycloak_rename_job.sh").read_text()

_VARS = {
    "keycloak_namespace": "keycloak",
    "chattool_keycloak_path_prefix": "",
    "chattool_keycloak_realm": "ChatbotToolkit",
    "chattool_keycloak_previous_realm": "EnterpriseRAG",
    "keycloak_admin_user": "admin",
    "keycloak_admin_secret_name": "keycloak-admin-secret",
    "_keycloak_config_fixed_uids": False,
    "init_container_image": "registry/init:1",
    "configurator_job_name": "keycloak-configurator-1",
    "rename_job_name": "keycloak-rename-1",
    "minio_domain": "s3.example",
    "ui_domain": "ui.example",
    "credentials_secret_name": "chattool-credentials",
    "role_path": str(_ROLE),
}


def _render(template, **extra):
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(_ROLE / "templates"), trim_blocks=True,
                             undefined=jinja2.StrictUndefined)
    env.filters.update(to_json=json.dumps, bool=lambda v: str(v).lower() in ("true", "1", "yes"),
                       b64encode=lambda v: base64.b64encode(v.encode()).decode())
    env.globals["lookup"] = lambda plugin, path: Path(path).read_text()
    text = env.get_template(template).render({**_VARS, **extra})
    return {(d["kind"], d["metadata"]["name"]): d for d in yaml.safe_load_all(text) if d}


def _job(manifest):
    return next(d for (kind, _), d in manifest.items() if kind == "Job")


@pytest.mark.parametrize("template, app, retries", [
    ("configurator-job.yaml.j2", "keycloak-configurator", (3, "OnFailure")),
    ("rename-job.yaml.j2", "keycloak-rename", (0, "Never")),
])
@pytest.mark.parametrize("extra", [{}, {"http_proxy": "http://proxy:3128", "no_proxy": "localhost",
                                        "istio_pod_annotations": {"sidecar.istio.io/inject": "true"}}])
def test_job_runs_as_its_own_account_and_script(template, app, retries, extra):
    manifest = _render(template, **extra)
    assert {("ServiceAccount", app), ("Role", app), ("RoleBinding", app), ("ConfigMap", f"{app}-script")} \
        <= set(manifest)
    job = _job(manifest)
    pod = job["spec"]["template"]["spec"]
    assert job["metadata"]["labels"]["app.kubernetes.io/name"] == app
    assert (job["spec"]["backoffLimit"], pod["restartPolicy"]) == retries
    assert pod["serviceAccountName"] == app
    assert pod["volumes"][0]["configMap"]["name"] == f"{app}-script"
    script = pod["containers"][0]["command"][-1].removeprefix("/scripts/")
    assert script in manifest[("ConfigMap", f"{app}-script")]["data"]
    env = {e["name"] for e in pod["containers"][0]["env"]}
    assert {"KEYCLOAK_URL", "KEYCLOAK_REALM", "KEYCLOAK_ADMIN_USER", "KEYCLOAK_ADMIN_PASSWORD"} <= env
    assert ("http_proxy" in env) == ("http_proxy" in extra)


def test_rename_job_passes_every_variable_the_script_requires():
    required = set(re.findall(r"\$\{(\w+):\?\}", _SCRIPT))
    env = {e["name"] for e in _job(_render("rename-job.yaml.j2"))["spec"]["template"]["spec"]["containers"][0]["env"]}
    assert required and required <= env, f"not passed: {required - env}"


def test_rename_role_grants_the_names_the_script_uses():
    names = dict(re.findall(r"\b(OLD_SECRET_NAME|NEW_SECRET_NAME|STATE_CONFIGMAP)=([\w-]+)", _SCRIPT))
    rules = _render("rename-job.yaml.j2")[("Role", "keycloak-rename")]["rules"]
    named = {r["resources"][0]: set(r["resourceNames"]) for r in rules if "resourceNames" in r}
    assert named == {"secrets": {names["OLD_SECRET_NAME"], names["NEW_SECRET_NAME"]},
                     "configmaps": {names["STATE_CONFIGMAP"]}}


def _rename_runs(resources):
    tasks = yaml.safe_load((_ROLE / "tasks" / "rename_realm.yaml").read_text())
    conditions = next(t for t in tasks if "block" in t)["when"]
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    scope = {"_kc_rename_old_secret": {"resources": resources}}
    return all(env.compile_expression(c)(**scope) for c in conditions)


@pytest.mark.parametrize("resources, runs", [
    ([], False),  # installed with the new names
    ([{"metadata": {"name": "erag-credentials"}}], True),
    ([{"metadata": {"name": "erag-credentials", "labels": {"app.kubernetes.io/name": "keycloak-configurator"}}}],
     True),
    ([{"metadata": {"name": "erag-credentials", "labels": {"chatbot-toolkit/keycloak-renamed": "true"}}}], False),
])
def test_rename_runs_until_the_old_secret_is_marked(resources, runs):
    assert _rename_runs(resources) is runs


def test_teardown_removes_what_the_jobs_create():
    tasks = yaml.safe_load((_ROLE / "tasks" / "teardown.yaml").read_text())
    deleted = {(i["kind"], i["name"]) for t in tasks for i in (t.get("loop") or []) if isinstance(i, dict) and "name" in i}
    selectors = [s for t in tasks for s in (t.get("kubernetes.core.k8s_info") or {}).get("label_selectors", [])]
    created = set()
    for template in ("configurator-job.yaml.j2", "rename-job.yaml.j2"):
        manifest = _render(template, chattool_keycloak_oidc_endpoint="https://idp.example",
                           chattool_keycloak_oidc_client_secret="s")
        created |= {key for key in manifest if key[0] != "Job"}
    assert created <= deleted, f"left behind by teardown: {created - deleted}"
    assert any(all(app in s for app in ("keycloak-configurator", "keycloak-rename")) for s in selectors)


def test_teardown_keeps_old_credentials_until_renamed():
    tasks = yaml.safe_load((_ROLE / "tasks" / "teardown.yaml").read_text())
    delete = next(t for t in tasks if {"kind": "Secret", "name": "erag-credentials"} in (t.get("loop") or []))
    assert {"kind": "ConfigMap", "name": "keycloak-rename-state"} in delete["loop"]
    env = jinja2.Environment(undefined=jinja2.StrictUndefined)
    for labels, deleted in [({}, False), ({"chatbot-toolkit/keycloak-renamed": "true"}, True)]:
        scope = {"_kc_old_secret": {"resources": [{"metadata": {"labels": labels}}]}}
        assert bool(env.compile_expression(delete["when"])(**scope)) is deleted


@pytest.mark.parametrize("status, finished, succeeded", [
    ({"resources": [{"status": {"succeeded": 1}}]}, True, True),
    ({"resources": [{"status": {"failed": 1}}]}, True, False),
    ({"resources": [{"status": {"active": 1}}]}, False, False),
    ({"resources": [{}]}, False, False),
    ({"resources": []}, False, False),
    (None, False, False),
])
def test_runner_keeps_the_account_of_a_job_still_running(status, finished, succeeded):
    tasks = yaml.safe_load((_ROLE / "tasks" / "_run_admin_api_job.yaml").read_text())
    always = next(t for t in tasks if "block" in t)["always"]
    read = next(t for t in always if "ansible.builtin.set_fact" in t)["ansible.builtin.set_fact"]["_kc_job_state"]
    delete = next(t for t in always if "kubernetes.core.k8s" in t)
    check = next(t for t in tasks if "ansible.builtin.fail" in t)
    env = jinja2.Environment(undefined=jinja2.ChainableUndefined)  # as Ansible's, for an unregistered status
    env.filters["bool"] = bool
    state = env.compile_expression(read.strip("{} "))(**({"_kc_job_status": status} if status else {}))
    scope = {"_kc_job_state": state, "_kc_job_delete_account": True}
    assert all(env.compile_expression(c)(**scope) for c in delete["when"]) is finished
    assert bool(env.compile_expression(check["when"])(**scope)) is not succeeded


def test_runner_forgets_the_status_of_the_previous_job():
    tasks = yaml.safe_load((_ROLE / "tasks" / "_run_admin_api_job.yaml").read_text())
    assert tasks[0]["ansible.builtin.set_fact"] == {"_kc_job_status": {}}
