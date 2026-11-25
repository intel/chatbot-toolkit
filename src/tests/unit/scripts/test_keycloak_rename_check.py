#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the refusal of an in-place upgrade from a release before the Chatbot Toolkit rename.

deployment/upgrade_checks/keycloak_rename.yaml records a blocking verdict when the
deployment record holds a version below 3.1.0: such a release created the Keycloak
realm EnterpriseRAG, which this release does not rename. The resolve points to the
reinstall section of docs/deployment/install-rag.md. A fresh install (no record) and a
record from a 3.1.0 build, including a dev or rc build, pass.

The check runs on localhost with a fixture record. No cluster is used; the test is
skipped when Ansible is not installed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytest.importorskip("ansible")

_REPO = Path(__file__).resolve().parents[4]
_CHECK = _REPO / "deployment" / "upgrade_checks" / "keycloak_rename.yaml"
_SECTION = "Upgrade from a release before the Chatbot Toolkit rename"

_PLAY = """
- hosts: localhost
  gather_facts: false
  tasks:
    - ansible.builtin.set_fact:
        _upgrade_verdicts: []
    - ansible.builtin.include_tasks: "{{ check }}"
      vars:
        upgrade_check_layer: erag
    - ansible.builtin.copy:
        content: "{{ _upgrade_verdicts | to_json }}"
        dest: "{{ out }}"
        mode: "0644"
"""


def _check(tmp_path, records):
    (tmp_path / "play.yaml").write_text(_PLAY)
    extra = {"check": str(_CHECK), "out": str(tmp_path / "out.json"), "_layer_records": records}
    (tmp_path / "vars.json").write_text(json.dumps(extra))
    proc = subprocess.run(
        [sys.executable, "-m", "ansible", "playbook", "-i", "localhost,", "-c", "local",
         "-e", f"@{tmp_path / 'vars.json'}", str(tmp_path / "play.yaml")],
        env={"ANSIBLE_NOCOLOR": "1", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (verdict,) = json.loads((tmp_path / "out.json").read_text())
    assert (verdict["layer"], verdict["check"]) == ("erag", "keycloak-rename")
    return verdict


@pytest.mark.parametrize("recorded", ["3.0.0", "3.0.1", "3.0.1-rc.1", "3.0.0.dev111901"])
def test_record_before_the_rename_blocks_with_the_reinstall(tmp_path, recorded):
    verdict = _check(tmp_path, {"erag": {"layer": "erag", "version": recorded, "status": "complete"}})
    assert verdict["verdict"] == "blocking"
    assert recorded in verdict["reason"]
    assert "docs/deployment/install-rag.md" in verdict["resolve"]
    assert _SECTION in verdict["resolve"]


@pytest.mark.parametrize("records", [
    {},
    {"erag": {"layer": "erag", "status": "deleting"}},
    {"erag": {"layer": "erag", "version": "3.1.0", "status": "complete"}},
    {"erag": {"layer": "erag", "version": "3.1.0-rc.1", "status": "complete"}},
    {"erag": {"layer": "erag", "version": "3.1.1", "status": "complete"}},
])
def test_fresh_install_or_a_record_after_the_rename_passes(tmp_path, records):
    assert _check(tmp_path, records)["verdict"] == "pass"


def test_resolve_names_an_existing_section():
    assert f"## {_SECTION}" in (_REPO / "docs" / "deployment" / "install-rag.md").read_text()
