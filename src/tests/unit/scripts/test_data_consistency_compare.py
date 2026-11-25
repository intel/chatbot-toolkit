#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the comparison of the data snapshots taken before and after an upgrade.

Runs app_data_consistency's record_failure.yaml, compare.yaml and report.yaml on
localhost against fixture snapshots, then reads the result fact and parses the
upgrade report. A store whose count failed after the upgrade must be reported as
lost (no count, no delta), a decrease as a violation, and a failure reason that
contains quotes must still leave a valid YAML report. No cluster is used; the
test is skipped when Ansible is not installed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ansible")

_ROLES_DIR = Path(__file__).resolve().parents[4] / "deployment" / "roles"

_PLAY = """
- hosts: localhost
  gather_facts: false
  tasks:
    - ansible.builtin.set_fact:
        _data_snapshot: "{{ fixture_post_ok }}"
    # The way snapshot.yaml records a store whose count failed.
    - when: fixture_failed_store | length > 0
      block:
        - ansible.builtin.fail:
            msg: "{{ fixture_failed_reason }}"
      rescue:
        - ansible.builtin.include_role:
            name: app_data_consistency
            tasks_from: record_failure.yaml
          vars:
            dcr_store: "{{ fixture_failed_store }}"
    - ansible.builtin.set_fact:
        _upgrade_pre_snapshot: "{{ fixture_pre }}"
        _upgrade_post_snapshot: "{{ _data_snapshot }}"
    - ansible.builtin.include_role:
        name: app_data_consistency
        tasks_from: compare.yaml
    - ansible.builtin.include_role:
        name: app_data_consistency
        tasks_from: report.yaml
      vars:
        data_consistency_report_kind: upgrade
    - ansible.builtin.copy:
        content: "{{ {'result': _upgrade_consistency_result, 'report': _data_consistency_report_path} | to_json }}"
        dest: "{{ env_dir }}/out.json"
        mode: "0644"
"""

_REASON = 'pods "edp-count" is forbidden: C:\\path'


def _ok(count):
    return {"count": count, "status": "ok", "method": "fixture"}


def _run(tmp_path, pre, post_ok, failed_store="", failed_reason=""):
    (tmp_path / "play.yaml").write_text(_PLAY)
    # Prefixed: extra vars outrank the role's own vars of the same name (pre, post).
    extra = {"env_dir": str(tmp_path), "fixture_pre": pre, "fixture_post_ok": post_ok,
             "fixture_failed_store": failed_store, "fixture_failed_reason": failed_reason}
    (tmp_path / "vars.json").write_text(json.dumps(extra))
    proc = subprocess.run(
        [sys.executable, "-m", "ansible", "playbook", "-i", "localhost,", "-c", "local",
         "-e", f"@{tmp_path / 'vars.json'}", str(tmp_path / "play.yaml")],
        env={"ANSIBLE_ROLES_PATH": str(_ROLES_DIR), "ANSIBLE_NOCOLOR": "1",
             "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    out = json.loads((tmp_path / "out.json").read_text())
    return out["result"], yaml.safe_load(Path(out["report"]).read_text())


def test_each_store_gets_its_own_verdict(tmp_path):
    result, report = _run(
        tmp_path,
        pre={"vdb": _ok(5), "chat_history": _ok(4), "edp": _ok(3), "keycloak": _ok(0)},
        post_ok={"vdb": _ok(6), "chat_history": _ok(2), "keycloak": _ok(0)},
        failed_store="edp",
        failed_reason=_REASON,
    )
    stores = result["stores"]
    assert {k: v["verdict"] for k, v in stores.items()} == {
        "vdb": "pass", "chat_history": "violation", "edp": "lost", "keycloak": "baseline"}
    assert stores["vdb"]["delta"] == 1
    assert stores["chat_history"]["delta"] == -2
    assert stores["edp"]["post"] is None and stores["edp"]["delta"] is None
    assert result["verdict"] == "fail"
    assert result["violations"] == [
        "chat_history: decreased from 4 to 2",
        f"edp: counted 3 before the upgrade but not after it: {_REASON}",
    ]
    # The report parses and carries the same facts.
    assert report["post_snapshot"]["edp"]["count"] is None
    assert report["post_snapshot"]["edp"]["reason"] == _REASON
    assert report["consistency"]["stores"]["edp"]["delta"] is None
    assert report["consistency"]["violations"] == result["violations"]


def test_skipped_store_is_not_a_violation(tmp_path):
    result, report = _run(
        tmp_path,
        pre={"vdb": {"count": 0, "status": "skipped", "method": "none"}, "edp": _ok(3)},
        post_ok={"vdb": _ok(7), "edp": _ok(3)},
    )
    assert {k: v["verdict"] for k, v in result["stores"].items()} == {"vdb": "skipped", "edp": "pass"}
    assert result["verdict"] == "pass"
    assert report["consistency"]["violations"] == []
