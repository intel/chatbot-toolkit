#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the data check around a run that changes the erag version.

app_pre_install/tasks/version_change.yaml decides from Toolkit Manager's run class
and record whether the run changes the version, counts the stores before anything
changes and keeps that snapshot until app_post_install/tasks/data_consistency.yaml
has compared against it. A resume reuses the kept snapshot; a store that lost data
fails the run unless --force bypasses it, and a bypass is recorded in the
installer's _upgrade_gate fact.

Both task files run on localhost. The snapshot task of app_data_consistency is
replaced by a stub that returns fixture counts in call order; its comparison and
report tasks are the real ones. No cluster is used; the test is skipped when
Ansible is not installed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ansible")

_ROLES_DIR = Path(__file__).resolve().parents[4] / "deployment" / "roles"

_STUB_SNAPSHOT = """
- ansible.builtin.set_fact:
    _stub_snapshot_calls: "{{ (_stub_snapshot_calls | default(0) | int) + 1 }}"
- ansible.builtin.set_fact:
    _data_snapshot: "{{ fixture_snapshots[_stub_snapshot_calls | int - 1] }}"
"""

# Inputs are set as facts, not passed as extra vars: extra vars outrank set_fact,
# so _upgrade_gate could not be updated by the role.
_PLAY = """
- hosts: localhost
  gather_facts: false
  tasks:
    - ansible.builtin.set_fact:
        _layer_class: "{{ fixture_layer_class }}"
        _layer_records: "{{ fixture_layer_records }}"
        _layer_release: {erag: "3.1.0"}
        _upgrade_force: "{{ fixture_force }}"
    - ansible.builtin.set_fact:
        _upgrade_gate: "{{ fixture_gate }}"
      when: fixture_gate is not none
    - block:
        - ansible.builtin.include_role:
            name: app_pre_install
            tasks_from: version_change.yaml
        - ansible.builtin.include_role:
            name: app_post_install
            tasks_from: data_consistency.yaml
      always:
        - ansible.builtin.copy:
            content: >-
              {{ {'version_change': erag_version_change | default(none),
                  'snapshot_calls': _stub_snapshot_calls | default(0) | int,
                  'verdict': (_upgrade_consistency_result | default({})).verdict | default(none),
                  'gate': _upgrade_gate | default(none)} | to_json }}
            dest: "{{ env_dir }}/out.json"
            mode: "0644"
"""

_PENDING = "pending-pre-snapshot-3.0.1-to-3.1.0.yaml"


def _counts(vdb):
    return {"vdb": {"count": vdb, "status": "ok", "method": "fixture"}}


def _roles(tmp_path):
    """The real roles, with app_data_consistency's snapshot task stubbed."""
    roles = tmp_path / "roles"
    stub = roles / "app_data_consistency"
    (stub / "tasks").mkdir(parents=True)
    real = _ROLES_DIR / "app_data_consistency"
    for sub in ("defaults", "templates"):
        (stub / sub).symlink_to(real / sub)
    for task in ("compare.yaml", "report.yaml"):
        (stub / "tasks" / task).symlink_to(real / "tasks" / task)
    (stub / "tasks" / "snapshot.yaml").write_text(_STUB_SNAPSHOT)
    return f"{roles}:{_ROLES_DIR}"


def _run(tmp_path, run_class, snapshots, record_version="3.0.1", target=None, force=False, gate=None):
    (tmp_path / "play.yaml").write_text(_PLAY)
    extra = {
        "env_dir": str(tmp_path),
        "tmp_dir": str(tmp_path / "tmp"),
        "fixture_layer_class": {"erag": run_class},
        "fixture_layer_records": {"erag": {"version": record_version, "targetVersion": target or "3.1.0"}},
        "fixture_force": force,
        "fixture_gate": gate,
        "fixture_snapshots": snapshots,
    }
    (tmp_path / "vars.json").write_text(json.dumps(extra))
    proc = subprocess.run(
        [sys.executable, "-m", "ansible", "playbook", "-i", "localhost,", "-c", "local",
         "-e", f"@{tmp_path / 'vars.json'}", str(tmp_path / "play.yaml")],
        env={"ANSIBLE_ROLES_PATH": _roles(tmp_path), "ANSIBLE_NOCOLOR": "1",
             "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True, text=True, check=False,
    )
    return proc, json.loads((tmp_path / "out.json").read_text())


def _reports(tmp_path, prefix):
    return sorted((tmp_path / "reports").glob(f"{prefix}*"))


def test_upgrade_counts_before_and_after_and_drops_the_snapshot(tmp_path):
    proc, out = _run(tmp_path, "upgrade", [_counts(5), _counts(5)])
    assert proc.returncode == 0, proc.stdout
    assert out["version_change"] is True
    assert out["snapshot_calls"] == 2
    assert out["verdict"] == "pass"
    assert not (tmp_path / "reports" / _PENDING).exists()
    assert len(_reports(tmp_path, "upgrade-report-")) == 1


def test_loss_refuses_the_run_and_keeps_the_snapshot(tmp_path):
    proc, out = _run(tmp_path, "upgrade", [_counts(5), _counts(3)])
    assert proc.returncode != 0
    assert "Data consistency check failed" in proc.stdout
    assert out["verdict"] == "fail"
    assert yaml.safe_load((tmp_path / "reports" / _PENDING).read_text()) == _counts(5)


def test_resume_compares_against_the_kept_snapshot(tmp_path):
    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / _PENDING).write_text(yaml.safe_dump(_counts(5)))
    # Only the count after the version change is taken; it shows the loss.
    proc, out = _run(tmp_path, "resume", [_counts(3)])
    assert proc.returncode != 0
    assert out["version_change"] is True
    assert out["snapshot_calls"] == 1
    assert out["verdict"] == "fail"


def test_force_passes_and_records_the_bypass(tmp_path):
    proc, out = _run(tmp_path, "upgrade", [_counts(5), _counts(3)], force=True,
                     gate={"forcedChecks": ["platform/other"]})
    assert proc.returncode == 0, proc.stdout
    assert out["gate"]["forcedChecks"] == ["platform/other", "erag/data_consistency"]
    assert not (tmp_path / "reports" / _PENDING).exists()


def test_force_without_the_gate_fact_fails(tmp_path):
    proc, _ = _run(tmp_path, "upgrade", [_counts(5), _counts(3)], force=True, gate=None)
    assert proc.returncode != 0
    assert "_upgrade_gate" in proc.stdout


def test_same_version_run_writes_only_the_data_report(tmp_path):
    proc, out = _run(tmp_path, "reinstall", [_counts(5)], record_version="3.1.0")
    assert proc.returncode == 0, proc.stdout
    assert out["version_change"] is False
    assert out["snapshot_calls"] == 1
    assert len(_reports(tmp_path, "data-report-")) == 1
    assert not _reports(tmp_path, "pending-pre-snapshot-")


def test_a_fresh_count_drops_snapshots_of_abandoned_version_changes(tmp_path):
    (tmp_path / "reports").mkdir()
    stale = tmp_path / "reports" / "pending-pre-snapshot-3.0.1-to-3.0.9.yaml"
    stale.write_text(yaml.safe_dump(_counts(9)))
    proc, out = _run(tmp_path, "upgrade", [_counts(5), _counts(5)])
    assert proc.returncode == 0, proc.stdout
    assert out["verdict"] == "pass"
    assert not _reports(tmp_path, "pending-pre-snapshot-")
