#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Guard the configuration rules Toolkit Manager runs before anything changes.

deployment/upgrade_checks/configuration.yaml runs app_pre_install's
config_rules.yaml against the layer's version.yaml and records one verdict. The
image-tag rule takes its default override from version.yaml: between releases
(default_image_tag "main", default_tag_override true) any tag passes; in a
release (override false) a tag without the solution version is reported, unless
the configuration sets tag_override.

The check runs on localhost against a copy of the layer directory with a fixture
version.yaml. No cluster is used; the test is skipped when Ansible is not
installed.
"""

import json
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

pytest.importorskip("ansible")

_DEPLOYMENT = Path(__file__).resolve().parents[4] / "deployment"

_PLAY = """
- hosts: localhost
  gather_facts: false
  tasks:
    - ansible.builtin.set_fact:
        _upgrade_verdicts: []
    - ansible.builtin.include_tasks: "{{ layer_dir }}/upgrade_checks/configuration.yaml"
      vars:
        upgrade_check_layer: erag
        _layer_dirs: {erag: "{{ layer_dir }}"}
    - ansible.builtin.copy:
        content: "{{ _upgrade_verdicts | to_json }}"
        dest: "{{ out }}"
        mode: "0644"
"""

_BETWEEN_RELEASES = {"solution_version": "3.1.0", "default_image_tag": "main", "default_tag_override": True}
_RELEASE = {"solution_version": "3.1.0", "default_image_tag": "3.1.0", "default_tag_override": False}


def _check(tmp_path, version, **config):
    layer = tmp_path / "layer"
    layer.mkdir(parents=True)
    for sub in ("roles", "upgrade_checks"):
        (layer / sub).symlink_to(_DEPLOYMENT / sub)
    (layer / "version.yaml").write_text(yaml.safe_dump(version))
    (tmp_path / "play.yaml").write_text(_PLAY)
    extra = {"layer_dir": str(layer), "out": str(tmp_path / "out.json"), "base_domain_name": "example.com"}
    extra.update(config)
    (tmp_path / "vars.json").write_text(json.dumps(extra))
    proc = subprocess.run(
        [sys.executable, "-m", "ansible", "playbook", "-i", "localhost,", "-c", "local",
         "-e", f"@{tmp_path / 'vars.json'}", str(tmp_path / "play.yaml")],
        env={"ANSIBLE_NOCOLOR": "1", "PATH": "/usr/bin:/bin", "HOME": str(tmp_path)},
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    (verdict,) = json.loads((tmp_path / "out.json").read_text())
    assert (verdict["layer"], verdict["check"]) == ("erag", "configuration")
    return verdict


def test_between_releases_any_tag_passes(tmp_path):
    assert _check(tmp_path, _BETWEEN_RELEASES, tag="3.0.1")["verdict"] == "pass"


def test_release_reports_a_tag_without_the_version(tmp_path):
    verdict = _check(tmp_path, _RELEASE, tag="3.0.1")
    assert verdict["verdict"] == "blocking"
    assert 'Image tag "3.0.1" does not contain the solution version "3.1.0"' in verdict["reason"]


def test_release_default_tag_passes(tmp_path):
    assert _check(tmp_path, _RELEASE)["verdict"] == "pass"


def test_configured_tag_override_wins_over_version_yaml(tmp_path):
    assert _check(tmp_path, _BETWEEN_RELEASES, tag="3.0.1", tag_override=False)["verdict"] == "blocking"
    assert _check(tmp_path / "b", _RELEASE, tag="3.0.1", tag_override=True)["verdict"] == "pass"


def test_missing_base_domain_name_blocks(tmp_path):
    verdict = _check(tmp_path, _BETWEEN_RELEASES, base_domain_name="")
    assert verdict["verdict"] == "blocking"
    assert "base_domain_name is required" in verdict["reason"]
