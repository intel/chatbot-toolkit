# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import authentication, paths
from tests.e2e.benchmarks.chatqna.pysweep.authentication import (
    BUILD_CONFIG_FILES,
    BuildConfigError,
    build_config_dir,
)


@allure.feature("pysweep")
def test_repo_root_is_the_checkout():
    assert (paths.REPO_ROOT / "src" / "tests" / "e2e").is_dir()
    assert (paths.REPO_ROOT / "deployment").is_dir()


@allure.feature("pysweep")
def test_deploy_root_is_derived_from_an_installer_layout(monkeypatch, tmp_path):
    repo = tmp_path / "ai-solutions" / "ext" / "enterprise.ai-erag"
    monkeypatch.setattr(paths, "REPO_ROOT", repo)
    assert paths.deploy_roots() == (tmp_path / "ai-solutions", paths.ROOT_DEPLOY_ROOT)


@allure.feature("pysweep")
def test_a_plain_checkout_only_falls_back_to_root(monkeypatch, tmp_path):
    monkeypatch.setattr(paths, "REPO_ROOT", tmp_path / "PycharmProjects" / "ai-erag")
    assert paths.deploy_roots() == (paths.ROOT_DEPLOY_ROOT,)


@allure.feature("pysweep")
@pytest.mark.parametrize("probe", [paths.is_dir, paths.is_file], ids=["is_dir", "is_file"])
def test_an_unreadable_path_counts_as_absent(monkeypatch, probe):
    def denied(self, *args, **kwargs):
        raise PermissionError(13, "Permission denied", str(self))

    monkeypatch.setattr(Path, "stat", denied)
    assert probe(Path("/root/ai-solutions/env/local")) is False


@allure.feature("pysweep")
def test_build_config_dir_skips_an_unreadable_fallback(monkeypatch, tmp_path):
    readable = tmp_path / "env" / "local"
    readable.mkdir(parents=True)
    unreadable = Path("/root/ai-solutions/env/local")
    real_stat = Path.stat

    def stat(self, *args, **kwargs):
        if self == unreadable:
            raise PermissionError(13, "Permission denied", str(self))
        return real_stat(self, *args, **kwargs)

    monkeypatch.delenv(authentication.BUILD_CONFIG_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(Path, "stat", stat)
    monkeypatch.setattr(authentication, "BUILD_CONFIG_FALLBACKS", (str(unreadable), str(readable)))
    assert build_config_dir() == readable


@allure.feature("pysweep")
def test_no_build_config_dir_anywhere_is_an_error(monkeypatch):
    monkeypatch.delenv(authentication.BUILD_CONFIG_DIR_ENV_VAR, raising=False)
    monkeypatch.setattr(authentication, "BUILD_CONFIG_FALLBACKS", ())
    with pytest.raises(BuildConfigError, match="Cannot find"):
        build_config_dir()


@allure.feature("pysweep")
def test_build_config_files_match_the_e2e_suite():
    conftest = (paths.REPO_ROOT / "src" / "tests" / "e2e" / "conftest.py").read_text()
    for filename in BUILD_CONFIG_FILES:
        assert f'"{filename}"' in conftest, filename
