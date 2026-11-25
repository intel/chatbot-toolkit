# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import shutil
import subprocess

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import commit_info
from tests.e2e.benchmarks.chatqna.pysweep.commit_info import (
    COMMIT_INFO_FIELDS,
    CommitInfoError,
    read_commit_info,
)

needs_git = pytest.mark.skipif(shutil.which("git") is None, reason="git is not installed")


def _git(repo, *args):
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


@pytest.fixture
def repo(tmp_path):
    _git(tmp_path, "init", "-q", "-b", "feature/x")
    _git(tmp_path, "config", "user.name", "A. Author")
    _git(tmp_path, "config", "user.email", "author@example.com")
    (tmp_path / "f.txt").write_text("one\n")
    _git(tmp_path, "add", "f.txt")
    _git(tmp_path, "commit", "-q", "-m", "feat: first change", "-m", "Body that must not appear.")
    return tmp_path


@allure.feature("pysweep")
@needs_git
def test_commit_info_describes_the_committed_head(repo):
    info = read_commit_info(repo)
    assert tuple(info) == COMMIT_INFO_FIELDS
    assert len(info["sha"]) == 40
    assert info["author"] == "A. Author"
    assert info["description"] == "feat: first change"
    assert info["branch"] == "feature/x"
    assert "T" in info["datetime"]


@allure.feature("pysweep")
@needs_git
def test_uncommitted_changes_do_not_count(repo):
    before = read_commit_info(repo)
    (repo / "f.txt").write_text("edited\n")
    assert read_commit_info(repo) == before


@allure.feature("pysweep")
@needs_git
def test_a_detached_head_is_an_error(repo):
    _git(repo, "checkout", "-q", "--detach")
    with pytest.raises(CommitInfoError, match="detached"):
        read_commit_info(repo)


@allure.feature("pysweep")
@needs_git
def test_a_directory_without_git_is_an_error(tmp_path):
    with pytest.raises(CommitInfoError, match="readable .git"):
        read_commit_info(tmp_path)


@allure.feature("pysweep")
def test_missing_git_is_an_error(monkeypatch, tmp_path):
    monkeypatch.setattr(commit_info.shutil, "which", lambda name: None)
    with pytest.raises(CommitInfoError, match="git is not installed"):
        read_commit_info(tmp_path)
