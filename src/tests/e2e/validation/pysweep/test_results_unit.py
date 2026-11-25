# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.config import load_config
from tests.e2e.benchmarks.chatqna.pysweep.results import (
    PER_CASE_DIR_NAME,
    SUMMARY_COLUMNS,
    SUMMARY_FILE_NAME,
    RequestRecord,
    SummariesLayoutError,
    SweepSummary,
    percentile,
    prepare_summaries_dir,
    read_requests,
    summarize_requests,
)

CLIENT = {
    "question_file": "questions-short.csv",
    "concurrency": 4,
    "duration": "3m",
    "worker_startup_delay": "0s",
    "failures_allowed": 200,
    "input_tokens": 128,
    "warmup_requests": 1,
}


def _result_file(tmp_path, rows, name="bench_x.result.csv"):
    """Write a benchmark.py-shaped result CSV: no header, one row per request."""
    path = tmp_path / name
    with path.open("w", newline="") as handle:
        csv.writer(handle).writerows(rows)
    return path


@allure.feature("pysweep")
def test_tpot_excludes_the_first_token():
    """11 tokens, 2.0s after the first: 10 gaps of 0.2s, not parse.py's 2.0/11."""
    record = RequestRecord(128, 11, 1.0, 3.0, "", 200)
    assert record.tpot == pytest.approx(0.2)


@allure.feature("pysweep")
def test_tpot_is_undefined_for_a_single_token_answer():
    assert RequestRecord(128, 1, 1.0, 1.0, "", 200).tpot is None
    assert RequestRecord(128, 0, 1.0, 1.0, "", 200).tpot is None


@allure.feature("pysweep")
def test_percentile_interpolates_like_pandas():
    """Must match numpy's default method, as parse.py reports."""
    assert percentile(list(range(1, 11)), 0.9) == pytest.approx(9.1)
    assert percentile([5.0], 0.9) == 5.0
    assert percentile([], 0.9) is None


@allure.feature("pysweep")
def test_kpis_are_computed_over_the_recorded_requests(tmp_path):
    path = _result_file(
        tmp_path,
        [
            [128, 11, 1.0, 3.0, "", 200],
            [128, 11, 2.0, 6.0, "", 200],
        ],
    )
    summary = summarize_requests(read_requests(path))
    assert summary["requests"] == 2
    assert summary["mean_ttft"] == pytest.approx(1.5)
    assert summary["p90_ttft"] == pytest.approx(1.9)
    assert summary["mean_tpot"] == pytest.approx(0.3)
    assert summary["p90_tpot"] == pytest.approx(0.38)


@allure.feature("pysweep")
def test_unsuccessful_and_implausible_rows_are_excluded(tmp_path):
    path = _result_file(
        tmp_path,
        [
            [128, 11, 1.0, 3.0, "", 200],
            [128, 11, 1.0, 3.0, "", 500],
            [128, 11, 1.0, 3.0, "connection reset", 200],
            [128, 11, 0.0, 3.0, "", 200],  # no first chunk ever timed
            [128, 11, 5.0, 1.0, "", 200],  # overall precedes the first chunk
            ["not", "a", "row"],
        ],
    )
    records = read_requests(path)
    assert len(records) == 1
    assert summarize_requests(records)["requests"] == 1


@allure.feature("pysweep")
def test_a_run_that_recorded_nothing_still_gets_a_row(tmp_path):
    summary = SweepSummary(["client.concurrency"], path=tmp_path / "s.csv")
    row = summary.add_run({"client.concurrency": 128}, tmp_path / "missing.csv", error=1)
    assert row["requests"] == 0
    assert row["mean_ttft"] is None
    written = list(csv.DictReader((tmp_path / "s.csv").open()))
    assert written == [
        {
            "client.concurrency": "128",
            "requests": "0",
            "mean_ttft": "",
            "p90_ttft": "",
            "mean_tpot": "",
            "p90_tpot": "",
            "served_model": "",
            "error": "1",
            "result_file": "missing.csv",
        }
    ]


@allure.feature("pysweep")
def test_result_file_is_recorded_relative_to_the_summary(tmp_path):
    per_case = tmp_path / PER_CASE_DIR_NAME
    per_case.mkdir()
    result = _result_file(per_case, [[128, 11, 1.0, 3.0, "", 200]])
    summary = SweepSummary(["client.concurrency"], path=tmp_path / SUMMARY_FILE_NAME)

    row = summary.add_run({"client.concurrency": 4}, result)

    assert row["result_file"] == f"{PER_CASE_DIR_NAME}/bench_x.result.csv"
    assert (summary.path.parent / row["result_file"]).exists()


@allure.feature("pysweep")
def test_a_result_file_from_outside_the_directory_is_recorded_verbatim(tmp_path):
    elsewhere = _result_file(tmp_path, [[128, 11, 1.0, 3.0, "", 200]])
    root = tmp_path / "results"
    root.mkdir()
    summary = SweepSummary([], path=root / SUMMARY_FILE_NAME)

    row = summary.add_run({}, elsewhere)

    assert row["result_file"] == str(elsewhere)


@allure.feature("pysweep")
def test_summary_columns_are_the_declared_knobs_in_dot_notation(tmp_path):
    """Single-valued knobs get columns too."""
    cfg = load_config(
        {
            "client": {**CLIENT, "concurrency": [1, 16], "input_tokens": [128, 256]},
            "llms": {"max_new_tokens": [128, 256]},
        }
    )
    summary = SweepSummary(cfg.knob_paths(), path=tmp_path / "s.csv")
    assert summary.columns == [
        "llms.max_new_tokens",
        "client.concurrency",
        "client.duration",
        "client.failures_allowed",
        "client.input_tokens",
        "client.question_file",
        "client.warmup_requests",
        "client.worker_startup_delay",
        *SUMMARY_COLUMNS,
    ]


@allure.feature("pysweep")
def test_header_exists_before_the_first_run_finishes(tmp_path):
    path = tmp_path / "sweep_summary.csv"
    summary = SweepSummary(["client.concurrency"], path=path)
    assert path.exists()
    assert next(csv.reader(path.open())) == summary.columns
    assert list(csv.DictReader(path.open())) == []


@allure.feature("pysweep")
def test_summary_is_readable_after_every_run(tmp_path):
    path = tmp_path / "sweep_summary.csv"
    summary = SweepSummary(["client.concurrency"], path=path)
    for index, concurrency in enumerate((1, 16), start=1):
        result = _result_file(
            tmp_path, [[128, 11, 1.0, 3.0, "", 200]], name=f"bench_{concurrency}.csv"
        )
        summary.add_run({"client.concurrency": concurrency}, result)
        rows = list(csv.DictReader(path.open()))
        assert len(rows) == index
    assert [row["client.concurrency"] for row in rows] == ["1", "16"]
    assert [row["mean_tpot"] for row in rows] == ["0.2", "0.2"]


@allure.feature("pysweep")
def test_unset_knobs_render_blank_rather_than_shifting_columns(tmp_path):
    summary = SweepSummary(["llms.max_new_tokens", "client.concurrency"], path=tmp_path / "s.csv")
    row = summary.add_run({"client.concurrency": 4}, tmp_path / "none.csv")
    assert row["llms.max_new_tokens"] == ""
    header = next(csv.reader((tmp_path / "s.csv").open()))
    assert header[:2] == ["llms.max_new_tokens", "client.concurrency"]


@allure.feature("pysweep")
def test_the_output_layout_is_created_when_nothing_is_there(tmp_path):
    """Including intermediate directories."""
    layout = prepare_summaries_dir(tmp_path / "runs" / "today")

    assert layout.root.is_dir()
    assert layout.per_case_dir.is_dir()
    assert layout.summary_file == layout.root / SUMMARY_FILE_NAME
    assert layout.per_case_dir == layout.root / PER_CASE_DIR_NAME


@allure.feature("pysweep")
def test_the_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    layout = prepare_summaries_dir("results")

    assert layout.root.is_absolute()
    assert layout.summary_file.is_absolute()
    assert layout.per_case_dir.is_absolute()


@allure.feature("pysweep")
def test_the_previous_sweep_is_discarded(tmp_path):
    root = tmp_path / "results"
    per_case = root / PER_CASE_DIR_NAME
    per_case.mkdir(parents=True)
    (root / SUMMARY_FILE_NAME).write_text("stale header\n")
    (per_case / "bench_old.result.csv").write_text("stale\n")
    (per_case / "nested").mkdir()
    (per_case / "nested" / "deeper.csv").write_text("stale\n")

    layout = prepare_summaries_dir(root)

    assert not layout.summary_file.exists()
    assert list(layout.per_case_dir.iterdir()) == []


@allure.feature("pysweep")
def test_nothing_else_in_the_directory_is_touched(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / "sweep.yaml").write_text("client: {}\n")
    (root / "notes").mkdir()
    (root / "notes" / "why.md").write_text("because\n")

    prepare_summaries_dir(root)

    assert (root / "sweep.yaml").exists()
    assert (root / "notes" / "why.md").exists()


@allure.feature("pysweep")
def test_a_symlinked_per_case_directory_is_refused(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "precious.csv").write_text("keep\n")
    root = tmp_path / "results"
    root.mkdir()
    (root / PER_CASE_DIR_NAME).symlink_to(outside, target_is_directory=True)

    with pytest.raises(SummariesLayoutError):
        prepare_summaries_dir(root)

    assert (outside / "precious.csv").exists()


@allure.feature("pysweep")
def test_a_file_where_a_directory_belongs_is_refused(tmp_path):
    as_file = tmp_path / "results"
    as_file.write_text("not a directory\n")

    with pytest.raises(SummariesLayoutError):
        prepare_summaries_dir(as_file)

    assert as_file.read_text() == "not a directory\n"


@allure.feature("pysweep")
def test_nothing_is_deleted_when_the_layout_is_refused(tmp_path):
    root = tmp_path / "results"
    root.mkdir()
    (root / SUMMARY_FILE_NAME).write_text("previous sweep\n")
    (root / PER_CASE_DIR_NAME).write_text("not a directory\n")

    with pytest.raises(SummariesLayoutError):
        prepare_summaries_dir(root)

    assert (root / SUMMARY_FILE_NAME).read_text() == "previous sweep\n"
