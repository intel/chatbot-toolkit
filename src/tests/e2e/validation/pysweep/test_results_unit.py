# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the sweep summary post-processing (``pysweep/results.py``).

Pure computation over CSV files, so nothing here needs a cluster, transformers,
or a load test — the inputs are hand-written result files whose KPIs can be
checked by arithmetic.
"""

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
    """TPOT is the *inter-token* gap, so neither TTFT nor its token counts in it.

    With 11 output tokens, 1.0s to the first and 3.0s overall, decode produced
    the remaining 10 tokens in 2.0s: 0.2s per token. Dividing by 11 instead (as
    ``parse.py``'s next_token_lat does) would report 0.1818 and understate the
    decode cost, which is the distinction this KPI exists to make.
    """
    record = RequestRecord(128, 11, 1.0, 3.0, "", 200)
    assert record.tpot == pytest.approx(0.2)


@allure.feature("pysweep")
def test_tpot_is_undefined_for_a_single_token_answer():
    """One token means no gap between tokens; its whole latency is TTFT."""
    assert RequestRecord(128, 1, 1.0, 1.0, "", 200).tpot is None
    assert RequestRecord(128, 0, 1.0, 1.0, "", 200).tpot is None


@allure.feature("pysweep")
def test_percentile_interpolates_like_pandas():
    """p90 must agree with what parse.py reports, i.e. numpy's default method.

    For 1..10 the 0.9 quantile sits between the 9th and 10th values at 9.1 —
    not 9 (nearest-rank) and not 10 (ceiling).
    """
    assert percentile(list(range(1, 11)), 0.9) == pytest.approx(9.1)
    assert percentile([5.0], 0.9) == 5.0
    assert percentile([], 0.9) is None


@allure.feature("pysweep")
def test_kpis_are_computed_over_the_recorded_requests(tmp_path):
    """The four KPIs come out of the per-request rows by plain arithmetic.

    Two requests, TTFT 1.0 and 2.0 -> mean 1.5, p90 1.9 (interpolated). Each has
    11 output tokens and 2.0s/4.0s of decode after the first token -> TPOT 0.2
    and 0.4, mean 0.3, p90 0.38.
    """
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
    """A summary must describe successful requests only.

    A non-200, an errored row, one that never timed a first chunk, and a
    malformed line would each poison the mean if counted. Dropping them (rather
    than raising) also means one damaged line cannot cost an hours-long sweep its
    summary.
    """
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
    """"This configuration produced no data" is a result, not a reason to abort.

    Blank KPIs with ``requests`` 0 says exactly that; dropping the row would
    leave a hole in the sweep that looks like a configuration never tried.
    """
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
    """The pointer column is relative, so the results directory can be moved.

    A sweep runs on the benchmark host and is read somewhere else. An absolute
    path in the column would name a directory that only ever existed on the host;
    ``per_case_results/bench_x.result.csv`` still resolves after the directory is
    copied off it or renamed.
    """
    per_case = tmp_path / PER_CASE_DIR_NAME
    per_case.mkdir()
    result = _result_file(per_case, [[128, 11, 1.0, 3.0, "", 200]])
    summary = SweepSummary(["client.concurrency"], path=tmp_path / SUMMARY_FILE_NAME)

    row = summary.add_run({"client.concurrency": 4}, result)

    assert row["result_file"] == f"{PER_CASE_DIR_NAME}/bench_x.result.csv"
    assert (summary.path.parent / row["result_file"]).exists()


@allure.feature("pysweep")
def test_a_result_file_from_outside_the_directory_is_recorded_verbatim(tmp_path):
    """There is no meaningful relative name for it, so the absolute one is kept.

    Not a shape the sweep produces, but the column must stay resolvable rather
    than becoming a ``../..``-prefixed guess at where the reader will stand.
    """
    elsewhere = _result_file(tmp_path, [[128, 11, 1.0, 3.0, "", 200]])
    root = tmp_path / "results"
    root.mkdir()
    summary = SweepSummary([], path=root / SUMMARY_FILE_NAME)

    row = summary.add_run({}, elsewhere)

    assert row["result_file"] == str(elsewhere)


@allure.feature("pysweep")
def test_summary_columns_are_the_declared_knobs_in_dot_notation(tmp_path):
    """Header = every declared knob as its dotted config path, then the KPIs.

    Dotted paths are the point: the column name is the key the sweep YAML used,
    so a reader can go from a column back to the config with no lookup table.
    Constants get columns too (``llms.model`` is single-valued here), because a
    summary that only named the varying knobs would not say what the numbers
    were measured under.
    """
    cfg = load_config(
        {
            "client": {**CLIENT, "concurrency": [1, 16], "input_tokens": [128, 256]},
            "llms": {"max_new_tokens": [128, 256], "model": "some/model"},
        }
    )
    summary = SweepSummary(cfg.knob_paths(), path=tmp_path / "s.csv")
    assert summary.columns == [
        "llms.max_new_tokens",
        "llms.model",
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
    """A sweep's first row can be an hour away; the file must not be missing.

    The first run sits behind a multi-replica rollout, and until it completes a
    missing file is indistinguishable from a harness that never summarizes at
    all. A header-only file says "running, and here is the shape of the results".
    """
    path = tmp_path / "sweep_summary.csv"
    summary = SweepSummary(["client.concurrency"], path=path)
    assert path.exists()
    assert next(csv.reader(path.open())) == summary.columns
    assert list(csv.DictReader(path.open())) == []


@allure.feature("pysweep")
def test_summary_is_readable_after_every_run(tmp_path):
    """Rewritten per run, so a sweep killed at hour three still leaves a table.

    An hours-long sweep that only wrote its summary at the end would lose
    everything to a failed rollout in the middle.
    """
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
    """A knob with no value for a run leaves an empty cell.

    The columns are fixed from the config up front, so a missing value must not
    be able to misalign a row — that would silently attribute one knob's value
    to another column.
    """
    summary = SweepSummary(["llms.model", "client.concurrency"], path=tmp_path / "s.csv")
    row = summary.add_run({"client.concurrency": 4}, tmp_path / "none.csv")
    assert row["llms.model"] == ""
    header = next(csv.reader((tmp_path / "s.csv").open()))
    assert header[:2] == ["llms.model", "client.concurrency"]


@allure.feature("pysweep")
def test_the_output_layout_is_created_when_nothing_is_there(tmp_path):
    """A sweep must not fail because its output directory does not exist yet.

    That includes intermediate directories: ``--summaries runs/today`` is a
    reasonable thing to type on a fresh host.
    """
    layout = prepare_summaries_dir(tmp_path / "runs" / "today")

    assert layout.root.is_dir()
    assert layout.per_case_dir.is_dir()
    assert layout.summary_file == layout.root / SUMMARY_FILE_NAME
    assert layout.per_case_dir == layout.root / PER_CASE_DIR_NAME


@allure.feature("pysweep")
def test_the_paths_are_absolute(tmp_path, monkeypatch):
    """``benchmark.py`` opens its output with a bare ``open()``.

    A relative per-case path would land against the working directory of whatever
    process happened to do the opening, so the layout is resolved once, here.
    """
    monkeypatch.chdir(tmp_path)

    layout = prepare_summaries_dir("results")

    assert layout.root.is_absolute()
    assert layout.summary_file.is_absolute()
    assert layout.per_case_dir.is_absolute()


@allure.feature("pysweep")
def test_the_previous_sweep_is_discarded(tmp_path):
    """A sweep's artifacts only make sense as a set, so the slate is wiped clean.

    Keeping them would mix two sweeps: per-case CSVs with no row in the current
    summary, or — if the previous sweep was longer — rows in a stale summary
    pointing at files this one never overwrote.
    """
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
    """The deletion is confined to the summary and the per-case subdirectory.

    ``--summaries`` is a path a user types, and it may well point at a directory
    that already holds notes, a copy of the sweep YAML, or someone's plots. Only
    what this sweep is about to rewrite is removed.
    """
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
    """Emptying it would delete files outside the directory the caller named.

    The flag is destructive by design, but only within the tree it was pointed at;
    following a link out of that tree is not something to do quietly.
    """
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
    """``--summaries`` names a directory; a file there means the wrong path.

    Most likely the old ``--summary sweep_summary.csv`` habit. Failing beats
    guessing at the parent directory, and beats deleting the file.
    """
    as_file = tmp_path / "results"
    as_file.write_text("not a directory\n")

    with pytest.raises(SummariesLayoutError):
        prepare_summaries_dir(as_file)

    assert as_file.read_text() == "not a directory\n"


@allure.feature("pysweep")
def test_nothing_is_deleted_when_the_layout_is_refused(tmp_path):
    """The checks all run before the first deletion.

    Otherwise a directory that turns out to be unusable would still have lost the
    previous sweep's results by the time the error was raised.
    """
    root = tmp_path / "results"
    root.mkdir()
    (root / SUMMARY_FILE_NAME).write_text("previous sweep\n")
    (root / PER_CASE_DIR_NAME).write_text("not a directory\n")

    with pytest.raises(SummariesLayoutError):
        prepare_summaries_dir(root)

    assert (root / SUMMARY_FILE_NAME).read_text() == "previous sweep\n"
