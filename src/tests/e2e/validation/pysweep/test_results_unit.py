# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import json

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.results import (
    PER_CASE_DIR_NAME,
    RESULTS_FILE_NAME,
    SCHEMA_VERSION,
    SUMMARY_FILE_NAME,
    RequestRecord,
    SummariesLayoutError,
    SweepResults,
    flatten,
    nest,
    percentile,
    prepare_summaries_dir,
    read_requests,
    run_record,
    summarize_requests,
)

COMMIT = {
    "sha": "0123456789abcdef0123456789abcdef01234567",
    "datetime": "2026-11-25T18:02:44+01:00",
    "author": "A. Author",
    "description": "feat: something",
    "branch": "kkurzacz/pysweep-core",
}
METADATA = {"commit_info": COMMIT, "hardware": {"nodes": ["node-1", "node-2"], "cpu": {"family": "SPR"}}}

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
def test_results_are_computed_over_the_recorded_requests(tmp_path):
    path = _result_file(
        tmp_path,
        [
            [128, 11, 1.0, 3.0, "", 200],
            [128, 11, 2.0, 6.0, "", 200],
        ],
    )
    results = summarize_requests(read_requests(path), duration_s=10)
    assert results["requests"] == {"ok": 2}
    assert results["ttft_s"]["mean"] == pytest.approx(1.5)
    assert results["ttft_s"]["p50"] == pytest.approx(1.5)
    assert results["ttft_s"]["p90"] == pytest.approx(1.9)
    assert results["tpot_s"]["mean"] == pytest.approx(0.3)
    assert results["tpot_s"]["p90"] == pytest.approx(0.38)
    assert results["e2e_s"]["mean"] == pytest.approx(4.5)
    assert results["input_tokens"] == {"mean": 128}
    assert results["output_tokens"] == {"mean": 11, "total": 22}
    assert results["throughput"] == {"requests_per_s": 0.2, "output_tokens_per_s": 2.2}


@allure.feature("pysweep")
def test_results_keep_their_shape_when_nothing_was_recorded():
    results = summarize_requests([], duration_s=10)
    assert results["requests"] == {"ok": 0}
    assert results["ttft_s"] == {"mean": None, "p50": None, "p90": None, "p99": None}
    assert results["output_tokens"] == {"mean": None, "total": 0}
    assert results["throughput"] == {"requests_per_s": 0.0, "output_tokens_per_s": 0.0}


@allure.feature("pysweep")
def test_throughput_is_undefined_without_a_duration():
    results = summarize_requests([RequestRecord(128, 11, 1.0, 3.0, "", 200)], duration_s=0)
    assert results["throughput"] == {"requests_per_s": None, "output_tokens_per_s": None}


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
    assert summarize_requests(records, duration_s=10)["requests"] == {"ok": 1}


@allure.feature("pysweep")
def test_nest_and_flatten_are_inverse():
    dotted = {"llms.max_new_tokens": 128, "retriever.k": 5, "top": 1}
    assert nest(dotted) == {"llms": {"max_new_tokens": 128}, "retriever": {"k": 5}, "top": 1}
    assert flatten(nest(dotted)) == dotted


def _record(index="1.1", result_file="per_case_results/r.csv", concurrency=4, results=None):
    return run_record(
        index=index,
        started_at="2026-11-26T09:14:03Z",
        finished_at="2026-11-26T09:29:05Z",
        exit_code=0,
        result_file=result_file,
        deployment={"llms.max_new_tokens": 128},
        client={"client.concurrency": concurrency, "client.tokenizer": "org/tok"},
        results=results or summarize_requests([], duration_s=60),
    )


@allure.feature("pysweep")
def test_a_record_groups_settings_into_deployment_and_client():
    record = _record()
    assert list(record) == ["run", "settings", "results"]
    assert record["run"] == {
        "index": "1.1",
        "started_at": "2026-11-26T09:14:03Z",
        "finished_at": "2026-11-26T09:29:05Z",
        "exit_code": 0,
        "result_file": "per_case_results/r.csv",
    }
    assert record["settings"] == {
        "deployment": {"llms": {"max_new_tokens": 128}},
        "client": {"concurrency": 4, "tokenizer": "org/tok"},
    }


@allure.feature("pysweep")
def test_edp_settings_sit_next_to_deployment_and_client():
    record = run_record(
        index="1.1", started_at="s", finished_at="f", exit_code=0, result_file="r.csv",
        deployment={}, client={"client.concurrency": 1},
        results=summarize_requests([], duration_s=60),
        edp={"vector_db.num_vectors": 55312},
    )
    assert record["settings"]["edp"] == {"vector_db": {"num_vectors": 55312}}
    assert flatten(record)["settings.edp.vector_db.num_vectors"] == 55312


@allure.feature("pysweep")
def test_without_metadata_the_document_has_no_metadata_key(tmp_path):
    layout = prepare_summaries_dir(tmp_path / "out")
    sweep = SweepResults(layout, date="d", metadata=None)
    sweep.add_run(_record())
    assert "metadata" not in json.loads(layout.results_file.read_text())
    header = next(csv.reader(layout.summary_file.open()))
    assert header[:2] == ["date", "run.index"]


@pytest.fixture
def layout(tmp_path):
    return prepare_summaries_dir(tmp_path / "out")


@allure.feature("pysweep")
def test_results_json_exists_before_the_first_run_finishes(layout):
    SweepResults(layout, date="2026-11-26T09:02:11Z", metadata=METADATA)
    assert json.loads(layout.results_file.read_text()) == {
        "schema_version": SCHEMA_VERSION,
        "date": "2026-11-26T09:02:11Z",
        "metadata": METADATA,
        "runs": [],
    }


@allure.feature("pysweep")
def test_results_json_holds_every_run_after_each_one(layout):
    sweep = SweepResults(layout, date="2026-11-26T09:02:11Z", metadata=METADATA)
    for index in (1, 2):
        sweep.add_run(_record(index=f"1.{index}"))
        document = json.loads(layout.results_file.read_text())
        assert [run["run"]["index"] for run in document["runs"]] == [f"1.{i}" for i in range(1, index + 1)]


@allure.feature("pysweep")
def test_the_csv_has_one_dotted_row_per_run(layout):
    sweep = SweepResults(layout, date="2026-11-26T09:02:11Z", metadata=METADATA)
    sweep.add_run(_record(concurrency=1))
    sweep.add_run(_record(index="1.2", concurrency=16))

    rows = list(csv.DictReader(layout.summary_file.open()))
    assert [row["settings.client.concurrency"] for row in rows] == ["1", "16"]
    header = next(csv.reader(layout.summary_file.open()))
    assert header[:9] == [
        "date",
        "metadata.commit_info.sha",
        "metadata.commit_info.datetime",
        "metadata.commit_info.author",
        "metadata.commit_info.description",
        "metadata.commit_info.branch",
        "metadata.hardware.nodes",
        "metadata.hardware.cpu.family",
        "run.index",
    ]
    assert "settings.deployment.llms.max_new_tokens" in header
    assert "results.ttft_s.p90" in header
    assert rows[0]["metadata.commit_info.branch"] == "kkurzacz/pysweep-core"
    assert rows[0]["metadata.hardware.nodes"] == "node-1;node-2"


@allure.feature("pysweep")
def test_missing_values_render_blank_in_the_csv(layout):
    sweep = SweepResults(layout, date="2026-11-26T09:02:11Z", metadata=METADATA)
    sweep.add_run(_record())
    (row,) = csv.DictReader(layout.summary_file.open())
    assert row["results.ttft_s.mean"] == ""
    assert row["results.requests.ok"] == "0"


@allure.feature("pysweep")
def test_result_file_pointer_is_relative_to_the_summaries_directory(layout):
    sweep = SweepResults(layout, date="d", metadata=METADATA)
    inside = layout.per_case_dir / "bench_x.result.csv"
    assert sweep.pointer(inside) == f"{PER_CASE_DIR_NAME}/bench_x.result.csv"


@allure.feature("pysweep")
def test_a_result_file_from_outside_the_directory_is_recorded_verbatim(layout, tmp_path):
    sweep = SweepResults(layout, date="d", metadata=METADATA)
    elsewhere = tmp_path / "elsewhere.csv"
    assert sweep.pointer(elsewhere) == str(elsewhere)


@allure.feature("pysweep")
def test_the_output_layout_is_created_when_nothing_is_there(tmp_path):
    """Including intermediate directories."""
    layout = prepare_summaries_dir(tmp_path / "runs" / "today")

    assert layout.root.is_dir()
    assert layout.per_case_dir.is_dir()
    assert layout.summary_file == layout.root / SUMMARY_FILE_NAME
    assert layout.results_file == layout.root / RESULTS_FILE_NAME
    assert layout.per_case_dir == layout.root / PER_CASE_DIR_NAME


@allure.feature("pysweep")
def test_the_paths_are_absolute(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    layout = prepare_summaries_dir("results")

    assert layout.root.is_absolute()
    assert layout.summary_file.is_absolute()
    assert layout.results_file.is_absolute()
    assert layout.per_case_dir.is_absolute()


@allure.feature("pysweep")
def test_the_previous_sweep_is_discarded(tmp_path):
    root = tmp_path / "results"
    per_case = root / PER_CASE_DIR_NAME
    per_case.mkdir(parents=True)
    (root / SUMMARY_FILE_NAME).write_text("stale header\n")
    (root / RESULTS_FILE_NAME).write_text("{}\n")
    (per_case / "bench_old.result.csv").write_text("stale\n")
    (per_case / "nested").mkdir()
    (per_case / "nested" / "deeper.csv").write_text("stale\n")

    layout = prepare_summaries_dir(root)

    assert not layout.summary_file.exists()
    assert not layout.results_file.exists()
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
