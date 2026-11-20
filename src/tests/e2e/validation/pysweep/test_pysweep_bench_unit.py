# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pysweep_bench orchestration.

These cover the glue logic — argument splitting, property building, the
apply/converge/verify step, and the nesting of the client loop inside the
deployment loop — with a fake fingerprint client, so no cluster, transformers, or
benchmark load is needed. The actual benchmark run (``benchmark.run_benchmark``)
is exercised by the existing benchmark tooling, not re-tested here.
"""

import argparse
import ast
import csv
import itertools
import logging
from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import pysweep_bench as psb
from tests.e2e.benchmarks.chatqna.pysweep.base import PropertyApplyError, PropertyReadError
from tests.e2e.benchmarks.chatqna.pysweep.client_args import BenchmarkClientArgs
from tests.e2e.benchmarks.chatqna.pysweep import results
from tests.e2e.benchmarks.chatqna.pysweep.config import load_config
from tests.e2e.benchmarks.chatqna.pysweep.properties.llm_model import LlmModel
from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens

# The client section is mandatory; these tests care about the deployment side, so
# it stays fixed and minimal except where a test overrides a knob.
#
# question_file is the real corpus path (relative to pysweep_bench.py, hence the
# '../'), because the tests that call main() run preflight(), which
# resolves and stats it. A placeholder name would fail there rather than in the
# logic under test.
CLIENT = {
    "question_file": "../questions-short.csv",
    "concurrency": 4,
    "duration": "3m",
    "worker_startup_delay": "0s",
    "failures_allowed": 200,
    "input_tokens": 128,
    "warmup_requests": 1,
}


def _sweep(**sections):
    return {"client": dict(CLIENT), **sections}


def _client(**overrides):
    return {"client": {**CLIENT, **overrides}}


def _parse_benchmark_argv(argv):
    """Parse argv with a parser rebuilt from ``benchmark.parse_args``'s source.

    Predates the lazy ``transformers`` import inside ``run_benchmark``: importing
    ``benchmark`` used to mean importing ``transformers``, which this suite does not
    install, so the parser was rebuilt from the file instead. That is no longer a
    constraint — this could call ``benchmark.parse_args`` directly — but rebuilding
    from the real source is still a genuine check that the adapter's argv is
    acceptable: a flag or type the tool does not actually declare would not be
    reconstructed here either.
    """
    source = (
        Path(__file__).resolve().parents[3] / "e2e/benchmarks/chatqna/benchmark.py"
    ).read_text()
    tree = ast.parse(source)
    func = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    parser = argparse.ArgumentParser()
    for node in ast.walk(func):
        if not (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr == "add_argument"
        ):
            continue
        flag = node.args[0].value
        kwargs = {kw.arg: kw.value for kw in node.keywords}
        arg_type = {"int": int, "str": str}[kwargs["type"].id]
        default = ast.literal_eval(kwargs["default"]) if "default" in kwargs else None
        parser.add_argument(flag, type=arg_type, default=default)
    return parser.parse_args(argv)


class FakeFingerprintClient:
    """In-memory stand-in for FingerprintClient that actually applies writes.

    The real reader is per-component; the properties under test here all read the
    values of the component they write, so one dict is enough and the component
    name is ignored.
    """

    def __init__(self, initial=999):
        self.params = {"max_new_tokens": initial}

    def read_values(self, component):
        return dict(self.params)

    def change_arguments(self, body):
        for entry in body:
            self.params.update(entry["data"])


class IgnoringClient(FakeFingerprintClient):
    """Client whose writes never take effect (simulates a non-converging plane)."""

    def change_arguments(self, body):
        pass


@allure.feature("pysweep")
def test_parse_args_splits_benchmark_args_after_double_dash():
    args = psb.parse_args(["--sweep", "s.yaml", "--", "-s", "http://x", "-c", "32"])
    assert args.sweep == "s.yaml"
    assert args.benchmark_args == ["-s", "http://x", "-c", "32"]


@allure.feature("pysweep")
def test_parse_args_without_benchmark_args():
    args = psb.parse_args(["--sweep", "s.yaml"])
    assert args.benchmark_args == []


@allure.feature("pysweep")
def test_parse_args_run_config_is_optional():
    """``--sweep`` alone is valid; ``--config`` is the optional run config."""
    assert psb.parse_args(["--sweep", "s.yaml"]).config is None
    args = psb.parse_args(["--sweep", "s.yaml", "--config", "c.yaml"])
    assert (args.sweep, args.config) == ("s.yaml", "c.yaml")


@allure.feature("pysweep")
def test_build_property_dispatches_runtime_api_to_fingerprint_client():
    cfg = load_config(_sweep(llms={"max_new_tokens": 128}))
    setting = cfg.deployment_combinations()[0][0]
    prop = psb.build_property(setting, FakeFingerprintClient())
    assert isinstance(prop, MaxNewTokens)


@allure.feature("pysweep")
def test_build_property_dispatches_k8s_plane_without_fingerprint_client():
    """K8s-plane properties reach the cluster themselves, so take no client."""
    cfg = load_config(_sweep(llms={"model": "some/model"}))
    setting = cfg.deployment_combinations()[0][0]
    prop = psb.build_property(setting, FakeFingerprintClient())
    assert isinstance(prop, LlmModel)


@allure.feature("pysweep")
def test_apply_combination_reports_unestablishable_state_as_sweep_error(monkeypatch):
    """A property that cannot establish its value aborts the sweep cleanly.

    E.g. a model the deployment has no profile for: apply raises
    PropertyApplyError and apply_combination must surface it as a SweepError
    naming the knob, rather than letting it escape as a crash.
    """
    cfg = load_config(_sweep(llms={"model": "some/other-model"}))

    def refuse(self, desired):
        raise PropertyApplyError("no profile for that model")

    monkeypatch.setattr(LlmModel, "apply", refuse)
    with pytest.raises(psb.SweepError, match="llms.model"):
        psb.apply_combination(cfg.deployment_combinations()[0], FakeFingerprintClient())


@allure.feature("pysweep")
def test_apply_combination_reports_failed_convergence_as_sweep_error(monkeypatch):
    """Applied but never ready must abort, not benchmark the old config."""
    cfg = load_config(_sweep(llms={"model": "some/other-model"}))
    monkeypatch.setattr(LlmModel, "apply", lambda self, desired: None)

    def never_ready(self, desired, timeout=None):
        raise PropertyApplyError("did not converge")

    monkeypatch.setattr(LlmModel, "wait_until_ready", never_ready)
    with pytest.raises(psb.SweepError, match="llms.model"):
        psb.apply_combination(cfg.deployment_combinations()[0], FakeFingerprintClient())


@allure.feature("pysweep")
def test_apply_combination_applies_and_snapshots_observed():
    cfg = load_config(_sweep(llms={"max_new_tokens": [128, 256]}))
    client = FakeFingerprintClient()
    snapshots = [psb.apply_combination(c, client) for c in cfg.deployment_combinations()]
    assert snapshots == [
        {"llms.max_new_tokens": 128},
        {"llms.max_new_tokens": 256},
    ]


@allure.feature("pysweep")
def test_apply_combination_raises_when_not_converged():
    cfg = load_config(_sweep(llms={"max_new_tokens": 128}))
    with pytest.raises(psb.SweepError):
        psb.apply_combination(cfg.deployment_combinations()[0], IgnoringClient())


@allure.feature("pysweep")
def test_combination_slug_is_unique_per_combination():
    """Each combination must map to a distinct result filename fragment.

    Regression guard: combinations previously shared one timestamped CSV, so a
    later combination silently clobbered an earlier one's results.
    """
    cfg = load_config(_sweep(llms={"max_new_tokens": [128, 256]}))
    slugs = [psb.combination_slug(c) for c in cfg.deployment_combinations()]
    assert slugs == ["max_new_tokens-128", "max_new_tokens-256"]
    assert len(set(slugs)) == len(slugs)


# -- client knobs: the innermost loop -----------------------------------------


class FakeBenchmark:
    """Stand-in for the benchmark module, recording each invocation.

    Lets the nesting be asserted without transformers, a cluster, or minutes of
    load: what matters is *which* argv each run got and in what order.
    """

    def __init__(self, rows=1):
        self.calls = []
        self._rows = rows

    #: Mirrors benchmark.parse_args's signature without importing it (that pulls
    #: in transformers, which only a real load test needs). The faithfulness of
    #: the mirror is what test_rendered_argv_is_accepted_by_benchmark_parse_args
    #: checks against the real parser.
    parse_args = staticmethod(lambda argv: _parse_benchmark_argv(argv))

    def run_benchmark(self, args, output_file=None):
        self.calls.append((args, output_file))
        if self._rows:
            Path(output_file).write_text("128,128,0.8,6.9,,200\n" * self._rows)
        return 0, output_file


@allure.feature("pysweep")
def test_client_combination_renders_to_benchmark_argv():
    """Every declared client knob reaches benchmark.py as its own flag.

    ``-m`` is absent by design: it is not declared, so the client combination
    cannot carry it. The runner appends it from the served model (see
    ``test_tokenizer_comes_from_the_served_model``).
    """
    cfg = load_config(_client(concurrency=16))
    argv = BenchmarkClientArgs.from_settings(cfg.client_combinations()[0]).to_argv()
    # The DECLARED path, verbatim: the adapter is a pure translation. Anchoring it
    # to an absolute path happens later, at the argv boundary
    # (``resolve_question_file_argv``), so that the recorded value stays the one
    # the config author wrote.
    assert argv == [
        "-f", "../questions-short.csv",
        "-c", "16",
        "-d", "3m",
        "-u", "0s",
        "-e", "200",
        "-x", "128",
        "-w", "1",
    ]


@allure.feature("pysweep")
def test_rendered_argv_is_accepted_by_benchmark_parse_args():
    """The adapter's output must actually parse — the mirror has to be faithful.

    Types matter as much as flag names: every value is rendered through ``str``,
    so a knob bound to an option benchmark.py declares as ``type=int`` has to
    round-trip back to an int.
    """
    cfg = load_config(_client(concurrency=16, input_tokens=0))
    argv = BenchmarkClientArgs.from_settings(cfg.client_combinations()[0]).to_argv()
    parsed = _parse_benchmark_argv(argv)
    assert (parsed.c, parsed.d, parsed.x, parsed.f) == (16, "3m", 0, "../questions-short.csv")
    assert (parsed.e, parsed.w, parsed.u) == (200, 1, "0s")


@allure.feature("pysweep")
def test_client_knobs_vary_innermost(monkeypatch, tmp_path):
    """The whole point of the nesting: load varies within a fixed deployment.

    Each deployment combination must be applied once and then benchmarked under
    every client combination before the next (expensive) deployment change.
    """
    monkeypatch.chdir(tmp_path)
    cfg = load_config(
        {"client": {**CLIENT, "concurrency": [1, 16]}, "llms": {"max_new_tokens": [128, 256]}}
    )
    fake = FakeBenchmark()
    applied = []

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)

    def record_apply(combination, client):
        applied.append(tuple(s.value for s in combination))
        return {s.path: s.value for s in combination}

    monkeypatch.setattr(psb, "apply_combination", record_apply)
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda snapshot: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    # Two deployment combinations, applied once each — not once per load level.
    assert applied == [(128,), (256,)]
    # ...each benchmarked under both concurrencies before the next apply.
    assert [args.c for args, _ in fake.calls] == [1, 16, 1, 16]
    assert [args.d for args, _ in fake.calls] == ["3m"] * 4
    # The derived tokenizer reaches every run, though no config declares it.
    assert [args.m for args, _ in fake.calls] == ["served/model"] * 4


@allure.feature("pysweep")
def test_sweep_writes_one_summary_row_per_run(monkeypatch, tmp_path):
    """The sweep's own artifact: every run, its knob values, and its KPIs.

    Checked through ``main`` rather than against ``SweepSummary`` directly
    (``test_results.py`` covers the arithmetic), because what could realistically
    break here is the wiring: a row per run, the observed deployment value and
    the declared client value in their dotted-path columns, and the served model
    recorded even though nothing declares it.
    """
    monkeypatch.chdir(tmp_path)
    cfg = load_config(
        {"client": {**CLIENT, "concurrency": [1, 16]}, "llms": {"max_new_tokens": [128, 256]}}
    )
    fake = FakeBenchmark(rows=2)

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(
        psb, "apply_combination", lambda combination, client: {s.path: s.value for s in combination}
    )
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda snapshot: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml", "--summaries", "out"]) == 0

    summary_file = tmp_path / "out" / results.SUMMARY_FILE_NAME
    rows = list(csv.DictReader(summary_file.open()))
    assert len(rows) == 4
    assert [(r["llms.max_new_tokens"], r["client.concurrency"]) for r in rows] == [
        ("128", "1"), ("128", "16"), ("256", "1"), ("256", "16"),
    ]
    # Constant knobs get columns too, so a row says what it was measured under.
    assert {r["client.duration"] for r in rows} == {"3m"}
    # FakeBenchmark writes 128,128,0.8,6.9,,200 twice per run.
    assert {r["requests"] for r in rows} == {"2"}
    assert {r["mean_ttft"] for r in rows} == {"0.8"}
    assert {r["mean_tpot"] for r in rows} == {str(round((6.9 - 0.8) / 127, 6))}
    assert {r["served_model"] for r in rows} == {"served/model"}
    # And each row points back at the per-request CSV it summarises, by a path
    # relative to the summary itself — so the whole directory can be copied off the
    # benchmark host with its pointers intact.
    assert all(not Path(r["result_file"]).is_absolute() for r in rows)
    assert all((summary_file.parent / r["result_file"]).exists() for r in rows)
    # ...and that CSV lives in the per-case subdirectory, not beside the summary.
    assert {Path(r["result_file"]).parent.name for r in rows} == {results.PER_CASE_DIR_NAME}


@allure.feature("pysweep")
def test_the_previous_sweeps_results_are_gone_before_this_one_starts(monkeypatch, tmp_path):
    """``--summaries`` is a clean slate per sweep, wired through ``main``.

    ``test_results.py`` covers what gets deleted; what matters here is that the
    sweep does it at all. Without it, a per-case CSV from a longer previous sweep
    survives with no row in the new summary, and reads exactly like a result of it.
    """
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_client())
    stale = tmp_path / "out" / results.PER_CASE_DIR_NAME / "bench_from_last_week.result.csv"
    stale.parent.mkdir(parents=True)
    stale.write_text("128,128,9.9,99.9,,200\n")

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: {})
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda snapshot: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, FakeBenchmark())

    assert psb.main(["--sweep", "ignored.yaml", "--summaries", "out"]) == 0

    assert not stale.exists()
    # And this sweep's own run is there, so the wipe happened before it, not after.
    assert [p.name for p in stale.parent.iterdir()] == ["bench_single.result.csv"]


def _passthrough_provision(benchmark_args, workers=1):
    """provision_benchmark_target with no cluster: yield the args unchanged.

    ``workers`` is accepted and ignored — nothing is minted here, but the runner
    passes the sweep's widest concurrency, so the signature has to match.
    """
    from contextlib import contextmanager

    @contextmanager
    def cm():
        yield list(benchmark_args) + ["-s", "http://127.0.0.1:1/api", "-b", "uat.txt"]

    return cm()


def _install_fake_benchmark(monkeypatch, fake):
    """Point the runner's module-level ``benchmark`` at ``fake``.

    Without it, a test that runs ``main()`` would start a real load test against
    whatever ``-s`` happened to be set. One binding is enough: ``main()`` reads the
    name off this module, and monkeypatch restores it afterwards.
    """
    monkeypatch.setattr(psb, "benchmark", fake)


@allure.feature("pysweep")
def test_result_file_names_are_unique_per_run():
    """Deployment x client must not collide: each run gets its own CSV.

    Regression guard in the same spirit as the per-combination slug — with the
    client loop added, a name built from the deployment alone would have every
    load level overwrite the previous one's results.
    """
    cfg = load_config(
        {"client": {**CLIENT, "concurrency": [1, 16]}, "llms": {"max_new_tokens": [128, 256]}}
    )
    varying = cfg.varying_paths()
    names = [
        psb.result_file_name(c, cc, only=varying)
        for c in cfg.deployment_combinations()
        for cc in cfg.client_combinations()
    ]
    assert names == [
        "bench_max_new_tokens-128_concurrency-1.result.csv",
        "bench_max_new_tokens-128_concurrency-16.result.csv",
        "bench_max_new_tokens-256_concurrency-1.result.csv",
        "bench_max_new_tokens-256_concurrency-16.result.csv",
    ]
    assert len(set(names)) == len(names)


@allure.feature("pysweep")
def test_result_file_name_omits_constant_knobs():
    """Only the knobs that vary are named; the rest are constant and logged."""
    cfg = load_config(_client(concurrency=[1, 16]))
    name = psb.result_file_name((), cfg.client_combinations()[0], only=cfg.varying_paths())
    assert name == "bench_concurrency-1.result.csv"


@allure.feature("pysweep")
def test_result_file_name_stays_within_the_filesystem_limit():
    """Naming every declared knob can overflow the 255-byte filename limit.

    Why names carry only the varying knobs. The values here are all plausible —
    an HF repo id and a question file under a real path — and together they push
    the all-knobs name past what the filesystem accepts, while the filtered name
    stays short regardless of how many knobs are held constant.
    """
    cfg = load_config(
        {
            "client": {
                **CLIENT,
                "concurrency": [1, 16],
                "question_file": "datasets/pubmed/questions-pubmed-long-form-2026-01.csv",
            },
            "llms": {
                "model": "hugging-quants/Meta-Llama-3.1-8B-Instruct-AWQ-INT4",
                "max_new_tokens": [128, 256],
            },
        }
    )
    combination = cfg.deployment_combinations()[0]
    client_combination = cfg.client_combinations()[0]

    assert len(Path(psb.result_file_name(combination, client_combination)).name) > 255
    filtered = psb.result_file_name(combination, client_combination, only=cfg.varying_paths())
    assert len(Path(filtered).name) <= 255
    # And the shipped example, whatever it declares, must also be nameable.
    example = (
        Path(__file__).resolve().parents[3]
        / "e2e/benchmarks/chatqna/pysweep/examples/sweep.yaml"
    )
    shipped = load_config(example)
    for combo in shipped.deployment_combinations():
        for client_combo in shipped.client_combinations():
            name = psb.result_file_name(combo, client_combo, only=shipped.varying_paths())
            assert len(Path(name).name) <= 255


@allure.feature("pysweep")
def test_result_file_name_falls_back_when_nothing_varies():
    """A single-combination sweep still needs a name."""
    cfg = load_config(_client())
    name = psb.result_file_name((), cfg.client_combinations()[0], only=cfg.varying_paths())
    assert name == "bench_single.result.csv"


@allure.feature("pysweep")
def test_tokenizer_comes_from_the_observed_model_in_the_snapshot(monkeypatch):
    """When the sweep varies llms.model, the snapshot already holds the answer.

    It is the value read back from the pipeline after convergence, so using it
    needs no second read and cannot disagree with what is actually loaded.
    """
    monkeypatch.setattr(psb.model_catalog, "tokenizer_for", lambda name: f"hf/{name}")
    assert psb.resolve_tokenizer({"llms.model": "llama3-8b-awq"}) == "hf/llama3-8b-awq"
    assert (
        psb.resolve_tokenizer({"llms.max_new_tokens": 128, "llms.model": "qwen3-8b"})
        == "hf/qwen3-8b"
    )


@allure.feature("pysweep")
def test_tokenizer_is_the_catalog_model_id_not_the_served_name(monkeypatch, tmp_path):
    """The two names differ, and using the wrong one breaks every token count.

    What the deployment serves is a catalog name ('llama3-8b-awq' — the value of
    LLM_MODEL_NAME and the name of the LLMInferenceService); AutoTokenizer needs a
    HuggingFace repo id. This goes through the real catalog lookup, so a wiring
    mistake that passed the served name straight to -m fails here.
    """
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        "models:\n"
        "  - name: llama3-8b-awq\n"
        "    model_id: casperhansen/llama-3-8b-instruct-awq\n"
        "    category: llm\n"
    )
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(catalog))

    assert (
        psb.resolve_tokenizer({"llms.model": "llama3-8b-awq"})
        == "casperhansen/llama-3-8b-instruct-awq"
    )


@allure.feature("pysweep")
def test_tokenizer_is_read_from_the_deployment_when_the_model_is_not_swept(monkeypatch):
    """A sweep that does not touch the model still needs to know what is served."""
    monkeypatch.setattr(LlmModel, "read", lambda self: "llama3-8b-awq")
    monkeypatch.setattr(psb.model_catalog, "tokenizer_for", lambda name: f"hf/{name}")
    assert psb.resolve_tokenizer({"llms.max_new_tokens": 128}) == "hf/llama3-8b-awq"


@allure.feature("pysweep")
def test_unreadable_model_aborts_rather_than_guessing_a_tokenizer(monkeypatch):
    """No served model means no defensible token counts, so the sweep stops.

    Falling back to benchmark.py's default (an embedding model's tokenizer) would
    produce numbers on a basis nobody chose, which is worse than failing.
    """
    def unreadable(self):
        raise PropertyReadError("no llm-usvc pods")

    monkeypatch.setattr(LlmModel, "read", unreadable)
    with pytest.raises(psb.SweepError, match="served model"):
        psb.resolve_tokenizer({"llms.max_new_tokens": 128})


@allure.feature("pysweep")
def test_a_served_model_missing_from_the_catalog_aborts(monkeypatch, tmp_path):
    """Same reasoning one step later: a name with no model_id has no tokenizer.

    This is reachable without a config typo — the deployment may serve a model the
    local catalog does not list — so it must fail rather than tokenize on a guess.
    """
    catalog = tmp_path / "models.yaml"
    catalog.write_text("models:\n  - name: other-model\n    model_id: org/other\n")
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(catalog))

    with pytest.raises(psb.SweepError, match="tokenizer"):
        psb.resolve_tokenizer({"llms.model": "llama3-8b-awq"})


@allure.feature("pysweep")
def test_forwarded_tokenizer_flag_is_refused():
    """``-m`` cannot be passed after ``--``: it is derived, not configurable.

    Unlike -s/-b there is no "omit it from the config to pass it here" escape —
    there is no config key at all, and any value other than the served model
    misattributes the token counts.
    """
    cfg = load_config(_client())
    with pytest.raises(psb.SweepError, match=r"-m"):
        psb.resolve_benchmark_argv(cfg, ["-m", "nomic-ai/nomic-embed-text-v1"])


@allure.feature("pysweep")
def test_forwarded_flag_that_the_config_declares_is_refused():
    """A load flag in both places has no defensible winner, so it is an error.

    Honouring the YAML would ignore an explicit command line; honouring the
    command line would silently collapse a declared sweep dimension. Both leave
    "what was tested" unanswerable, so the sweep refuses to start.
    """
    cfg = load_config(_client(concurrency=[1, 16]))
    with pytest.raises(psb.SweepError, match=r"-c"):
        psb.resolve_benchmark_argv(cfg, ["-c", "8"])


@allure.feature("pysweep")
def test_forwarded_server_is_allowed_when_the_config_omits_it():
    """Passing -s/-b on the command line is the supported provisioning bypass."""
    cfg = load_config(_client())
    assert psb.resolve_benchmark_argv(cfg, ["-s", "https://erag.com/api/v1/chatqna"]) == [
        "-s", "https://erag.com/api/v1/chatqna",
    ]


@allure.feature("pysweep")
def test_config_declared_server_and_bearer_reach_benchmark():
    """Declared connection settings render once and suppress auto-provisioning."""
    cfg = load_config(_client(server="https://erag.com/api/v1/chatqna", bearer_file="uat.txt"))
    argv = psb.resolve_benchmark_argv(cfg, [])
    assert argv == ["-s", "https://erag.com/api/v1/chatqna", "-b", "uat.txt"]
    # provision_benchmark_target sees both as supplied and touches neither.
    with psb.provision_benchmark_target(argv) as provisioned:
        assert provisioned == argv


@allure.feature("pysweep")
def test_count_result_rows_counts_and_tolerates_missing(tmp_path):
    """An absent file counts as zero rows rather than raising."""
    assert psb.count_result_rows(tmp_path / "nope.csv") == 0
    csv_file = tmp_path / "r.csv"
    csv_file.write_text("128,128,0.8,6.9,,200\n128,128,1.2,7.1,,200\n")
    assert psb.count_result_rows(csv_file) == 2


@allure.feature("pysweep")
def _fake_keycloak(monkeypatch, domain="box.example"):
    """Point provisioning at a fake deployment whose every token mint differs."""
    from tests.e2e.benchmarks.chatqna.pysweep import authentication

    minted = itertools.count(1)

    # access_token must yield a *fresh* value per read, as the real property does
    # (each read is a new password grant), which is what makes the tokens distinct.
    class _Helper:
        @property
        def access_token(self):
            return f"token-{next(minted)}"

    monkeypatch.setattr(
        authentication, "load_build_config", lambda: {"base_domain_name": domain}
    )
    monkeypatch.setattr(authentication, "keycloak_helper", _Helper)


@allure.feature("pysweep")
def test_provision_defaults_to_the_deployments_chatqna_endpoint(monkeypatch):
    """With no -s, the endpoint is built from the deployment's own domain.

    The host must come from the build config rather than a constant, or a sweep
    would drive load at a different cluster than the one it configures.
    """
    _fake_keycloak(monkeypatch)
    with psb.provision_benchmark_target([]) as args:
        assert args[args.index("-s") + 1] == "https://box.example/api/v1/chatqna"


@allure.feature("pysweep")
def test_provision_mints_one_distinct_token_per_worker(monkeypatch):
    """The bearer file holds ``workers`` *different* tokens, one per line.

    benchmark.py quits a worker whose line is empty, so the file must cover every
    worker index. They must also be distinct: APISIX rate-limits /api/v1/chatqna
    per Authorization header value, so one repeated token would put all workers in
    a single bucket and cap the sweep's throughput regardless of concurrency.
    """
    _fake_keycloak(monkeypatch)
    with psb.provision_benchmark_target([], workers=8) as args:
        lines = Path(args[args.index("-b") + 1]).read_text().splitlines()
        assert len(lines) == 8
        assert len(set(lines)) == 8, "workers must not share a rate-limit bucket"
        assert all(line for line in lines), "an empty line makes that worker quit"


@allure.feature("pysweep")
def test_configure_kubernetes_client_logs_quiets_httpx():
    """Disabled: the kr8s/httpx loggers stop emitting INFO but keep warnings."""
    levels = {n: logging.getLogger(n).level for n in psb.KUBERNETES_CLIENT_LOGGERS}
    asyncio_log = logging.getLogger("asyncio")
    installed = list(asyncio_log.filters)
    try:
        psb.configure_kubernetes_client_logs(enabled=False)
        for name in psb.KUBERNETES_CLIENT_LOGGERS:
            log = logging.getLogger(name)
            assert not log.isEnabledFor(logging.INFO)
            assert log.isEnabledFor(logging.WARNING)
        # asyncio keeps its level (real errors matter) but gains the noise filter.
        assert asyncio_log.isEnabledFor(logging.ERROR)
        assert any(
            isinstance(f, psb._DropPortForwardTeardownNoise) for f in asyncio_log.filters
        )
    finally:
        for name, level in levels.items():
            logging.getLogger(name).setLevel(level)
        asyncio_log.filters = installed


@allure.feature("pysweep")
def test_kubernetes_client_log_filter_drops_only_teardown_noise():
    """kr8s's port-forward resets are dropped; other asyncio errors survive."""
    log_filter = psb._DropPortForwardTeardownNoise()

    def record(msg, exc=None):
        rec = logging.LogRecord("asyncio", logging.ERROR, __file__, 1, msg, (), None)
        if exc is not None:
            rec.exc_info = (type(exc), exc, None)
        return rec

    assert not log_filter.filter(record("Unhandled exception in client_connected_cb"))
    assert not log_filter.filter(record("boom", ConnectionResetError("Connection lost")))
    # A genuine problem must still be reported.
    assert log_filter.filter(record("Task exception was never retrieved"))


@allure.feature("pysweep")
def test_configure_kubernetes_client_logs_enabled_is_a_noop():
    """Enabled: levels are left exactly as the caller/CLI configured them."""
    levels = {n: logging.getLogger(n).level for n in psb.KUBERNETES_CLIENT_LOGGERS}
    psb.configure_kubernetes_client_logs(enabled=True)
    assert {n: logging.getLogger(n).level for n in psb.KUBERNETES_CLIENT_LOGGERS} == levels


@allure.feature("pysweep")
def test_provision_preserves_caller_supplied_server_and_bearer(tmp_path):
    """Explicit -s/-b are left untouched (no port-forward, no token minting)."""
    uat = tmp_path / "uat.txt"
    uat.write_text("real-token\n")
    original = ["-s", "https://erag.com/api/v1/chatqna", "-b", str(uat), "-c", "8"]
    with psb.provision_benchmark_target(list(original)) as args:
        assert args == original


# -- preflight: question_file resolution & verification -----------------------
# The CWD-relative bug these cover: benchmark.py opens question_file with a bare
# open(), so before anchoring, the documented repo-root invocation could not find
# the corpus at all, and a same-named file in some other CWD would have been used
# silently while the summary recorded only the declared string.


@allure.feature("pysweep")
def test_question_file_resolves_against_pysweep_dir_not_cwd(monkeypatch, tmp_path):
    """The anchor is the harness directory, whatever the process's CWD is."""
    monkeypatch.chdir(tmp_path)
    resolved = psb.resolve_question_file("../questions-short.csv")

    assert resolved.is_absolute()
    assert resolved == (psb.PYSWEEP_DIR / ".." / "questions-short.csv")
    # The shipped corpora really are one level up, beside benchmark.py.
    assert resolved.resolve().is_file()


@allure.feature("pysweep")
def test_question_file_absolute_path_is_untouched(tmp_path):
    """An absolute path is already unambiguous; rewriting it would surprise."""
    corpus = tmp_path / "questions.csv"
    corpus.write_text("q\n")
    assert psb.resolve_question_file(str(corpus)) == corpus


@allure.feature("pysweep")
def test_preflight_accepts_the_shipped_example_config(monkeypatch):
    """The example must pass its own preflight, or the docs ship a broken config.

    The model half of the preflight is stubbed: the example declares ``llms.model``,
    and ``model_verify`` shells out to ``model-manager``/``kubectl``, which these
    off-cluster tests cannot do (its own behaviour is covered by
    ``test_model_manager_unit.py``). What is checkable here — that every path the
    example declares really resolves — is what this asserts.
    """
    monkeypatch.setattr(psb, "model_verify", lambda config: None)
    example = psb.PYSWEEP_DIR / "examples" / "sweep.yaml"
    psb.preflight_check(load_config(example))


@allure.feature("pysweep")
def test_preflight_rejects_a_missing_question_file():
    cfg = load_config(_client(question_file="../no-such-corpus.csv"))
    with pytest.raises(psb.PreflightError, match="no-such-corpus.csv"):
        psb.preflight_check(cfg)


@allure.feature("pysweep")
def test_preflight_error_names_the_resolved_path_and_the_anchor():
    """The message has to be actionable: what was looked for, and relative to what."""
    cfg = load_config(_client(question_file="questions-short.csv"))  # missing '../'
    with pytest.raises(psb.PreflightError) as excinfo:
        psb.preflight_check(cfg)

    message = str(excinfo.value)
    assert str(psb.PYSWEEP_DIR) in message
    assert "not the current directory" in message


@allure.feature("pysweep")
def test_preflight_checks_every_declared_corpus_not_just_the_first():
    """question_file is sweepable: a bad path in the last value must fail up front,
    not after the earlier runs have already spent their load."""
    cfg = load_config(_client(question_file=["../questions-short.csv", "../gone.csv"]))
    with pytest.raises(psb.PreflightError, match="gone.csv"):
        psb.preflight_check(cfg)


@allure.feature("pysweep")
def test_preflight_error_is_a_sweep_error():
    """Callers catching SweepError keep working; the subclass only adds meaning."""
    assert issubclass(psb.PreflightError, psb.SweepError)


@allure.feature("pysweep")
def test_preflight_runs_before_anything_is_applied(monkeypatch, tmp_path):
    """The whole point of the up-front call: a bad path costs no rollout.

    A model swap takes hours, so a config typo must fail before apply, not after.
    """
    monkeypatch.chdir(tmp_path)
    cfg = load_config({"client": {**CLIENT, "question_file": "../gone.csv"}})
    applied = []

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: applied.append(c) or {})
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)

    with pytest.raises(psb.PreflightError):
        psb.main(["--sweep", "ignored.yaml"])

    assert applied == []


@allure.feature("pysweep")
def test_benchmark_receives_the_resolved_absolute_question_file(monkeypatch, tmp_path):
    """benchmark.py opens the value relative to ITS CWD, so it must get an absolute
    path — that is what makes the anchor authoritative without editing benchmark.py."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_client())
    fake = FakeBenchmark()

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: {})
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda snapshot: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    (args, _), = fake.calls
    assert Path(args.f).is_absolute()
    assert Path(args.f).is_file()


@allure.feature("pysweep")
def test_summary_records_the_declared_path_not_the_resolved_one(monkeypatch, tmp_path):
    """Provenance shows what the config asked for; an absolute path would be
    machine-specific and make two identical configs look different."""
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_client())
    fake = FakeBenchmark()

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: {})
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda snapshot: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    summary = tmp_path / results.DEFAULT_SUMMARIES_DIR / results.SUMMARY_FILE_NAME
    rows = list(csv.DictReader(summary.read_text().splitlines()))
    assert rows[0]["client.question_file"] == "../questions-short.csv"


@allure.feature("pysweep")
def test_preflight_runs_once_for_the_whole_sweep(monkeypatch, tmp_path):
    """Checked before the sweep and not again: every check is about the config and
    its environment, so no run can change the answer and repeating it per load test
    would only spend a cluster round trip (model_verify shells out) to re-learn it.
    """
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "questions.csv"
    corpus.write_text("q\n")
    cfg = load_config(_client(question_file=str(corpus), concurrency=[1, 16]))
    fake = FakeBenchmark()

    calls = []

    def counting_preflight(config):
        calls.append(len(fake.calls))

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "preflight_check", counting_preflight)
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: {})
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda snapshot: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    # Once, with no run done yet — and both client combinations still ran.
    assert calls == [0]
    assert len(fake.calls) == 2


@allure.feature("pysweep")
def test_preflight_covers_both_the_corpus_and_the_model_checks(monkeypatch):
    """One entry point, so a new precondition is wired in one place and callers
    (and this test) do not have to know how many checks there are."""
    called = []
    monkeypatch.setattr(psb, "question_file_verify", lambda config: called.append("corpus"))
    monkeypatch.setattr(psb, "model_verify", lambda config: called.append("model"))

    psb.preflight_check(load_config(_client()))

    # Cheapest first: a stat per corpus before model-manager's dry run.
    assert called == ["corpus", "model"]


# -- logging: what a run is announced as --------------------------------------


@allure.feature("pysweep")
def test_format_settings_yaml_nests_dotted_paths_with_four_spaces():
    """The block has to read like the sweep config it came from.

    Dotted paths back into sections, four-space indentation to match the config
    files — so the logged text is comparable to (and pasteable into) a sweep YAML
    rather than being a Python repr of the runner's internals.
    """
    block = psb.format_settings_yaml({"client.concurrency": 16, "client.duration": "3m"})
    assert block == "client:\n    concurrency: 16\n    duration: 3m"


@allure.feature("pysweep")
def test_format_settings_yaml_groups_sections_and_keeps_kpis_top_level():
    """A summary row mixes knob paths with undotted KPIs; both have to read well.

    Sections group (in the caller's order, not sorted — for load parameters that
    order is benchmark.py's own), and the KPI keys have no section to belong to,
    so they stay at the top level instead of being invented one.
    """
    block = psb.format_settings_yaml(
        {
            "llms.max_new_tokens": 1024,
            "retriever.k": 5,
            "reranker.top_n": 3,
            "client.concurrency": 128,
            "requests": 3225,
            "error": 0,
        }
    )
    assert block.splitlines() == [
        "llms:",
        "    max_new_tokens: 1024",
        "retriever:",
        "    k: 5",
        "reranker:",
        "    top_n: 3",
        "client:",
        "    concurrency: 128",
        "requests: 3225",
        "error: 0",
    ]


@allure.feature("pysweep")
def test_format_settings_yaml_is_empty_for_no_settings():
    """So a caller can say '(unchanged)' instead of logging an empty block."""
    assert psb.format_settings_yaml({}) == ""


@allure.feature("pysweep")
def test_format_settings_yaml_stringifies_what_yaml_cannot_represent():
    """A log line must never be what kills a four-hour sweep.

    Property read-backs are ints and strings today, but yaml.safe_dump raises on
    anything else, so unknown types are rendered rather than propagated.
    """
    block = psb.format_settings_yaml({"client.question_file": Path("/tmp/q.csv")})
    assert block == "client:\n    question_file: /tmp/q.csv"


@allure.feature("pysweep")
def test_load_parameters_are_logged_as_yaml(caplog, tmp_path):
    """The 'load parameters' step announces the client section, not a dict repr.

    That line is how a reader of a multi-hour log knows what the next load test
    is about to do, so it is asserted here: YAML block, four-space indent, the
    declared corpus path (not the anchored absolute one), and no dict/JSON braces.
    """
    cfg = load_config(_client(concurrency=16))

    with caplog.at_level(logging.INFO, logger=psb.logger.name):
        psb.run_client_combination(
            FakeBenchmark(), [], (), cfg.client_combinations()[0], "Run 1.1",
            results_dir=tmp_path, tokenizer="served/model",
        )

    message = next(
        record.getMessage() for record in caplog.records
        if "load parameters" in record.getMessage()
    )
    assert "\nclient:\n" in message
    assert "    question_file: ../questions-short.csv" in message
    assert "    concurrency: 16" in message
    assert "{" not in message and "'" not in message


@allure.feature("pysweep")
def test_load_parameters_line_names_the_derived_tokenizer():
    """It is a load parameter too (-m), but it has no config key.

    So it is reported in the prose rather than inside the YAML block: a block
    containing 'tokenizer:' would look declarable, and 'client.tokenizer' is a
    ConfigError by design.
    """
    cfg = load_config(_client())
    block = psb.format_settings_yaml(
        BenchmarkClientArgs.from_settings(cfg.client_combinations()[0]).snapshot()
    )
    assert "tokenizer" not in block
