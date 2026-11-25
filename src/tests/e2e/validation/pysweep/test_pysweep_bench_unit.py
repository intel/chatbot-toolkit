# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import ast
import csv
import itertools
import linecache
import logging
from contextlib import contextmanager
from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import pysweep_bench as psb
from tests.e2e.benchmarks.chatqna.pysweep.base import (
    Plane,
    Property,
    PropertyApplyError,
    PropertyReadError,
)
from tests.e2e.benchmarks.chatqna.pysweep.client_args import BenchmarkClientArgs
from tests.e2e.benchmarks.chatqna.pysweep import results
from tests.e2e.benchmarks.chatqna.pysweep.config import PropertySetting, load_config
from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens
from tests.e2e.benchmarks.chatqna.pysweep.served_model import ServedModel

# question_file must exist (relative to pysweep_bench.py): main() runs preflight, which stats it.
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
    """Parse argv with a parser rebuilt from ``benchmark.parse_args``'s source."""
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
    """In-memory FingerprintClient that applies writes; the component name is ignored."""

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

    class _K8sKnob(Property):
        plane = Plane.K8S_SPEC

        def __init__(self):
            pass

        def read(self):
            return None

        def apply(self, desired):
            pass

    setting = PropertySetting(path="fake.knob", property_cls=_K8sKnob, value=1)
    assert isinstance(psb.build_property(setting, FakeFingerprintClient()), _K8sKnob)


@allure.feature("pysweep")
def test_apply_combination_reports_unestablishable_state_as_sweep_error(monkeypatch):
    cfg = load_config(_sweep(llms={"max_new_tokens": 128}))

    def refuse(self, desired):
        raise PropertyApplyError("value rejected")

    monkeypatch.setattr(MaxNewTokens, "apply", refuse)
    with pytest.raises(psb.SweepError, match="llms.max_new_tokens"):
        psb.apply_combination(cfg.deployment_combinations()[0], FakeFingerprintClient())


@allure.feature("pysweep")
def test_apply_combination_reports_failed_convergence_as_sweep_error(monkeypatch):
    cfg = load_config(_sweep(llms={"max_new_tokens": 128}))
    monkeypatch.setattr(MaxNewTokens, "apply", lambda self, desired: None)

    def never_ready(self, desired, timeout=None):
        raise PropertyApplyError("did not converge")

    monkeypatch.setattr(MaxNewTokens, "wait_until_ready", never_ready)
    with pytest.raises(psb.SweepError, match="llms.max_new_tokens"):
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
    cfg = load_config(_sweep(llms={"max_new_tokens": [128, 256]}))
    slugs = [psb.combination_slug(c) for c in cfg.deployment_combinations()]
    assert slugs == ["max_new_tokens-128", "max_new_tokens-256"]
    assert len(set(slugs)) == len(slugs)


class FakeBenchmark:
    """Stand-in for the benchmark module that records each invocation."""

    def __init__(self, rows=1):
        self.calls = []
        self._rows = rows

    parse_args = staticmethod(lambda argv: _parse_benchmark_argv(argv))

    def run_benchmark(self, args, output_file=None):
        self.calls.append((args, output_file))
        if self._rows:
            Path(output_file).write_text("128,128,0.8,6.9,,200\n" * self._rows)
        return 0, output_file


@allure.feature("pysweep")
def test_client_combination_renders_to_benchmark_argv():
    """``-m`` is absent: the runner appends it from the served model."""
    cfg = load_config(_client(concurrency=16))
    argv = BenchmarkClientArgs.from_settings(cfg.client_combinations()[0]).to_argv()
    # The declared path verbatim; anchoring happens later, at the argv boundary.
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
    """Values rendered through ``str`` must round-trip to benchmark.py's declared types."""
    cfg = load_config(_client(concurrency=16, input_tokens=0))
    argv = BenchmarkClientArgs.from_settings(cfg.client_combinations()[0]).to_argv()
    parsed = _parse_benchmark_argv(argv)
    assert (parsed.c, parsed.d, parsed.x, parsed.f) == (16, "3m", 0, "../questions-short.csv")
    assert (parsed.e, parsed.w, parsed.u) == (200, 1, "0s")


@allure.feature("pysweep")
def test_client_knobs_vary_innermost(monkeypatch, tmp_path):
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
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    assert applied == [(128,), (256,)]
    assert [args.c for args, _ in fake.calls] == [1, 16, 1, 16]
    assert [args.d for args, _ in fake.calls] == ["3m"] * 4
    # The derived tokenizer reaches every run, though no config declares it.
    assert [args.m for args, _ in fake.calls] == ["served/model"] * 4


@allure.feature("pysweep")
def test_sweep_writes_one_summary_row_per_run(monkeypatch, tmp_path):
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
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml", "--summaries", "out"]) == 0

    summary_file = tmp_path / "out" / results.SUMMARY_FILE_NAME
    rows = list(csv.DictReader(summary_file.open()))
    assert len(rows) == 4
    assert [(r["llms.max_new_tokens"], r["client.concurrency"]) for r in rows] == [
        ("128", "1"), ("128", "16"), ("256", "1"), ("256", "16"),
    ]
    assert {r["client.duration"] for r in rows} == {"3m"}
    # FakeBenchmark writes 128,128,0.8,6.9,,200 twice per run.
    assert {r["requests"] for r in rows} == {"2"}
    assert {r["mean_ttft"] for r in rows} == {"0.8"}
    assert {r["mean_tpot"] for r in rows} == {str(round((6.9 - 0.8) / 127, 6))}
    assert {r["served_model"] for r in rows} == {"served/model"}
    # Relative to the summary, so the directory can be copied off the host intact.
    assert all(not Path(r["result_file"]).is_absolute() for r in rows)
    assert all((summary_file.parent / r["result_file"]).exists() for r in rows)
    assert {Path(r["result_file"]).parent.name for r in rows} == {results.PER_CASE_DIR_NAME}


@allure.feature("pysweep")
def test_the_previous_sweeps_results_are_gone_before_this_one_starts(monkeypatch, tmp_path):
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
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, FakeBenchmark())

    assert psb.main(["--sweep", "ignored.yaml", "--summaries", "out"]) == 0

    assert not stale.exists()
    assert [p.name for p in stale.parent.iterdir()] == ["bench_single.result.csv"]


def _passthrough_provision(benchmark_args, workers=1):
    """provision_benchmark_target without a cluster."""
    from contextlib import contextmanager

    @contextmanager
    def cm():
        yield list(benchmark_args) + ["-s", "http://127.0.0.1:1/api", "-b", "uat.txt"]

    return cm()


def _install_fake_benchmark(monkeypatch, fake):
    monkeypatch.setattr(psb, "benchmark", fake)


@allure.feature("pysweep")
def test_result_file_names_are_unique_per_run():
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
    cfg = load_config(_client(concurrency=[1, 16]))
    name = psb.result_file_name((), cfg.client_combinations()[0], only=cfg.varying_paths())
    assert name == "bench_concurrency-1.result.csv"


@allure.feature("pysweep")
def test_result_file_name_stays_within_the_filesystem_limit():
    """Naming every declared knob would overflow the 255-byte filename limit."""
    cfg = load_config(
        {
            "client": {
                **CLIENT,
                "concurrency": [1, 16],
                "question_file": (
                    "datasets/pubmed/long-form/2026-01/reviewed/"
                    "questions-pubmed-long-form-with-reference-answers-and-citations-2026-01.csv"
                ),
            },
            "llms": {"max_new_tokens": [128, 256]},
        }
    )
    combination = cfg.deployment_combinations()[0]
    client_combination = cfg.client_combinations()[0]

    assert len(Path(psb.result_file_name(combination, client_combination)).name) > 255
    filtered = psb.result_file_name(combination, client_combination, only=cfg.varying_paths())
    assert len(Path(filtered).name) <= 255
    # The shipped example must also be nameable.
    example = (
        Path(__file__).resolve().parents[3]
        / "e2e/benchmarks/chatqna/pysweep/configs/sweep.yaml"
    )
    shipped = load_config(example)
    for combo in shipped.deployment_combinations():
        for client_combo in shipped.client_combinations():
            name = psb.result_file_name(combo, client_combo, only=shipped.varying_paths())
            assert len(Path(name).name) <= 255


@allure.feature("pysweep")
def test_result_file_name_falls_back_when_nothing_varies():
    cfg = load_config(_client())
    name = psb.result_file_name((), cfg.client_combinations()[0], only=cfg.varying_paths())
    assert name == "bench_single.result.csv"


@allure.feature("pysweep")
def test_tokenizer_is_the_catalog_model_id_not_the_served_name(monkeypatch, tmp_path):
    """AutoTokenizer needs the catalog's HF repo id, not the served catalog name."""
    catalog = tmp_path / "models.yaml"
    catalog.write_text(
        "models:\n"
        "  - name: llama3-8b-awq\n"
        "    model_id: casperhansen/llama-3-8b-instruct-awq\n"
        "    category: llm\n"
    )
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(catalog))
    monkeypatch.setattr(ServedModel, "read", lambda self: "llama3-8b-awq")

    assert psb.resolve_tokenizer() == "casperhansen/llama-3-8b-instruct-awq"


@allure.feature("pysweep")
def test_unreadable_model_aborts_rather_than_guessing_a_tokenizer(monkeypatch):
    def unreadable(self):
        raise PropertyReadError("no llm-usvc pods")

    monkeypatch.setattr(ServedModel, "read", unreadable)
    with pytest.raises(psb.SweepError, match="served model"):
        psb.resolve_tokenizer()


@allure.feature("pysweep")
def test_a_served_model_missing_from_the_catalog_aborts(monkeypatch, tmp_path):
    catalog = tmp_path / "models.yaml"
    catalog.write_text("models:\n  - name: other-model\n    model_id: org/other\n")
    monkeypatch.setenv("PYSWEEP_MODEL_CATALOG", str(catalog))
    monkeypatch.setattr(ServedModel, "read", lambda self: "llama3-8b-awq")

    with pytest.raises(psb.SweepError, match="tokenizer"):
        psb.resolve_tokenizer()


@allure.feature("pysweep")
def test_forwarded_tokenizer_flag_is_refused():
    cfg = load_config(_client())
    with pytest.raises(psb.SweepError, match=r"-m"):
        psb.resolve_benchmark_argv(cfg, ["-m", "nomic-ai/nomic-embed-text-v1"])


@allure.feature("pysweep")
def test_forwarded_flag_that_the_config_declares_is_refused():
    cfg = load_config(_client(concurrency=[1, 16]))
    with pytest.raises(psb.SweepError, match=r"-c"):
        psb.resolve_benchmark_argv(cfg, ["-c", "8"])


@allure.feature("pysweep")
def test_forwarded_server_is_allowed_when_the_config_omits_it():
    cfg = load_config(_client())
    assert psb.resolve_benchmark_argv(cfg, ["-s", "https://erag.com/api/v1/chatqna"]) == [
        "-s", "https://erag.com/api/v1/chatqna",
    ]


@allure.feature("pysweep")
def test_config_declared_server_and_bearer_reach_benchmark():
    """Declared connection settings also suppress auto-provisioning."""
    cfg = load_config(_client(server="https://erag.com/api/v1/chatqna", bearer_file="uat.txt"))
    argv = psb.resolve_benchmark_argv(cfg, [])
    assert argv == ["-s", "https://erag.com/api/v1/chatqna", "-b", "uat.txt"]
    with psb.provision_benchmark_target(argv) as provisioned:
        assert provisioned == argv


@allure.feature("pysweep")
def test_count_result_rows_counts_and_tolerates_missing(tmp_path):
    assert psb.count_result_rows(tmp_path / "nope.csv") == 0
    csv_file = tmp_path / "r.csv"
    csv_file.write_text("128,128,0.8,6.9,,200\n128,128,1.2,7.1,,200\n")
    assert psb.count_result_rows(csv_file) == 2


@allure.feature("pysweep")
def _fake_keycloak(monkeypatch, domain="box.example"):
    """Point provisioning at a fake deployment whose every token mint differs."""
    from tests.e2e.benchmarks.chatqna.pysweep import authentication

    minted = itertools.count(1)

    # Like the real property, each read is a new grant and yields a fresh token.
    class _Helper:
        @property
        def access_token(self):
            return f"token-{next(minted)}"

    monkeypatch.setattr(
        authentication, "load_build_config", lambda: {"base_domain_name": domain}
    )
    monkeypatch.setattr(authentication, "keycloak_helper", _Helper)

    # Record the lifespan override instead of calling the Keycloak admin API.
    lifespan_calls = []

    @contextmanager
    def _lifespan(helper, seconds=authentication.SWEEP_TOKEN_LIFESPAN_S):
        lifespan_calls.append(("set", seconds))
        try:
            yield
        finally:
            lifespan_calls.append(("restored", seconds))

    monkeypatch.setattr(authentication, "client_token_lifespan", _lifespan)
    return lifespan_calls


@allure.feature("pysweep")
def test_provision_defaults_to_the_deployments_chatqna_endpoint(monkeypatch):
    _fake_keycloak(monkeypatch)
    with psb.provision_benchmark_target([]) as args:
        assert args[args.index("-s") + 1] == "https://box.example/api/v1/chatqna"


@allure.feature("pysweep")
def test_provision_mints_one_distinct_token_per_worker(monkeypatch):
    """Distinct because APISIX rate-limits per Authorization header value."""
    _fake_keycloak(monkeypatch)
    with psb.provision_benchmark_target([], workers=8) as args:
        lines = Path(args[args.index("-b") + 1]).read_text().splitlines()
        assert len(lines) == 8
        assert len(set(lines)) == 8, "workers must not share a rate-limit bucket"
        assert all(line for line in lines), "an empty line makes that worker quit"


@allure.feature("pysweep")
def test_provision_refresh_mints_new_tokens_benchmark_can_see(monkeypatch):
    """Without invalidating linecache, benchmark.py would reuse the first run's tokens."""
    _fake_keycloak(monkeypatch)
    with psb.provision_benchmark_target([], workers=2) as args:
        bearer = args[args.index("-b") + 1]
        first = [linecache.getline(bearer, n).strip() for n in (1, 2)]
        args.refresh()
        second = [linecache.getline(bearer, n).strip() for n in (1, 2)]
    assert all(first) and all(second)
    assert not set(first) & set(second), "a refresh must hand out new tokens"


@allure.feature("pysweep")
def test_provision_lengthens_tokens_for_the_sweep_and_restores(monkeypatch):
    calls = _fake_keycloak(monkeypatch)
    with psb.provision_benchmark_target([], workers=1):
        assert calls == [("set", 3600)]
    assert calls == [("set", 3600), ("restored", 3600)]


@allure.feature("pysweep")
def test_a_supplied_bearer_is_neither_refreshed_nor_lengthened(monkeypatch, tmp_path):
    calls = _fake_keycloak(monkeypatch)
    uat = tmp_path / "uat.txt"
    uat.write_text("real-token\n")
    with psb.provision_benchmark_target(["-b", str(uat)]) as args:
        args.refresh()
    assert uat.read_text() == "real-token\n"
    assert calls == []


@allure.feature("pysweep")
def test_main_refreshes_tokens_before_every_run(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(
        {"client": {**CLIENT, "concurrency": [1, 16]}, "llms": {"max_new_tokens": [128, 256]}}
    )
    events = []

    class _RecordingBenchmark(FakeBenchmark):
        def run_benchmark(self, args, output_file=None):
            events.append("run")
            return super().run_benchmark(args, output_file=output_file)

    @contextmanager
    def _provision(benchmark_args, workers=1):
        yield psb.ProvisionedArgv(
            list(benchmark_args) + ["-s", "http://127.0.0.1:1/api", "-b", "uat.txt"],
            refresh=lambda: events.append("refresh"),
        )

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(
        psb, "apply_combination", lambda combination, client: {s.path: s.value for s in combination}
    )
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _provision)
    _install_fake_benchmark(monkeypatch, _RecordingBenchmark(rows=1))

    assert psb.main(["--sweep", "ignored.yaml", "--summaries", "out"]) == 0
    assert events == ["refresh", "run"] * 4


@allure.feature("pysweep")
def test_configure_kubernetes_client_logs_quiets_httpx():
    """kr8s/httpx loggers drop INFO but keep warnings; asyncio gains the noise filter."""
    levels = {n: logging.getLogger(n).level for n in psb.KUBERNETES_CLIENT_LOGGERS}
    asyncio_log = logging.getLogger("asyncio")
    installed = list(asyncio_log.filters)
    try:
        psb.configure_kubernetes_client_logs(enabled=False)
        for name in psb.KUBERNETES_CLIENT_LOGGERS:
            log = logging.getLogger(name)
            assert not log.isEnabledFor(logging.INFO)
            assert log.isEnabledFor(logging.WARNING)
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
    log_filter = psb._DropPortForwardTeardownNoise()

    def record(msg, exc=None):
        rec = logging.LogRecord("asyncio", logging.ERROR, __file__, 1, msg, (), None)
        if exc is not None:
            rec.exc_info = (type(exc), exc, None)
        return rec

    assert not log_filter.filter(record("Unhandled exception in client_connected_cb"))
    assert not log_filter.filter(record("boom", ConnectionResetError("Connection lost")))
    assert log_filter.filter(record("Task exception was never retrieved"))


@allure.feature("pysweep")
def test_configure_kubernetes_client_logs_enabled_is_a_noop():
    levels = {n: logging.getLogger(n).level for n in psb.KUBERNETES_CLIENT_LOGGERS}
    psb.configure_kubernetes_client_logs(enabled=True)
    assert {n: logging.getLogger(n).level for n in psb.KUBERNETES_CLIENT_LOGGERS} == levels


@allure.feature("pysweep")
def test_provision_preserves_caller_supplied_server_and_bearer(tmp_path):
    uat = tmp_path / "uat.txt"
    uat.write_text("real-token\n")
    original = ["-s", "https://erag.com/api/v1/chatqna", "-b", str(uat), "-c", "8"]
    with psb.provision_benchmark_target(list(original)) as args:
        assert args == original


# benchmark.py opens question_file with a bare open(), i.e. relative to its CWD.


@allure.feature("pysweep")
def test_question_file_resolves_against_pysweep_dir_not_cwd(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    resolved = psb.resolve_question_file("../questions-short.csv")

    assert resolved.is_absolute()
    assert resolved == (psb.PYSWEEP_DIR / ".." / "questions-short.csv")
    assert resolved.resolve().is_file()


@allure.feature("pysweep")
def test_question_file_absolute_path_is_untouched(tmp_path):
    corpus = tmp_path / "questions.csv"
    corpus.write_text("q\n")
    assert psb.resolve_question_file(str(corpus)) == corpus


@allure.feature("pysweep")
def test_preflight_accepts_the_shipped_example_config(monkeypatch):
    example = psb.PYSWEEP_DIR / "configs" / "sweep.yaml"
    psb.preflight_check(load_config(example))


@allure.feature("pysweep")
def test_preflight_rejects_a_missing_question_file():
    cfg = load_config(_client(question_file="../no-such-corpus.csv"))
    with pytest.raises(psb.PreflightError, match="no-such-corpus.csv"):
        psb.preflight_check(cfg)


@allure.feature("pysweep")
def test_preflight_error_names_the_resolved_path_and_the_anchor():
    cfg = load_config(_client(question_file="questions-short.csv"))  # missing '../'
    with pytest.raises(psb.PreflightError) as excinfo:
        psb.preflight_check(cfg)

    message = str(excinfo.value)
    assert str(psb.PYSWEEP_DIR) in message
    assert "not the current directory" in message


@allure.feature("pysweep")
def test_preflight_checks_every_declared_corpus_not_just_the_first():
    cfg = load_config(_client(question_file=["../questions-short.csv", "../gone.csv"]))
    with pytest.raises(psb.PreflightError, match="gone.csv"):
        psb.preflight_check(cfg)


@allure.feature("pysweep")
def test_preflight_error_is_a_sweep_error():
    assert issubclass(psb.PreflightError, psb.SweepError)


@allure.feature("pysweep")
def test_preflight_runs_before_anything_is_applied(monkeypatch, tmp_path):
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
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_client())
    fake = FakeBenchmark()

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: {})
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    (args, _), = fake.calls
    assert Path(args.f).is_absolute()
    assert Path(args.f).is_file()


@allure.feature("pysweep")
def test_summary_records_the_declared_path_not_the_resolved_one(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    cfg = load_config(_client())
    fake = FakeBenchmark()

    run_config = psb.load_run_config({})
    monkeypatch.setattr(psb, "load_config", lambda _: cfg)
    monkeypatch.setattr(psb, "load_run_config", lambda _: run_config)
    monkeypatch.setattr(psb, "FingerprintClient", lambda transport: None)
    monkeypatch.setattr(psb, "default_transport", lambda: None)
    monkeypatch.setattr(psb, "apply_combination", lambda c, f: {})
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    summary = tmp_path / results.DEFAULT_SUMMARIES_DIR / results.SUMMARY_FILE_NAME
    rows = list(csv.DictReader(summary.read_text().splitlines()))
    assert rows[0]["client.question_file"] == "../questions-short.csv"


@allure.feature("pysweep")
def test_preflight_runs_once_for_the_whole_sweep(monkeypatch, tmp_path):
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
    monkeypatch.setattr(psb, "resolve_tokenizer", lambda: "served/model")
    monkeypatch.setattr(psb, "provision_benchmark_target", _passthrough_provision)
    _install_fake_benchmark(monkeypatch, fake)

    assert psb.main(["--sweep", "ignored.yaml"]) == 0

    assert calls == [0]
    assert len(fake.calls) == 2


@allure.feature("pysweep")
def test_preflight_runs_the_corpus_check(monkeypatch):
    called = []
    monkeypatch.setattr(psb, "question_file_verify", lambda config: called.append("corpus"))

    psb.preflight_check(load_config(_client()))

    assert called == ["corpus"]


@allure.feature("pysweep")
def test_format_settings_yaml_nests_dotted_paths_with_four_spaces():
    block = psb.format_settings_yaml({"client.concurrency": 16, "client.duration": "3m"})
    assert block == "client:\n    concurrency: 16\n    duration: 3m"


@allure.feature("pysweep")
def test_format_settings_yaml_groups_sections_and_keeps_kpis_top_level():
    """Sections keep the caller's order rather than being sorted."""
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
    assert psb.format_settings_yaml({}) == ""


@allure.feature("pysweep")
def test_format_settings_yaml_stringifies_what_yaml_cannot_represent():
    block = psb.format_settings_yaml({"client.question_file": Path("/tmp/q.csv")})
    assert block == "client:\n    question_file: /tmp/q.csv"


@allure.feature("pysweep")
def test_load_parameters_are_logged_as_yaml(caplog, tmp_path):
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
    """Named in the prose, not the YAML block: ``client.tokenizer`` is a ConfigError."""
    cfg = load_config(_client())
    block = psb.format_settings_yaml(
        BenchmarkClientArgs.from_settings(cfg.client_combinations()[0]).snapshot()
    )
    assert "tokenizer" not in block
