# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import ast
import textwrap
from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import client_args
from tests.e2e.benchmarks.chatqna.pysweep.properties.base import Plane
from tests.e2e.benchmarks.chatqna.pysweep.config import (
    ConfigError,
    SweepConfig,
    load_config,
    load_run_config,
)
from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens

CLIENT = {
    "question_file": "questions-short.csv",
    "concurrency": 4,
    "duration": "3m",
    "worker_startup_delay": "0s",
    "failures_allowed": 200,
    "input_tokens": 128,
    "warmup_requests": 1,
}

CLIENT_YAML = """
        client:
          question_file: questions-short.csv
          concurrency: 4
          duration: 3m
          worker_startup_delay: 0s
          failures_allowed: 200
          input_tokens: 128
          warmup_requests: 1
"""


def _sweep(**sections):
    """A valid config: the required client section plus whatever is under test."""
    return {"client": dict(CLIENT), **sections}


def _client(**overrides):
    """A config whose client section differs from the default in ``overrides``."""
    return {"client": {**CLIENT, **overrides}}


def _write(tmp_path, text):
    path = tmp_path / "sweep.yaml"
    path.write_text(textwrap.dedent(text))
    return path


def _benchmark_option_flags():
    """Parsed from source: importing benchmark.py pulls in transformers."""
    source = (
        Path(__file__).resolve().parents[3] / "e2e/benchmarks/chatqna/benchmark.py"
    ).read_text()
    tree = ast.parse(source)
    parse_args = next(
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "parse_args"
    )
    return {
        node.args[0].value
        for node in ast.walk(parse_args)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
    }


@allure.feature("pysweep")
def test_config_with_only_client_knobs_is_valid():
    cfg = load_config(_sweep())
    assert isinstance(cfg, SweepConfig)
    assert list(cfg.iter_settings()) == []
    assert cfg.deployment_combinations() == [()]
    assert len(cfg.client_combinations()) == 1


@allure.feature("pysweep")
def test_config_without_client_section_raises():
    with pytest.raises(ConfigError, match="client"):
        load_config({})


@allure.feature("pysweep")
def test_single_value_binds_to_property_and_yields_one_combination():
    cfg = load_config(_sweep(llms={"max_new_tokens": 256}))

    settings = list(cfg.iter_settings())
    assert len(settings) == 1
    path, property_cls, values = settings[0]
    assert path == "llms.max_new_tokens"
    assert property_cls is MaxNewTokens
    assert values == [256]

    combos = cfg.deployment_combinations()
    assert len(combos) == 1
    (setting,) = combos[0]
    assert setting.path == "llms.max_new_tokens"
    assert setting.property_cls is MaxNewTokens
    assert setting.value == 256


@allure.feature("pysweep")
def test_list_value_expands_to_product():
    cfg = load_config(_sweep(llms={"max_new_tokens": [128, 256, 512]}))
    combos = cfg.deployment_combinations()
    assert [c[0].value for c in combos] == [128, 256, 512]
    assert all(c[0].property_cls is MaxNewTokens for c in combos)


@allure.feature("pysweep")
def test_omitted_knob_is_not_swept():
    cfg = load_config(_sweep(llms={}))
    assert list(cfg.iter_settings()) == []
    assert cfg.deployment_combinations() == [()]


@allure.feature("pysweep")
def test_unknown_top_level_key_raises():
    with pytest.raises(ConfigError):
        load_config(_sweep(llmsss={"max_new_tokens": 256}))


@allure.feature("pysweep")
def test_unknown_knob_key_raises():
    with pytest.raises(ConfigError):
        load_config(_sweep(llms={"max_new_tokenz": 256}))


@allure.feature("pysweep")
@pytest.mark.parametrize("bad", ["abc", -1, 0, 1.5, [128, "oops"]])
def test_wrong_value_type_or_range_raises(bad):
    with pytest.raises(ConfigError):
        load_config(_sweep(llms={"max_new_tokens": bad}))


@allure.feature("pysweep")
def test_top_level_must_be_mapping(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text("- 1\n- 2\n- 3\n")
    with pytest.raises(ConfigError):
        load_config(path)


@allure.feature("pysweep")
def test_load_from_yaml_file(tmp_path):
    path = _write(
        tmp_path,
        """
        llms:
          max_new_tokens: [128, 256]
        """
        + CLIENT_YAML,
    )
    cfg = load_config(path)
    assert [c[0].value for c in cfg.deployment_combinations()] == [128, 256]


@allure.feature("pysweep")
def test_malformed_yaml_raises(tmp_path):
    path = tmp_path / "sweep.yaml"
    path.write_text("llms: [unbalanced\n")
    with pytest.raises(ConfigError):
        load_config(path)


@allure.feature("pysweep")
def test_max_new_tokens_is_runtime_api_plane():
    """Guards the outer/inner loop ordering assumption in combinations()."""
    assert MaxNewTokens.plane == Plane.RUNTIME_API


@allure.feature("pysweep")
@pytest.mark.parametrize("missing", sorted(CLIENT))
def test_every_scenario_client_knob_is_required(missing):
    section = {k: v for k, v in CLIENT.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        load_config({"client": section})


@allure.feature("pysweep")
def test_client_knobs_bind_to_benchmark_flags():
    cfg = load_config(_client())
    bound = {path: arg.flag for path, arg, _ in cfg.iter_client_settings()}
    assert bound == {
        "client.question_file": "-f",
        "client.concurrency": "-c",
        "client.duration": "-d",
        "client.worker_startup_delay": "-u",
        "client.failures_allowed": "-e",
        "client.input_tokens": "-x",
        "client.warmup_requests": "-w",
    }


@allure.feature("pysweep")
def test_every_benchmark_option_is_mirrored():
    mirrored = [a.flag for a in client_args.CLIENT_ARGS]
    assert set(mirrored) == _benchmark_option_flags()
    assert len(mirrored) == len(set(mirrored)), "a benchmark option is mirrored twice"
    # -m is derived from the served model, so it has no ClientConfig field.
    declarable = {a.yaml_name for a in client_args.CLIENT_ARGS if not a.derived}
    assert set(SweepConfig.model_fields["client"].annotation.model_fields) == declarable


@allure.feature("pysweep")
def test_client_list_expands_to_its_own_product():
    cfg = load_config(_client(concurrency=[1, 16]))
    assert cfg.deployment_combinations() == [()]
    combos = cfg.client_combinations()
    assert len(combos) == 2
    assert [s.value for c in combos for s in c if s.path == "client.concurrency"] == [1, 16]


@allure.feature("pysweep")
def test_max_concurrency_is_the_widest_declared_value():
    assert load_config(_client(concurrency=[1, 16, 128])).max_concurrency() == 128
    assert load_config(_client(concurrency=[128, 16, 1])).max_concurrency() == 128
    assert load_config(_client(concurrency=4)).max_concurrency() == 4


@allure.feature("pysweep")
def test_client_product_multiplies_dimensions():
    cfg = load_config(_client(concurrency=[1, 16], input_tokens=[128, 256]))
    combos = cfg.client_combinations()
    pairs = [
        (
            next(s.value for s in c if s.path == "client.concurrency"),
            next(s.value for s in c if s.path == "client.input_tokens"),
        )
        for c in combos
    ]
    assert pairs == [(1, 128), (1, 256), (16, 128), (16, 256)]


@allure.feature("pysweep")
def test_client_knobs_are_not_properties():
    cfg = load_config(_client(concurrency=[1, 16]))
    assert list(cfg.iter_settings()) == []
    assert not hasattr(cfg.client_combinations()[0][0], "property_cls")


@allure.feature("pysweep")
def test_server_and_bearer_file_are_optional_and_not_swept():
    cfg = load_config(_client())
    assert cfg.client_provisioning_argv() == []
    assert "server" not in cfg.declared_client_names()
    cfg = load_config(_client(server="https://erag.com/api/v1/chatqna", bearer_file="uat.txt"))
    assert cfg.client_provisioning_argv() == [
        "-s", "https://erag.com/api/v1/chatqna", "-b", "uat.txt",
    ]
    assert [p for p, _, _ in cfg.iter_client_settings() if "server" in p] == []


@allure.feature("pysweep")
@pytest.mark.parametrize("bad", ["3 minutes", "3", "m", "3d", 3, ""])
def test_duration_must_be_a_parsable_duration(bad):
    """benchmark.duration_to_seconds raises KeyError on anything else."""
    with pytest.raises(ConfigError):
        load_config(_client(duration=bad))


@allure.feature("pysweep")
@pytest.mark.parametrize("bad", [0, -1, "many", 1.5])
def test_concurrency_must_be_a_positive_integer(bad):
    with pytest.raises(ConfigError):
        load_config(_client(concurrency=bad))


@allure.feature("pysweep")
@pytest.mark.parametrize("knob", ["input_tokens", "warmup_requests", "failures_allowed"])
def test_zero_is_allowed_where_it_is_meaningful(knob):
    cfg = load_config(_client(**{knob: 0}))
    assert next(v for p, _, v in cfg.iter_client_settings() if p.endswith(knob)) == [0]


@allure.feature("pysweep")
def test_unknown_client_knob_raises():
    with pytest.raises(ConfigError):
        load_config(_client(concurency=4))


@allure.feature("pysweep")
def test_tokenizer_cannot_be_declared():
    with pytest.raises(ConfigError):
        load_config(_client(tokenizer="nomic-ai/nomic-embed-text-v1"))
    assert client_args.TOKENIZER.derived
    assert client_args.TOKENIZER in client_args.CLIENT_ARGS
    assert not client_args.TOKENIZER.sweepable
    cfg = load_config(_client())
    assert [p for p, _, _ in cfg.iter_client_settings() if "tokenizer" in p] == []


@allure.feature("pysweep")
def test_varying_paths_reports_only_multi_valued_knobs():
    cfg = load_config(
        {
            "client": {**CLIENT, "concurrency": [1, 16]},
            "llms": {"max_new_tokens": [128, 256]},
        }
    )
    assert cfg.varying_paths() == {"client.concurrency", "llms.max_new_tokens"}


@allure.feature("pysweep")
def test_shipped_example_sweep_config_is_valid():
    example = (
        Path(__file__).resolve().parents[3]
        / "e2e/benchmarks/chatqna/pysweep/configs/sweep.yaml"
    )
    cfg = load_config(example)
    assert cfg.deployment_combinations()
    assert cfg.client_combinations()


@allure.feature("pysweep")
def test_run_config_reads_display_kubernetes_client_logs(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("display_kubernetes_client_logs: true\n")
    assert load_run_config(path).display_kubernetes_client_logs is True


@allure.feature("pysweep")
def test_kubernetes_client_logs_default_to_off():
    assert load_run_config({}).display_kubernetes_client_logs is False


@allure.feature("pysweep")
def test_run_config_rejects_unknown_key():
    with pytest.raises(ConfigError):
        load_run_config({"display_kubernetes_client_log": False})


@allure.feature("pysweep")
def test_run_config_rejects_non_boolean():
    with pytest.raises(ConfigError):
        load_run_config({"display_kubernetes_client_logs": "sometimes"})


@allure.feature("pysweep")
def test_shipped_example_run_config_is_valid():
    example = (
        Path(__file__).resolve().parents[3]
        / "e2e/benchmarks/chatqna/pysweep/configs/general.yaml"
    )
    assert load_run_config(example).display_kubernetes_client_logs is False


def _pre_bench(num_vectors):
    return {**_sweep(), "pre_bench": {"edp": {"vector_db": {"num_vectors": num_vectors}}}}


@allure.feature("pysweep")
def test_pre_bench_settings_are_dotted_paths():
    assert load_config(_pre_bench(50000)).pre_bench_settings() == {"edp.vector_db.num_vectors": 50000}
    assert load_config(_sweep()).pre_bench_settings() == {}


@allure.feature("pysweep")
@pytest.mark.parametrize("bad", [[50000, 100000], 0, -1, "many"])
def test_num_vectors_is_a_single_positive_integer(bad):
    with pytest.raises(ConfigError):
        load_config(_pre_bench(bad))


@allure.feature("pysweep")
def test_pre_bench_rejects_unknown_keys():
    with pytest.raises(ConfigError):
        load_config({**_sweep(), "pre_bench": {"edp": {"vector_db": {"vectors": 1}}}})


@allure.feature("pysweep")
def test_pre_bench_is_neither_a_knob_nor_a_dimension():
    cfg = load_config({**_pre_bench(50000), "llms": {"max_new_tokens": [128, 256]}})
    assert not any(path.startswith(("pre_bench", "edp")) for path in cfg.knob_paths())
    assert len(cfg.deployment_combinations()) == 2
