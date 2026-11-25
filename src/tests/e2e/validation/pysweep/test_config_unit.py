# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the pysweep sweep-config validator.

These test the schema/validation/expansion logic only — no cluster is needed,
so they run anywhere the requirements are installed. They assert three guarantees
the design leans on: (1) any YAML that violates the schema raises ConfigError,
(2) each declared deployment knob is bound to the correct Property class and
expands into the right cartesian product, and (3) client knobs are required,
bound to the right ``benchmark.py`` flag, and expand into their own separate
product (the runner's inner loop).
"""

import ast
import textwrap
from pathlib import Path

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import client_args
from tests.e2e.benchmarks.chatqna.pysweep.base import Plane
from tests.e2e.benchmarks.chatqna.pysweep.config import (
    ConfigError,
    SweepConfig,
    load_config,
    load_run_config,
)
from tests.e2e.benchmarks.chatqna.pysweep.properties.max_tokens import MaxNewTokens

# The client section is mandatory, so every config in these tests carries one.
# Kept minimal and boring: these values are never asserted on except by the
# client-specific tests below, which override what they care about.
CLIENT = {
    "question_file": "questions-short.csv",
    "concurrency": 4,
    "duration": "3m",
    "worker_startup_delay": "0s",
    "failures_allowed": 200,
    "input_tokens": 128,
    "warmup_requests": 1,
}

# Same, as YAML, for the file-loading tests.
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
    """The flags ``benchmark.parse_args`` adds, read from its source.

    Parsed rather than imported: ``benchmark.py`` pulls in ``transformers``, which
    is only needed to *run* a load test and is not in the validation suite's
    requirements (the runner imports it lazily for the same reason). Reading the
    source keeps this check on the real parser without that dependency.
    """
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
    """A sweep may vary load alone: no deployment knobs is a valid sweep.

    Nothing is declared, so the deployment product is the single empty
    combination — one pass with nothing to apply, benchmarking the deployment
    exactly as it stands.
    """
    cfg = load_config(_sweep())
    assert isinstance(cfg, SweepConfig)
    assert list(cfg.iter_settings()) == []
    assert cfg.deployment_combinations() == [()]
    # The client product is never empty either — the knobs are required.
    assert len(cfg.client_combinations()) == 1


@allure.feature("pysweep")
def test_config_without_client_section_raises():
    """Load parameters must be declared; an empty config is no longer valid.

    An omitted deployment knob means "leave it alone", which is observable. An
    omitted client knob has no such reading — the load test would silently run on
    benchmark.py's defaults and no record of it would exist.
    """
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
    """Optional *deployment* knobs left out contribute nothing to the sweep."""
    cfg = load_config(_sweep(llms={}))
    assert list(cfg.iter_settings()) == []
    # No dimension to expand, so the product stays at its one empty combination.
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


# -- client knobs -------------------------------------------------------------


@allure.feature("pysweep")
@pytest.mark.parametrize("missing", sorted(CLIENT))
def test_every_scenario_client_knob_is_required(missing):
    """Omitting any load parameter is a config error, naming the missing key.

    Parametrised over the whole set rather than spot-checking one, because the
    point is that none of them silently falls back to benchmark.py's default.
    """
    section = {k: v for k, v in CLIENT.items() if k != missing}
    with pytest.raises(ConfigError, match=missing):
        load_config({"client": section})


@allure.feature("pysweep")
def test_client_knobs_bind_to_benchmark_flags():
    """Each client knob is bound to the benchmark.py option it renders to."""
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
    """The adapter must cover benchmark.parse_args exactly — no drift either way.

    A ``benchmark.py`` option missing here cannot be swept; one that exists here
    but not there would be rendered into argv and rejected at run time. Compared
    against the real parser so the two cannot diverge silently.
    """
    mirrored = [a.flag for a in client_args.CLIENT_ARGS]
    assert set(mirrored) == _benchmark_option_flags()
    assert len(mirrored) == len(set(mirrored)), "a benchmark option is mirrored twice"
    # Mirrored != declarable: every option is covered, but -m is supplied by the
    # runner (from the served model) and so has no field on ClientConfig.
    declarable = {a.yaml_name for a in client_args.CLIENT_ARGS if not a.derived}
    assert set(SweepConfig.model_fields["client"].annotation.model_fields) == declarable


@allure.feature("pysweep")
def test_client_list_expands_to_its_own_product():
    """Client knobs form a separate product — the runner's inner loop."""
    cfg = load_config(_client(concurrency=[1, 16]))
    # The deployment product is untouched by client knobs: still one empty pass.
    assert cfg.deployment_combinations() == [()]
    combos = cfg.client_combinations()
    assert len(combos) == 2
    assert [s.value for c in combos for s in c if s.path == "client.concurrency"] == [1, 16]


@allure.feature("pysweep")
def test_max_concurrency_is_the_widest_declared_value():
    """One bearer file serves every combination, so it is sized to the widest one.

    Taking the max rather than the last/first value is the point: a sweep that
    ramps 1 -> 128 would otherwise leave the 128-worker run short of tokens, and
    benchmark.py quits any worker whose bearer line is empty.
    """
    assert load_config(_client(concurrency=[1, 16, 128])).max_concurrency() == 128
    assert load_config(_client(concurrency=[128, 16, 1])).max_concurrency() == 128
    # A scalar is the whole sweep's concurrency.
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
    """Client settings must never reach the apply/converge/verify path.

    ``iter_settings`` (which feeds ``combinations()`` and thus ``build_property``)
    yields only Property-bound knobs, so a client knob cannot be mistaken for
    something the runner should apply to the cluster.
    """
    cfg = load_config(_client(concurrency=[1, 16]))
    assert list(cfg.iter_settings()) == []
    assert not hasattr(cfg.client_combinations()[0][0], "property_cls")


@allure.feature("pysweep")
def test_server_and_bearer_file_are_optional_and_not_swept():
    """The auto-provisioned connection settings default to absent."""
    cfg = load_config(_client())
    assert cfg.client_provisioning_argv() == []
    assert "server" not in cfg.declared_client_names()
    # Declared, they render once for the whole sweep rather than as a dimension.
    cfg = load_config(_client(server="https://erag.com/api/v1/chatqna", bearer_file="uat.txt"))
    assert cfg.client_provisioning_argv() == [
        "-s", "https://erag.com/api/v1/chatqna", "-b", "uat.txt",
    ]
    assert [p for p, _, _ in cfg.iter_client_settings() if "server" in p] == []


@allure.feature("pysweep")
@pytest.mark.parametrize("bad", ["3 minutes", "3", "m", "3d", 3, ""])
def test_duration_must_be_a_parsable_duration(bad):
    """benchmark.duration_to_seconds raises KeyError on anything else.

    Catching the shape here turns a crash an hour into a sweep into a config
    error before anything is applied.
    """
    with pytest.raises(ConfigError):
        load_config(_client(duration=bad))


@allure.feature("pysweep")
@pytest.mark.parametrize("bad", [0, -1, "many", 1.5])
def test_concurrency_must_be_a_positive_integer(bad):
    """``-c 0`` starts no workers and would produce an empty result CSV."""
    with pytest.raises(ConfigError):
        load_config(_client(concurrency=bad))


@allure.feature("pysweep")
@pytest.mark.parametrize("knob", ["input_tokens", "warmup_requests", "failures_allowed"])
def test_zero_is_allowed_where_it_is_meaningful(knob):
    """0 is a real setting for these: no input fixing, no warmup, no failures."""
    cfg = load_config(_client(**{knob: 0}))
    assert next(v for p, _, v in cfg.iter_client_settings() if p.endswith(knob)) == [0]


@allure.feature("pysweep")
def test_unknown_client_knob_raises():
    with pytest.raises(ConfigError):
        load_config(_client(concurency=4))  # missing 'r'


@allure.feature("pysweep")
def test_tokenizer_cannot_be_declared():
    """``-m`` is not a knob: it must be the served model, so it is derived.

    Rejecting the key rather than accepting and overriding it is the point — the
    only thing declaring it could express is a tokenizer that disagrees with the
    model whose tokens are being counted.
    """
    with pytest.raises(ConfigError):
        load_config(_client(tokenizer="nomic-ai/nomic-embed-text-v1"))
    assert client_args.TOKENIZER.derived
    # Still mirrored, so the runner can render it; just not swept or declared.
    assert client_args.TOKENIZER in client_args.CLIENT_ARGS
    assert not client_args.TOKENIZER.sweepable
    cfg = load_config(_client())
    assert [p for p, _, _ in cfg.iter_client_settings() if "tokenizer" in p] == []


@allure.feature("pysweep")
def test_varying_paths_reports_only_multi_valued_knobs():
    """Result files are named by what varies; scalars distinguish nothing."""
    cfg = load_config(
        {
            "client": {**CLIENT, "concurrency": [1, 16]},
            "llms": {"max_new_tokens": [128, 256]},
        }
    )
    assert cfg.varying_paths() == {"client.concurrency", "llms.max_new_tokens"}


@allure.feature("pysweep")
def test_shipped_example_sweep_config_is_valid():
    """The committed configs/sweep.yaml must load (docs stay honest)."""
    example = (
        Path(__file__).resolve().parents[3]
        / "e2e/benchmarks/chatqna/pysweep/configs/sweep.yaml"
    )
    cfg = load_config(example)
    assert cfg.deployment_combinations()
    assert cfg.client_combinations()


# -- run config (--config) ----------------------------------------------------


@allure.feature("pysweep")
def test_run_config_reads_display_kubernetes_client_logs(tmp_path):
    path = tmp_path / "config.yaml"
    path.write_text("display_kubernetes_client_logs: true\n")
    assert load_run_config(path).display_kubernetes_client_logs is True


@allure.feature("pysweep")
def test_kubernetes_client_logs_default_to_off():
    """The kr8s/httpx per-call chatter is noise by default (one line per request)."""
    assert load_run_config({}).display_kubernetes_client_logs is False


@allure.feature("pysweep")
def test_run_config_rejects_unknown_key():
    """Same strictness as the sweep config: a typo must not be ignored."""
    with pytest.raises(ConfigError):
        load_run_config({"display_kubernetes_client_log": False})  # missing trailing 's'


@allure.feature("pysweep")
def test_run_config_rejects_non_boolean():
    with pytest.raises(ConfigError):
        load_run_config({"display_kubernetes_client_logs": "sometimes"})


@allure.feature("pysweep")
def test_shipped_example_run_config_is_valid():
    """The committed configs/general.yaml must load (docs stay honest)."""
    example = (
        Path(__file__).resolve().parents[3]
        / "e2e/benchmarks/chatqna/pysweep/configs/general.yaml"
    )
    assert load_run_config(example).display_kubernetes_client_logs is False
