#!/bin/python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import argparse
import csv
import linecache
import logging
import re
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml

from tests.e2e.benchmarks.chatqna import benchmark
from tests.e2e.benchmarks.chatqna.benchmark import duration_to_seconds
from tests.e2e.benchmarks.chatqna.pysweep.properties.base import (
    Plane,
    Property,
    PropertyApplyError,
    PropertyReadError,
)
from tests.e2e.benchmarks.chatqna.pysweep.client_args import (
    DERIVED_FLAGS,
    QUESTION_FILE,
    TOKENIZER,
    BenchmarkClientArgs,
    conflicting_flags,
    derived_flag_reason,
)
from tests.e2e.benchmarks.chatqna.pysweep.config import (
    ClientCombination,
    Combination,
    PropertySetting,
    SweepConfig,
    load_config,
    load_run_config,
)
from tests.e2e.benchmarks.chatqna.pysweep.cluster import model_catalog
from tests.e2e.benchmarks.chatqna.pysweep.pre_bench import NUM_VECTORS, run_pre_bench, seed
from tests.e2e.benchmarks.chatqna.pysweep.provenance.commit_info import CommitInfoError, read_commit_info
from tests.e2e.benchmarks.chatqna.pysweep.results import (
    DEFAULT_SUMMARIES_DIR,
    SweepResults,
    prepare_summaries_dir,
    read_requests,
    run_record,
    summarize_requests,
)
from tests.e2e.benchmarks.chatqna.pysweep.cluster.fingerprint import (
    FingerprintClient,
    default_transport,
)
from tests.e2e.benchmarks.chatqna.pysweep.cluster.served_model import ServedModel

logger = logging.getLogger(__name__)

CHATQNA_PATH = "/api/v1/chatqna"

# Relative sweep-config paths resolve against this directory, not the CWD.
PYSWEEP_DIR = Path(__file__).resolve().parent


class SweepError(Exception):
    """Raised when a combination cannot be applied or fails verification."""


class PreflightError(SweepError):
    """Raised before anything is applied when a sweep precondition is not met."""


def build_property(setting: PropertySetting, fingerprint_client: FingerprintClient) -> Property:
    """Instantiate the Property for a config leaf, dispatching on its plane."""
    if setting.property_cls.plane == Plane.RUNTIME_API:
        return setting.property_cls(fingerprint_client)
    if setting.property_cls.plane == Plane.K8S_SPEC:
        return setting.property_cls()
    raise SweepError(
        f"No builder for property '{setting.property_cls.__name__}' "
        f"(plane {setting.property_cls.plane.value}); wire it up in build_property()."
    )


def apply_combination(
    combination: Combination, fingerprint_client: FingerprintClient
) -> dict:
    """Apply, converge and verify one combination; return the observed snapshot."""
    properties = [(setting, build_property(setting, fingerprint_client)) for setting in combination]

    for setting, prop in properties:
        logger.info("Applying %s = %r", setting.path, setting.value)
        try:
            prop.apply(setting.value)
        except (PropertyApplyError, NotImplementedError) as exc:
            raise SweepError(f"{setting.path}: {exc}") from exc

    for setting, prop in properties:
        try:
            prop.wait_until_ready(setting.value)
        except PropertyApplyError as exc:
            raise SweepError(f"{setting.path}: {exc}") from exc

    snapshot = {}
    for setting, prop in properties:
        observed = prop.read()
        if not prop.is_satisfied(setting.value):
            raise SweepError(
                f"{setting.path}: desired {setting.value!r} not satisfied; "
                f"observed {observed!r} after apply/converge."
            )
        snapshot[setting.path] = observed
    return snapshot


def resolve_tokenizer() -> str:
    """Return the HF tokenizer of the observed served model."""
    try:
        served = ServedModel().read()
    except PropertyReadError as exc:
        raise SweepError(
            f"Cannot determine the served model to tokenize with: {exc}. "
            f"The benchmark's -m must be the tokenizer of the model the "
            f"deployment serves."
        ) from exc
    try:
        return model_catalog.tokenizer_for(served)
    except model_catalog.CatalogError as exc:
        raise SweepError(
            f"Cannot map the served model '{served}' to a tokenizer: {exc}"
        ) from exc


def _slug(value) -> str:
    return re.sub(r"[^0-9A-Za-z_]+", "-", str(value)).strip("-")


def _settings_slug(settings, only=None) -> str:
    return "_".join(
        f"{_slug(s.path.rsplit('.', 1)[-1])}-{_slug(s.value)}"
        for s in settings
        if only is None or s.path in only
    )


def combination_slug(combination: Combination, only=None) -> str:
    """A short, unique-per-combination tag like ``max_new_tokens-128``."""
    return _settings_slug(combination, only)


def client_combination_slug(client_combination: ClientCombination, only=None) -> str:
    """A short tag for one client combination, e.g. ``concurrency-4_duration-3m``."""
    return _settings_slug(client_combination, only)


def result_file_name(
    combination: Combination,
    client_combination: ClientCombination,
    only=None,
    fallback: str = "single",
) -> str:
    """Result CSV file name for one run; names only the varying knobs to stay under 255 bytes."""
    slug = "_".join(
        part
        for part in (
            combination_slug(combination, only),
            client_combination_slug(client_combination, only),
        )
        if part
    )
    return f"bench_{slug or fallback}.result.csv"


def count_result_rows(output_file) -> int:
    """Number of data rows the benchmark wrote (0 if the file is missing)."""
    path = Path(output_file)
    if not path.exists():
        return 0
    with path.open(newline="") as handle:
        return sum(1 for _ in csv.reader(handle))


KUBERNETES_CLIENT_LOGGERS = ("httpx", "httpcore", "kr8s", "websockets")


# kr8s's port-forward task writes to closed sockets on teardown; asyncio logs harmless reset tracebacks.
class _DropPortForwardTeardownNoise(logging.Filter):
    _NOISE = (
        "ConnectionResetError",
        "Connection lost",
        "Connection reset by peer",
        "client_connected_cb",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        text = record.getMessage()
        if record.exc_info and record.exc_info[1] is not None:
            text = f"{text} {record.exc_info[1]!r}"
        return not any(marker in text for marker in self._NOISE)


def configure_kubernetes_client_logs(enabled: bool) -> None:
    if enabled:
        return
    for name in KUBERNETES_CLIENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("asyncio").addFilter(_DropPortForwardTeardownNoise())


def resolve_question_file(value: str) -> Path:
    """Resolve a client.question_file against PYSWEEP_DIR; absolute paths pass through."""
    path = Path(value)
    return path if path.is_absolute() else (PYSWEEP_DIR / path)


def question_file_verify(config: SweepConfig) -> None:
    """Raise PreflightError unless every declared client.question_file is a readable file."""
    for value in config.declared_question_files():
        path = resolve_question_file(value)
        if not path.is_file():
            raise PreflightError(
                f"client.question_file '{value}' does not resolve to a readable file "
                f"(looked for {path}). Relative paths are resolved against "
                f"{PYSWEEP_DIR}, not the current directory — the corpora live one "
                f"level up beside benchmark.py, so a bare name needs a '../' prefix "
                f"(e.g. '../questions-short.csv')."
            )
        logger.debug("Preflight: question_file '%s' -> %s", value, path)


def commit_info_verify() -> dict[str, str]:
    """Raise PreflightError unless the checked-out commit can be identified; return it."""
    try:
        info = read_commit_info()
    except CommitInfoError as exc:
        raise PreflightError(f"commit_info: {exc}") from exc
    logger.info(
        "Preflight: results are attributed to %s on %s (%s).",
        info["sha"][:12], info["branch"], info["description"],
    )
    return info


def pre_bench_verify(config: SweepConfig) -> None:
    """Raise PreflightError unless the inputs of every declared pre_bench step exist."""
    if NUM_VECTORS in config.pre_bench_settings():
        for path in (seed.DOCUMENTS_CSV, seed.WIKIPARSE):
            if not path.is_file():
                raise PreflightError(f"pre_bench.{NUM_VECTORS} needs {path}, which does not exist.")


def preflight_check(config: SweepConfig) -> dict[str, str]:
    """Run every preflight check; return the commit the results are attributed to."""
    question_file_verify(config)
    pre_bench_verify(config)
    return commit_info_verify()


def chatqna_url() -> str:
    """Return the ChatQNA URL on the deployment's base_domain_name."""
    from tests.e2e.benchmarks.chatqna.pysweep.cluster.auth import load_build_config

    return f"https://{load_build_config()['base_domain_name']}{CHATQNA_PATH}"


class ProvisionedArgv(list):
    """Benchmark argv with refresh() to re-mint the bearer file, if pysweep owns it."""

    def __init__(self, argv, refresh=None):
        super().__init__(argv)
        self._refresh = refresh

    def refresh(self) -> None:
        if self._refresh is not None:
            self._refresh()


def write_bearer_file(path: str, helper, workers: int) -> None:
    """Write one distinct token per worker; APISIX rate-limits per Authorization header value."""
    tokens = [helper.access_token for _ in range(workers)]
    Path(path).write_text("".join(token + "\n" for token in tokens))
    # benchmark.py reads tokens via linecache; checkcache() misses same-size, same-mtime rewrites.
    linecache.clearcache()


@contextmanager
def provision_benchmark_target(benchmark_args: list[str], workers: int = 1):
    """Yield the argv with -s and a minted bearer file (one token per worker) added unless supplied."""
    args = list(benchmark_args)
    refresh = None

    with ExitStack() as stack:
        if "-s" not in args:
            url = chatqna_url()
            logger.info("No -s given; using the deployment's ChatQNA endpoint %s", url)
            args += ["-s", url]

        if "-b" not in args:
            tmp = stack.enter_context(
                tempfile.NamedTemporaryFile("w", suffix=".uat", delete=True)
            )
            from tests.e2e.benchmarks.chatqna.pysweep.cluster.auth import (
                client_token_lifespan,
                keycloak_helper,
            )

            helper = keycloak_helper()
            stack.enter_context(client_token_lifespan(helper))

            def refresh():
                logger.info(
                    "Minting %d Keycloak token(s), one per worker.", workers
                )
                write_bearer_file(tmp.name, helper, workers)

            refresh()
            args += ["-b", tmp.name]

        yield ProvisionedArgv(args, refresh)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the ChatQNA benchmark across a pysweep sweep config.",
        epilog="Arguments after '--' are passed verbatim to benchmark.py.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to the sweep config YAML (what to vary; see configs/sweep.yaml).",
    )
    parser.add_argument(
        "--config",
        help="Path to the pysweep run config YAML (how to run; see configs/general.yaml).",
    )
    parser.add_argument(
        "--summaries",
        default=DEFAULT_SUMMARIES_DIR,
        help=(
            "Directory a sweep writes into: results.json (the sweep's run records), "
            "benchmark_summary_results.csv (the same records, one dotted-column row "
            "per run) and per_case_results/ (one per-request CSV per run). Created if "
            "missing. WARNING: the previous sweep's outputs there are DELETED before "
            f"this one starts (default: {DEFAULT_SUMMARIES_DIR})."
        ),
    )
    parser.add_argument(
        "benchmark_args",
        nargs=argparse.REMAINDER,
        help=(
            "Extra arguments forwarded to benchmark.py (prefix with '--'). "
            "Load parameters belong in the sweep config's 'client:' section."
        ),
    )
    args = parser.parse_args(argv)
    # argparse.REMAINDER keeps the leading '--'; drop it so it isn't forwarded.
    if args.benchmark_args and args.benchmark_args[0] == "--":
        args.benchmark_args = args.benchmark_args[1:]
    return args


def resolve_benchmark_argv(config: SweepConfig, benchmark_args: list[str]) -> list[str]:
    """Merge the config's client knobs with the `--` args; raise SweepError on duplicated flags."""
    conflicts = conflicting_flags(benchmark_args, config.declared_client_names())
    derived = [flag for flag in conflicts if flag in DERIVED_FLAGS]
    if derived:
        raise SweepError(
            "benchmark flag(s) "
            + ", ".join(f"{flag} ({derived_flag_reason(flag)})" for flag in derived)
            + " cannot be passed after '--': pysweep derives them from the "
            "deployment and there is no config key for them either."
        )
    if conflicts:
        raise SweepError(
            f"benchmark flag(s) {' '.join(conflicts)} passed after '--' are already "
            f"declared in the sweep config's 'client:' section. Remove them from the "
            f"command line (the sweep config is the source of truth for load "
            f"parameters), or remove the client knob to set it per-invocation."
        )
    return config.client_provisioning_argv() + list(benchmark_args)


def resolve_question_file_argv(argv: list[str]) -> list[str]:
    """Rewrite the -f value to an absolute path; benchmark.py opens it relative to the CWD."""
    argv = list(argv)
    if QUESTION_FILE.flag in argv:
        index = argv.index(QUESTION_FILE.flag) + 1
        argv[index] = str(resolve_question_file(argv[index]))
    return argv


LOG_YAML_INDENT = 4


def format_settings_yaml(values: dict, indent: int = LOG_YAML_INDENT) -> str:
    """Render a {dotted_path: value} mapping as sectioned YAML for logging; "" if empty."""
    if not values:
        return ""

    tree: dict = {}
    for path, value in values.items():
        if not isinstance(value, (str, int, float, bool, type(None))):
            value = str(value)
        section, _, leaf = str(path).rpartition(".")
        if section:
            tree.setdefault(section, {})[leaf] = value
        else:
            tree[leaf] = value

    return yaml.safe_dump(
        tree, indent=indent, sort_keys=False, default_flow_style=False
    ).rstrip("\n")


def format_record_yaml(record: dict, indent: int = LOG_YAML_INDENT) -> str:
    return yaml.safe_dump(
        record, indent=indent, sort_keys=False, default_flow_style=False
    ).rstrip("\n")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def run_client_combination(
    benchmark,
    bench_argv: list[str],
    combination: Combination,
    client_combination: ClientCombination,
    label: str,
    results_dir,
    varying_paths=None,
    tokenizer: str | None = None,
) -> tuple[int, Path]:
    """Run one load test; return (error, output_file)."""
    client_args = BenchmarkClientArgs.from_settings(client_combination)
    argv = resolve_question_file_argv(bench_argv + client_args.to_argv())
    if tokenizer is not None:
        argv += [TOKENIZER.flag, tokenizer]
    logger.info(
        "%s load parameters (tokenizer: %s):\n%s",
        label, tokenizer, format_settings_yaml(client_args.snapshot()),
    )

    bench_args = benchmark.parse_args(argv)
    output_file = Path(results_dir) / result_file_name(
        combination, client_combination, only=varying_paths
    )
    error, output_file = benchmark.run_benchmark(bench_args, output_file=output_file)

    rows = count_result_rows(output_file)
    if rows == 0:
        logger.error(
            "%s produced NO result rows in %s "
            "(endpoint unreachable, all requests failed, or duration too short).",
            label, output_file,
        )
        error = error or 1
    logger.info("%s finished (error=%s, rows=%d); results in %s", label, error, rows, output_file)
    return error, output_file


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)

    run_config = load_run_config(args.config) if args.config else load_run_config({})
    configure_kubernetes_client_logs(run_config.display_kubernetes_client_logs)

    config = load_config(args.sweep)
    bench_argv = resolve_benchmark_argv(config, args.benchmark_args)

    commit_info = preflight_check(config)

    deployment_combinations = config.deployment_combinations()
    client_combinations = config.client_combinations()
    varying_paths = config.varying_paths()
    pre_bench = config.pre_bench_settings()
    run_pre_bench(pre_bench)

    layout = prepare_summaries_dir(args.summaries)
    sweep_results = SweepResults(layout, date=utc_now(), commit_info=commit_info)

    fingerprint_client = FingerprintClient(default_transport())

    total = len(deployment_combinations) * len(client_combinations)
    logger.info(
        "Sweep: %d deployment combination(s) x %d client combination(s) = %d run(s).",
        len(deployment_combinations), len(client_combinations), total,
    )
    overall_error = 0
    with provision_benchmark_target(bench_argv, config.max_concurrency()) as provisioned_argv:
        for index, combination in enumerate(deployment_combinations, start=1):
            desired = {s.path: s.value for s in combination}
            logger.info(
                "=== Deployment combination %d/%d: %s ===",
                index, len(deployment_combinations), desired or "(unchanged)",
            )

            snapshot = apply_combination(combination, fingerprint_client)
            logger.info(
                "Observed configuration under test:\n%s", format_settings_yaml(snapshot)
            )

            tokenizer = resolve_tokenizer()
            logger.info("Counting tokens with the served model's tokenizer: %s", tokenizer)

            for client_index, client_combination in enumerate(client_combinations, start=1):
                label = (
                    f"Run {index}.{client_index} "
                    f"(deployment {index}/{len(deployment_combinations)}, "
                    f"client {client_index}/{len(client_combinations)})"
                )
                # Test doubles may yield a plain list.
                if isinstance(provisioned_argv, ProvisionedArgv):
                    provisioned_argv.refresh()
                started_at = utc_now()
                error, output_file = run_client_combination(
                    benchmark, provisioned_argv, combination, client_combination, label,
                    results_dir=layout.per_case_dir,
                    varying_paths=varying_paths, tokenizer=tokenizer,
                )
                finished_at = utc_now()
                overall_error = overall_error or error

                client = {s.path: s.value for s in client_combination}
                record = sweep_results.add_run(
                    run_record(
                        index=f"{index}.{client_index}",
                        started_at=started_at,
                        finished_at=finished_at,
                        exit_code=error,
                        result_file=sweep_results.pointer(output_file),
                        deployment=snapshot,
                        client={**client, "client.tokenizer": tokenizer},
                        pre_bench=pre_bench,
                        results=summarize_requests(
                            read_requests(output_file),
                            duration_to_seconds(str(client["client.duration"])),
                        ),
                    )
                )
                logger.info("%s record:\n%s", label, format_record_yaml(record))

    logger.info(
        "Sweep results (%d run(s)) written to %s", len(sweep_results.runs), sweep_results.write()
    )
    return overall_error


if __name__ == "__main__":
    sys.exit(main())
