#!/bin/python3
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""pysweep benchmark runner.

Ties the pieces together (DESIGN.md "Execution flow"):

1. **Load & validate** a sweep config (``config.load_config`` — invalid YAML
   fails here, before anything is applied), then **preflight** the preconditions
   the schema cannot express (``preflight``) — corpora and models both, once,
   while stopping is still free.
2. For each **deployment** combination in the cartesian ``product`` (outer loop):
   a. **apply** every declared property's value (idempotent),
   b. **wait_until_ready** on each,
   c. **verify** via *observed* state (``is_satisfied(read())``) — a mismatch
      aborts that combination rather than benchmarking the wrong config,
   d. **snapshot** the observed values (so "what was tested" is recorded),
   e. **resolve the tokenizer** from the observed served model, so token counts
      come from the tokenizer of the model that actually produced them,
   f. for each **client** combination (inner loop), **run the benchmark** by
      reusing ``benchmark.run_benchmark`` unchanged, then
   g. **summarize** that run into the sweep's summary CSV (``results.py``): one
      row per run, a column per declared knob, and the TTFT/TPOT KPIs.

The nesting is the point. A deployment knob costs a rollout — minutes to hours
for a model switch — so it is established once and then held still while every
load scenario runs against it. Client knobs (concurrency, input length, duration)
change nothing on the cluster, so each additional value costs only another load
test; they are the innermost dimension and need no apply, no convergence wait,
and no read-back (see ``client_args.py``). One deployment combination therefore
yields ``len(client_combinations)`` result CSVs, all directly comparable because
they ran against a deployment that was verified once and never touched between
them.

Those per-run CSVs are per-*request* rows and carry no configuration, so a sweep
also writes one **summary CSV** holding a row per run: every declared knob as a
dotted-path column, then the mean and p90 of TTFT and TPOT. It is rewritten after
each run, so a sweep killed partway still leaves a readable table of what it got
through.

Everything a sweep writes goes under one directory (``--summaries``, default
``./results``)::

    results/
      benchmark_summary_results.csv    # the table: one row per run
      per_case_results/                # one per-request CSV per run
        bench_max_new_tokens-64_k-3_concurrency-1.result.csv
        ...

That directory is **cleared of the previous run's artifacts before the sweep
starts** — see :func:`~.results.prepare_summaries_dir` for why a merge would be
worse than a clean slate.

Usage::

    PYTHONPATH=src python -m tests.e2e.benchmarks.chatqna.pysweep.pysweep_bench \
        --sweep sweep.yaml [--config config.yaml]

``--sweep`` declares *what to vary* (the knobs and their values, including the
required ``client:`` section); ``--config`` declares *how to run* (harness
behaviour, e.g. whether benchmark.py's own logs are displayed) and is optional.
Arguments after ``--`` are still forwarded to benchmark.py, but a flag the sweep
config already owns is rejected rather than silently overriding a declared sweep
dimension — the config is the single source of truth for load parameters.

**Zero-prerequisite runs.** With no benchmark ``-s`` given, pysweep_bench drives
the deployment's own ChatQNA endpoint — the same path ``benchmark.py`` defaults
to, on the domain read from the installed environment's build config (see
:func:`chatqna_url`). With no ``-b`` (bearer file) given it provisions one
automatically, minting one real Keycloak token per worker. The only thing a
caller must supply is ``--sweep``.
"""

import argparse
import csv
import logging
import re
import sys
import tempfile
from contextlib import ExitStack, contextmanager
from pathlib import Path

import yaml

from tests.e2e.benchmarks.chatqna import benchmark
from tests.e2e.benchmarks.chatqna.pysweep.base import (
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
from tests.e2e.benchmarks.chatqna.pysweep.properties.llm_model import LlmModel
from tests.e2e.benchmarks.chatqna.pysweep.properties.sub import model_catalog, model_manager
from tests.e2e.benchmarks.chatqna.pysweep.results import (
    DEFAULT_SUMMARIES_DIR,
    SweepSummary,
    prepare_summaries_dir,
)
from tests.e2e.benchmarks.chatqna.pysweep.runtime_api import (
    FingerprintClient,
    default_transport,
)

logger = logging.getLogger(__name__)

# ChatQNA entrypoint: the same API path benchmark.py drives, so a sweep measures
# what a plain benchmark.py run measures. Only the host differs, and it is not a
# constant — see chatqna_url().
CHATQNA_PATH = "/api/v1/chatqna"

# Relative paths in a sweep config are resolved against THIS file's directory,
# never the process's working directory. A sweep YAML is checked in next to the
# harness, so a path in it means the same thing wherever the sweep is launched
# from; resolving against the CWD would instead make the same config name a
# different file per caller — and `benchmark.py`'s bare open() does exactly that,
# which is the bug this anchor exists to close.
PYSWEEP_DIR = Path(__file__).resolve().parent


class SweepError(Exception):
    """Raised when a combination cannot be applied or fails verification."""


class PreflightError(SweepError):
    """Raised when a precondition for running the sweep is not met.

    Distinct from a bare :class:`SweepError` because it is raised *before*
    anything is applied or benchmarked: nothing has to be undone, and the fix is
    always in the caller's config or environment rather than in the cluster.
    """


def build_property(setting: PropertySetting, fingerprint_client: FingerprintClient) -> Property:
    """Instantiate the ``Property`` bound to a config leaf with the resources it needs.

    Properties differ in what they need to reach their plane. Runtime-API
    properties take the fingerprint client; K8s-plane properties talk to the
    cluster themselves (kr8s reads its own kubeconfig) and so take no argument.
    Dispatch on the class's declared plane so adding a property of an existing
    plane needs no change here.
    """
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
    """Apply, converge, and verify one combination; return the observed snapshot.

    Raises :class:`SweepError` if any property fails to converge to its desired
    value, so a benchmark never runs against a configuration we could not
    actually establish.
    """
    properties = [(setting, build_property(setting, fingerprint_client)) for setting in combination]

    for setting, prop in properties:
        logger.info("Applying %s = %r", setting.path, setting.value)
        try:
            prop.apply(setting.value)
        except (PropertyApplyError, NotImplementedError) as exc:
            # The property could not establish the requested state (a value the
            # deployment cannot run, a manifest it no longer recognises, or a
            # knob that is not writable). That is a sweep problem, not a crash.
            raise SweepError(f"{setting.path}: {exc}") from exc

    for setting, prop in properties:
        try:
            prop.wait_until_ready(setting.value)
        except PropertyApplyError as exc:
            # Applied but never took effect — never benchmark on that.
            raise SweepError(f"{setting.path}: {exc}") from exc

    # Verify against OBSERVED state (DESIGN.md: the set/verify split).
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


def resolve_tokenizer(snapshot: dict) -> str:
    """The tokenizer to count tokens with: the one of the *observed* served model.

    ``benchmark.py``'s ``-m`` is not a sweep knob, so nobody declares it. It has
    exactly one correct value — the served model's tokenizer — because it produces
    the ``input_tokens`` / ``output_tokens`` columns and pads questions to
    ``client.input_tokens``. Counted with any other model's tokenizer, those
    numbers describe a tokenization the deployment never performed, and tokens/s
    stops being comparable to anything.

    Two steps, because the two names differ. What the deployment serves is a
    *catalog* name (``llama3-8b-awq``, the value of ``LLM_MODEL_NAME`` and the name
    of the ``LLMInferenceService``); ``AutoTokenizer.from_pretrained`` needs a
    HuggingFace repo id. The model catalog holds both, so the observed name is
    resolved through it (:func:`model_catalog.tokenizer_for`) rather than guessed.

    Preferring the ``snapshot`` avoids a second read: when the sweep varies
    ``llms.model``, :func:`apply_combination` has already read the value back from
    the pipeline for this combination, so it is both current and observed. When the
    sweep does not vary the model there is no entry, and the deployment is asked
    directly — it is still serving *something*, and that something is what the
    benchmark is about to load-test.

    Reads observed state in both cases rather than echoing a desired value, for
    the same reason properties do: a rollout that half-succeeded would otherwise
    be tokenized as though it had worked.
    """
    served = None
    for path, observed in snapshot.items():
        if path.endswith(".model"):
            served = observed
            break
    if served is None:
        try:
            served = LlmModel().read()
        except (PropertyReadError, NotImplementedError) as exc:
            # Without it every token count would be on an unknown basis, so this
            # is not something to fall back from silently.
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
    """Filesystem-safe fragment for a config value (used in result filenames).

    Underscores are preserved so knob names stay readable (``max_new_tokens``).
    """
    return re.sub(r"[^0-9A-Za-z_]+", "-", str(value)).strip("-")


def _settings_slug(settings, only=None) -> str:
    """Join ``<knob>-<value>`` fragments for the given settings.

    ``only`` restricts the output to those dotted paths — see
    :func:`result_file_name` for why a filename names only what varies.
    """
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
    """Result CSV *file name* for one (deployment, client) pair.

    A bare name, not a path: where it goes is the caller's business (the
    ``per_case_results`` subdirectory of ``--summaries``), and what it is called is
    this function's.

    Both slugs contribute, because both identify the run and either alone repeats
    across the sweep — a name built from one would clobber the other's results.

    ``only`` (the sweep's varying paths) is what keeps the name usable. Naming
    every declared knob is tempting since the filename is the record of what
    produced the numbers, but with every option declared an HF repo id and a
    question-file path push it past the 255-byte filename limit. So the name
    carries what *distinguishes*
    this run, and the full configuration — deployment snapshot and load parameters
    both — is logged per run instead. ``fallback`` names the degenerate case where
    the sweep has a single combination and so nothing varies at all.
    """
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


# Loggers belonging to the Kubernetes client stack. kr8s talks to the API server
# over httpx/httpcore and logs a line per call; with a port-forward under load
# that is one '101 Switching Protocols' per benchmark request.
KUBERNETES_CLIENT_LOGGERS = ("httpx", "httpcore", "kr8s", "websockets")


class _DropPortForwardTeardownNoise(logging.Filter):
    """Drop kr8s's port-forward connection-reset tracebacks.

    When the load test closes a connection (and again when the sweep tears the
    forward down), kr8s's background asyncio task may still be writing to the
    now-closed socket. The resulting ``ConnectionResetError`` is not awaited by
    anyone, so asyncio's default handler dumps a multi-frame ExceptionGroup
    traceback. It is a teardown race *inside kr8s* with no effect on the
    benchmark — verified: every recorded request still returns 200 — but it
    swamps the output. Only these reset/closed records are dropped; any other
    asyncio error still surfaces.
    """

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
    """Quiet the kr8s/httpx client loggers unless explicitly enabled.

    Applied once at startup (not scoped to the benchmark) because the port-forward
    spans the whole sweep. Raising the level rather than filtering keeps genuine
    warnings and errors — only the per-call INFO chatter goes away. The asyncio
    logger keeps its level (real errors matter) but loses kr8s's teardown resets.
    """
    if enabled:
        return
    for name in KUBERNETES_CLIENT_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)
    logging.getLogger("asyncio").addFilter(_DropPortForwardTeardownNoise())


def resolve_question_file(value: str) -> Path:
    """Resolve a declared ``client.question_file`` to a concrete path.

    Relative values are anchored at :data:`PYSWEEP_DIR` (this file's directory),
    so ``../questions-short.csv`` names the corpus beside ``benchmark.py``
    regardless of where the sweep was launched from. Absolute paths are returned
    unchanged — an explicit absolute path is already unambiguous, and rewriting
    it would be surprising.
    """
    path = Path(value)
    return path if path.is_absolute() else (PYSWEEP_DIR / path)


def question_file_verify(config: SweepConfig) -> None:
    """Check every declared ``client.question_file`` names a readable file.

    ``question_file`` is sweepable, so a config may declare a list of corpora;
    all of them are checked, since a bad path in the *last* one would otherwise
    surface only after the earlier runs had already spent their load.

    The schema cannot do this: it validates the value as a non-empty string
    (``PathValue``), and whether that string resolves to a file is a fact about
    the filesystem rather than about the config's shape.
    """
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


def model_verify(config: SweepConfig) -> None:
    """Check every declared ``llms.model`` is deployable before the sweep starts.

    Two checks per declared model, in increasing cost order:

    1. It is a **name in the model catalog**. The catalog is what maps the short
       served-model name to a HuggingFace repo id and a resource profile; a name
       that is not in it can neither be deployed nor tokenized. The schema cannot
       check this — it validates a non-empty string, and the set of valid names is
       a fact about the environment's catalog file.
    2. ``model-manager deploy --dry-run`` accepts it. That is the same code path
       the apply will take, minus the side effects: it renders the manifest and
       runs model-manager's own preflight (NUMA/balloons capacity, node fit), so a
       model the cluster cannot host fails here rather than after the first
       combination has already run.

    Raises :class:`PreflightError` on the first failure.
    """
    models = config.declared_models()
    if not models:
        logger.debug("Preflight: sweep declares no llms.model; skipping model checks.")
        return

    try:
        known = model_catalog.known_models()
    except model_catalog.CatalogError as exc:
        raise PreflightError(
            f"Cannot read the model catalog needed to validate llms.model: {exc}"
        ) from exc

    for model in models:
        if model not in known:
            raise PreflightError(
                f"llms.model '{model}' is not in the model catalog "
                f"({model_catalog.catalog_path()}); available: {', '.join(sorted(known))}. "
                f"A model is named by its catalog name (e.g. 'llama3-8b-awq'), not by "
                f"its HuggingFace repo id."
            )
        try:
            model_manager.dry_run(model)
        except model_manager.ModelManagerError as exc:
            raise PreflightError(
                f"llms.model '{model}' cannot be deployed: model-manager's dry run "
                f"rejected it: {exc}"
            ) from exc
        logger.info("Preflight: model '%s' is in the catalog and passes a dry run.", model)


def preflight_check(config: SweepConfig) -> None:
    question_file_verify(config)
    model_verify(config)


def chatqna_url() -> str:
    """The deployed ChatQNA endpoint, with the domain read from the deployment.

    The host is **derived, not declared**: it comes from ``base_domain_name`` in
    the installed environment's build config — the same value, read from the same
    files, that :class:`~.runtime_api.IngressTransport` and ``KeycloakHelper``
    use. Hardcoding a domain here would let a sweep drive load at one cluster
    while configuring another, and the minted token would not even be valid for
    it. Point ``PYSWEEP_BUILD_CONFIG_DIR`` at a different ``env/<name>/`` to
    benchmark a different deployment, exactly as for every other plane.
    """
    from tests.e2e.benchmarks.chatqna.pysweep.authentication import load_build_config

    # load_build_config guarantees a non-empty base_domain_name or raises.
    return f"https://{load_build_config()['base_domain_name']}{CHATQNA_PATH}"


@contextmanager
def provision_benchmark_target(benchmark_args: list[str], workers: int = 1):
    """Ensure the benchmark has a reachable endpoint and a bearer file.

    Yields the ``benchmark_args`` to actually use. If the caller supplied no
    ``-s``, inject :func:`chatqna_url` (the deployment's own ChatQNA endpoint —
    the same path ``benchmark.py`` defaults to). If the caller supplied no
    ``-b``, write a bearer file holding ``workers`` freshly-minted Keycloak
    tokens, one per line.

    ``workers`` must be the *widest* concurrency the sweep will reach (see
    :meth:`~.config.SweepConfig.max_concurrency`), because the file is written
    once and shared by every combination while ``benchmark.py`` indexes into it
    per worker.

    "Supplied" covers both sources: a flag after ``--`` and a ``client.server`` /
    ``client.bearer_file`` in the sweep config, since :func:`main` renders the
    latter into these args before calling. Entered once per sweep — the endpoint
    and the minted tokens are shared by every combination.
    """
    args = list(benchmark_args)
    has_bearer = "-b" in args

    with ExitStack() as stack:
        if "-s" not in args:
            url = chatqna_url()
            logger.info("No -s given; using the deployment's ChatQNA endpoint %s", url)
            args += ["-s", url]

        if not has_bearer:
            tmp = stack.enter_context(
                tempfile.NamedTemporaryFile("w", suffix=".uat", delete=True)
            )
            # The target is the authenticated ERAG ingress, so mint real Keycloak
            # user tokens.
            from tests.e2e.benchmarks.chatqna.pysweep.authentication import (
                keycloak_helper,
            )

            logger.info(
                "No -b given; minting %d Keycloak token(s), one per worker.", workers
            )
            helper = keycloak_helper()
            # One DISTINCT token per worker, matching generate_uat_to_file.sh —
            # which the shell benchmark calls with the concurrency for exactly this
            # reason. Each `access_token` read is a fresh password grant, so the
            # strings differ. That matters because APISIX's limit-count plugin on
            # /api/v1/chatqna buckets by the Authorization header *value*: one token
            # repeated would put every worker in a single 30-requests-per-60s
            # bucket, capping the whole sweep near 0.5 req/s no matter what
            # concurrency it declares.
            tokens = [helper.access_token for _ in range(workers)]
            tmp.write("".join(token + "\n" for token in tokens))
            tmp.flush()
            args += ["-b", tmp.name]

        yield args


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Run the ChatQNA benchmark across a pysweep sweep config.",
        epilog="Arguments after '--' are passed verbatim to benchmark.py.",
    )
    parser.add_argument(
        "--sweep",
        required=True,
        help="Path to the sweep config YAML (what to vary; see examples/sweep.yaml).",
    )
    parser.add_argument(
        "--config",
        help="Path to the pysweep run config YAML (how to run; see examples/config.yaml).",
    )
    parser.add_argument(
        "--summaries",
        default=DEFAULT_SUMMARIES_DIR,
        help=(
            "Directory a sweep writes into: benchmark_summary_results.csv (a row "
            "per run, a column per declared knob, TTFT/TPOT mean/p90) and "
            "per_case_results/ (one per-request CSV per run). Created if missing. "
            "WARNING: the previous sweep's summary and per-case CSVs there are "
            f"DELETED before this one starts (default: {DEFAULT_SUMMARIES_DIR})."
        ),
    )
    # Everything after '--' is forwarded to the existing benchmark CLI. Load
    # parameters belong in the sweep config's `client:` section; a flag here that
    # the config already declares is an error, not an override.
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
    """Merge the sweep config's whole-sweep client knobs with the caller's ``--`` args.

    Raises :class:`SweepError` if a forwarded flag duplicates a client knob the
    config declares. Silently letting one win would mean the command line and the
    config disagree about what was tested, which is the failure mode this harness
    exists to prevent — so it is refused before anything is applied.
    """
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
    # Config-declared -s/-b first so provision_benchmark_target sees them as
    # "supplied" and skips auto-provisioning for those.
    return config.client_provisioning_argv() + list(benchmark_args)


def resolve_question_file_argv(argv: list[str]) -> list[str]:
    """Rewrite the ``-f`` value in ``argv`` to its anchored absolute path.

    Done at the argv boundary because ``benchmark.py`` opens the value with a
    bare ``open()``, i.e. against the process's working directory. Handing it an
    absolute path is what makes the anchor authoritative without modifying
    ``benchmark.py``, whose other callers rely on its current behaviour.

    Only the value pysweep rendered from the config is rewritten; the declared
    string is what the summary CSV records, so provenance still shows what the
    config asked for rather than a machine-specific absolute path.
    """
    argv = list(argv)
    if QUESTION_FILE.flag in argv:
        index = argv.index(QUESTION_FILE.flag) + 1
        argv[index] = str(resolve_question_file(argv[index]))
    return argv


#: Indentation of the logged YAML blocks. Four spaces, matching the sweep config
#: files those blocks are meant to be read against.
LOG_YAML_INDENT = 4


def format_settings_yaml(values: dict, indent: int = LOG_YAML_INDENT) -> str:
    """Render a ``{dotted_path: value}`` mapping as a YAML block, for logging.

    A dict repr of a run's knobs is one unreadable line quoted by Python's rules;
    the same values grouped back into their config sections read like the sweep
    YAML they came from::

        client:
            question_file: ../questions-short.csv
            concurrency: 16

    Undotted keys (the summary row's KPIs) stay at the top level. Key order is the
    caller's, not sorted, and values ``yaml.safe_dump`` cannot represent are
    stringified rather than raised — a log line must not end a long sweep.
    Returns ``""`` for an empty mapping, so a caller can say "(unchanged)".
    """
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
    """Run one load test: the innermost step of the sweep.

    No apply, no convergence wait, no verification — the client knobs are
    arguments to the process about to start, so establishing them *is* rendering
    them into argv. Returns ``(error, output_file)``.

    ``tokenizer`` is the one client argument nobody declares: it is the served
    model, resolved once per deployment combination by :func:`resolve_tokenizer`
    and passed in here rather than re-read per load test, since the deployment is
    held still across this whole inner loop.

    ``results_dir`` is where the per-request CSV goes — the ``per_case_results``
    subdirectory of ``--summaries``. It is joined here, and absolute, because
    ``benchmark.py`` opens its output file with a bare ``open()``: a relative name
    would land against the process's working directory instead.
    """
    client_args = BenchmarkClientArgs.from_settings(client_combination)
    argv = resolve_question_file_argv(bench_argv + client_args.to_argv())
    if tokenizer is not None:
        argv += [TOKENIZER.flag, tokenizer]
    # The tokenizer stays out of the YAML block: it is derived from the served
    # model, and 'client.tokenizer' is a ConfigError, so a block containing it
    # would not be pasteable into a sweep config.
    logger.info(
        "%s load parameters (tokenizer: %s):\n%s",
        label, tokenizer, format_settings_yaml(client_args.snapshot()),
    )

    # Reuse benchmark.py unchanged: parse its args, then run one load test into a
    # CSV named for both combinations so results are never clobbered.
    bench_args = benchmark.parse_args(argv)
    output_file = Path(results_dir) / result_file_name(
        combination, client_combination, only=varying_paths
    )
    error, output_file = benchmark.run_benchmark(bench_args, output_file=output_file)

    rows = count_result_rows(output_file)
    if rows == 0:
        # A load test that recorded nothing is a failed run, not a pass:
        # surface it instead of exiting 0 with an empty CSV.
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

    # Run config is optional; its defaults are the previous behaviour.
    run_config = load_run_config(args.config) if args.config else load_run_config({})
    configure_kubernetes_client_logs(run_config.display_kubernetes_client_logs)

    config = load_config(args.sweep)
    bench_argv = resolve_benchmark_argv(config, args.benchmark_args)

    preflight_check(config)

    deployment_combinations = config.deployment_combinations()
    client_combinations = config.client_combinations()
    # Result files are named by what varies; the rest is constant and logged.
    varying_paths = config.varying_paths()
    # Last thing before the sweep proper, and the first destructive one: the
    # output directory is cleared here rather than at startup, so a bad config or a
    # failed preflight does not cost the previous sweep's results.
    layout = prepare_summaries_dir(args.summaries)
    # ...and everything declared, varying or not, becomes a column of the summary,
    # which is the artifact that has to stand on its own once the sweep is over.
    summary = SweepSummary(config.knob_paths(), path=layout.summary_file)

    fingerprint_client = FingerprintClient(default_transport())

    total = len(deployment_combinations) * len(client_combinations)
    logger.info(
        "Sweep: %d deployment combination(s) x %d client combination(s) = %d run(s).",
        len(deployment_combinations), len(client_combinations), total,
    )
    overall_error = 0
    # Provision the endpoint/bearer once for the whole sweep: the endpoint and the
    # minted tokens are shared by every combination's load test. Sized to the
    # widest concurrency, since one bearer file has to serve every combination.
    with provision_benchmark_target(bench_argv, config.max_concurrency()) as provisioned_argv:
        for index, combination in enumerate(deployment_combinations, start=1):
            desired = {s.path: s.value for s in combination}
            logger.info(
                "=== Deployment combination %d/%d: %s ===",
                index, len(deployment_combinations), desired or "(unchanged)",
            )

            # Applied and verified ONCE, then held still for every client
            # combination below — which is what makes those results comparable.
            snapshot = apply_combination(combination, fingerprint_client)
            logger.info(
                "Observed configuration under test:\n%s", format_settings_yaml(snapshot)
            )

            # Resolved here, not per load test: the deployment is held still for
            # the whole inner loop, so the served model cannot change within it.
            tokenizer = resolve_tokenizer(snapshot)
            logger.info("Counting tokens with the served model's tokenizer: %s", tokenizer)

            for client_index, client_combination in enumerate(client_combinations, start=1):
                label = (
                    f"Run {index}.{client_index} "
                    f"(deployment {index}/{len(deployment_combinations)}, "
                    f"client {client_index}/{len(client_combinations)})"
                )
                error, output_file = run_client_combination(
                    benchmark, provisioned_argv, combination, client_combination, label,
                    results_dir=layout.per_case_dir,
                    varying_paths=varying_paths, tokenizer=tokenizer,
                )
                overall_error = overall_error or error

                # Observed values for the deployment half, declared ones for the
                # client half — the same asymmetry as the per-run log line, for
                # the same reason (see BenchmarkClientArgs.snapshot).
                row = summary.add_run(
                    knob_values={
                        **snapshot,
                        **{s.path: s.value for s in client_combination},
                    },
                    result_file=output_file,
                    error=error,
                    served_model=tokenizer,
                )
                # The summary row's KPI keys are undotted, so they stay top-level
                # under the knob sections rather than nesting.
                logger.info("%s summary:\n%s", label, format_settings_yaml(row))

    logger.info("Sweep summary (%d run(s)) written to %s", len(summary.rows), summary.write())
    return overall_error


if __name__ == "__main__":
    sys.exit(main())
