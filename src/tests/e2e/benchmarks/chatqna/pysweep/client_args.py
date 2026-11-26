# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ClientArg:
    """One ``benchmark.py`` CLI option; pydantic field metadata and its argv rendering rule."""

    flag: str
    dest: str
    yaml_name: str
    summary: str
    sweepable: bool = True
    # Supplied by pysweep with no config key; passing its flag is refused.
    derived: bool = False

    @property
    def path(self) -> str:
        return f"client.{self.yaml_name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ClientArg({self.flag} -> client.{self.yaml_name})"


# Keep in step with benchmark.parse_args.
QUESTION_FILE = ClientArg(
    flag="-f",
    dest="f",
    yaml_name="question_file",
    summary="Question file location; the corpus the load is drawn from.",
)
SERVER = ClientArg(
    flag="-s",
    dest="s",
    yaml_name="server",
    summary="ChatQA endpoint URL. Omit to use /api/v1/chatqna on the deployment's domain.",
    sweepable=False,
)
CONCURRENCY = ClientArg(
    flag="-c",
    dest="c",
    yaml_name="concurrency",
    summary="Number of concurrent users (worker threads).",
)
DURATION = ClientArg(
    flag="-d",
    dest="d",
    yaml_name="duration",
    summary="How long each combination generates load for, e.g. '3m'.",
)
WORKER_STARTUP_DELAY = ClientArg(
    flag="-u",
    dest="u",
    yaml_name="worker_startup_delay",
    summary="Per-worker startup stagger, e.g. '0s' to start all at once.",
)
TOKENIZER = ClientArg(
    flag="-m",
    dest="m",
    yaml_name="tokenizer",
    summary="Tokenizer counting input/output tokens; always the served model.",
    sweepable=False,
    derived=True,
)
FAILURES_ALLOWED = ClientArg(
    flag="-e",
    dest="e",
    yaml_name="failures_allowed",
    summary="Failed requests tolerated before the run aborts.",
)
BEARER_FILE = ClientArg(
    flag="-b",
    dest="b",
    yaml_name="bearer_file",
    summary="File of bearer tokens, one per worker. Omit to let pysweep provision one.",
    sweepable=False,
)
INPUT_TOKENS = ClientArg(
    flag="-x",
    dest="x",
    yaml_name="input_tokens",
    summary="Pad/trim every question to exactly N tokens (0 = leave as written).",
)
WARMUP_REQUESTS = ClientArg(
    flag="-w",
    dest="w",
    yaml_name="warmup_requests",
    summary="Requests per worker excluded from the results as warmup.",
)

CLIENT_ARGS: tuple[ClientArg, ...] = (
    QUESTION_FILE,
    SERVER,
    CONCURRENCY,
    DURATION,
    WORKER_STARTUP_DELAY,
    TOKENIZER,
    FAILURES_ALLOWED,
    BEARER_FILE,
    INPUT_TOKENS,
    WARMUP_REQUESTS,
)

PROVISIONED_ARGS: tuple[ClientArg, ...] = (SERVER, BEARER_FILE)

# Read from the deployment, so unlike PROVISIONED_ARGS these can never be overridden.
DERIVED_ARGS: tuple[ClientArg, ...] = tuple(arg for arg in CLIENT_ARGS if arg.derived)
DERIVED_FLAGS: frozenset[str] = frozenset(arg.flag for arg in DERIVED_ARGS)

CLIENT_ARGS_BY_FLAG = {arg.flag: arg for arg in CLIENT_ARGS}
CLIENT_ARGS_BY_NAME = {arg.yaml_name: arg for arg in CLIENT_ARGS}


class ClientArgsError(Exception):
    """Raised when declared client knobs cannot be turned into a valid invocation."""


class BenchmarkClientArgs:
    """Resolved per-combination client knob values, renderable as argv."""

    def __init__(self, values: dict[ClientArg, Any]):
        self._values = dict(values)

    @classmethod
    def from_settings(cls, settings: Iterable) -> "BenchmarkClientArgs":
        """Build from an iterable of :class:`~.config.ClientSetting`; duplicates raise."""
        values: dict[ClientArg, Any] = {}
        for setting in settings:
            if setting.arg in values:
                raise ClientArgsError(
                    f"'{setting.arg.path}' was resolved twice in one combination "
                    f"({values[setting.arg]!r} then {setting.value!r})."
                )
            values[setting.arg] = setting.value
        return cls(values)

    def value_of(self, arg: ClientArg, default: Any = None) -> Any:
        return self._values.get(arg, default)

    def to_argv(self, include: Iterable[ClientArg] | None = None) -> list[str]:
        """Render the held values as ``benchmark.py`` arguments, in ``CLIENT_ARGS`` order."""
        allowed = set(CLIENT_ARGS if include is None else include)
        argv: list[str] = []
        for arg in CLIENT_ARGS:
            if arg not in allowed or arg not in self._values:
                continue
            value = self._values[arg]
            if value is None:
                continue
            argv += [arg.flag, str(value)]
        return argv

    def snapshot(self) -> dict[str, Any]:
        """Declared (not observed) values keyed by dotted path."""
        return {arg.path: self._values[arg] for arg in CLIENT_ARGS if arg in self._values}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rendered = " ".join(self.to_argv())
        return f"<BenchmarkClientArgs {rendered}>"


def conflicting_flags(benchmark_argv: Iterable[str], declared_names: Iterable[str]) -> list[str]:
    """Return flags after ``--`` owned by the config: declared knobs and every derived flag."""
    argv = list(benchmark_argv)
    declared = set(declared_names)
    conflicts = []
    for arg in CLIENT_ARGS:
        if arg.flag not in argv:
            continue
        if arg in PROVISIONED_ARGS and arg.yaml_name not in declared:
            continue
        conflicts.append(arg.flag)
    return conflicts


def derived_flag_reason(flag: str) -> str:
    if flag == TOKENIZER.flag:
        return (
            "pysweep sets -m from the model the LLM engine is observed to be "
            "serving, so token counts come from that model's own tokenizer"
        )
    return "pysweep derives this value from the deployment"
