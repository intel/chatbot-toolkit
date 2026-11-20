# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Adapter between sweep-declared client knobs and ``benchmark.py``'s CLI.

The load generator's parameters (concurrency, input length, duration, ...) are
sweep knobs in the sense that matters — each distinct value is a different test
scenario and they expand into the cartesian product like any other knob. But
they are **not** :class:`~..base.Property` instances, and deliberately so:

- There is nothing to ``apply``. The value is an argument to a process pysweep
  is about to start, not a mutation of the deployed RAG.
- There is nothing to ``wait_until_ready`` on. No rollout, no reconcile, no
  propagation delay.
- There is nothing to ``read``. ``Property.read`` is contractually *observed*
  state read back from the live system (DESIGN.md calls this non-negotiable —
  it is what makes the preflight meaningful). A client knob has no live system
  behind it; "reading" it could only echo the value pysweep itself holds, which
  is precisely the spec-read the invariant forbids.

So instead of weakening ``Property`` (or inventing a ``Plane`` whose ``apply`` /
``read`` / ``wait_until_ready`` are all vacuous), this module mirrors
``benchmark.py``'s options as :class:`ClientArg` descriptors and renders resolved
values back into that CLI. ``config.py`` binds YAML leaves to these descriptors
the same way it binds other leaves to property classes, so the knobs still get
schema validation, product expansion, filename slugs, and the recorded snapshot
— everything except the apply/converge machinery that does not apply to them.

**Every option is mirrored and the scenario-shaping ones are required** in a
sweep config. A client parameter that silently fell back to ``benchmark.py``'s
own default (``-c 32``, ``-d 30m``, ``-x 0``) would reintroduce exactly the
ambiguity pysweep exists to remove: "what configuration was actually tested".
Requiring them also makes the sweep config the single source of truth, so there
is no precedence question against flags passed after ``--``; passing one of
these flags there is an error (see :func:`conflicting_flags`).

Three options are exceptions, and none of them is a sweep knob:

- ``server`` and ``bearer_file`` stay optional because pysweep auto-provisions
  them (the deployment's own endpoint + a minted token) to keep the "no
  preparation steps required" promise. They are whole-sweep values rather than
  per-combination ones — one endpoint and one token carry every combination — so
  they are not sweepable.
- ``tokenizer`` (``-m``) is **derived, not declared**. It must name the model
  being served, or the ``input_tokens`` / ``output_tokens`` columns (and every
  tokens/s figure derived from them) are counted on a different basis than the
  model actually processed. That model is not a value a config author can be
  trusted to keep in step with ``llms.model`` — it changes with every deployment
  combination — so pysweep reads it back from the running engine
  (``LlmVllmModel.read()``, the observed value) after each combination converges
  and renders it itself. There is no ``client.tokenizer`` key; declaring one is a
  config error, and passing ``-m`` after ``--`` is refused.
"""

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class ClientArg:
    """One ``benchmark.py`` CLI option, mirrored as a sweep knob.

    Used both as pydantic field metadata (``Annotated[..., CONCURRENCY]``, which
    is how ``config.py`` recovers the flag for a YAML leaf) and as the rendering
    rule that turns a resolved value back into argv.
    """

    #: The flag ``benchmark.py`` parses, e.g. ``-c``.
    flag: str
    #: ``benchmark.py``'s argparse destination, e.g. ``c`` for ``args.c``.
    dest: str
    #: The key this knob has in a sweep YAML's ``client:`` section.
    yaml_name: str
    #: What the option does, in ``benchmark.py``'s own terms.
    summary: str
    #: Whether a list may be given (i.e. whether it is a product dimension).
    #: False for whole-sweep values that cannot differ between combinations.
    sweepable: bool = True
    #: True when pysweep supplies this value itself and no config key exists for
    #: it, so declaring it is impossible and passing its flag is refused.
    derived: bool = False

    @property
    def path(self) -> str:
        """The knob's dotted config path, e.g. ``client.concurrency``."""
        return f"client.{self.yaml_name}"

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"ClientArg({self.flag} -> client.{self.yaml_name})"


# -- the mirror of benchmark.py's parse_args ----------------------------------
# One entry per option in benchmark.parse_args. Keep this table in step with it:
# an option here that benchmark.py does not accept fails at load time, and an
# option benchmark.py grows that is missing here simply cannot be swept.

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

#: Every mirrored option, in ``benchmark.py``'s declaration order.
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

#: The two options pysweep provisions itself when they are not declared. They
#: are whole-sweep values, so they are resolved once rather than per combination.
PROVISIONED_ARGS: tuple[ClientArg, ...] = (SERVER, BEARER_FILE)

#: Options pysweep supplies itself, with no config key to declare them. Unlike
#: PROVISIONED_ARGS these cannot be overridden at all: the value is read from the
#: deployment, so any other value would describe a run that did not happen.
DERIVED_ARGS: tuple[ClientArg, ...] = tuple(arg for arg in CLIENT_ARGS if arg.derived)
DERIVED_FLAGS: frozenset[str] = frozenset(arg.flag for arg in DERIVED_ARGS)

CLIENT_ARGS_BY_FLAG = {arg.flag: arg for arg in CLIENT_ARGS}
CLIENT_ARGS_BY_NAME = {arg.yaml_name: arg for arg in CLIENT_ARGS}


class ClientArgsError(Exception):
    """Raised when declared client knobs cannot be turned into a valid invocation."""


class BenchmarkClientArgs:
    """Resolved per-combination client knob values, renderable as argv.

    Built from the client settings of one combination. Holds no cluster state and
    performs no I/O — it is a pure translation from "what the sweep declared" to
    "what ``benchmark.py`` is invoked with", which is the whole reason client
    knobs need no plane.
    """

    def __init__(self, values: dict[ClientArg, Any]):
        self._values = dict(values)

    @classmethod
    def from_settings(cls, settings: Iterable) -> "BenchmarkClientArgs":
        """Build from an iterable of :class:`~.config.ClientSetting`.

        Duplicate declarations of the same option are a programming error rather
        than a config error (the schema permits each key at most once), so they
        raise instead of silently keeping the last value.
        """
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
        """Render the held values as ``benchmark.py`` arguments.

        Emitted in ``CLIENT_ARGS`` order for stable, diffable command lines.
        ``include`` restricts the output to a subset of options, which is how the
        per-combination knobs are kept separate from the whole-sweep endpoint
        ones (those are resolved once, before the port-forward is opened).
        """
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
        """The recorded 'what was tested' contribution for these knobs.

        Unlike a property's snapshot entry this is the *declared* value, not an
        observed read-back — there is no live system to observe. It is recorded
        anyway because it is half of what identifies a result: the same
        deployment under two different loads produces two different numbers.
        """
        return {arg.path: self._values[arg] for arg in CLIENT_ARGS if arg in self._values}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        rendered = " ".join(self.to_argv())
        return f"<BenchmarkClientArgs {rendered}>"


def conflicting_flags(benchmark_argv: Iterable[str], declared_names: Iterable[str]) -> list[str]:
    """Return flags passed after ``--`` that a declared client knob already owns.

    The sweep config is the single source of truth for client parameters, so a
    flag appearing in both places has no defensible resolution: honouring the
    YAML would silently ignore an explicit command line, and honouring the
    command line would silently collapse a declared sweep dimension. Both are
    the "you cannot tell what was tested" failure pysweep exists to prevent, so
    the caller reports these and refuses to start.

    ``server`` / ``bearer_file`` are only conflicts when actually declared —
    passing ``-s`` / ``-b`` on the command line while the YAML omits them is the
    supported way to bypass auto-provisioning.

    Derived options (``-m``) are *always* conflicts. They cannot be declared, so
    there is nothing to omit to make room for them: pysweep reads the value from
    the deployment, and a caller-supplied one would count tokens with a tokenizer
    other than the model that produced them.
    """
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
    """Why a derived flag cannot be passed, for the error message."""
    if flag == TOKENIZER.flag:
        return (
            "pysweep sets -m from the model the LLM engine is observed to be "
            "serving, so token counts come from that model's own tokenizer"
        )
    return "pysweep derives this value from the deployment"
