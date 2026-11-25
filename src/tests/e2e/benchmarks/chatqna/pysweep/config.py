# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Schema-validated sweep configuration for pysweep.

A sweep is declared in YAML (DESIGN.md "Sweep config (YAML)"). This module is
the single place that:

1. **Validates** that YAML against a Pydantic schema — unknown keys and wrong
   value types raise :class:`ConfigError` *before* any rollout is triggered, so
   a typo never silently tests the wrong configuration.
2. **Binds** each config leaf to the thing that knows how to realise it. The
   binding lives on the field itself, so the YAML key is statically tied to its
   implementation and there is no name-string lookup table to drift out of sync.
   Two kinds of leaf exist:

   - ``Annotated[..., PropertyBinding(MaxNewTokens)]`` — a **deployment** knob,
     driven by a :class:`~..base.Property` that applies it to the cluster and
     reads it back.
   - ``Annotated[..., CONCURRENCY]`` — a **client** knob, a
     :class:`~.client_args.ClientArg` that is rendered into ``benchmark.py``'s
     CLI. It has no plane: nothing to apply, converge on, or read back (see
     ``client_args.py`` for why that distinction is kept rather than folded into
     ``Property``).

3. **Expands** the declared values into cartesian ``product``\\ s of sweep
   combinations (DESIGN.md combination strategy). Deployment and client knobs
   expand into *separate* products, because they are nested rather than
   interleaved: the runner walks deployment combinations in the outer loop
   (each costs a rollout) and, for each one, every client combination in the
   inner loop (each costs only another load test against the deployment already
   standing). Within the deployment product, expensive K8s-plane knobs are
   themselves the outer dimension.

Every deployment property is **optional**: omit a key to leave that knob
untouched. Client knobs are **required** — see ``client_args.py``; falling back
to ``benchmark.py``'s own defaults would leave "what load was applied"
unanswerable from the config. A value may be a single scalar (fixed across the
sweep) or a list (one product dimension).

Adding a deployment knob = add one annotated, optional field to the relevant
section model (and, per DESIGN.md objective 6, ship the property's e2e test).
Adding a client knob = add a :class:`~.client_args.ClientArg` mirroring the new
``benchmark.py`` option and one annotated field on :class:`ClientConfig`. No
other wiring — unless the option has only one correct value given the deployment
(as ``-m`` does), in which case it is not a knob: mark the ``ClientArg``
``derived=True``, add no field here, and have the runner resolve it.
"""

import itertools
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any, Iterator

import yaml
from pydantic import (
    BaseModel,
    ConfigDict,
    NonNegativeInt,
    PositiveInt,
    StringConstraints,
    ValidationError,
)

from .base import Plane, Property
from .client_args import (
    BEARER_FILE,
    CONCURRENCY,
    DURATION,
    FAILURES_ALLOWED,
    INPUT_TOKENS,
    QUESTION_FILE,
    SERVER,
    WARMUP_REQUESTS,
    WORKER_STARTUP_DELAY,
    ClientArg,
)
from .properties.max_tokens import MaxNewTokens

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when a sweep config file is invalid (bad schema, key, or value)."""


class PropertyBinding:
    """Field metadata tying a config leaf to the ``Property`` class it drives.

    Attached via ``Annotated[<type>, PropertyBinding(SomeProperty)]``. Pydantic
    v2 preserves it in ``FieldInfo.metadata``, so :class:`SweepConfig` can walk
    its own fields and recover, for each declared knob, the class that applies
    it — without a separate name→class registry.
    """

    def __init__(self, property_cls: type[Property]):
        self.property_cls = property_cls

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PropertyBinding({self.property_cls.__name__})"


@dataclass(frozen=True)
class PropertySetting:
    """One resolved deployment knob value within a single sweep combination."""

    path: str  # dotted config path, e.g. "llms.max_new_tokens"
    property_cls: type[Property]  # the Property class that applies/reads it
    value: Any  # the concrete value for this combination


@dataclass(frozen=True)
class ClientSetting:
    """One resolved client knob value within a single client combination.

    Deliberately a separate type from :class:`PropertySetting` rather than the
    same type with a null ``property_cls``: the runner's apply/converge/verify
    loop takes ``PropertySetting``\\ s, and keeping client knobs out of that type
    means they cannot reach it by accident. What they carry instead is the
    ``benchmark.py`` flag they render to.
    """

    path: str  # dotted config path, e.g. "client.concurrency"
    arg: ClientArg  # the benchmark.py option it renders to
    value: Any  # the concrete value for this combination


# A sweep combination is an ordered tuple of per-property settings.
Combination = tuple[PropertySetting, ...]
# A client combination is an ordered tuple of per-client-knob settings.
ClientCombination = tuple[ClientSetting, ...]


# -- value types --------------------------------------------------------------
# A knob's YAML value is either a single value or a list of values to sweep.
# Constraints (e.g. PositiveInt) are enforced by Pydantic and reported as
# ConfigError. `| None` + default None makes every knob optional.
MaxTokensValue = PositiveInt | list[PositiveInt]

# Client knob value types. These mirror benchmark.py's argparse types, tightened
# where argparse accepts more than the tool can actually use: `-c 0` starts no
# workers and `-e -1`/`-w -1` are nonsense, so they are constrained here and
# rejected as a ConfigError instead of producing an empty result CSV an hour later.
CountValue = PositiveInt | list[PositiveInt]
NonNegativeCountValue = NonNegativeInt | list[NonNegativeInt]
PathValue = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
PathListValue = PathValue | list[PathValue]
# benchmark.duration_to_seconds parses "<number><s|m|h>" and raises KeyError on
# anything else. Validating the shape here turns that late crash into a config error.
DurationValue = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^\d+(\.\d+)?[smh]$")
]
DurationListValue = DurationValue | list[DurationValue]


# -- section models -----------------------------------------------------------
# One model per config section. `extra="forbid"` makes any unknown key an error.


class _StrictModel(BaseModel):
    """Only pre-defined fields are allowed."""
    model_config = ConfigDict(extra="forbid")


class LlmsConfig(_StrictModel):
    """LLM knobs."""

    max_new_tokens: Annotated[MaxTokensValue | None, PropertyBinding(MaxNewTokens)] = None


class ClientConfig(_StrictModel):
    """Load-generator knobs — the full set of ``benchmark.py``'s options.

    Every field mirrors one option of ``benchmark.parse_args``; the annotated
    :class:`~.client_args.ClientArg` is what carries the flag it renders back to.

    **These are required, unlike deployment knobs.** A deployment knob that is
    omitted means "leave it alone", which is a meaningful and observable state.
    An omitted client knob has no such reading — the load test runs regardless,
    silently on ``benchmark.py``'s defaults (``-c 32``, ``-d 30m``, ``-x 0``),
    and the result CSV records none of it. Making them mandatory keeps every
    result attributable to a configuration someone wrote down.

    ``server`` and ``bearer_file`` are the two optional ones: pysweep provisions
    both when absent (the deployment's ingress URL, plus a bearer file of freshly
    minted tokens) and that is what keeps a sweep
    runnable with no preparation steps. They are connection plumbing rather than
    test scenarios, so they are also single-valued — one endpoint serves the
    whole sweep, and a list is rejected.

    **There is no ``tokenizer`` field**, though ``benchmark.py`` has a ``-m``
    option, because it is not a knob: it must always match the model being
    served, so its value is fully determined by what the deployment runs. The
    runner reads the served model back per combination via
    ``ServedModel.read()`` and looks up its HuggingFace repo id in the model
    catalog. Declaring ``client.tokenizer``
    therefore fails as an unknown key — deliberately, since the only thing a
    config author could express with it is a mismatch, and a mismatched tokenizer
    silently corrupts every token count in the results.
    """

    question_file: Annotated[PathListValue, QUESTION_FILE]
    concurrency: Annotated[CountValue, CONCURRENCY]
    duration: Annotated[DurationListValue, DURATION]
    worker_startup_delay: Annotated[DurationListValue, WORKER_STARTUP_DELAY]
    failures_allowed: Annotated[NonNegativeCountValue, FAILURES_ALLOWED]
    # 0 means "use the questions as written"; any other value pads/trims every
    # question to exactly that many tokens.
    input_tokens: Annotated[NonNegativeCountValue, INPUT_TOKENS]
    warmup_requests: Annotated[NonNegativeCountValue, WARMUP_REQUESTS]

    # Auto-provisioned when omitted; never a product dimension.
    server: Annotated[PathValue | None, SERVER] = None
    bearer_file: Annotated[PathValue | None, BEARER_FILE] = None


class RunConfig(_StrictModel):
    """pysweep **run** configuration — how a sweep executes, not what it varies.

    Kept separate from :class:`SweepConfig` (and passed via ``--config`` while
    the sweep comes from ``--sweep``) because these settings are not sweep
    dimensions: they describe the harness's behaviour and never expand into
    combinations. Same strictness applies — an unknown key raises
    :class:`ConfigError`.
    """

    # The kr8s/httpx Kubernetes client logs one line per API call, including a
    # '101 Switching Protocols' for *every* connection the port-forward carries
    # — i.e. one per benchmark request, which is pure noise at load. Off by
    # default; set true to debug cluster connectivity.
    display_kubernetes_client_logs: bool = False


class SweepConfig(_StrictModel):
    """Top-level pysweep sweep configuration.

    Deployment sections are optional, and within them each knob is optional: the
    set of *declared* (non-None) knobs defines what the sweep varies on the
    cluster. The ``client`` section is required, because a load test always
    applies *some* load and pysweep will not leave that undeclared.
    """

    llms: LlmsConfig | None = None
    client: ClientConfig

    # -- introspection --------------------------------------------------------

    def iter_settings(self) -> Iterator[tuple[str, type[Property], list[Any]]]:
        """Yield ``(dotted_path, property_cls, values)`` for every declared *deployment* knob.

        Only knobs actually present in the config are yielded; a single scalar
        is normalised to a one-element list so callers treat every knob
        uniformly. Client knobs are not properties and are yielded by
        :meth:`iter_client_settings` instead.
        """
        for path, binding, values in _walk_bindings(self):
            if isinstance(binding, PropertyBinding):
                yield path, binding.property_cls, values

    def iter_client_settings(self) -> Iterator[tuple[str, ClientArg, list[Any]]]:
        """Yield ``(dotted_path, client_arg, values)`` for every sweepable client knob.

        Non-sweepable knobs (``server``, ``bearer_file``) are excluded: they are
        whole-sweep connection settings, not product dimensions, and are rendered
        once by :meth:`client_provisioning_argv`.
        """
        for path, binding, values in _walk_bindings(self):
            if isinstance(binding, ClientArg) and binding.sweepable:
                yield path, binding, values

    def deployment_combinations(self) -> list[Combination]:
        """Expand the *deployment* knobs into the cartesian product of combinations.

        Combinations order by config-plane (later is more inner in the loop):
        1. Kubernetes
        2. Fingerprint

        Never empty. A sweep with no declared deployment knob yields exactly one
        combination — the empty one, meaning "apply nothing, benchmark the
        deployment as it stands" — which is what an empty cartesian product
        already means, so the runner iterates one list either way instead of
        special-casing the no-knobs sweep.
        """
        settings = list(self.iter_settings())
        # K8s-plane knobs (slow rollout) as the outer loop; itertools.product
        # varies its leftmost argument slowest.
        settings.sort(key=lambda s: (0 if s[1].plane == Plane.K8S_SPEC else 1, s[0]))

        value_lists = [values for _, _, values in settings]
        combos: list[Combination] = []
        for chosen in itertools.product(*value_lists):
            combos.append(
                tuple(
                    PropertySetting(path=path, property_cls=cls, value=value)
                    for (path, cls, _), value in zip(settings, chosen)
                )
            )
        return combos

    def client_combinations(self) -> list[ClientCombination]:
        """Expand the client knobs into their own cartesian product.

        A *separate* product from :meth:`deployment_combinations` rather than extra
        dimensions of it, because the runner nests the two: each deployment
        combination is established once, then every client combination is run
        against it. That nesting is what makes these knobs cheap — a new
        concurrency level costs one more load test, not one more rollout — and
        expressing it as nesting means the ordering is structural rather than
        something a sort key has to keep getting right.

        Ordered by dotted path so the sequence is stable and diffable across
        runs; there is no cost gradient among client knobs to order by, since
        none of them touches the deployment. Never empty: the knobs are required,
        so the product is at least the single all-scalars combination.
        """
        settings = sorted(self.iter_client_settings(), key=lambda s: s[0])
        if not settings:
            return []

        value_lists = [values for _, _, values in settings]
        combos: list[ClientCombination] = []
        for chosen in itertools.product(*value_lists):
            combos.append(
                tuple(
                    ClientSetting(path=path, arg=arg, value=value)
                    for (path, arg, _), value in zip(settings, chosen)
                )
            )
        return combos

    def client_provisioning_argv(self) -> list[str]:
        """Render the declared whole-sweep client knobs (``-s`` / ``-b``) as argv.

        Empty when neither is declared, which is the default and the path where
        pysweep provisions both itself.
        """
        argv: list[str] = []
        for path, binding, values in _walk_bindings(self):
            del path
            if isinstance(binding, ClientArg) and not binding.sweepable:
                argv += [binding.flag, str(values[0])]
        return argv

    def max_concurrency(self) -> int:
        """The largest ``client.concurrency`` any combination of this sweep runs.

        Whole-sweep provisioning needs this before the product is expanded:
        ``benchmark.py`` reads one bearer token per worker index, so the bearer
        file — written once, shared by every combination — has to cover the widest
        combination the sweep will reach.

        ``client.concurrency`` is a required field, so the max is always defined.
        """
        return max(
            int(value)
            for _, arg, values in self.iter_client_settings()
            if arg is CONCURRENCY
            for value in values
        )

    def varying_paths(self) -> set[str]:
        """Dotted paths of the knobs declared with more than one value.

        These are the sweep's actual dimensions — the knobs whose value differs
        between combinations. Knobs declared as a scalar are constant across the
        whole sweep and so distinguish nothing. Used to name result files by what
        varies (see :func:`~.pysweep_bench.result_file_name`).
        """
        return {path for path, _, values in _walk_bindings(self) if len(values) > 1}

    def knob_paths(self) -> list[str]:
        """Dotted paths of every knob this sweep declares, deployment first.

        The sweep's own description of itself, used as the knob columns of the
        summary CSV (see :mod:`~.results`). *Every* declared knob is included,
        not only the varying ones, because a summary that dropped the constants
        would not say what the numbers were measured under — and because it makes
        two sweeps' summaries concatenable when they declare the same knobs.

        Ordered deployment-then-client, each group by path, so the columns read
        outer-loop to inner-loop and stay stable across runs. ``server`` and
        ``bearer_file`` are excluded: they are connection plumbing rather than a
        test scenario, identical for every row, and one of them is a file path
        that says nothing about what was tested.
        """
        deployment = sorted(path for path, _, _ in self.iter_settings())
        client = sorted(path for path, _, _ in self.iter_client_settings())
        return deployment + client

    def declared_client_names(self) -> set[str]:
        """YAML names of every client knob this config declares.

        Used to detect a client flag passed after ``--`` that the config already
        owns (see :func:`~.client_args.conflicting_flags`).
        """
        return {
            binding.yaml_name
            for _, binding, _ in _walk_bindings(self)
            if isinstance(binding, ClientArg)
        }

    def declared_question_files(self) -> list[str]:
        """Every value declared for ``client.question_file``, in declared order.

        A list, not a set, because the knob is sweepable: a config may name
        several corpora and the caller checks all of them. Whether each names a
        readable file is not a schema question (the value is validated only as a
        non-empty string), so it is checked by the runner's preflight — see
        ``pysweep_bench.question_file_verify``.
        """
        return [
            str(value)
            for _, binding, values in _walk_bindings(self)
            if binding is QUESTION_FILE
            for value in values
        ]


def _binding_of(field_info) -> PropertyBinding | ClientArg | None:
    """The binding attached to a field, whichever kind it is."""
    for meta in field_info.metadata:
        if isinstance(meta, (PropertyBinding, ClientArg)):
            return meta
    return None


def _walk_bindings(
    model: BaseModel, prefix: str = ""
) -> Iterator[tuple[str, PropertyBinding | ClientArg, list[Any]]]:
    """Recurse a validated model, yielding declared bindings of either kind.

    Callers filter by binding type; walking both kinds in one pass keeps this the
    only place that knows how a config's shape maps to dotted paths.
    """
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        path = f"{prefix}{name}"
        binding = _binding_of(field)
        if binding is not None:
            if value is not None:
                yield path, binding, _as_list(value)
        elif isinstance(value, BaseModel):
            yield from _walk_bindings(value, prefix=f"{path}.")


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


# -- loading ------------------------------------------------------------------


def _load_model(source: str | Path | dict, model: type[BaseModel], what: str):
    """Load YAML (or take a dict) and validate it against ``model``.

    Shared by the sweep and run configs so both get identical strictness and
    error reporting. ``what`` names the config in error messages.
    """
    if isinstance(source, dict):
        data: Any = source
    else:
        path = Path(source)
        try:
            text = path.read_text()
        except OSError as exc:
            raise ConfigError(f"Cannot read {what} config '{path}': {exc}") from exc
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise ConfigError(f"{what.capitalize()} config '{path}' is not valid YAML: {exc}") from exc

    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ConfigError(
            f"{what.capitalize()} config must be a mapping at the top level, "
            f"got {type(data).__name__}."
        )

    try:
        return model.model_validate(data)
    except ValidationError as exc:
        raise ConfigError(f"Invalid {what} config:\n{exc}") from exc


def load_config(source: str | Path | dict) -> SweepConfig:
    """Load and validate a sweep config from a YAML file path or a dict.

    Raises :class:`ConfigError` on anything that violates the schema — an
    unknown key, a wrong type, an out-of-range value, or malformed YAML — so
    invalid config fails loudly before a sweep starts.
    """
    return _load_model(source, SweepConfig, "sweep")


def load_run_config(source: str | Path | dict) -> RunConfig:
    """Load and validate a pysweep run config (``--config``).

    Same strictness as :func:`load_config`: an unknown key or wrong type raises
    :class:`ConfigError` rather than being silently ignored.
    """
    return _load_model(source, RunConfig, "run")
