# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

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

from .properties.base import Plane, Property
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
from .properties.reranker_top_n import RerankerTopN
from .properties.retriever_k import RetrieverK

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Raised when a sweep config file is invalid (bad schema, key, or value)."""


class PropertyBinding:
    """Field metadata tying a config leaf to the ``Property`` class it drives."""

    def __init__(self, property_cls: type[Property]):
        self.property_cls = property_cls

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"PropertyBinding({self.property_cls.__name__})"


@dataclass(frozen=True)
class PropertySetting:
    """One resolved deployment knob value within a single sweep combination."""

    path: str
    property_cls: type[Property]
    value: Any


@dataclass(frozen=True)
class ClientSetting:
    """One resolved client knob value within a single client combination."""

    path: str
    arg: ClientArg
    value: Any


Combination = tuple[PropertySetting, ...]
ClientCombination = tuple[ClientSetting, ...]


MaxTokensValue = PositiveInt | list[PositiveInt]
# k=0 retrieves nothing; PositiveInt is also the app's own type.
RetrieverKValue = PositiveInt | list[PositiveInt]
# top_n=0 leaves the prompt without context; PositiveInt is also the app's own type.
RerankerTopNValue = PositiveInt | list[PositiveInt]

# Tighter than benchmark.py's argparse, which accepts -c 0, -e -1 and -w -1.
CountValue = PositiveInt | list[PositiveInt]
NonNegativeCountValue = NonNegativeInt | list[NonNegativeInt]
PathValue = Annotated[str, StringConstraints(min_length=1, strip_whitespace=True)]
PathListValue = PathValue | list[PathValue]
# benchmark.duration_to_seconds raises KeyError on anything but <number><s|m|h>.
DurationValue = Annotated[
    str, StringConstraints(strip_whitespace=True, pattern=r"^\d+(\.\d+)?[smh]$")
]
DurationListValue = DurationValue | list[DurationValue]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class VectorDbPreBench(_StrictModel):
    num_vectors: PositiveInt | None = None


class EdpPreBench(_StrictModel):
    vector_db: VectorDbPreBench | None = None


class PreBenchConfig(_StrictModel):
    """Done once before the first run; single values only, never part of the product."""

    edp: EdpPreBench | None = None


class LlmsConfig(_StrictModel):
    max_new_tokens: Annotated[MaxTokensValue | None, PropertyBinding(MaxNewTokens)] = None


class RetrieverConfig(_StrictModel):
    k: Annotated[RetrieverKValue | None, PropertyBinding(RetrieverK)] = None


class RerankerConfig(_StrictModel):
    top_n: Annotated[RerankerTopNValue | None, PropertyBinding(RerankerTopN)] = None


class ClientConfig(_StrictModel):
    """Load-generator knobs mirroring ``benchmark.py``; ``-m`` is derived, not declared."""

    question_file: Annotated[PathListValue, QUESTION_FILE]
    concurrency: Annotated[CountValue, CONCURRENCY]
    duration: Annotated[DurationListValue, DURATION]
    worker_startup_delay: Annotated[DurationListValue, WORKER_STARTUP_DELAY]
    failures_allowed: Annotated[NonNegativeCountValue, FAILURES_ALLOWED]
    input_tokens: Annotated[NonNegativeCountValue, INPUT_TOKENS]
    warmup_requests: Annotated[NonNegativeCountValue, WARMUP_REQUESTS]

    # Auto-provisioned when omitted; never a product dimension.
    server: Annotated[PathValue | None, SERVER] = None
    bearer_file: Annotated[PathValue | None, BEARER_FILE] = None


class RunConfig(_StrictModel):
    """How a sweep executes, as opposed to what it varies."""

    # kr8s/httpx logs a '101 Switching Protocols' line for every port-forwarded request.
    display_kubernetes_client_logs: bool = False
    metadata: bool = True


class SweepConfig(_StrictModel):
    pre_bench: PreBenchConfig | None = None
    llms: LlmsConfig | None = None
    retriever: RetrieverConfig | None = None
    reranker: RerankerConfig | None = None
    client: ClientConfig

    def iter_settings(self) -> Iterator[tuple[str, type[Property], list[Any]]]:
        """Yield ``(path, property_cls, values)`` for every declared deployment knob."""
        for path, binding, values in _walk_bindings(self):
            if isinstance(binding, PropertyBinding):
                yield path, binding.property_cls, values

    def iter_client_settings(self) -> Iterator[tuple[str, ClientArg, list[Any]]]:
        """Yield ``(path, client_arg, values)`` for every sweepable client knob."""
        for path, binding, values in _walk_bindings(self):
            if isinstance(binding, ClientArg) and binding.sweepable:
                yield path, binding, values

    def deployment_combinations(self) -> list[Combination]:
        """Cartesian product of deployment knobs, K8s-plane outermost; never empty."""
        settings = list(self.iter_settings())
        # itertools.product varies its leftmost argument slowest.
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
        """Cartesian product of sweepable client knobs, ordered by path."""
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
        """Render the declared whole-sweep client knobs (``-s`` / ``-b``) as argv."""
        argv: list[str] = []
        for path, binding, values in _walk_bindings(self):
            del path
            if isinstance(binding, ClientArg) and not binding.sweepable:
                argv += [binding.flag, str(values[0])]
        return argv

    def max_concurrency(self) -> int:
        """Largest ``client.concurrency`` in the sweep; sizes the shared bearer file."""
        return max(
            int(value)
            for _, arg, values in self.iter_client_settings()
            if arg is CONCURRENCY
            for value in values
        )

    def varying_paths(self) -> set[str]:
        """Dotted paths of the knobs declared with more than one value."""
        return {path for path, _, values in _walk_bindings(self) if len(values) > 1}

    def pre_bench_settings(self) -> dict[str, Any]:
        """Declared pre_bench values as dotted paths, e.g. ``{"edp.vector_db.num_vectors": 50000}``."""
        return _declared_values(self.pre_bench) if self.pre_bench else {}

    def knob_paths(self) -> list[str]:
        """Declared knob paths (minus ``server``/``bearer_file``), deployment first."""
        deployment = sorted(path for path, _, _ in self.iter_settings())
        client = sorted(path for path, _, _ in self.iter_client_settings())
        return deployment + client

    def declared_client_names(self) -> set[str]:
        """YAML names of every client knob this config declares."""
        return {
            binding.yaml_name
            for _, binding, _ in _walk_bindings(self)
            if isinstance(binding, ClientArg)
        }

    def declared_question_files(self) -> list[str]:
        return [
            str(value)
            for _, binding, values in _walk_bindings(self)
            if binding is QUESTION_FILE
            for value in values
        ]


def _binding_of(field_info) -> PropertyBinding | ClientArg | None:
    for meta in field_info.metadata:
        if isinstance(meta, (PropertyBinding, ClientArg)):
            return meta
    return None


def _walk_bindings(
    model: BaseModel, prefix: str = ""
) -> Iterator[tuple[str, PropertyBinding | ClientArg, list[Any]]]:
    for name, field in type(model).model_fields.items():
        value = getattr(model, name)
        path = f"{prefix}{name}"
        binding = _binding_of(field)
        if binding is not None:
            if value is not None:
                yield path, binding, _as_list(value)
        elif isinstance(value, BaseModel):
            yield from _walk_bindings(value, prefix=f"{path}.")


def _declared_values(model: BaseModel, prefix: str = "") -> dict[str, Any]:
    values: dict[str, Any] = {}
    for name in type(model).model_fields:
        value = getattr(model, name)
        if isinstance(value, BaseModel):
            values.update(_declared_values(value, f"{prefix}{name}."))
        elif value is not None:
            values[f"{prefix}{name}"] = value
    return values


def _as_list(value: Any) -> list[Any]:
    return list(value) if isinstance(value, list) else [value]


def _load_model(source: str | Path | dict, model: type[BaseModel], what: str):
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
    """Load and validate a sweep config from a YAML file path or a dict."""
    return _load_model(source, SweepConfig, "sweep")


def load_run_config(source: str | Path | dict) -> RunConfig:
    """Load and validate a pysweep run config (``--config``)."""
    return _load_model(source, RunConfig, "run")
