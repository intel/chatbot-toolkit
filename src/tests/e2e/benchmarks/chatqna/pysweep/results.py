# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence


logger = logging.getLogger(__name__)

DEFAULT_SUMMARIES_DIR = "./results"

# Not bench_*.csv: parse.py globs that as per-request data.
SUMMARY_FILE_NAME = "benchmark_summary_results.csv"
RESULTS_FILE_NAME = "results.json"

PER_CASE_DIR_NAME = "per_case_results"

SCHEMA_VERSION = 2
PERCENTILES = (("p50", 0.5), ("p90", 0.9), ("p99", 0.99))

_PRECISION = 6


class SummariesLayoutError(Exception):
    """The ``--summaries`` directory cannot be prepared for a sweep."""


class SummariesLayout(NamedTuple):
    """Where a sweep's artifacts go, as absolute paths (benchmark.py opens them bare)."""

    root: Path
    summary_file: Path
    results_file: Path
    per_case_dir: Path


def _clear_directory(directory: Path) -> int:
    removed = 0
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed += 1
    return removed


def prepare_summaries_dir(root=DEFAULT_SUMMARIES_DIR) -> SummariesLayout:
    """Create the sweep's output layout, discarding the previous run's summary and per-case CSVs."""
    root = Path(root)
    if root.exists() and not root.is_dir():
        raise SummariesLayoutError(
            f"--summaries {root} exists and is not a directory; it names the "
            f"directory a sweep writes into (holding {SUMMARY_FILE_NAME} and "
            f"{PER_CASE_DIR_NAME}/), not a file."
        )

    summary_file = root / SUMMARY_FILE_NAME
    results_file = root / RESULTS_FILE_NAME
    per_case_dir = root / PER_CASE_DIR_NAME
    for output in (summary_file, results_file):
        if output.is_dir():
            raise SummariesLayoutError(
                f"{output} is a directory; a sweep needs to write its summary there."
            )
    if per_case_dir.is_symlink():
        raise SummariesLayoutError(
            f"{per_case_dir} is a symlink. It is emptied before every sweep, and "
            "following it would delete files outside the directory you named; "
            "point --summaries at a real directory instead."
        )
    if per_case_dir.exists() and not per_case_dir.is_dir():
        raise SummariesLayoutError(
            f"{per_case_dir} exists and is not a directory; a sweep needs to write "
            "one CSV per run there."
        )

    root.mkdir(parents=True, exist_ok=True)

    for output in (summary_file, results_file):
        if output.exists():
            output.unlink()
            logger.info("Discarded the previous summary %s.", output)

    if per_case_dir.is_dir():
        removed = _clear_directory(per_case_dir)
        if removed:
            logger.info(
                "Discarded %d previous result(s) from %s.", removed, per_case_dir
            )
    else:
        per_case_dir.mkdir()

    layout = SummariesLayout(
        root=root.resolve(),
        summary_file=summary_file.resolve(),
        results_file=results_file.resolve(),
        per_case_dir=per_case_dir.resolve(),
    )
    logger.info(
        "Writing results to %s (%s, %s, per-case CSVs: %s/).",
        layout.root, RESULTS_FILE_NAME, SUMMARY_FILE_NAME, PER_CASE_DIR_NAME,
    )
    return layout


class RequestRecord(NamedTuple):
    """One completed request, as ``benchmark.py`` recorded it."""

    input_tokens: int
    output_tokens: int
    ttft: float
    end_to_end: float
    err: str
    code: int

    @property
    def tpot(self) -> float | None:
        """Mean inter-token gap during decode; ``None`` for a single-token answer."""
        if self.output_tokens < 2:
            return None
        return (self.end_to_end - self.ttft) / (self.output_tokens - 1)


def read_requests(result_file) -> list[RequestRecord]:
    """Parse one ``benchmark.py`` result CSV, dropping malformed or unsuccessful rows."""
    path = Path(result_file)
    if not path.exists():
        return []

    records: list[RequestRecord] = []
    with path.open(newline="") as handle:
        for line, row in enumerate(csv.reader(handle), start=1):
            if len(row) < 6:
                logger.debug("%s:%d: expected 6 columns, got %d; skipped", path, line, len(row))
                continue
            try:
                record = RequestRecord(
                    input_tokens=int(row[0]),
                    output_tokens=int(row[1]),
                    ttft=float(row[2]),
                    end_to_end=float(row[3]),
                    err=row[4],
                    code=int(row[5]),
                )
            except ValueError as exc:
                logger.debug("%s:%d: unparsable (%s); skipped", path, line, exc)
                continue
            if record.code != 200 or record.err:
                logger.debug("%s:%d: not a successful request; skipped", path, line)
                continue
            if record.ttft <= 0 or record.end_to_end < record.ttft:
                logger.debug("%s:%d: implausible timings %r; skipped", path, line, record)
                continue
            records.append(record)
    return records


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile (numpy's default); ``None`` for an empty sample."""
    ordered = sorted(values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * quantile
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _round(value: float | None) -> float | None:
    return None if value is None else round(value, _PRECISION)


def _distribution(values: Sequence[float]) -> dict[str, float | None]:
    stats = {"mean": _round(_mean(values))}
    stats.update({name: _round(percentile(values, q)) for name, q in PERCENTILES})
    return stats


def summarize_requests(records: Iterable[RequestRecord], duration_s: float) -> dict[str, Any]:
    """Aggregate request records into a run's ``results``; rates are per declared duration."""
    records = list(records)
    output_total = sum(record.output_tokens for record in records)
    return {
        "requests": {"ok": len(records)},
        "ttft_s": _distribution([record.ttft for record in records]),
        "tpot_s": _distribution([record.tpot for record in records if record.tpot is not None]),
        "e2e_s": _distribution([record.end_to_end for record in records]),
        "input_tokens": {"mean": _round(_mean([record.input_tokens for record in records]))},
        "output_tokens": {
            "mean": _round(_mean([record.output_tokens for record in records])),
            "total": output_total,
        },
        "throughput": {
            "requests_per_s": _round(len(records) / duration_s) if duration_s > 0 else None,
            "output_tokens_per_s": _round(output_total / duration_s) if duration_s > 0 else None,
        },
    }


def nest(dotted: dict[str, Any]) -> dict[str, Any]:
    """Turn ``{"a.b": 1}`` into ``{"a": {"b": 1}}``, keeping key order."""
    tree: dict[str, Any] = {}
    for path, value in dotted.items():
        *parents, leaf = path.split(".")
        node = tree
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = value
    return tree


def flatten(tree: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    """Turn nested dicts into dotted keys, keeping key order."""
    flat: dict[str, Any] = {}
    for key, value in tree.items():
        path = f"{prefix}{key}"
        if isinstance(value, dict):
            flat.update(flatten(value, f"{path}."))
        else:
            flat[path] = value
    return flat


def run_record(
    index: str,
    started_at: str,
    finished_at: str,
    exit_code: int,
    result_file: str,
    deployment: dict[str, Any],
    client: dict[str, Any],
    results: dict[str, Any],
    edp: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One run's record; ``deployment``/``client``/``edp`` take dotted setting paths."""
    return {
        "run": {
            "index": index,
            "started_at": started_at,
            "finished_at": finished_at,
            "exit_code": exit_code,
            "result_file": result_file,
        },
        "settings": {
            "deployment": nest(deployment),
            "client": nest({path.removeprefix("client."): value for path, value in client.items()}),
            **({"edp": nest(edp)} if edp else {}),
        },
        "results": results,
    }


class SweepResults:
    """Accumulates run records and rewrites results.json and the summary CSV after each."""

    def __init__(self, layout: SummariesLayout, date: str, metadata: dict[str, Any] | None):
        self.layout = layout
        self.date = date
        self.metadata = metadata
        self.runs: list[dict[str, Any]] = []
        self.write()

    def pointer(self, result_file) -> str:
        """``result_file`` relative to the summaries directory, so the directory can be moved."""
        path = Path(result_file)
        try:
            return str(path.relative_to(self.layout.root))
        except ValueError:
            return str(result_file)

    def document(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "date": self.date,
            **({"metadata": self.metadata} if self.metadata else {}),
            "runs": self.runs,
        }

    def add_run(self, record: dict[str, Any]) -> dict[str, Any]:
        self.runs.append(record)
        self.write()
        return record

    def _csv_rows(self) -> list[dict[str, Any]]:
        sweep = flatten({"date": self.date, **({"metadata": self.metadata} if self.metadata else {})})
        return [{**sweep, **flatten(record)} for record in self.runs]

    def write(self) -> Path:
        self.layout.results_file.write_text(json.dumps(self.document(), indent=2) + "\n")
        rows = self._csv_rows()
        columns = list(rows[0]) if rows else []
        with self.layout.summary_file.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            if columns:
                writer.writeheader()
            for row in rows:
                writer.writerow({column: _csv_value(row.get(column)) for column in columns})
        return self.layout.results_file


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, list):
        return ";".join(str(item) for item in value)
    return value
