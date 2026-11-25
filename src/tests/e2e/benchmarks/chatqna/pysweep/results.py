# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import csv
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

logger = logging.getLogger(__name__)

DEFAULT_SUMMARIES_DIR = "./results"

# Not bench_*.csv: parse.py globs that as per-request data.
SUMMARY_FILE_NAME = "benchmark_summary_results.csv"

PER_CASE_DIR_NAME = "per_case_results"

SUMMARY_COLUMNS: tuple[str, ...] = (
    "requests",
    "mean_ttft",
    "p90_ttft",
    "mean_tpot",
    "p90_tpot",
    "served_model",
    "error",
    "result_file",
)

_PRECISION = 6


class SummariesLayoutError(Exception):
    """The ``--summaries`` directory cannot be prepared for a sweep."""


class SummariesLayout(NamedTuple):
    """Where a sweep's artifacts go, as absolute paths (benchmark.py opens them bare)."""

    root: Path
    summary_file: Path
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
    per_case_dir = root / PER_CASE_DIR_NAME
    if summary_file.is_dir():
        raise SummariesLayoutError(
            f"{summary_file} is a directory; a sweep needs to write its summary "
            "table there."
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

    if summary_file.exists():
        summary_file.unlink()
        logger.info("Discarded the previous summary %s.", summary_file)

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
        per_case_dir=per_case_dir.resolve(),
    )
    logger.info(
        "Writing results to %s (summary: %s, per-case CSVs: %s/).",
        layout.root, SUMMARY_FILE_NAME, PER_CASE_DIR_NAME,
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


def _round(value: float | None) -> str:
    return "" if value is None else f"{round(value, _PRECISION)}"


def summarize_requests(records: Iterable[RequestRecord]) -> dict[str, Any]:
    """Aggregate request records into the KPI columns."""
    records = list(records)
    ttfts = [record.ttft for record in records]
    tpots = [record.tpot for record in records if record.tpot is not None]
    return {
        "requests": len(records),
        "mean_ttft": _mean(ttfts),
        "p90_ttft": percentile(ttfts, 0.9),
        "mean_tpot": _mean(tpots),
        "p90_tpot": percentile(tpots, 0.9),
    }


class SweepSummary:
    """Accumulates one row per run and rewrites the summary CSV after each."""

    def __init__(
        self,
        knob_paths: Sequence[str],
        path=Path(DEFAULT_SUMMARIES_DIR) / SUMMARY_FILE_NAME,
    ):
        self.knob_paths = list(knob_paths)
        self.path = Path(path)
        self.rows: list[dict[str, Any]] = []
        self.write()

    def _pointer(self, result_file) -> str:
        path = Path(result_file)
        try:
            return str(path.relative_to(self.path.parent))
        except ValueError:
            return str(result_file)

    @property
    def columns(self) -> list[str]:
        return self.knob_paths + list(SUMMARY_COLUMNS)

    def add_run(
        self,
        knob_values: dict[str, Any],
        result_file,
        error: int = 0,
        served_model: str | None = None,
    ) -> dict[str, Any]:
        """Summarize one finished run and return the recorded row."""
        summary = summarize_requests(read_requests(result_file))
        row: dict[str, Any] = {path: knob_values.get(path, "") for path in self.knob_paths}
        row.update(summary)
        row["served_model"] = served_model or ""
        row["error"] = error
        row["result_file"] = self._pointer(result_file)
        self.rows.append(row)
        self.write()
        return row

    def write(self) -> Path:
        with self.path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.columns)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(
                    {
                        column: (
                            _round(row[column])
                            if column in ("mean_ttft", "p90_ttft", "mean_tpot", "p90_tpot")
                            else row.get(column, "")
                        )
                        for column in self.columns
                    }
                )
        return self.path
