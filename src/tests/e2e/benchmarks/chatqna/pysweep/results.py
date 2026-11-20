# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Post-processing: one summary CSV for a whole sweep.

``benchmark.py`` writes one CSV per load test, with one row per completed
request and no header::

    input_tokens,output_tokens,ttft,end_to_end,err,code

That is the right raw format — it is written incrementally under load and loses
nothing — but it answers nothing on its own. Which configuration produced it
lives in the *filename*, and only partially: :func:`~.pysweep_bench.result_file_name`
names a run by what *varies*, so the constant half of the configuration is not
there at all, and neither is the served model.

So this module turns a sweep into a single table: one row per run, a column per
declared knob (named by its dotted config path, e.g. ``client.input_tokens``),
and the KPIs that make runs comparable — TTFT and TPOT, mean and p90. Knob
columns come first so the file reads as "these settings produced these numbers",
and every declared knob gets a column, not just the varying ones: a summary that
omitted the constants would be unreadable a month later and impossible to
concatenate with a different sweep's.

The per-request CSVs are kept as they are. This is a summary, not a replacement:
anything the four KPIs do not answer (distribution shape, drift over the run,
the tail past p90) is still in them, and the ``result_file`` column is the
pointer back.

**Definitions** (seconds, since that is what ``benchmark.py`` measures):

``ttft``
    Time to first token — ``first_chunk``, as recorded: the wall time from
    issuing the request to the first streamed content chunk. Prefill plus every
    queueing delay ahead of it, which is why it is the KPI that moves with
    concurrency.
``tpot``
    Time per output token — ``(end_to_end - ttft) / (output_tokens - 1)``: the
    average inter-token gap during decode. The first token is excluded from both
    sides, because its cost is TTFT; dividing by ``output_tokens`` instead (as
    ``parse.py``'s ``next_token_lat`` does) charges the decode phase for a token
    it did not generate, which understates TPOT by ``1/output_tokens``. The
    difference is under a percent at 128 tokens and material at 8.

p90 uses linear interpolation between the two nearest ranks (the numpy/pandas
default), so it agrees with what ``parse.py`` reports and needs no dependency
beyond the standard library — pysweep stays importable without pandas, which
matters because the config/apply half of it runs in the validation suite.
"""

import csv
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Iterable, NamedTuple, Sequence

logger = logging.getLogger(__name__)

#: Everything a sweep writes goes under one directory, given by ``--summaries``.
#: Relative to the CWD, like any other command-line path.
DEFAULT_SUMMARIES_DIR = "./results"

#: The sweep's own table, at the root of that directory. Deliberately not named
#: ``bench_*.csv``: that glob is what ``parse.py`` consumes as per-request data,
#: and this file is not that shape.
SUMMARY_FILE_NAME = "benchmark_summary_results.csv"

#: The per-run CSVs, in a subdirectory of it. Separated because there is one per
#: run and they are the raw data, while the summary is one file and is the
#: artifact people read — mixing them made the interesting file hard to find, and
#: made "wipe the previous run's results" impossible to express safely.
PER_CASE_DIR_NAME = "per_case_results"

#: KPI and provenance columns, in the order they appear after the knob columns.
#: ``requests`` comes first of them because it is what licenses reading the
#: rest — a p90 over four requests is a number, not a measurement.
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

#: Seconds are recorded to microseconds. Enough to keep sub-millisecond TPOT
#: differences visible, few enough that the CSV stays readable.
_PRECISION = 6


class SummariesLayoutError(Exception):
    """The ``--summaries`` directory cannot be prepared for a sweep."""


class SummariesLayout(NamedTuple):
    """Where a sweep's artifacts go, resolved once and passed down.

    Resolved to absolute paths: the load test runs through ``benchmark.py``, which
    opens its output with a bare ``open()``, so anything relative would land
    against the process's working directory instead.
    """

    root: Path
    summary_file: Path
    per_case_dir: Path


def _clear_directory(directory: Path) -> int:
    """Delete every entry inside ``directory``, returning how many were removed.

    Symlinks are unlinked, not followed: a link inside the directory names
    something outside it, which is not this sweep's to delete.
    """
    removed = 0
    for entry in directory.iterdir():
        if entry.is_dir() and not entry.is_symlink():
            shutil.rmtree(entry)
        else:
            entry.unlink()
        removed += 1
    return removed


def prepare_summaries_dir(root=DEFAULT_SUMMARIES_DIR) -> SummariesLayout:
    """Create the sweep's output layout, **discarding any previous run's results**.

    A sweep's artifacts only make sense as a set: the summary indexes the per-case
    CSVs, and a per-case CSV means nothing without the row that says what produced
    it. Leaving the previous run's files in place would silently mix two sweeps —
    stale per-case CSVs with no row in the current summary, or rows pointing at
    files a shorter sweep never overwrote. So this is a clean slate, not a merge:

    * ``<root>/`` is created if missing (parents included).
    * ``<root>/benchmark_summary_results.csv`` is deleted if present.
    * ``<root>/per_case_results/`` is emptied if present, created if not.

    **Nothing else under ``<root>`` is touched** — the deletion is confined to the
    one file and the one subdirectory this function owns, so pointing
    ``--summaries`` at a directory that holds other things loses none of them. Two
    shapes are refused outright rather than resolved:

    * ``<root>`` or the summary path already being the wrong kind of file, which
      means this directory is not a results directory and the caller has the wrong
      path;
    * ``per_case_results`` being a symlink, since emptying it would delete
      whatever it points at, outside the tree the caller named.

    Raises :class:`SummariesLayoutError` in those cases, before any deletion.
    """
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
        """Mean inter-token gap during decode, or ``None`` if undefined.

        Undefined for a single-token answer: there is no gap *between* tokens to
        average, and the whole latency is already reported as TTFT.
        """
        if self.output_tokens < 2:
            return None
        return (self.end_to_end - self.ttft) / (self.output_tokens - 1)


def read_requests(result_file) -> list[RequestRecord]:
    """Parse one ``benchmark.py`` result CSV into records.

    Rows that are malformed, unsuccessful, or nonsensical are dropped rather
    than crashing the summary: a sweep is hours long and the summary is written
    after every run, so one damaged line must not cost the rest of the table.
    Every drop is logged at debug level. ``benchmark.py`` only enqueues 200s with
    no error, so in practice nothing is dropped — the filtering is here so the
    summary keeps the same meaning if that ever changes.
    """
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
                # No first chunk was ever timed, or the totals contradict each
                # other. Either way the latencies are not measurements.
                logger.debug("%s:%d: implausible timings %r; skipped", path, line, record)
                continue
            records.append(record)
    return records


def percentile(values: Sequence[float], quantile: float) -> float | None:
    """Linear-interpolated percentile, matching numpy/pandas' default method.

    Returns ``None`` for an empty sample rather than raising, so a run that
    recorded nothing still produces a row (with blank KPIs) instead of aborting
    the summary — "this configuration produced no data" is itself a result worth
    keeping in the table.
    """
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
    """Render a KPI for the CSV; an undefined one renders as an empty cell."""
    return "" if value is None else f"{round(value, _PRECISION)}"


def summarize_requests(records: Iterable[RequestRecord]) -> dict[str, Any]:
    """Aggregate request records into the KPI columns.

    ``requests`` counts the sample the TTFT statistics are over. The TPOT
    statistics are over the subset with at least two output tokens (see
    :attr:`RequestRecord.tpot`); that subset is normally the whole sample, and
    when it is not, a blank TPOT with a non-zero ``requests`` is the honest
    reading — the run happened, the decode rate is not defined for it.
    """
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
    """Accumulates one row per run and writes the sweep's summary CSV.

    Rewritten in full after every run rather than appended to. A sweep of a
    dozen runs is a few hundred bytes, and the rewrite means the file on disk is
    always a complete, valid CSV of everything finished so far — so a sweep that
    is killed at hour three (or a model rollout that fails midway) still leaves a
    readable table instead of a truncated last line.

    ``knob_paths`` fixes the columns up front, from the config rather than from
    the rows: every run of a sweep has the same knobs, and taking them from the
    config means the header is right before the first run finishes. It is written
    at construction for that reason — the first run of a sweep can be an hour of
    load behind a multi-replica rollout, and until then a missing file is
    indistinguishable from a harness that forgot to summarize. A header-only file
    says "this sweep is running and here is the shape of its results".
    """

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
        """How the ``result_file`` column names a per-case CSV.

        Relative to the summary's own directory when the CSV is inside it, which
        is the normal case (``per_case_results/bench_….result.csv``). That keeps
        the whole results directory movable: copy it off the benchmark host or
        rename it, and every pointer still resolves. An absolute path would name a
        directory that only existed on the machine that ran the sweep.

        A CSV from outside the directory is recorded verbatim, since there is no
        relative name for it that means anything.
        """
        path = Path(result_file)
        try:
            return str(path.relative_to(self.path.parent))
        except ValueError:
            return str(result_file)

    @property
    def columns(self) -> list[str]:
        """Header: knob columns in dotted-path form, then the KPIs."""
        return self.knob_paths + list(SUMMARY_COLUMNS)

    def add_run(
        self,
        knob_values: dict[str, Any],
        result_file,
        error: int = 0,
        served_model: str | None = None,
    ) -> dict[str, Any]:
        """Summarize one finished run and return the row that was recorded.

        ``knob_values`` maps dotted paths to the values that run used — observed
        read-backs for deployment knobs (what the cluster actually did), declared
        values for client knobs (there is nothing to observe: they are arguments
        to the process). A path with no entry renders blank, which is what makes
        it safe to call before every knob has a value.
        """
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
        """Write the whole table, header included, replacing any previous copy."""
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
