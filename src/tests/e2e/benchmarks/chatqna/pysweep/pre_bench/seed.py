# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import bz2
import csv
import logging
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Callable, Iterable

import requests

from ..cluster.vector_db import VectorDbState, wait_until_idle

logger = logging.getLogger(__name__)

BENCHMARK_DIR = Path(__file__).resolve().parents[2]
DOCUMENTS_CSV = BENCHMARK_DIR / "documents.csv"
WIKIPARSE = BENCHMARK_DIR / "wikiparse.py"
WIKI_DUMP_URL = (
    "https://dumps.wikimedia.org/simplewiki/latest/simplewiki-latest-pages-articles-multistream.xml.bz2"
)
CACHE_ENV_VAR = "PYSWEEP_SEED_CACHE"
DEFAULT_CACHE_DIR = Path.home() / ".cache" / "pysweep" / "seed"

DOWNLOAD_TIMEOUT_S = 600
# EDP processes two files at a time, so pairs keep it busy without queueing.
WIKI_BATCH = 2


class SeedError(Exception):
    """The vector DB could not be seeded to the requested size."""


def upload(edp, path: Path) -> None:
    presigned = edp.generate_presigned_url(path.name)
    if presigned.status_code != 200:
        raise SeedError(f"No presigned URL for {path.name}: HTTP {presigned.status_code} {presigned.text[:300]}")
    response = edp.upload_file(str(path), presigned.json()["url"])
    if response.status_code not in (200, 201, 204):
        raise SeedError(f"Upload of {path.name} failed: HTTP {response.status_code} {response.text[:300]}")
    logger.info("Uploaded %s.", path.name)


def download(url: str, target: Path) -> Path:
    if target.is_file() and target.stat().st_size > 0:
        return target
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    logger.info("Downloading %s.", url)
    with requests.get(url, stream=True, timeout=DOWNLOAD_TIMEOUT_S) as response:
        if response.status_code != 200:
            raise SeedError(f"Download of {url} failed: HTTP {response.status_code}")
        with partial.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                handle.write(chunk)
    partial.rename(target)
    return target


def relevant_documents(cache_dir: Path) -> list[Path]:
    """The documents the benchmark's questions are about, in documents.csv order."""
    with DOCUMENTS_CSV.open(newline="") as handle:
        rows = [row for row in csv.reader(handle, delimiter=";") if row]
    return [download(url, cache_dir / "documents" / name) for url, name in rows]


def wikipedia_files(cache_dir: Path) -> list[Path]:
    """Simple English Wikipedia split by wikiparse.py into upload-sized text files."""
    out_dir = cache_dir / "wiki"
    parsed = sorted(out_dir.glob("wiki*.txt"))
    if parsed:
        return parsed
    dump = download(WIKI_DUMP_URL, cache_dir / WIKI_DUMP_URL.rsplit("/", 1)[1])
    xml = dump.with_suffix("")
    with bz2.open(dump, "rb") as source, xml.open("wb") as target:
        shutil.copyfileobj(source, target, 1 << 20)
    staging = cache_dir / "wiki.part"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    subprocess.run([sys.executable, str(WIKIPARSE), str(xml), str(staging)], check=True)
    xml.unlink()
    staging.rename(out_dir)
    return sorted(out_dir.glob("wiki*.txt"))


def cache_dir() -> Path:
    return Path(os.getenv(CACHE_ENV_VAR) or DEFAULT_CACHE_DIR)


def _batches(paths: Iterable[Path], size: int):
    batch = []
    for path in paths:
        batch.append(path)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch


def seed(
    num_vectors: int,
    edp,
    documents: Callable[[], list[Path]] | None = None,
    wikipedia: Callable[[], list[Path]] | None = None,
    sleep=time.sleep,
    clock=time.monotonic,
) -> VectorDbState:
    """Load the vector DB until it holds at least ``num_vectors`` vectors."""
    documents = documents or (lambda: relevant_documents(cache_dir()))
    wikipedia = wikipedia or (lambda: wikipedia_files(cache_dir()))

    def wait() -> VectorDbState:
        return wait_until_idle(edp, sleep=sleep, clock=clock)

    state = wait()
    logger.info("Vector DB holds %d vectors in %d file(s); seeding to %d.", state.vectors, len(state.files), num_vectors)
    if state.vectors >= num_vectors:
        return state

    for path in documents():
        if path.name not in state.files:
            upload(edp, path)
    state = wait()

    if state.vectors < num_vectors:
        for batch in _batches(wikipedia(), WIKI_BATCH):
            pending = [path for path in batch if path.name not in state.files]
            if not pending:
                continue
            for path in pending:
                upload(edp, path)
            state = wait()
            if state.vectors >= num_vectors:
                break

    if state.vectors < num_vectors:
        raise SeedError(
            f"Seeding ran out of data at {state.vectors} vectors; {num_vectors} were requested."
        )
    if state.failed:
        logger.warning("%d file(s) failed ingestion in EDP; they hold no vectors.", state.failed)
    logger.info("Vector DB seeded: %d vectors in %d file(s).", state.vectors, len(state.files))
    return state
