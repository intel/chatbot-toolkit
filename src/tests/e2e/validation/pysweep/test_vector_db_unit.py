# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.cluster import vector_db
from tests.e2e.benchmarks.chatqna.pysweep.cluster.vector_db import (
    VectorDbError,
    observed_settings,
    read_state,
    wait_until_idle,
)


class Edp:
    """EDP whose /files answers come from a list of listings, one per call."""

    def __init__(self, *listings, status=200):
        self.listings = list(listings)
        self.status = status

    def list_files(self):
        listing = self.listings.pop(0) if len(self.listings) > 1 else self.listings[0]
        return SimpleNamespace(status_code=self.status, json=lambda: listing, text="boom")


def _file(name, status, chunks=0):
    return {"object_name": name, "status": status, "chunks_processed": chunks}


@allure.feature("pysweep")
def test_read_state_counts_ingested_chunks_and_open_work():
    edp = Edp([
        _file("a.pdf", "ingested", 7),
        _file("Docs/b.pdf", "ingested", 5),
        _file("c.pdf", "error"),
        _file("d.pdf", "embedding"),
    ])
    state = read_state(edp)
    assert (state.vectors, state.failed, state.in_progress) == (12, 1, 1)
    assert state.files == {"a.pdf", "b.pdf", "c.pdf", "d.pdf"}


@allure.feature("pysweep")
def test_an_edp_error_is_a_vector_db_error():
    with pytest.raises(VectorDbError, match="HTTP 503"):
        read_state(Edp([], status=503))


@allure.feature("pysweep")
def test_wait_until_idle_polls_while_files_are_processed():
    edp = Edp([_file("a.pdf", "embedding")], [_file("a.pdf", "ingested", 4)])
    sleeps = []
    state = wait_until_idle(edp, sleep=sleeps.append, clock=lambda: 0.0)
    assert sleeps == [vector_db.POLL_INTERVAL_S]
    assert state.vectors == 4


@allure.feature("pysweep")
def test_wait_until_idle_gives_up_after_the_timeout():
    now = iter([0.0, 10.0, 20.0])
    with pytest.raises(VectorDbError, match="still processes 1 file"):
        wait_until_idle(Edp([_file("a.pdf", "embedding")]), timeout=15, sleep=lambda s: None, clock=lambda: next(now))


@allure.feature("pysweep")
def test_observed_settings_report_the_ingested_vectors_once_idle():
    edp = Edp([_file("a.pdf", "embedding")], [_file("a.pdf", "ingested", 55312)])
    assert observed_settings(edp, sleep=lambda s: None) == {"vector_db.num_vectors": 55312}
