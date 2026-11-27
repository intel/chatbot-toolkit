# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep import pre_bench
from tests.e2e.benchmarks.chatqna.pysweep.pre_bench import seed as seed_module
from tests.e2e.benchmarks.chatqna.pysweep.pre_bench.seed import (
    SeedError,
    download,
    read_state,
    seed,
    wait_until_idle,
)


def _response(status=200, payload=None, text=""):
    return SimpleNamespace(status_code=status, json=lambda: payload, text=text)


class FakeEdp:
    """EDP whose uploads move from processing to ingested on the next /files poll."""

    def __init__(self, files=None, chunks=10):
        self.files = {name: dict(item) for name, item in (files or {}).items()}
        self.chunks = chunks
        self.uploads = []
        self.presigned_status = 200
        self.upload_status = 204
        self.list_status = 200

    def list_files(self):
        listing = [dict(object_name=name, **item) for name, item in self.files.items()]
        for item in self.files.values():
            if item["status"] == "processing":
                item.update(status="ingested", chunks_processed=self.chunks)
        return _response(self.list_status, listing, "boom")

    def generate_presigned_url(self, object_name):
        return _response(self.presigned_status, {"url": f"http://s3/{object_name}"}, "denied")

    def upload_file(self, path, url):
        name = url.rsplit("/", 1)[1]
        self.uploads.append(name)
        self.files[name] = {"status": "processing", "chunks_processed": 0}
        return _response(self.upload_status, text="nope")


def _paths(tmp_path, *names):
    paths = []
    for name in names:
        path = tmp_path / name
        path.write_text(name)
        paths.append(path)
    return paths


def _seed(edp, num_vectors, documents=(), wiki=()):
    return seed(
        num_vectors, edp,
        documents=lambda: list(documents), wikipedia=lambda: list(wiki),
        sleep=lambda seconds: None,
    )


@allure.feature("pysweep")
def test_read_state_counts_ingested_chunks_and_open_work():
    edp = FakeEdp({
        "a.pdf": {"status": "ingested", "chunks_processed": 7},
        "Docs/b.pdf": {"status": "ingested", "chunks_processed": 5},
        "c.pdf": {"status": "error", "chunks_processed": 0},
        "d.pdf": {"status": "embedding", "chunks_processed": 0},
    })
    state = read_state(edp)
    assert (state.vectors, state.failed, state.in_progress) == (12, 1, 1)
    assert state.files == {"a.pdf", "b.pdf", "c.pdf", "d.pdf"}


@allure.feature("pysweep")
def test_an_edp_error_is_a_seed_error():
    edp = FakeEdp()
    edp.list_status = 503
    with pytest.raises(SeedError, match="HTTP 503"):
        read_state(edp)


@allure.feature("pysweep")
def test_nothing_is_uploaded_when_the_target_is_already_met(tmp_path):
    edp = FakeEdp({"a.pdf": {"status": "ingested", "chunks_processed": 60}})
    state = _seed(edp, 50, documents=_paths(tmp_path, "doc.pdf"))
    assert edp.uploads == []
    assert state.vectors == 60


@allure.feature("pysweep")
def test_documents_first_then_wikipedia_in_pairs_until_the_target(tmp_path):
    edp = FakeEdp(chunks=10)
    documents = _paths(tmp_path, "d1.pdf", "d2.pdf")
    wiki = _paths(tmp_path, "wiki0000.txt", "wiki0001.txt", "wiki0002.txt", "wiki0003.txt", "wiki0004.txt", "wiki0005.txt")
    state = _seed(edp, 50, documents=documents, wiki=wiki)
    assert edp.uploads == ["d1.pdf", "d2.pdf", "wiki0000.txt", "wiki0001.txt", "wiki0002.txt", "wiki0003.txt"]
    assert state.vectors == 60


@allure.feature("pysweep")
def test_wikipedia_is_not_needed_when_the_documents_suffice(tmp_path):
    edp = FakeEdp(chunks=30)
    state = _seed(edp, 50, documents=_paths(tmp_path, "d1.pdf", "d2.pdf"), wiki=_paths(tmp_path, "wiki0000.txt"))
    assert edp.uploads == ["d1.pdf", "d2.pdf"]
    assert state.vectors == 60


@allure.feature("pysweep")
def test_files_already_in_edp_are_not_uploaded_again(tmp_path):
    edp = FakeEdp({"d1.pdf": {"status": "ingested", "chunks_processed": 10}}, chunks=10)
    _seed(edp, 30, documents=_paths(tmp_path, "d1.pdf", "d2.pdf"), wiki=_paths(tmp_path, "wiki0000.txt"))
    assert edp.uploads == ["d2.pdf", "wiki0000.txt"]


@allure.feature("pysweep")
def test_running_out_of_data_is_an_error(tmp_path):
    edp = FakeEdp(chunks=10)
    with pytest.raises(SeedError, match="ran out of data at 30 vectors; 100"):
        _seed(edp, 100, documents=_paths(tmp_path, "d1.pdf"), wiki=_paths(tmp_path, "wiki0000.txt", "wiki0001.txt"))


@allure.feature("pysweep")
@pytest.mark.parametrize("failure", ["presigned_status", "upload_status"])
def test_a_failed_upload_stops_seeding(tmp_path, failure):
    edp = FakeEdp()
    setattr(edp, failure, 500)
    with pytest.raises(SeedError, match="HTTP 500"):
        _seed(edp, 10, documents=_paths(tmp_path, "d1.pdf"))


@allure.feature("pysweep")
def test_wait_until_idle_polls_while_files_are_processed():
    edp = FakeEdp({"a.pdf": {"status": "processing", "chunks_processed": 0}}, chunks=4)
    sleeps = []
    state = wait_until_idle(edp, sleep=sleeps.append, clock=lambda: 0.0)
    assert sleeps == [seed_module.POLL_INTERVAL_S]
    assert state.vectors == 4


@allure.feature("pysweep")
def test_wait_until_idle_gives_up_after_the_timeout():
    class Stuck(FakeEdp):
        def list_files(self):
            return _response(200, [{"object_name": "a.pdf", "status": "embedding"}])

    now = iter([0.0, 10.0, 20.0])
    with pytest.raises(SeedError, match="still processes 1 file"):
        wait_until_idle(Stuck(), timeout=15, sleep=lambda s: None, clock=lambda: next(now))


@allure.feature("pysweep")
def test_a_downloaded_file_is_reused(tmp_path, monkeypatch):
    target = tmp_path / "doc.pdf"
    target.write_bytes(b"cached")

    def no_network(*args, **kwargs):
        raise AssertionError("downloaded again")

    monkeypatch.setattr(seed_module.requests, "get", no_network)
    assert download("http://example/doc.pdf", target) == target


@allure.feature("pysweep")
def test_the_seed_inputs_exist_in_the_repo():
    assert seed_module.DOCUMENTS_CSV.is_file()
    assert seed_module.WIKIPARSE.is_file()


@allure.feature("pysweep")
def test_run_pre_bench_does_nothing_without_settings(monkeypatch):
    monkeypatch.setattr(pre_bench, "edp_client", lambda: pytest.fail("EDP contacted"))
    pre_bench.run_pre_bench({})


@allure.feature("pysweep")
def test_run_pre_bench_seeds_the_declared_number_of_vectors(monkeypatch):
    calls = []
    monkeypatch.setattr(pre_bench.seed, "seed", lambda num_vectors, edp: calls.append((num_vectors, edp)))
    pre_bench.run_pre_bench({pre_bench.NUM_VECTORS: 50000}, edp="edp")
    assert calls == [(50000, "edp")]
