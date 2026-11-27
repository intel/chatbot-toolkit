# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import allure
import pytest

from tests.e2e.benchmarks.chatqna.pysweep.provenance import hardware
from tests.e2e.benchmarks.chatqna.pysweep.provenance.hardware import (
    HardwareError,
    parse_cpuinfo,
    read_hardware,
)

NFD = hardware.NFD


def cpuinfo(model="Intel(R) Xeon(R) Platinum 8468H", sockets=2, cores=2, threads_per_core=2, numa=2):
    blocks, processor = [], 0
    for socket in range(sockets):
        for core in range(cores):
            for _ in range(threads_per_core):
                blocks.append(
                    f"processor\t: {processor}\nmodel name\t: {model}\n"
                    f"physical id\t: {socket}\ncore id\t\t: {core}\nflags\t\t: fpu amx_tile"
                )
                processor += 1
    return "\n\n".join(blocks) + f"\n\n#numa\n{numa}\n"


def node(name="node-1", model_id="143", vendor="Intel", family="6", memory="2109061508Ki", labels=None):
    nfd = {f"{NFD}cpu-model.vendor_id": vendor, f"{NFD}cpu-model.family": family,
           f"{NFD}cpu-model.id": model_id, f"{NFD}cpu-cpuid.AMXTILE": "true"}
    return SimpleNamespace(
        name=name,
        metadata={"labels": nfd if labels is None else labels},
        status={"capacity": {"memory": memory},
                "nodeInfo": {"osImage": "Ubuntu 24.10", "kernelVersion": "6.11.0", "kubeletVersion": "v1.35.4"}},
    )


class Pod:
    def __init__(self, name, node_name, stdout, phase="Running"):
        self.name = name
        self.spec = {"nodeName": node_name}
        self.status = {"phase": phase}
        self.metadata = {}
        self.stdout = stdout
        self.exec_calls = []

    def exec(self, command, container=None):
        self.exec_calls.append(container)
        return SimpleNamespace(stdout=self.stdout.encode())


def install(monkeypatch, pods, nodes):
    by_name = {n.name: n for n in nodes}
    selectors = []

    def get(resource, namespace=None, label_selector=None):
        selectors.append((resource, namespace, label_selector))
        return pods

    monkeypatch.setattr(hardware, "kr8s", SimpleNamespace(
        get=get, objects=SimpleNamespace(Node=SimpleNamespace(get=lambda name: by_name[name])),
    ))
    return selectors


@allure.feature("pysweep")
def test_cpuinfo_gives_the_model_and_topology():
    assert parse_cpuinfo(cpuinfo(sockets=2, cores=48, threads_per_core=2)) == {
        "model": "Intel(R) Xeon(R) Platinum 8468H",
        "sockets": 2,
        "cores": 96,
        "threads": 192,
        "numa_nodes": 2,
    }


@allure.feature("pysweep")
def test_cpuinfo_without_a_model_name_is_an_error():
    with pytest.raises(HardwareError, match="model name"):
        parse_cpuinfo("processor\t: 0\n\n#numa\n1\n")


@allure.feature("pysweep")
def test_the_served_models_node_is_described(monkeypatch):
    pod = Pod("llama-0", "node-1", cpuinfo())
    selectors = install(monkeypatch, [pod], [node()])
    assert read_hardware("llama3-8b-awq") == {
        "nodes": ["node-1"],
        "cpu": {"vendor": "Intel", "model": "Intel(R) Xeon(R) Platinum 8468H", "family": "SPR",
                "model_id": 143, "sockets": 2, "cores": 4, "threads": 8, "numa_nodes": 2, "amx": True},
        "memory_gib": 2011,
        "os": "Ubuntu 24.10",
        "kernel": "6.11.0",
        "kubernetes": "v1.35.4",
    }
    assert selectors[0][2] == {"app.kubernetes.io/name": "llama3-8b-awq", "kserve.io/component": "workload"}
    assert pod.exec_calls == [hardware.MODEL_SERVER_CONTAINER]


@allure.feature("pysweep")
@pytest.mark.parametrize("model_id", ["173", "174"])
def test_granite_rapids_is_gnr(monkeypatch, model_id):
    install(monkeypatch, [Pod("p", "node-1", cpuinfo("Intel(R) Xeon(R) 6980P"))], [node(model_id=model_id)])
    assert read_hardware("m")["cpu"]["family"] == "GNR"


@allure.feature("pysweep")
@pytest.mark.parametrize("vendor, model_id", [("Intel", "207"), ("Intel", "175"), ("AMD", "143")])
def test_any_other_cpu_is_unrecognized_and_named(monkeypatch, vendor, model_id):
    install(monkeypatch, [Pod("p", "node-1", cpuinfo("Some CPU"))], [node(vendor=vendor, model_id=model_id)])
    with pytest.raises(HardwareError, match=f"Unrecognized CPU family.*vendor {vendor}, family 6, model id {model_id} \\(Some CPU\\); supported: GNR, SPR"):
        read_hardware("m")


@allure.feature("pysweep")
def test_a_node_without_nfd_labels_is_an_error(monkeypatch):
    install(monkeypatch, [Pod("p", "node-1", cpuinfo())], [node(labels={})])
    with pytest.raises(HardwareError, match="Node Feature Discovery"):
        read_hardware("m")


@allure.feature("pysweep")
def test_no_running_model_server_is_an_error(monkeypatch):
    install(monkeypatch, [Pod("p", "node-1", cpuinfo(), phase="Pending")], [node()])
    with pytest.raises(HardwareError, match="No running model-server pod"):
        read_hardware("m")


@allure.feature("pysweep")
def test_each_node_is_read_once(monkeypatch):
    first, second = Pod("a", "node-1", cpuinfo()), Pod("b", "node-1", cpuinfo())
    install(monkeypatch, [first, second], [node()])
    assert read_hardware("m")["nodes"] == ["node-1"]
    assert second.exec_calls == []


@allure.feature("pysweep")
def test_nodes_with_different_hardware_are_an_error(monkeypatch):
    pods = [Pod("a", "node-1", cpuinfo(sockets=2)), Pod("b", "node-2", cpuinfo(sockets=1))]
    install(monkeypatch, pods, [node("node-1"), node("node-2")])
    with pytest.raises(HardwareError, match="different hardware"):
        read_hardware("m")


@allure.feature("pysweep")
def test_nodes_with_the_same_hardware_are_listed(monkeypatch):
    pods = [Pod("a", "node-2", cpuinfo()), Pod("b", "node-1", cpuinfo())]
    install(monkeypatch, pods, [node("node-1"), node("node-2")])
    assert read_hardware("m")["nodes"] == ["node-1", "node-2"]
