# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import kr8s

from tests.e2e.validation.constants import LLM_INFERENCE_NAMESPACE

logger = logging.getLogger(__name__)

MODEL_SERVER_CONTAINER = "main"
NFD = "feature.node.kubernetes.io/"
CPUINFO_COMMAND = [
    "sh", "-c",
    "cat /proc/cpuinfo; echo '#numa'; ls -d /sys/devices/system/node/node[0-9]* 2>/dev/null | wc -l",
]

# The supported CPUs: Intel family 6 model ids, as NFD's cpu-model.id label reports them.
CPU_FAMILIES = {
    ("Intel", "6", 143): "SPR",
    ("Intel", "6", 173): "GNR",
    ("Intel", "6", 174): "GNR",
}


class HardwareError(Exception):
    """The hardware the served model runs on cannot be identified."""


def model_server_pods(model: str, namespace: str = LLM_INFERENCE_NAMESPACE) -> list:
    pods = [
        pod
        for pod in kr8s.get(
            "pods", namespace=namespace,
            label_selector={"app.kubernetes.io/name": model, "kserve.io/component": "workload"},
        )
        if pod.status.get("phase") == "Running" and not pod.metadata.get("deletionTimestamp")
    ]
    if not pods:
        raise HardwareError(f"No running model-server pod of {model!r} in ns/{namespace}.")
    return pods


def parse_cpuinfo(text: str) -> dict:
    cpuinfo, _, numa = text.partition("#numa")
    blocks = [dict(line.split(":", 1) for line in block.splitlines() if ":" in line)
              for block in cpuinfo.strip().split("\n\n")]
    blocks = [{k.strip(): v.strip() for k, v in block.items()} for block in blocks if block]
    if not blocks or "model name" not in blocks[0]:
        raise HardwareError("/proc/cpuinfo has no 'model name'.")
    sockets = {block.get("physical id") for block in blocks}
    cores = {(block.get("physical id"), block.get("core id")) for block in blocks}
    numa_nodes = int(numa.strip() or 0)
    if not numa_nodes:
        raise HardwareError("No NUMA nodes under /sys/devices/system/node.")
    return {
        "model": blocks[0]["model name"],
        "sockets": len(sockets),
        "cores": len(cores),
        "threads": len(blocks),
        "numa_nodes": numa_nodes,
    }


def node_facts(node) -> tuple[dict, dict]:
    """CPU facts from the node's Node Feature Discovery labels, and the node's memory and OS."""
    labels = node.metadata.get("labels", {})
    try:
        vendor = labels[f"{NFD}cpu-model.vendor_id"]
        family_id = labels[f"{NFD}cpu-model.family"]
        model_id = int(labels[f"{NFD}cpu-model.id"])
    except (KeyError, ValueError) as exc:
        raise HardwareError(
            f"Node {node.name} has no Node Feature Discovery cpu-model labels ({exc})."
        ) from exc
    memory = node.status.get("capacity", {}).get("memory", "")
    if not memory.endswith("Ki"):
        raise HardwareError(f"Node {node.name} reports memory capacity {memory!r}, not in Ki.")
    info = node.status.get("nodeInfo", {})
    cpu = {
        "vendor": vendor,
        "family": CPU_FAMILIES.get((vendor, family_id, model_id)),
        "family_id": family_id,
        "model_id": model_id,
        "amx": labels.get(f"{NFD}cpu-cpuid.AMXTILE") == "true",
    }
    system = {
        "memory_gib": round(int(memory[:-2]) / 1024 ** 2),
        "os": info.get("osImage"),
        "kernel": info.get("kernelVersion"),
        "kubernetes": info.get("kubeletVersion"),
    }
    return cpu, system


def read_hardware(model: str) -> dict:
    """The CPU, memory and OS of the node(s) serving ``model``."""
    nodes: dict[str, dict] = {}
    for pod in model_server_pods(model):
        node_name = pod.spec.get("nodeName")
        if node_name in nodes:
            continue
        try:
            result = pod.exec(CPUINFO_COMMAND, container=MODEL_SERVER_CONTAINER)
        except Exception as exc:  # noqa: BLE001
            raise HardwareError(f"Cannot read /proc/cpuinfo in pod {pod.name}: {exc}") from exc
        stdout = result.stdout.decode() if isinstance(result.stdout, bytes) else result.stdout
        cpu, system = node_facts(kr8s.objects.Node.get(node_name))
        topology = parse_cpuinfo(stdout)
        if cpu["family"] is None:
            raise HardwareError(
                f"Unrecognized CPU family on node {node_name}: vendor {cpu['vendor']}, family "
                f"{cpu['family_id']}, model id {cpu['model_id']} ({topology['model']}); "
                f"supported: {', '.join(sorted(set(CPU_FAMILIES.values())))}."
            )
        nodes[node_name] = {
            "cpu": {
                "vendor": cpu["vendor"], "model": topology.pop("model"), "family": cpu["family"],
                "model_id": cpu["model_id"], **topology, "amx": cpu["amx"],
            },
            **system,
        }

    if len({repr(hardware) for hardware in nodes.values()}) > 1:
        raise HardwareError(f"{model!r} is served from nodes with different hardware: {sorted(nodes)}.")
    hardware = next(iter(nodes.values()))
    logger.info("Model %r is served on %s: %s.", model, sorted(nodes), hardware["cpu"]["model"])
    return {"nodes": sorted(nodes), **hardware}
