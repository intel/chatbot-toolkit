#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""
Render the edp chart's text extractor autoscaling with `helm template`: the HPA
(hpa_enabled, Prometheus Adapter) or KEDA's ScaledObject (keda.enabled), never both,
with the same targets and behavior. Skipped where helm is not installed.
"""

import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

_CHART = Path(__file__).resolve().parents[4] / "deployment" / "components" / "edp"
_KEDA = ["ServiceAccount", "Role", "RoleBinding", "ConfigMap", "TriggerAuthentication", "ScaledObject"]

pytestmark = pytest.mark.skipif(shutil.which("helm") is None, reason="helm is not installed")


def _render(*values):
    args = ["helm", "template", "t", str(_CHART), "-n", "edp"]
    for value in values:
        args += ["--set", value]
    out = subprocess.run(args, check=True, capture_output=True, text=True).stdout
    return [d for d in yaml.safe_load_all(out) if d]


def _keda(docs):
    return {(d["kind"], d["metadata"]["name"]): d for d in docs
            if d["kind"] in _KEDA and d["metadata"]["name"].startswith("edp-text-extractor")
            and (d["kind"] in ("ScaledObject", "TriggerAuthentication") or "-keda" in d["metadata"]["name"])}


def _quantity(value):
    value = str(value)
    return float(value[:-1]) / 1000 if value.endswith("m") else float(value)


def _hpa(docs):
    return [d for d in docs if d["kind"] == "HorizontalPodAutoscaler"
            and d["metadata"]["name"] == "edp-text-extractor"]


def test_default_renders_neither():
    docs = _render()
    assert not _hpa(docs) and not _keda(docs)


def test_hpa_enabled_wins_over_keda():
    docs = _render("hpa_enabled=true", "keda.enabled=true")
    assert len(_hpa(docs)) == 1 and not _keda(docs)


def test_keda_objects():
    objects = _keda(_render("keda.enabled=true", "keda.prometheus.serverAddress=https://thanos:9092",
                            "keda.operatorNamespace=openshift-keda"))
    assert sorted(objects) == sorted([
        ("ServiceAccount", "edp-text-extractor-keda"),
        ("Role", "edp-text-extractor-keda-metrics"), ("RoleBinding", "edp-text-extractor-keda-metrics"),
        ("Role", "edp-text-extractor-keda-token"), ("RoleBinding", "edp-text-extractor-keda-token"),
        ("ConfigMap", "edp-text-extractor-keda-ca"),
        ("TriggerAuthentication", "edp-text-extractor-prometheus"),
        ("ScaledObject", "edp-text-extractor"),
    ])
    token = objects[("RoleBinding", "edp-text-extractor-keda-token")]
    assert token["subjects"] == [{"kind": "ServiceAccount", "name": "keda-operator", "namespace": "openshift-keda"}]
    auth = objects[("TriggerAuthentication", "edp-text-extractor-prometheus")]["spec"]
    assert auth["configMapTargetRef"] == [{"parameter": "ca", "name": "edp-text-extractor-keda-ca", "key": "service-ca.crt"}]
    triggers = objects[("ScaledObject", "edp-text-extractor")]["spec"]["triggers"]
    # The HPA dashboard names KEDA's s0 and s1 metrics after this order.
    assert [t["name"] for t in triggers] == ["files-to-upload", "files-active"]
    assert [float(t["metadata"]["threshold"]) for t in triggers] == [_quantity(m["object"]["target"]["value"])
                                                                      for m in _hpa(_render("hpa_enabled=true"))[0]["spec"]["metrics"]]
    assert [float(t["metadata"]["threshold"]) for t in _keda(_render(
        "keda.enabled=true", "textExtractor.hpa.scaleUpTargetValue=3", "textExtractor.hpa.scaleDownTargetValue=500m",
    ))[("ScaledObject", "edp-text-extractor")]["spec"]["triggers"]] == [3, 0.5]
    assert {t["metadata"]["serverAddress"] for t in triggers} == {"https://thanos:9092"}
    assert {t["metadata"]["namespace"] for t in triggers} == {"edp"}


@pytest.mark.parametrize("values", [
    [],
    ["textExtractor.hpa.behavior.scaleDown.stabilizationWindowSeconds=120"],
    ["textExtractor.hpa.behavior=null"],  # the chart's built-in behavior
], ids=["values", "override", "unset"])
def test_keda_behaves_like_the_hpa(values):
    hpa = _hpa(_render("hpa_enabled=true", *values))[0]["spec"]
    scaled = _keda(_render("keda.enabled=true", *values))[("ScaledObject", "edp-text-extractor")]["spec"]
    assert scaled["advanced"]["horizontalPodAutoscalerConfig"]["behavior"] == hpa["behavior"]
    assert hpa["behavior"]["scaleDown"]["stabilizationWindowSeconds"] == (120 if values and "120" in values[0] else 60)
    assert (scaled["minReplicaCount"], scaled["maxReplicaCount"]) == (hpa["minReplicas"], hpa["maxReplicas"])
