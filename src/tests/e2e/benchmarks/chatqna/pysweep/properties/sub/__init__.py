# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Support modules for the properties in the parent package.

Nothing here is a :class:`~pysweep.base.Property` and nothing here is a sweep
knob: these modules are **not** addressable from a sweep YAML (only fields
carrying a ``PropertyBinding`` in ``config.py`` are, and ``extra="forbid"``
rejects everything else). They exist to keep the parent package's rule intact —
one module per knob, one knob per class — by holding the machinery a single
property needs to do its job.

Current contents, all three used only by :mod:`..llm_model`, which is the one
knob whose value is realised by two systems rather than one API call:

- :mod:`.model_manager` — the ``model-manager`` CLI that serves the model in
  ``llm-inference`` (weights, sizing, gateway route). pysweep decides *which*
  model; this decides how it gets served.
- :mod:`.llm_usvc_manifest` — the line-precise rewrite of ``LLM_MODEL_NAME`` in
  ``gmc-config``'s ``llm-usvc.yaml``, which is how the ChatQNA pipeline is
  repointed at the model.
- :mod:`.model_catalog` — the catalog those short model names come from, and the
  HuggingFace repo id behind each (needed for the benchmark's tokenizer).

They are plain functions over strings, dicts, and one subprocess, which is what
lets the interesting ones be tested against verbatim copies of the real deployed
manifest and catalog with no cluster (``test_llm_usvc_manifest_unit.py``,
``test_model_catalog_unit.py``).
"""
