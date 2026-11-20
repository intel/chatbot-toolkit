# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Line-precise rewrite of ``gmc-config``'s ``llm-usvc.yaml`` model name.

``gmc-config`` (ns ``system``) holds the GMC operator's desired state as one
rendered multi-document YAML string per microservice. For the ChatQNA LLM
microservice that string is ``llm-usvc.yaml``, and the model the pipeline talks
to is a single field of the ``llm-usvc-config`` ConfigMap inside it::

    data:
      LLM_MODEL_NAME: "llama3-8b-awq"
      LLM_MODEL_SERVER: "vllm"
      LLM_MODEL_SERVER_ENDPOINT: "http://ai-gateway.envoy-gateway-system.svc"

**Why a text edit and not a YAML round-trip.** The value lives inside a string
in a ConfigMap, so writing it back means re-serialising the whole rendered
manifest. ``yaml.safe_load``/``safe_dump`` would not round-trip it: the manifest
is a multi-document stream carrying comments (``# Source: llm-usvc/templates/…``),
block scalars (the init container's shell script), and a key order that the
operator diffs against. A dump would rewrite all of it, turning a one-field
change into an unreviewable whole-file change. Editing the one line leaves every
other byte identical, so ``kubectl diff`` on the ConfigMap shows exactly what
pysweep changed.

**The discipline that makes a text edit safe** is that the anchor must match
*exactly once*. ``LLM_MODEL_NAME`` appears in the manifest in two very different
roles — once as the ConfigMap value being set, and several times as a shell
variable reference (``$LLM_MODEL_NAME``, ``${LLM_MODEL_NAME}``) inside the
init container's probe script. :data:`_MODEL_NAME_RE` is anchored to
"start of line, indented, followed by a colon", which the shell references can
never satisfy. If a chart change ever made the match ambiguous — or made it
disappear — :class:`ManifestError` is raised rather than a wrong line rewritten.

Everything here is a pure function over strings, which is what lets it be tested
against a verbatim copy of the deployed manifest with no cluster
(``test_llm_usvc_manifest_unit.py``).
"""

import re

# The ConfigMap key carrying the model name, anchored so it can only match the
# YAML mapping entry:
#   ^(\s+)          leading indentation, captured so the rewrite preserves it
#   LLM_MODEL_NAME  the key
#   :\s*            the YAML separator — a shell '$LLM_MODEL_NAME' has none
#   "?([^"\n]*)"?   the value, with or without quotes
# re.MULTILINE so '^' means "start of line" rather than "start of manifest".
_MODEL_NAME_RE = re.compile(
    r'^(?P<indent>[ \t]+)LLM_MODEL_NAME:[ \t]*"?(?P<value>[^"\n]*)"?[ \t]*$',
    re.MULTILINE,
)

# What the rewritten line looks like. The value is always quoted: model names are
# free-form and one that YAML would read as a number or a bool ('1.5', 'y')
# would otherwise change type on the operator's next parse.
_MODEL_NAME_LINE = '{indent}LLM_MODEL_NAME: "{model}"'


class ManifestError(Exception):
    """Raised when the manifest does not have exactly one model-name entry.

    Both directions are errors: no match means the chart no longer declares the
    model this way, and several matches mean the anchor is ambiguous. Either way
    pysweep cannot know which line to rewrite, so it refuses rather than guess.
    """


def _matches(manifest: str) -> list[re.Match]:
    return list(_MODEL_NAME_RE.finditer(manifest))


def _sole_match(manifest: str) -> re.Match:
    """Return the single model-name match, or raise :class:`ManifestError`."""
    found = _matches(manifest)
    if not found:
        raise ManifestError(
            "No 'LLM_MODEL_NAME: <value>' entry found in the llm-usvc manifest. "
            "The llm-usvc chart no longer sets the model through that ConfigMap "
            "key, so pysweep cannot repoint the pipeline by rewriting it."
        )
    if len(found) > 1:
        lines = [manifest[: m.start()].count("\n") + 1 for m in found]
        raise ManifestError(
            f"Found {len(found)} 'LLM_MODEL_NAME:' entries in the llm-usvc "
            f"manifest (lines {lines}); expected exactly one. Rewriting an "
            "ambiguous anchor could repoint the wrong ConfigMap."
        )
    return found[0]


def read_model(manifest: str) -> str:
    """Return the model name the manifest declares.

    This is the *declared* value — what the operator's next reconcile will put
    into ``llm-usvc-config`` — not what the running pods have loaded. See
    :meth:`~..llm_model.LlmModel.read` for the observed one.
    """
    return _sole_match(manifest).group("value").strip()


def rewrite_for_model(manifest: str, model: str) -> str:
    """Return ``manifest`` with its model name set to ``model``.

    Idempotent: rewriting a manifest that already names ``model`` returns the
    input string unchanged (byte for byte), which is what lets
    :meth:`~..llm_model.LlmModel.apply` detect "nothing to do" by comparison and
    skip the write — and therefore skip the rollout it would trigger.
    """
    model = model.strip()
    if not model:
        raise ManifestError("Refusing to rewrite the manifest with an empty model name.")
    if '"' in model or "\n" in model:
        raise ManifestError(
            f"Model name {model!r} contains a quote or newline; it cannot be "
            "written as a YAML scalar on one line."
        )

    match = _sole_match(manifest)
    replacement = _MODEL_NAME_LINE.format(indent=match.group("indent"), model=model)
    return manifest[: match.start()] + replacement + manifest[match.end():]
