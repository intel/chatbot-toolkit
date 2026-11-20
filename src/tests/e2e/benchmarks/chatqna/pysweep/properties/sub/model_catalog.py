# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The model-manager catalog: deployment names, and the HF repo behind each.

``model-manager`` deploys models from a YAML catalog (``models.yaml``) in which
every entry has a short **deployment name** and the **HuggingFace repo id** its
weights come from::

    models:
      - name: llama3-8b-awq
        model_id: casperhansen/llama-3-8b-instruct-awq
        category: llm

Both halves matter to pysweep, for different reasons.

**The name is what a sweep declares.** It is what ends up in
``--served-model-name``, and therefore what ``LLM_MODEL_NAME`` must hold and what
the served model answers to on the gateway. So ``llms.model`` in a sweep YAML
takes catalog names, not repo ids — a repo id there would deploy nothing and
match nothing.

**The repo id is what the benchmark's tokenizer needs.** ``benchmark.py -m``
counts tokens with ``transformers.AutoTokenizer``, which resolves a HF repo, not
a deployment name. The value is therefore *derived*, never configured: the runner
reads the observed model name off the deployment and looks its repo id up here
(see :func:`~..pysweep_bench.resolve_tokenizer`). Deriving it is the point —
a hand-set tokenizer that disagreed with the served model would silently mis-count
every input and output length in the results.

**Which catalog file.** The one the deployment actually used, which is not
necessarily the one in this repo: ``deployment/models.yaml`` here is the shipped
default, while an installed environment has its own copy under ``env/<name>/``
and the two do drift. The resolution order below therefore prefers explicit
configuration (``PYSWEEP_MODEL_CATALOG``) and then ``MM_CONFIG`` — the same
variable the ``model-manager`` wrapper exports — so pysweep reads the catalog
whose names the cluster was actually built from.
"""

import logging
import os
from pathlib import Path

import yaml

from ...paths import REPO_ROOT, deploy_roots, is_file

logger = logging.getLogger(__name__)

# Environment variables consulted for the catalog path, in order. MM_CONFIG is
# what the ai-solutions ./model-manager wrapper exports for the real CLI, so
# honouring it makes pysweep read the same catalog the deployment used without
# anyone having to configure pysweep separately.
CATALOG_ENV_VARS = ("PYSWEEP_MODEL_CATALOG", "MM_CONFIG")

# Catalog paths tried when nothing is configured. The installed environment's
# copy comes first, then this repo's shipped default.
CATALOG_FALLBACKS = (
    *(str(root / "env" / "local" / "models.yaml") for root in deploy_roots()),
    str(REPO_ROOT / "deployment" / "models.yaml"),
)


class CatalogError(Exception):
    """Raised when the model catalog is missing, malformed, or lacks a model."""


def catalog_path() -> Path:
    """Return the catalog file to read, or raise :class:`CatalogError`.

    Raising when nothing is found is deliberate: the alternative is an empty
    catalog, which would make every model name "unknown" and turn a
    configuration problem into a confusing per-model failure later on.
    """
    for var in CATALOG_ENV_VARS:
        configured = os.getenv(var)
        if configured:
            path = Path(configured)
            if not is_file(path):
                raise CatalogError(
                    f"{var}={configured!r} does not point at a readable file."
                )
            return path

    for candidate in CATALOG_FALLBACKS:
        path = Path(candidate)
        if is_file(path):
            return path

    raise CatalogError(
        "Cannot find a model-manager catalog. Set PYSWEEP_MODEL_CATALOG (or "
        "MM_CONFIG) to the models.yaml the deployment was built from; tried "
        f"{list(CATALOG_FALLBACKS)}."
    )


def load_catalog(path: str | Path | None = None) -> dict[str, dict]:
    """Return ``{name: entry}`` for every model in the catalog.

    Entries are returned verbatim, so callers can read whichever field they need
    (``model_id``, ``category``, ``cpu``…) without this module having to know
    about each one.
    """
    resolved = Path(path) if path is not None else catalog_path()
    try:
        raw = yaml.safe_load(resolved.read_text())
    except OSError as exc:
        raise CatalogError(f"Cannot read model catalog '{resolved}': {exc}") from exc
    except yaml.YAMLError as exc:
        raise CatalogError(f"Model catalog '{resolved}' is not valid YAML: {exc}") from exc

    if not isinstance(raw, dict):
        raise CatalogError(
            f"Model catalog '{resolved}' must be a mapping at the top level, "
            f"got {type(raw).__name__}."
        )
    models = raw.get("models") or []
    if not isinstance(models, list):
        raise CatalogError(
            f"Model catalog '{resolved}' has a 'models' key that is not a list."
        )

    catalog: dict[str, dict] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if name:
            catalog[str(name)] = entry
    if not catalog:
        raise CatalogError(f"Model catalog '{resolved}' declares no named models.")
    logger.debug("Loaded %d model(s) from catalog %s.", len(catalog), resolved)
    return catalog


def known_models(path: str | Path | None = None) -> list[str]:
    """Every deployment name the catalog offers, sorted — the valid ``llms.model`` values."""
    return sorted(load_catalog(path))


def model_entry(name: str, path: str | Path | None = None) -> dict:
    """Return one model's catalog entry, or raise naming what *is* available.

    Listing the alternatives matters here because the failure this catches is
    almost always a name typo or a repo id used where a name belongs, and both
    are obvious the moment the real names are on screen.
    """
    catalog = load_catalog(path)
    if name not in catalog:
        raise CatalogError(
            f"Model {name!r} is not in the model-manager catalog. "
            f"Available: {sorted(catalog)}. Note that llms.model takes catalog "
            "*names* (e.g. 'llama3-8b-awq'), not HuggingFace repo ids."
        )
    return catalog[name]


def tokenizer_for(name: str, path: str | Path | None = None) -> str:
    """Return the HF repo id whose tokenizer matches the model deployed as ``name``.

    Raises :class:`CatalogError` if the entry has no ``model_id``: a model served
    from a repo pysweep cannot name is one whose token counts it cannot compute,
    and reporting throughput per token off the wrong tokenizer is worse than not
    reporting it.
    """
    entry = model_entry(name, path)
    model_id = entry.get("model_id")
    if not model_id:
        raise CatalogError(
            f"Catalog entry for {name!r} has no 'model_id', so the matching "
            "HuggingFace tokenizer cannot be determined."
        )
    return str(model_id)
