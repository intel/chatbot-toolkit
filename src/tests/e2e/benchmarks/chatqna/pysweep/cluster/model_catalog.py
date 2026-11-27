# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging
import os
from pathlib import Path

import yaml

from ..provenance.paths import REPO_ROOT, deploy_roots, is_file

logger = logging.getLogger(__name__)

# MM_CONFIG is what the ai-solutions ./model-manager wrapper exports.
CATALOG_ENV_VARS = ("PYSWEEP_MODEL_CATALOG", "MM_CONFIG")

CATALOG_FALLBACKS = (
    *(
        str(root / "env" / "local" / name)
        for root in deploy_roots()
        for name in ("models-rag.yaml", "models.yaml")
    ),
    str(REPO_ROOT / "deployment" / "models.yaml"),
)


class CatalogError(Exception):
    """Raised when the model catalog is missing, malformed, or lacks a model."""


def catalog_path() -> Path:
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
    """Return ``{name: entry}`` for every model in the catalog."""
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
    """Sorted catalog names: the valid ``llms.model`` values."""
    return sorted(load_catalog(path))


def model_entry(name: str, path: str | Path | None = None) -> dict:
    """Return one model's catalog entry, or raise listing the available names."""
    catalog = load_catalog(path)
    if name not in catalog:
        raise CatalogError(
            f"Model {name!r} is not in the model-manager catalog. "
            f"Available: {sorted(catalog)}. Note that llms.model takes catalog "
            "*names* (e.g. 'llama3-8b-awq'), not HuggingFace repo ids."
        )
    return catalog[name]


def tokenizer_for(name: str, path: str | Path | None = None) -> str:
    """Return the HF repo id whose tokenizer matches the model deployed as ``name``."""
    entry = model_entry(name, path)
    model_id = entry.get("model_id")
    if not model_id:
        raise CatalogError(
            f"Catalog entry for {name!r} has no 'model_id', so the matching "
            "HuggingFace tokenizer cannot be determined."
        )
    return str(model_id)
