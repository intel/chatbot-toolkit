# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from .commit_info import CommitInfoError, read_commit_info
from .hardware import HardwareError, read_hardware


class MetadataError(Exception):
    """A piece of the results' metadata cannot be collected."""


def collect_metadata() -> dict:
    """``commit_info`` and the served model's ``hardware``; every part is mandatory."""
    from ..cluster.served_model import ServedModel
    from ..properties.base import PropertyReadError

    try:
        commit_info = read_commit_info()
    except CommitInfoError as exc:
        raise MetadataError(f"commit_info: {exc}") from exc
    try:
        hardware = read_hardware(ServedModel().read())
    except (HardwareError, PropertyReadError) as exc:
        raise MetadataError(f"hardware: {exc}") from exc
    return {"commit_info": commit_info, "hardware": hardware}
