# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class PropertyReadError(Exception):
    """Raised when a property cannot read a valid observed value."""


class PropertyApplyError(Exception):
    """Raised when a property cannot establish a desired value."""


class Plane(str, Enum):
    """The channel through which a knob's value lives and is changed."""

    K8S_SPEC = "k8s_spec"
    RUNTIME_API = "runtime_api"


class Property(ABC):
    """Interface every application (=not client's) configuration knob must implement."""

    # K8s-plane properties iterate in the outer loop, runtime-API ones inner.
    plane: Plane

    @abstractmethod
    def read(self) -> Any:
        """Return the knob's current observed value, not the last value written."""
        raise NotImplementedError

    def is_satisfied(self, desired: Any) -> bool:
        return self.read() == desired

    @abstractmethod
    def apply(self, desired: Any) -> None:
        """Drive the knob toward ``desired`` idempotently, without waiting for convergence."""
        raise NotImplementedError

    def wait_until_ready(self, desired: Any, timeout: float = 300.0) -> None:
        """Block until the applied change is live; a no-op for synchronous planes."""
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} plane={self.plane.value}>"
