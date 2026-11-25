# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""Core abstractions for pysweep: the Plane enum and the Property interface.

A **Property** owns exactly one configuration knob. It knows how to read the
knob's *observed* value from the live system, whether an observed value
satisfies a desired one, how to apply a desired value idempotently, and how to
wait until an applied change has actually converged.

A **Plane** is the channel through which a knob's value lives and is changed.
See DESIGN.md for the full model.
"""

from abc import ABC, abstractmethod
from enum import Enum
from typing import Any


class PropertyReadError(Exception):
    """Raised when a property cannot read a valid observed value.

    A missing or unparseable value is never an acceptable state for a sweep:
    it means we cannot know what configuration is live, so it must fail loudly
    rather than degrade to a ``None`` that silently never satisfies a desired
    value.
    """


class PropertyApplyError(Exception):
    """Raised when a property cannot establish a desired value.

    Distinct from :class:`PropertyReadError` (which is about not knowing the
    current state) and from ``NotImplementedError`` (which is about a knob that
    cannot be written at all). This is "the write was attempted and the desired
    state could not be reached" — a rejected value, a manifest the property no
    longer recognises, or a rollout that never converged.
    """


class Plane(str, Enum):
    """The channel through which a knob's value lives and is changed."""

    K8S_SPEC = "k8s_spec"           # Kubernetes objects - long wait for change
    RUNTIME_API = "runtime_api"     # Fingerprint - immediate change


class Property(ABC):
    """Interface every application (=not client's) configuration knob must implement."""

    # The way the Property is changed. Important for scheduling sweeps.
    # Kubernetes-based props are outer, fingerprint-based are innerly iterated
    plane: Plane

    @abstractmethod
    def read(self) -> Any:
        """Return the knob's current **observed** value from the live system.

        Must reflect actual runtime/observed state, not the last value written.
        """
        raise NotImplementedError

    def is_satisfied(self, desired: Any) -> bool:
        return self.read() == desired

    @abstractmethod
    def apply(self, desired: Any) -> None:
        """Drive the knob toward ``desired``.

        Idempotent, that means:
        - desired == observed --> do nothing
        - desired == applied --> do nothing

        Convergence (readiness) handled separately by `wait_until_ready()`
        """
        raise NotImplementedError

    def wait_until_ready(self, desired: Any, timeout: float = 300.0) -> None:
        """Block until the property itself is ready — the applied change is live.

        For synchronous planes (runtime API) this is a no-op. For async planes
        (K8s spec) this polls observed state until ``is_satisfied(desired)``
        holds or ``timeout`` seconds elapse.
        """
        return None

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} plane={self.plane.value}>"
