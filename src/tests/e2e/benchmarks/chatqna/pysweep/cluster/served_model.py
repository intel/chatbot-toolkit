# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import logging

import kr8s
from kr8s.objects import APIObject

from tests.e2e.validation.constants import CHATQNA_NAMESPACE, LLM_INFERENCE_NAMESPACE

from ..properties.base import PropertyReadError

logger = logging.getLogger(__name__)

# The label, unlike the Deployment name, is stable across chart versions.
LLM_USVC_LABEL_SELECTOR = {"app.kubernetes.io/name": "llm-usvc"}
# Named because the pod also has an init container.
LLM_USVC_CONTAINER = "llm-usvc"
MODEL_NAME_ENV = "LLM_MODEL_NAME"


class LLMInferenceService(APIObject):
    """Minimal kr8s binding for KServe's ``LLMInferenceService``."""

    kind = "LLMInferenceService"
    version = "serving.kserve.io/v1alpha2"
    endpoint = "llminferenceservices"
    plural = "llminferenceservices"
    singular = "llminferenceservice"
    namespaced = True


class ServedModel:
    """Reader for the model every ChatQNA LLM replica generates with."""

    def __init__(
        self,
        namespace: str = CHATQNA_NAMESPACE,
        inference_namespace: str = LLM_INFERENCE_NAMESPACE,
    ):
        self.namespace = namespace
        self.inference_namespace = inference_namespace

    def _pipeline_pods(self) -> list:
        try:
            pods = list(
                kr8s.get("pods", namespace=self.namespace, label_selector=LLM_USVC_LABEL_SELECTOR)
            )
        except Exception as exc:  # noqa: BLE001
            raise PropertyReadError(
                f"Cannot list llm-usvc pods in ns/{self.namespace} "
                f"(selector {LLM_USVC_LABEL_SELECTOR}): {exc}"
            ) from exc
        # Terminating pods are not part of the observed configuration.
        live = [pod for pod in pods if not pod.metadata.get("deletionTimestamp")]
        if not live:
            raise PropertyReadError(
                f"No live llm-usvc pods in ns/{self.namespace} matching "
                f"{LLM_USVC_LABEL_SELECTOR}; the ChatQNA LLM microservice is not running."
            )
        return live

    def _pod_model(self, pod) -> str:
        # Exec, not the ConfigMap: envFrom is resolved only at container start.
        command = ["sh", "-c", f"printf %s \"${MODEL_NAME_ENV}\""]
        try:
            result = pod.exec(command, container=LLM_USVC_CONTAINER)
        except Exception as exc:  # noqa: BLE001
            raise PropertyReadError(
                f"Cannot read ${MODEL_NAME_ENV} from pod {pod.name}: {exc}"
            ) from exc
        stdout = result.stdout
        if isinstance(stdout, bytes):
            stdout = stdout.decode()
        model = stdout.strip()
        if not model:
            raise PropertyReadError(
                f"Pod {pod.name} has an empty ${MODEL_NAME_ENV}; it is not pointed "
                "at any model."
            )
        return model

    def _model_ready(self, model: str) -> bool:
        try:
            llmisvc = LLMInferenceService.get(model, namespace=self.inference_namespace)
        except Exception as exc:  # noqa: BLE001
            logger.debug("LLMInferenceService %r not readable: %s", model, exc)
            return False
        conditions = (llmisvc.raw.get("status", {}) or {}).get("conditions", []) or []
        return any(
            cond.get("type") == "Ready" and cond.get("status") == "True"
            for cond in conditions
        )

    def read(self) -> str:
        """Return the model all live replicas agree on, raising unless it is Ready."""
        observed = {pod.name: self._pod_model(pod) for pod in self._pipeline_pods()}

        distinct = set(observed.values())
        if len(distinct) > 1:
            raise PropertyReadError(
                "llm-usvc replicas do not agree on the model (a rollout is likely "
                f"in progress): {observed}"
            )
        model = distinct.pop()

        if not self._model_ready(model):
            raise PropertyReadError(
                f"The pipeline points at model {model!r} but no Ready "
                f"LLMInferenceService of that name exists in "
                f"ns/{self.inference_namespace}; nothing is serving it."
            )
        logger.debug("Observed pipeline model %r across %d replica(s).", model, len(observed))
        return model
