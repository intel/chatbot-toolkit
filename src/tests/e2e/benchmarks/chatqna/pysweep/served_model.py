# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""The model the ChatQNA pipeline is **observed** to generate with.

Not a knob: nothing here writes. The runner needs the served model whether or not
a sweep varies it, because ``benchmark.py``'s ``-m`` must be that model's
tokenizer (see ``pysweep_bench.resolve_tokenizer``).

The model is read where it takes effect. ChatQNA's LLM microservice (``llm-usvc``)
sends ``LLM_MODEL_NAME`` as ``model=`` on every request, and that variable comes
from the ``llm-usvc-config`` ConfigMap through ``envFrom``, which is resolved once,
at container start. So :meth:`ServedModel.read` execs into every live replica for
the value its process actually has, and additionally requires that model to be a
Ready ``LLMInferenceService`` in ``llm-inference``: a pipeline pointed at a model
nothing serves answers nothing, and that is not a configuration to benchmark.

**Values are catalog names, not repo ids.** ``llama3-8b-awq``, not
``casperhansen/llama-3-8b-instruct-awq``: the short name is what the gateway routes
on and what ``LLM_MODEL_NAME`` contains. :mod:`.model_catalog` maps it to the
HuggingFace repo id the tokenizer needs.
"""

import logging

import kr8s
from kr8s.objects import APIObject

from tests.e2e.validation.constants import CHATQNA_NAMESPACE, LLM_INFERENCE_NAMESPACE

from .base import PropertyReadError

logger = logging.getLogger(__name__)

# The ChatQNA LLM microservice: the Deployment is 'llm-svc-deployment' but the
# stable handle across chart versions is the label, and the container name is
# needed because the pod also has an init container.
LLM_USVC_LABEL_SELECTOR = {"app.kubernetes.io/name": "llm-usvc"}
LLM_USVC_CONTAINER = "llm-usvc"
MODEL_NAME_ENV = "LLM_MODEL_NAME"


class LLMInferenceService(APIObject):
    """Minimal kr8s binding for KServe's ``LLMInferenceService``.

    Hand-written rather than looked up dynamically so the apiVersion this
    property depends on is stated in one visible place: if the CRD moves past
    ``v1alpha2``, the failure is a clear "not found" here rather than a subtle
    mismatch elsewhere.
    """

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
        """Return the LLM microservice pods, or raise if none are selectable."""
        try:
            pods = list(
                kr8s.get("pods", namespace=self.namespace, label_selector=LLM_USVC_LABEL_SELECTOR)
            )
        except Exception as exc:  # noqa: BLE001 - reported as an unreadable state
            raise PropertyReadError(
                f"Cannot list llm-usvc pods in ns/{self.namespace} "
                f"(selector {LLM_USVC_LABEL_SELECTOR}): {exc}"
            ) from exc
        # Pods being replaced are not part of the observed configuration; counting
        # them would make read() disagree with itself for the whole grace period.
        live = [pod for pod in pods if not pod.metadata.get("deletionTimestamp")]
        if not live:
            raise PropertyReadError(
                f"No live llm-usvc pods in ns/{self.namespace} matching "
                f"{LLM_USVC_LABEL_SELECTOR}; the ChatQNA LLM microservice is not running."
            )
        return live

    def _pod_model(self, pod) -> str:
        """Return the model name one replica's process actually has in its env.

        Read by exec, not from the ConfigMap or the pod spec, because ``envFrom``
        is resolved once at container start: the ConfigMap can already hold the
        new value while this process still runs with the old one, and that gap is
        precisely what must not be reported as converged.
        """
        command = ["sh", "-c", f"printf %s \"${MODEL_NAME_ENV}\""]
        try:
            result = pod.exec(command, container=LLM_USVC_CONTAINER)
        except Exception as exc:  # noqa: BLE001 - starting/terminating pod
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
        """Whether ``model`` is a Ready ``LLMInferenceService`` in the inference namespace."""
        try:
            llmisvc = LLMInferenceService.get(model, namespace=self.inference_namespace)
        except Exception as exc:  # noqa: BLE001 - absent or CRD missing
            logger.debug("LLMInferenceService %r not readable: %s", model, exc)
            return False
        conditions = (llmisvc.raw.get("status", {}) or {}).get("conditions", []) or []
        return any(
            cond.get("type") == "Ready" and cond.get("status") == "True"
            for cond in conditions
        )

    def read(self) -> str:
        """Return the model the pipeline is **observed** to be generating with.

        Raises :class:`PropertyReadError` unless the whole picture is consistent:
        every live replica names the same model, and that model is Ready in
        ``llm-inference``. Replicas disagreeing means a rollout is in flight;
        the model not being Ready means the pipeline points somewhere that cannot
        answer. Neither is a configuration a benchmark may be run against, so
        both fail loudly rather than being reported as a model.
        """
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
