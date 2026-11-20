# Copyright (C) 2026 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

"""``LlmModel`` — the model the ChatQNA pipeline generates with (K8s-spec plane).

Sweeping the model means changing two things that are owned by two different
systems, and this property's whole job is to drive both and then observe the
result:

1. **The model must be served.** Model serving lives in ``llm-inference`` as
   ``LLMInferenceService`` CRs behind the Envoy AI gateway, and it is
   ``model-manager`` that creates them (weights download, CPU/NUMA sizing, engine
   args, gateway route). pysweep delegates that step wholesale — see
   :mod:`.sub.model_manager` for why.

2. **The pipeline must be pointed at it.** ChatQNA's LLM microservice picks its
   model from ``LLM_MODEL_NAME`` in the ``llm-usvc-config`` ConfigMap, and that
   ConfigMap is rendered by the GMC operator from ``gmc-config`` in ns ``system``.
   So the write goes to ``gmc-config`` (see :mod:`.sub.llm_usvc_manifest`), never
   to ``llm-usvc-config`` directly: the operator re-applies the rendered manifest
   on every reconcile, so a direct edit would be reverted, and writing
   ``gmc-config`` is itself the trigger that makes the operator reconcile.

**Values are catalog names, not repo ids.** ``llama3-8b-awq``, not
``casperhansen/llama-3-8b-instruct-awq``. That short name is what model-manager
passes as ``--served-model-name``, so it is what the gateway routes on and what
``LLM_MODEL_NAME`` has to contain. The HuggingFace repo id is derived from it for
the benchmark's tokenizer only (:mod:`.sub.model_catalog`).

**Why the ConfigMap change is not the end of it.** ``llm-usvc``'s pod template
consumes the ConfigMap through ``envFrom``, and it carries no checksum annotation
— so when the operator updates ``llm-usvc-config``, the Deployment's pod template
is byte-identical and Kubernetes starts no rollout. Env vars are snapshotted at
container start, so the running pods keep the old model indefinitely. :meth:`apply`
therefore deletes the pods explicitly once the ConfigMap has caught up. Their
replacements block in an init container until the gateway actually answers for the
new model, which conveniently makes pod readiness a real signal here.

**Reading never reads what we wrote.** :meth:`read` execs into the live pods for
their effective ``LLM_MODEL_NAME`` and checks the named model is Ready in
``llm-inference``. Both halves are required: a pod pointed at a model that is not
being served answers nothing, and a model that is served but unreferenced changes
no benchmark. :meth:`declared` exposes the ``gmc-config`` side separately, which
is what lets :meth:`apply` stay idempotent mid-rollout.
"""

import logging
import time

import kr8s
from kr8s.objects import APIObject

from ..base import Plane, Property, PropertyApplyError, PropertyReadError
from .sub import model_manager
from .sub.llm_usvc_manifest import ManifestError, read_model, rewrite_for_model
from .sub.model_manager import ModelManagerError

logger = logging.getLogger(__name__)

# The ChatQNA LLM microservice: the Deployment is 'llm-svc-deployment' but the
# stable handle across chart versions is the label, and the container name is
# needed because the pod also has an init container.
PIPELINE_NAMESPACE = "chatqna"
LLM_USVC_LABEL_SELECTOR = {"app.kubernetes.io/name": "llm-usvc"}
LLM_USVC_CONTAINER = "llm-usvc"
LLM_USVC_CONFIGMAP = "llm-usvc-config"
MODEL_NAME_ENV = "LLM_MODEL_NAME"

# The GMC operator's desired state, and the key holding the LLM microservice's
# rendered manifest within it.
GMC_CONFIG_NAMESPACE = "system"
GMC_CONFIG_NAME = "gmc-config"
LLM_USVC_MANIFEST_KEY = "llm-usvc.yaml"

# Where model serving lives.
INFERENCE_NAMESPACE = "llm-inference"

# A model swap can be slow: model-manager may download weights before the engine
# loads them, and only then do the pipeline pods come up. model-manager --wait
# covers the serving half, so this ceiling covers the pipeline half plus slack.
CONVERGE_TIMEOUT = 3 * 60 * 60.0
CONVERGE_POLL_INTERVAL = 15.0
# The GMC operator reconciles within seconds of the gmc-config write; if it has
# not after this long, the operator is not running and waiting longer will not help.
RECONCILE_TIMEOUT = 300.0
RECONCILE_POLL_INTERVAL = 5.0


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


class LlmModel(Property):
    """The model the ChatQNA pipeline generates with."""

    plane = Plane.K8S_SPEC

    def __init__(
        self,
        namespace: str = PIPELINE_NAMESPACE,
        inference_namespace: str = INFERENCE_NAMESPACE,
    ):
        self.namespace = namespace
        self.inference_namespace = inference_namespace

    # -- reading (observed state) ---------------------------------------------

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
        both fail loudly and :meth:`wait_until_ready` is what tolerates them
        while they are transient.
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

    def is_satisfied(self, desired: str) -> bool:
        return self.read() == str(desired).strip()

    # -- writing --------------------------------------------------------------

    def _gmc_config(self):
        """Return the ``gmc-config`` ConfigMap holding the operator's desired state."""
        try:
            return kr8s.objects.ConfigMap.get(GMC_CONFIG_NAME, namespace=GMC_CONFIG_NAMESPACE)
        except Exception as exc:  # noqa: BLE001
            raise PropertyApplyError(
                f"Cannot read {GMC_CONFIG_NAME} in ns/{GMC_CONFIG_NAMESPACE}: {exc}"
            ) from exc

    def _llm_usvc_manifest(self, configmap) -> str:
        manifest = (configmap.data or {}).get(LLM_USVC_MANIFEST_KEY)
        if not manifest:
            raise PropertyApplyError(
                f"{GMC_CONFIG_NAME} has no '{LLM_USVC_MANIFEST_KEY}' entry; the "
                "ChatQNA LLM microservice is not managed by this ConfigMap, so "
                "pysweep cannot repoint it."
            )
        return manifest

    def declared(self) -> str:
        """Return the model ``gmc-config`` *declares* — what the next reconcile renders.

        Distinct from :meth:`read`, which observes what the pods run. During a
        swap the two differ, and that gap is what :meth:`wait_until_ready` closes.
        """
        try:
            return read_model(self._llm_usvc_manifest(self._gmc_config()))
        except ManifestError as exc:
            raise PropertyApplyError(str(exc)) from exc

    def _served_replicas(self, model: str) -> int | None:
        """The replica count ``model``'s LLMInferenceService runs at, if it exists."""
        try:
            llmisvc = LLMInferenceService.get(model, namespace=self.inference_namespace)
        except Exception:  # noqa: BLE001 - not deployed yet; nothing to carry over
            return None
        replicas = (llmisvc.raw.get("spec", {}) or {}).get("replicas")
        return int(replicas) if replicas is not None else None

    def _rendered_model(self) -> str | None:
        """The model in the operator's *rendered* output, ``llm-usvc-config``.

        The middle of the three states — declared (gmc-config), rendered
        (llm-usvc-config), observed (pod env) — and the one :meth:`apply` waits on
        before restarting pods, since restarting them any earlier would just have
        them pick the old value up again.
        """
        try:
            configmap = kr8s.objects.ConfigMap.get(LLM_USVC_CONFIGMAP, namespace=self.namespace)
        except Exception as exc:  # noqa: BLE001
            logger.debug("%s not readable: %s", LLM_USVC_CONFIGMAP, exc)
            return None
        return (configmap.data or {}).get(MODEL_NAME_ENV)

    def _wait_for_rendered(self, desired: str, timeout: float = RECONCILE_TIMEOUT) -> None:
        """Block until the operator has rendered ``desired`` into ``llm-usvc-config``."""
        deadline = time.monotonic() + timeout
        while True:
            rendered = self._rendered_model()
            if rendered == desired:
                logger.info("Operator rendered %s=%r into %s.", MODEL_NAME_ENV, desired, LLM_USVC_CONFIGMAP)
                return
            if time.monotonic() >= deadline:
                raise PropertyApplyError(
                    f"The GMC operator did not render {MODEL_NAME_ENV}={desired!r} "
                    f"into {self.namespace}/{LLM_USVC_CONFIGMAP} within {timeout:.0f}s "
                    f"(it still reads {rendered!r}). Check the gmc-controller "
                    f"deployment in ns/{GMC_CONFIG_NAMESPACE} is running."
                )
            logger.debug(
                "Waiting for the operator to render %s=%r (currently %r).",
                MODEL_NAME_ENV, desired, rendered,
            )
            time.sleep(RECONCILE_POLL_INTERVAL)

    def _restart_pipeline_pods(self) -> None:
        """Delete the LLM microservice pods so they re-read the ConfigMap.

        Necessary because the pod template consumes ``llm-usvc-config`` via
        ``envFrom`` with no checksum annotation: the operator's ConfigMap update
        leaves the template unchanged, so nothing rolls on its own and the running
        containers keep the env they started with. The Deployment recreates them,
        and their init container gates readiness on the new model actually
        answering on the gateway.
        """
        deleted = []
        for pod in kr8s.get("pods", namespace=self.namespace, label_selector=LLM_USVC_LABEL_SELECTOR):
            if pod.metadata.get("deletionTimestamp"):
                continue
            try:
                pod.delete()
            except Exception as exc:  # noqa: BLE001
                raise PropertyApplyError(f"Cannot delete pod {pod.name}: {exc}") from exc
            deleted.append(pod.name)
        if not deleted:
            raise PropertyApplyError(
                f"No llm-usvc pods to restart in ns/{self.namespace}; the "
                "microservice is not running, so the new model would never be picked up."
            )
        logger.info("Deleted %d llm-usvc pod(s) to pick up the new model: %s", len(deleted), deleted)

    def apply(self, desired: str) -> None:
        """Serve ``desired`` and point the pipeline at it (idempotent).

        Two owners, in the only order that works: the model has to be answering on
        the gateway *before* the pipeline is repointed, because the replacement
        ``llm-usvc`` pods' init container refuses to become Ready until it is.

        1. ``model-manager deploy`` — serves the model, carrying the outgoing
           model's replica count over so the two are benchmarked at the same scale.
        2. rewrite ``LLM_MODEL_NAME`` in ``gmc-config`` — the operator reconciles
           it into ``llm-usvc-config``.
        3. wait for that render, then delete the microservice's pods so they
           actually adopt it.

        Idempotence is checked against the **declared** value, not the observed
        one: re-applying mid-swap must not restart the swap from the beginning.
        When the manifest already names ``desired`` this returns without writing
        anything — though step 1 still runs, since a declared model that is not
        actually being served is exactly the state that needs repairing.

        Returns once the pods have been asked to restart; :meth:`wait_until_ready`
        is what waits for them to come back on the new model.
        """
        desired = str(desired).strip()

        current_model = None
        try:
            current_model = self.declared()
        except PropertyApplyError:
            # A manifest we cannot read is a fatal problem, but only for step 2 —
            # let the more specific error surface there.
            pass

        replicas = self._served_replicas(current_model) if current_model else None
        try:
            model_manager.deploy(desired, wait=True, replicas=replicas)
        except ModelManagerError as exc:
            raise PropertyApplyError(
                f"model-manager could not serve model {desired!r}: {exc}"
            ) from exc

        configmap = self._gmc_config()
        manifest = self._llm_usvc_manifest(configmap)
        try:
            updated = rewrite_for_model(manifest, desired)
        except ManifestError as exc:
            raise PropertyApplyError(
                f"Cannot rewrite {LLM_USVC_MANIFEST_KEY} for {desired!r}: {exc}"
            ) from exc

        if updated != manifest:
            logger.info(
                "Patching %s/%s '%s' to point the pipeline at %r (was %r).",
                GMC_CONFIG_NAMESPACE, GMC_CONFIG_NAME, LLM_USVC_MANIFEST_KEY,
                desired, read_model(manifest),
            )
            try:
                configmap.patch({"data": {LLM_USVC_MANIFEST_KEY: updated}})
            except Exception as exc:  # noqa: BLE001
                raise PropertyApplyError(
                    f"Failed to patch {GMC_CONFIG_NAME} in ns/{GMC_CONFIG_NAMESPACE}: {exc}"
                ) from exc
            # No operator restart is needed: gmc watches ConfigMaps in this
            # namespace, so the write is itself the reconcile trigger.
        else:
            logger.info("%s already declares %r; no patch needed.", GMC_CONFIG_NAME, desired)

        self._wait_for_rendered(desired)

        # Only restart if the pods are actually behind — a re-run of an already
        # converged combination must not bounce a healthy microservice.
        try:
            if self.read() == desired:
                logger.info("Pipeline pods already serve %r; no restart needed.", desired)
                return
        except PropertyReadError as exc:
            logger.debug("Observed state not readable before restart (%s); restarting.", exc)
        self._restart_pipeline_pods()

    def wait_until_ready(self, desired: str, timeout: float = CONVERGE_TIMEOUT) -> None:
        """Block until every replica generates with ``desired``.

        Polls the *observed* state, so it is the rollout — not the write — that
        decides when this returns. Everything mid-swap is tolerated while waiting
        (pods absent, replicas disagreeing, exec failing into a starting
        container, the model not yet Ready), because all of it is normal for a
        few minutes after :meth:`apply`.

        Raises :class:`PropertyApplyError` on timeout, naming what was last
        observed, so a sweep never benchmarks a configuration that never took
        effect.
        """
        desired = str(desired).strip()
        deadline = time.monotonic() + timeout
        while True:
            try:
                observed = self.read()
                last_state = repr(observed)
                if observed == desired:
                    logger.info("Pipeline now generates with %r on every replica.", desired)
                    return
            except PropertyReadError as exc:
                last_state = f"unreadable ({exc})"

            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise PropertyApplyError(
                    f"The ChatQNA pipeline did not converge to model {desired!r} within "
                    f"{timeout:.0f}s; last observed: {last_state}. Check the "
                    f"llm-usvc pods in ns/{self.namespace} (their init container "
                    f"blocks until the model answers on the gateway) and the "
                    f"model pods in ns/{self.inference_namespace}."
                )
            logger.info(
                "Waiting for model %r (last observed: %s); %.0fs of %.0fs remaining.",
                desired, last_state, remaining, timeout,
            )
            time.sleep(min(CONVERGE_POLL_INTERVAL, remaining))
