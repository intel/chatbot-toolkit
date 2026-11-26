<!--
Copyright (C) 2026 Intel Corporation
SPDX-License-Identifier: Apache-2.0
-->

# pysweep — Declarative Configuration Sweeps for ChatQNA Benchmarks

> **Status.** This document covers both config planes. On the Runtime-API plane
> `llms.max_new_tokens` is implemented; `retriever.k` and `reranker.top_n` are
> designed and land next. The K8s-spec plane's one designed knob, `llms.model`
> (`LlmModel`: model-manager + `gmc-config`), is **not implemented yet**; the
> sections on it describe the design it will follow. Until then the runner only *reads* the served model,
> to derive the benchmark's tokenizer (`served_model.py`).

## Problem

Benchmarking the ChatQNA microservice app requires running the same workload
against many different app configurations (resource allocation, retriever/reranker
knobs, model-server env vars, ...). Today this is done with a set of bash scripts:

- `prepare_change_*.sh` — **setters** that mutate one knob each, split across two
  different mechanisms (Kubernetes and a runtime REST API).
- `preflight_checks/run.bash` — a **checker** with the expected values hardcoded
  inline, mixing auth, side effects, and `kubectl get` reads.

The result: **it is hard to know what configuration was actually tested** in a
given run. Expectations live in one script, setters in another, and nothing ties
a declared desired state to the state the system was actually in when the
benchmark ran.

## Goals

1. **Declare** the desired configuration (or a sweep of configurations) in one
   place — a YAML file — decoupled from *how* each knob is applied.
2. **Apply** the desired state idempotently before each benchmark iteration.
3. **Verify** that the system actually reached the desired state before the
   benchmark runs — reading *observed* state, not just the value we wrote. Kept
   distinct from **preflight**, which checks config-level preconditions that
   belong to no property (see "Execution flow").
4. **Record** the observed, resolved configuration alongside each run's results,
   so "what was tested" is never in doubt.
5. **Scale**: adding a new knob = adding one small, self-contained Python class.
6. **Stay in sync with the deployment.** Every property class has an accompanying
   **e2e test** that exercises its `read()` / `apply()` / `wait_until_ready()`
   against a real (or realistic) deployment. These tests are the contract that
   proves pysweep still matches the actual deployment modules — if a plane's
   mechanism drifts (e.g. gmc-config keys renamed, the runtime API path changes,
   a workload's kind/label changes), the property's e2e test fails **before** a
   benchmark run silently tests the wrong configuration. New knob = new property
   class **and** its e2e test.

## Non-goals (initial version)

- ~~Running the benchmark itself~~ — since implemented: `pysweep_bench.py` drives
  the load too, reusing `benchmark.py` unchanged rather than replacing it. The
  reason is provenance: only the runner knows which configuration produced which
  result CSV, so only it can write the summary row (see "Execution flow").
- More than the two planes described below (env-var / helm-values knobs are
  deferred — see "Future: a third plane").
- Automatic teardown / restore of a baseline between runs (noted as a risk;
  not implemented initially).

## Core concepts

### Property

A **Property** owns exactly one configuration knob. It knows:

- how to **read** the knob's *observed* value from the live system,
- whether an observed value **satisfies** a desired value,
- how to **apply** a desired value (write only when not already satisfied —
  idempotent),
- how to **wait** until the applied change has actually converged.

Interface (conceptual):

```python
class Property:
    plane: Plane

    def read(self) -> Any: ...                 # observed state, NOT the spec we wrote
    def is_satisfied(self, desired) -> bool: ...
    def apply(self, desired) -> None: ...       # no-op if already satisfied
    def wait_until_ready(self, timeout) -> None: ...
```

Each concrete property (e.g. `MaxNewTokens`, `RetrieverK`, `LlmModel`,
`RerankerTopN`) is a small class bound to a config leaf via
`Annotated[..., PropertyBinding(SomeProperty)]`, so the YAML references it by
that leaf's dotted path rather than by a registered name.

### Plane

A **Plane** is *the channel through which a knob's value lives and is changed*.
The planes differ in set mechanism, propagation semantics, how convergence is
detected, and what "read" should return. A property delegates the mechanics to
its plane.

We support **two planes** in the initial version:

| | **K8s spec plane** | **Runtime API plane** |
|---|---|---|
| Example knobs | the served LLM model | reranker `top_n`, retriever `k`, `max_new_tokens` |
| Where value lives | `gmc-config` ConfigMap (`system` ns), which the GMC operator renders onto the pipeline's workloads | inside the running app |
| Set mechanism | delegate serving to the `model-manager` CLI, then patch the knob inside `gmc-config` so the operator repoints the pipeline | authenticated HTTP POST to `system_fingerprint/change_arguments` |
| Client library | `kr8s` (native ConfigMap + workload access), plus a subprocess for `model-manager` | `httpx`/`requests` |
| Propagation | **async** — the ConfigMap write returns instantly; the operator re-renders and the pods have to be replaced, over minutes to hours | ~synchronous — live once the call returns 200 |
| `wait_until_ready` | poll observed state until the property holds the desired value | typically a no-op |
| `read()` returns | **observed/actual** (the env of the live pods' processes + serving status), **not** the ConfigMap and **not** `.spec` | GET the value back from the API |
| Drift source | direct edits of the operator's *rendered* output are reverted on its next reconcile — so we don't do them; `gmc-config` is the durable source | value is in-process; lost on pod restart |
| Auth | kubeconfig | Keycloak user access token |

**The set/verify split is the crux of the design.** Setting a K8s knob returns
immediately, but the system has not converged until GMC has reconciled and the
new pods are `Ready`. Verifying against the value we just wrote — whether the
ConfigMap or the workload `.spec` — would falsely pass. Therefore getters read
*observed* state (live workload + pod status), and `wait_until_ready()` gates
verification.

**Why not patch the pipeline's workloads directly?** They are **GMC-owned**: the
operator `Update`s them from the `gmc-config` templates on every reconcile, and it
reconciles on *any* ConfigMap change in the namespaces it watches. A direct
`kr8s`/`kubectl` patch — of the workload, or of a ConfigMap the operator renders —
is therefore reverted, the classic silent mid-sweep rollback. `gmc-config` is the
value GMC itself reads, so writing it is durable, and the write is itself the
reconcile trigger. See the K8s-plane source-of-truth decision below.

**And why not create the serving workloads either?** Getting a model served is not
one field: it is an `LLMInferenceService`, a weight download, CPU/NUMA and balloon
sizing, engine args, and a gateway route, all coupled per model. The deployment
team owns that as the `model-manager` CLI — the tool this cluster was built with —
so the K8s plane **delegates** it instead of reimplementing it. See the decision
below.

### Sweep config (YAML)

Declares the knobs and the value(s) to test. A single value = one setting; a
list = multiple settings to sweep. The existing `config.yaml` already uses this
list-of-values shape.

**Combination strategy: `product` (decided).** The sweep is the cartesian
product of every property's declared values:

- A property given a **single value** is fixed across all combinations.
- A property given a **list** contributes one dimension to the product; every
  one of its values appears in combinations paired with every combination of the
  other list-valued properties.
- The total number of combinations is the product of the list lengths (properties
  with a single value contribute a factor of 1). This blows up fast — and each
  K8s-plane change is a multi-minute rollout — so keep the number of list-valued
  properties (and their lengths) small.

Considered and deferred: `zip`/paired (lists advanced together) and `scenarios`
(an explicit list of named, fully-specified combinations). Not implemented in the
initial version.

The sweep is **ordered so expensive K8s-plane changes are the outer loop and
cheap runtime-API changes are the inner loop**, minimizing rollouts across the
product.

#### Client knobs: sweep dimensions with no plane

The load generator's own parameters (concurrency, input token count, duration,
warmup, …) are sweep knobs in the sense that matters: each distinct value is a
different test scenario, and they expand into the product like anything else. But
they have **no plane**, and modelling them as a third one was considered and
rejected. A `Plane` exists to describe *how a value lives in the running system
and how a change propagates to it*; a client parameter has no such existence.
Concretely, all three `Property` methods would be vacuous:

- `apply()` — nothing to mutate; the value is an argument to a process pysweep is
  about to start.
- `wait_until_ready()` — nothing to propagate.
- `read()` — nothing to observe. It could only echo the value the harness already
  holds, which is exactly the spec-read that "getters read observed state, never
  the spec we wrote" forbids. The preflight for such a property would be
  tautological, and its snapshot entry would be a claim with nothing behind it.

Since the invariant is what makes "what was tested" trustworthy for the knobs
that *do* have observable state, the client knobs get a separate adapter
(`client_args.py`) that mirrors `benchmark.py`'s options and renders resolved
values back into its CLI. They keep everything a knob needs — schema validation,
product expansion, result-file naming, a recorded snapshot — and stay out of the
apply/converge/verify machinery entirely. `ClientSetting` is deliberately a
different type from `PropertySetting`, so a client knob cannot reach that path by
accident.

**Two nested products, not one.** Client knobs expand into their own product,
which the runner nests inside the deployment one (see "Execution flow"). The
nesting is the ordering: deployment combinations vary in the outer loop because
each costs a rollout, client combinations in the inner loop because each costs
only another load test against a deployment already standing. Expressed as
nesting rather than as extra product dimensions, "client knobs are innermost" is
structural — there is no sort key that has to keep getting it right.

**Client knobs are required in the config**, unlike deployment knobs. An omitted
deployment knob means "leave it alone", which is a meaningful and observable
state. An omitted client knob has no such reading: the load test would still run,
silently on `benchmark.py`'s defaults, and the result CSV records none of it —
reintroducing precisely the "what configuration was actually tested" ambiguity
this design exists to remove. For the same reason the config is the single source
of truth: a client flag passed after `--` that the config also declares is
rejected rather than silently winning in either direction.

**Not every `benchmark.py` option is a knob, though.** `-m` names the tokenizer
that produces the `input_tokens` / `output_tokens` columns, and it has exactly one
correct value: the model being served. Any other tokenizer reports counts for a
tokenization the deployment never performed. A value with one correct setting is
not a dimension to sweep and not a choice to delegate to a config author, so it
has no config key at all — the runner derives it from the *observed* served model
(`LlmModel.read()`, or the combination's snapshot when `llms.model` is swept)
and renders it per combination. Declaring it, or passing `-m` after `--`, is an
error; an unreadable model aborts rather than falling back to a default tokenizer.

This is the one place a client argument depends on deployment state, and it is
resolved by *reading a property* rather than by giving the argument a plane — the
observation happens where observation belongs, and the client adapter stays a pure
translation.

## Authorization & access

pysweep talks to the live system through two authenticated channels, plus a
possible network hop to reach pod APIs. These are hard prerequisites — the tool
cannot apply or verify anything without them, so they must be resolved (and
checked) up front, not discovered mid-sweep.

### 1. Kubernetes API access (K8s spec plane)

- **kubeconfig.** `kr8s` needs a working kubeconfig to reach the cluster API.
  Resolution order: explicit `KUBECONFIG` env var → path in the sweep config →
  default `~/.kube/config`. pysweep does **not** assume a particular file
  location; it fails fast with a clear message if no context is reachable. (On
  the bench box the real config lives at
  `/root/.kube/igklab_perf_pdt_bench_gnr_kube.config`.)
- **RBAC / permissions required.** The service account or user the kubeconfig
  points at must be able to:
  - `get`/`patch` the `gmc-config` ConfigMap in the `system` namespace (the
    K8s-plane setter),
  - `get`/`list` ConfigMaps in the pipeline namespace (`chatqna`) — to see what the
    operator *rendered*, the middle of the property's three states,
  - `get`/`list`/`watch` Deployments and Pods in `chatqna`, **`create` on
    `pods/exec`** there (the observed model is read out of the running containers'
    environment, which is the only place it is observable), and **`delete` Pods**
    there — the rendered ConfigMap is consumed via `envFrom` with no checksum
    annotation, so nothing rolls on its own,
  - `get`/`list` `llminferenceservices` in `llm-inference` — to confirm the model
    the pipeline names is actually being served and Ready,
  - `get` Secrets in the `keycloak` namespace (Keycloak credentials — see below).

  Two notes on grants the earlier plan expected and that are **not** needed:
  **no write access to `gmc-controller`** (the plan assumed a `rollout restart` was
  required to trigger a reconcile; it is not — see "Design decisions"), and **no
  access to Helm release Secrets** (the per-model profiles the setter used to read
  from them are now model-manager's concern). Both were dropped from
  `REQUIRED_ACCESS`. Note also that a sweep declaring `llms.model` needs whatever
  `model-manager` itself needs, which is broader than this list — it creates the
  serving CRs. pysweep does not model that: it shells out, and a permission failure
  surfaces as the CLI's own non-zero exit.
- A **preflight authorization check** should verify these up front (e.g. a
  `SelfSubjectAccessReview` per verb/resource, or a cheap probing `get`) and
  report exactly which permission is missing rather than failing on the first
  write mid-combination.

### 2. Runtime API auth (Runtime API plane)

- The `system_fingerprint` endpoints require a **Keycloak user access token**.
  pysweep reuses the existing token machinery (`load_secrets.bash` +
  `generate_uat_to_file.sh`) refactored into a **token provider**: it reads the
  Keycloak admin/user credentials from the `auth` namespace secrets (hence the
  Secrets RBAC above), obtains a UAT, and refreshes it on expiry.
- The provider must be **side-effect free** for reads. Today's flow mutates realm
  session/token timeouts and toggles the user's required-actions as a side effect
  of minting a token; pysweep should scope or restore any such mutation so a
  "check" never changes cluster state.

### 3. Reaching pod APIs (port-forwarding)

- Some knobs are read/verified by calling an API served *inside* a pod (the
  fingerprint API), and the load test itself has to reach the pipeline's router —
  neither of which has a stable externally-routable address on every deployment.
- pysweep establishes access on demand via a **`kr8s` port-forward** to the
  target Pod/Service, scoped to the operation and torn down afterward — rather
  than assuming a pre-existing ingress or a `kubectl port-forward` running in
  another terminal. `kr8s` supports port-forwarding natively, so this stays in
  the "everything is a Python object" model.
- Where an ingress route already exists (the fingerprint API is reachable via
  `https://${ERAG_DOMAIN_NAME}/v1/system_fingerprint/...`), prefer it and skip
  the forward; port-forward is the fallback for pod-local APIs with no route.
- `no_proxy`/`NO_PROXY` must include the ERAG domain (as `load_secrets.bash`
  already arranges) so token and API calls don't get routed through a proxy.

### Summary of access prerequisites

| Need | Used by | How obtained |
|---|---|---|
| kubeconfig + cluster reachability | K8s-plane `read`/`apply`/`converge` | `KUBECONFIG` / config / default path |
| RBAC on `gmc-config`, `gmc-controller`, workloads, pods, secrets | K8s-plane setters & getters, token provider | cluster RBAC on the kubeconfig identity |
| Keycloak user access token | Runtime-API `read`/`apply` | token provider (refactored from existing scripts) |
| Port-forward to pod/service | reading pod-local APIs with no ingress | `kr8s` port-forward, on demand |
| `no_proxy` for the ERAG domain | all HTTP calls | env, set by the token/secrets step |

All of the above are validated in a **single up-front access preflight** (before
any combination runs), so a missing credential or permission fails loudly and
early rather than halfway through a multi-minute rollout.

## Execution flow (per combination)

```
preflight_check(config)                # BEFORE ANY HEAVY WORK: config-level preconditions, once
for each deployment combination:                       # OUTER: costs a rollout
  validate(combination)          # cross-property constraints, fail fast
    └─ for each property whose desired value changed:
         apply(desired)          # idempotent: write only if not is_satisfied
    └─ for each plane touched:
         wait_until_ready()  # observed value == desired
    └─ verify: for each property:
         assert is_satisfied(read())   # verify via OBSERVED state
    └─ snapshot = {dotted path: prop.read() for all props}  # persist "what was tested"
    └─ tokenizer = observed served model            # -m: derived, never declared
    └─ for each client combination:                  # INNER: costs one load test
         run benchmark  (existing scripts, argv rendered from the client knobs)
         summarize      # one row: knob columns (dotted paths) + TTFT/TPOT KPIs
```

**Preflight is separate from verification.** They fail at different times and mean
different things. Verification asserts that a *property* reached its desired value,
so it can only run after that property was applied and is inherently per-property.
`preflight_check()` asserts preconditions that belong to no property and that the
schema cannot express — whether a declared `client.question_file` names a readable
file, for instance: the schema validates it as a non-empty string, but whether that
string resolves to a file is a fact about the filesystem, not about the config's
shape. It runs **once**, before the first apply, when stopping is still free: every
check it makes is about the config and its environment, so nothing the sweep itself
does can change the answer, and repeating it per load test would only spend a
`kubectl` round trip to re-learn what it already knows. Each check is a named
function (`question_file_verify()`, `model_verify()`) called from that one entry
point rather than an inline `if`, so the flow says what is being asserted and there
is a single place a new precondition is added.

**Paths inside a sweep config are anchored to the harness, not to the CWD.** A
relative `client.question_file` resolves against `pysweep_bench.py`'s directory, so
one config names the same corpus wherever it is launched from. Resolving against
the working directory — which is what `benchmark.py`'s bare `open()` does, and the
reason the runner hands it an absolute path — would let the same declared string
mean a different file per caller, while the summary CSV recorded only the string.
Two runs would then carry identical provenance and different inputs, which is the
unattributable-result failure this design exists to prevent. The summary still
records the *declared* value, not the resolved absolute path, so provenance shows
what the config asked for.

The deployment configuration is applied and verified **once** per outer iteration
and then held still for every load scenario — which is both what keeps those
results comparable to each other and what keeps the rollout count at the number of
deployment combinations rather than the number of runs. The inner loop needs no
apply, no convergence wait and no verification, for the reasons in "Client knobs:
sweep dimensions with no plane".

**Summarizing is part of the flow, not a separate tool.** The runner is the only
thing that knows which configuration produced which CSV — the per-request files
carry neither, and their names carry only what varied — so it writes the record
while it still holds both, with the commit it ran from. That is also why the summary is rewritten after every run
rather than at the end: a sweep is hours long, and a table of what finished is
worth more than a complete one that a failed rollout prevents. Aggregation stays
lossless-by-default: the per-request CSVs are untouched and every row points back
to its own.

**One output directory per sweep, cleared before the sweep starts.** `--summaries`
(default `./results`) holds `results.json`, `benchmark_summary_results.csv` and a
`per_case_results/` subdirectory, and both are wiped of the previous sweep's
artifacts first. The alternative — writing each run's CSV into the CWD and
accumulating across sweeps, which is what `benchmark.py` alone does — makes the two
halves of one result set independently deletable: a stale per-case CSV with no row
in the current summary is indistinguishable from a result of it, and a re-run
shorter than its predecessor leaves rows pointing at files it never overwrote.
Refusing to merge is a stronger guarantee than any naming scheme, because the
failure it prevents is misattribution, not collision. The wipe is confined to the
two files and the one subdirectory the harness owns, so the directory can hold other
things; it happens after the config is validated and the preflight has passed, so
a mistake that would have failed every run costs nothing. `result_file` is recorded relative to the summary
so the directory survives being copied off the benchmark host.

## Design decisions

- **The K8s plane has two owners, and pysweep drives both (decided).** Getting a
  model benchmarked means (1) *serving* it and (2) *pointing the pipeline at it*,
  and those live in different systems. Each half was decided separately:
    - **Serving: delegate to `model-manager`.** It is the deployment team's own
      actively-developed CLI and the tool this cluster was built with, so it already
      encodes the coupled per-model decisions — runtime and engine version,
      CPU/memory, NUMA and balloon placement, the `LLMInferenceService`, the gateway
      route. Reimplementing that as CR patches would mean re-deriving all of it from
      a catalog pysweep does not own, and drifting from it silently. Its `deploy` is
      already idempotent (`kubectl diff --server-side`), which is exactly the
      contract a `Property.apply` needs, and its `--dry-run` is side-effect free,
      which is what lets both an e2e test and a preflight exercise the real code
      path without touching the deployment. `properties/sub/model_manager.py` is a
      deliberately thin wrapper: the single place that knows where the executable
      lives and how its arguments are spelled, so a CLI change is a one-file fix
      with unit tests that pin the argv.
        - Trade-off accepted: a subprocess in an otherwise all-`kr8s` design, and
          errors that arrive as text rather than as typed API failures. Owning a
          second implementation of model serving is the worse trade.
    - **Pointing the pipeline: `gmc-config` (Option B, decided).** The pipeline's
      workloads are GMC-owned, so direct patches (Option A) are reverted by the
      operator, and a full helm re-render per combination (Option C) is too heavy and
      isn't a `kr8s` operation. Option B patches the knob *inside the `gmc-config`
      ConfigMap* (the value GMC actually reads), which is durable and `kr8s`-native.
        - `apply()` = edit the knob inside the relevant `gmc-config` key, then write
          the ConfigMap back. **No `gmc-controller` restart** — the operator's
          `configMapFilter` returns `true` for any ConfigMap change in the
          namespaces it watches, so the write is itself the reconcile trigger
          (verified live: the patch logs a `----RECONCILE REQUEST----` for
          `gmc-config`/`system` and runs all ten reconcile steps).
        - **Line-precise text edits, not a YAML round-trip.** The earlier plan said
          "parse the embedded YAML properly, not `sed`". Implementation reversed
          that: each `gmc-config` value is a rendered manifest that the operator
          `Update`s onto the whole pipeline, and a parse/re-emit reformats it
          wholesale. `properties/sub/llm_usvc_manifest.py` instead anchors on a
          regex that must match **exactly once** and raises rather than editing when
          it doesn't — bounded blast radius, byte-exact round-trip, both asserted
          against a verbatim copy of the real manifest. That manifest contains
          `LLM_MODEL_NAME` in two roles (the ConfigMap entry, and shell references
          in the init container's probe script), so "exactly once" is load-bearing:
          a looser anchor would break the readiness gate of every LLM pod.
    - **The ordering between the two halves is not free.** The model must be
      answering on the gateway *before* the pipeline is repointed, because the
      replacement pods' init container refuses to become Ready until it is. So
      `apply()` is: deploy, then patch `gmc-config`, then wait for the operator to
      render it, then delete the pods. The unit tests assert that order explicitly
      rather than trusting it to stay incidental.
    - **A ConfigMap change is not always a rollout.** The LLM microservice consumes
      the rendered ConfigMap through `envFrom` with no checksum annotation, so the
      operator's update leaves the pod template byte-identical and Kubernetes starts
      nothing; `envFrom` is resolved once at container start, so the running
      processes would keep the old model indefinitely. Hence the explicit pod
      deletion — a step that reads like a workaround and is actually the mechanism.
    - **Three states, not two.** *declared* (`gmc-config`) → *rendered*
      (`llm-usvc-config`) → *observed* (the pods' env). `read()`/`is_satisfied()`
      return the last, never the ConfigMap (which is *intended*, not *observed*).
      `declared()` exposes the first separately — `apply` needs it for idempotence,
      since mid-rollout the pods still serve the old value and an observed-state
      check alone would redeploy and restart the rollout on every retry.
    - Trade-off accepted: a swap is slow — a weight download plus a pipeline
      restart, so `wait_until_ready` budgets **3h** — acceptable because K8s knobs
      are the sweep's **outer** loop and change rarely.
- **Kubernetes client: `kr8s`.** All K8s reads/patches go through `kr8s` objects
  rather than shelling out to `kubectl`, keeping the "everything is a Python
  object" model. Helm is deliberately avoided in the per-combination path.
- **Getters read observed state, never the spec we wrote.** Non-negotiable; it is
  what makes the preflight meaningful.
- **Idempotent apply = read-then-write-if-different.** A reliable getter is a
  prerequisite for a cheap setter — build getters first.
- **Auth is a factored-out token provider.** The Keycloak token dance
  (`load_secrets.bash` + `generate_uat_to_file.sh`) becomes a reusable provider;
  the preflight/check step is **side-effect free** (unlike today's `run.bash`,
  which mutates realm timeouts as a side effect of "checking").
- **YAML is schema-validated (Pydantic).** Catches typos in property names and
  wrong value types before any rollout is triggered.
- **Every run persists the observed snapshot**, not the intended YAML.

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Convergence timing — spec set ≠ system ready | Split set / converged; getters read observed; explicit `wait_until_ready` |
| GMC operator reverts direct workload patches mid-sweep | Never patch the workload or the operator's rendered output; drive `gmc-config` (Option B), the value GMC reads |
| Combinatorial explosion of the sweep | Explicit combination strategy; expensive plane = outer loop, free client knobs = inner loop |
| A model swap that reimplements serving drifts from how the cluster was actually built | Delegate the deploy to the deployment team's `model-manager` CLI; pysweep owns only the pipeline's pointer to it, and observes the result |
| A ConfigMap update that starts no rollout leaves pods on the old value while every state pysweep can cheaply read says converged | `read()` execs into the containers for their effective env; `apply()` deletes the pods once the operator has rendered |
| Load parameters known only from the shell history of whoever ran it | Client knobs are **required** in the sweep config, logged per run, and a conflicting `--` flag is refused |
| Cross-property constraints (e.g. `TP_SIZE` divides device count; thread bind fits CPU limit) | Whole-combination `validate()` before apply |
| Token counts measured with a tokenizer that is not the served model's, making tokens/s incomparable | `-m` is not configurable at all: derived per combination from the **observed** served model (`LlmModel.read()`, mapped to a tokenizer repo id through model-manager's catalog), so it cannot drift from what is loaded; declaring it or passing it after `--` is refused, and an unreadable model aborts the sweep |
| Partial apply leaves dirty state for next combination | Reconcile *every* property each iteration; baseline snapshot/teardown (future) |
| Setter always writes → needless rollout | Idempotency via read-then-write-if-different |
| Auth token expiry / side effects | Token provider abstraction; side-effect-free checks |
| pysweep silently drifts from the deployment (renamed gmc-config keys, changed API path, changed workload kind/label) → benchmarks test the wrong config | Every property has an e2e test exercising `read`/`apply`/`converge` against a real deployment; drift fails the test, not the benchmark |

## Future: a third plane

Env-var / helm-values knobs (e.g. `VLLM_CPU_OMP_THREADS_BIND`,
`config_model_server_extra`) are neither instant (runtime API) nor a simple
`kubectl patch` rollout: the value lives in helm values *and* only takes effect
on the next pod start. This is a likely third plane, deferred for now.

## Mapping from existing scripts

| Existing | Plane | Property |
|---|---|---|
| `prepare_change_reranker.sh` (`top_n`) | Runtime API | `RerankerTopN` — **designed, not implemented** (`reranker.top_n`) |
| `prepare_change_retriever.sh` (`k`) | Runtime API | `RetrieverK` — **designed, not implemented** (`retriever.k`) |
| `prepare_change_max_tokens.sh` (`max_new_tokens`) | Runtime API | `MaxNewTokens` — **implemented** (`llms.max_new_tokens`) |
| `switch_llm_model.sh` (the served model) | K8s spec | `LlmModel` — **designed, not implemented** (`llms.model`), via `model-manager` + `gmc-config` |
| `run.bash` model-server replicas / CPU / memory reads | K8s spec | Not ported: these are now per-model facts in model-manager's catalog, so a sweep dimension over them would fight the tool that owns them. Sweep `llms.model` instead. |
| `load_secrets.bash` + `generate_uat_to_file.sh` | — | Bearer file minted by the runner, re-minted per run (`pysweep_bench.provision_benchmark_target`) |

## TODO / open questions

- [x] Decide combination strategy — **`product`** (cartesian product of all
      list-valued properties; single-valued properties are fixed).
- [x] Decide K8s-plane source of truth — **`gmc-config` ConfigMap (Option B)**,
      driven via `kr8s`; observed reads from the live workload/pod status.
- [x] Confirm whether the runtime API exposes a **read/GET** for each knob —
      **yes**: `GET system_fingerprint/config?params_key=<component>` returns
      `{"values": {...}}` with the component's current arguments. Both runtime-API
      properties read through it.
- [x] Confirm what serves the LLM and where — **not a GMC workload at all** on this
      architecture: serving moved out to `LLMInferenceService` CRs
      (`serving.kserve.io/v1alpha2`) in `llm-inference`, created by `model-manager`,
      behind the Envoy AI gateway. What lives in the pipeline namespace (`chatqna`)
      is the `llm-usvc` Deployment that *calls* it, and that is where the observed
      model is read from.
- [ ] Port `RetrieverK` (`prepare_change_retriever.sh`) and `RerankerTopN`
      (`prepare_change_reranker.sh`), the remaining runtime-API knobs from the
      original scripts.
