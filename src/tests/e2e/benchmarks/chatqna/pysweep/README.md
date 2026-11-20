<!-- Copyright (C) 2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# pysweep

Declarative configuration sweeps / benchmarking for the ChatQNA (Intel Enterprise
RAG) deployment. pysweep's core promise is that you always know **what
configuration was actually tested**: each tunable is modelled as a `Property`
that can `read` the deployment's observed state, `apply` a desired value, and
report whether the desired value `is_satisfied`.

See [`DESIGN.md`](./DESIGN.md) for the full architecture (the `Property` / `Plane`
model, the two config planes, the combination strategy, and the authorization &
access requirements). This README covers **how to run the tests** and the
environment gotchas.

## Layout

| Path | What it is |
|---|---|
| `base.py` | `Property` interface + `Plane` enum (K8s-spec plane vs Runtime-API plane). |
| `config.py` | Pydantic-validated configs: `load_config()` / `SweepConfig` (+ the `deployment_combinations()` and `client_combinations()` cartesian `product` expansions) and `load_run_config()` / `RunConfig`. |
| `client_args.py` | Adapter mirroring `benchmark.py`'s CLI options as sweep knobs. Not properties — no plane, nothing to apply or read back; see below. |
| `pysweep_bench.py` | Sweep runner: load config → apply/converge/verify each deployment combination → run one load test per client combination, reusing `benchmark.py`. |
| `examples/sweep.yaml` | An example sweep config (`--sweep`): what to vary. |
| `examples/config.yaml` | An example run config (`--config`): how to run. |
| `authentication.py` | `KubernetesAccess` (side-effect-free RBAC preflight via `SelfSubjectAccessReview`) and `KeycloakTokenProvider` (side-effect-free user-token minting). Both reuse the maintained `tests.e2e.helpers.k8s_helper.K8sHelper`. |
| `runtime_api.py` | `system_fingerprint` client. `HelperFingerprintClient` adapts the suite's `FingerprintApiHelper`; `PortForwardTransport` / `IngressTransport` cover deployments the helper can't reach. |
| `properties/max_tokens.py` | First property: `MaxNewTokens` (LLM `max_new_tokens`, Runtime-API plane). |
| `properties/retriever_k.py` | `RetrieverK` (the retriever's `k`, Runtime-API plane), migrated from `prepare_change_retriever.sh`. |
| `properties/reranker_top_n.py` | `RerankerTopN` (the reranker's `top_n`, Runtime-API plane), migrated from `prepare_change_reranker.sh`. Sets how much context reaches the prompt, so it is the LLM's input length. |
| `properties/llm_model.py` | `LlmModel` (the model the pipeline generates with, K8s-spec plane): reads the observed model out of the running pods, writes by delegating the deploy to `model-manager` and repointing the pipeline through `gmc-config` — see below. |
| `properties/sub/` | Support modules for the properties above — **not** properties and **not** sweep knobs (nothing here is addressable from a sweep YAML). Keeps `properties/` to one module per knob. |
| `properties/sub/model_manager.py` | Thin wrapper over the deployment team's `model-manager` CLI: the one place that knows where the executable is and how its arguments are spelled. |
| `properties/sub/model_catalog.py` | Reader for model-manager's catalog (`models.yaml`): validates a declared model name and maps it to the HuggingFace repo id the benchmark's tokenizer needs. |
| `properties/sub/llm_usvc_manifest.py` | Line-precise rewrite of `LLM_MODEL_NAME` in `gmc-config`'s `llm-usvc.yaml`. Text surgery, not a YAML round-trip: re-emitting would reformat a manifest the operator pushes to the whole pipeline. |

Only the modules directly under `properties/` define knobs. A file there means a
`Property` subclass bound to a YAML leaf in `config.py`; a file in `sub/` is
machinery one of those properties needs, reachable only through it.

The **e2e tests live with the rest of the validation suite** at
`src/tests/e2e/validation/pysweep/`, not inside this package. That keeps them in
the shared allure report and lets them reuse the suite's session fixtures
(`k8s_helper`, `keycloak_helper`, `fingerprint_api_helper`). It also avoids a
directory named `tests` shadowing the top-level `tests` package.

## Why these tests exist (DESIGN.md objective 6)

Every property has an e2e test that exercises `read` / `apply` / `is_satisfied`
against the **live deployment**. If the deployment's contract drifts (an endpoint
path, a payload shape, a `values` response key, an RBAC grant), the test
fails **here** — before a benchmark silently runs against the wrong
configuration. When the cluster or the API is unreachable the tests `skip`
(runtime `pytest.skip`, not a static marker), so they degrade gracefully in
environments without a live ChatQNA install.

## Sweep config (YAML)

A sweep is declared in YAML and validated by `config.load_config()`:

```yaml
llms:                                      # DEPLOYMENT knobs: optional, outer loop
  max_new_tokens: [128, 256, 512]          # a list -> one product dimension
  model: llama3-8b-awq                     # K8s-spec plane; a redeploy per value

retriever:
  k: 10                                    # documents fetched per query

reranker:
  top_n: 7                                 # of those, how many reach the prompt

client:                                    # CLIENT knobs: required, inner loop
  question_file: ../questions-short.csv    # relative to pysweep_bench.py, not the CWD
  concurrency: [1, 16, 128]                # a list -> the innermost dimension
  duration: 3m
  worker_startup_delay: 0s
  # no 'tokenizer': -m is derived from the served model (see below)
  failures_allowed: 200
  input_tokens: 128
  warmup_requests: 1
```

- A **single value** is fixed across the sweep; a **list** contributes one
  dimension to a cartesian `product` (see DESIGN.md).
- Each YAML leaf is **bound to its implementation on the field itself** — a
  `Property` class for deployment knobs
  (`Annotated[..., PropertyBinding(MaxNewTokens)]`, so `llms.max_new_tokens`
  statically resolves to `MaxNewTokens`) or a `benchmark.py` flag for client knobs
  (`Annotated[..., CONCURRENCY]`). No name-string registry to drift either way.
- **Every deployment knob is optional.** Omit a key to leave that knob untouched.
- **Every client knob is required** (except `server` / `bearer_file`) — see below.
- **Any deviation from the schema raises `ConfigError`**: an unknown key
  (`extra="forbid"`), a wrong type, an out-of-range value (`max_new_tokens` is a
  `PositiveInt`, `duration` must match `<number><s|m|h>`), or malformed YAML — all
  *before* a sweep starts.

```python
from tests.e2e.benchmarks.chatqna.pysweep.config import load_config

cfg = load_config("path/to/sweep.yaml")
for combination in cfg.deployment_combinations():  # deployment: apply + verify
  for setting in combination:  # setting.path, setting.property_cls, setting.value
    ...
  for client_combination in cfg.client_combinations():  # then load, per scenario
    for setting in client_combination:  # setting.path, setting.arg, setting.value
      ...
```

Adding a deployment knob = add one optional, `PropertyBinding`-annotated field to
the relevant section model in `config.py` (and ship the property's e2e test).
Adding a client knob = add a `ClientArg` for the new `benchmark.py` option and one
annotated field on `ClientConfig`.

### `client:` — the load generator's parameters

Every option of `benchmark.py` is mirrored in `client_args.py`, and each one that is
genuinely a choice appears as a `client:` knob. They behave like knobs where it
matters — each value is a distinct test scenario and they expand into a `product` —
but they are deliberately **not** `Property` instances and have **no `Plane`**:

| A `Property` does | A client knob |
|---|---|
| `apply()` mutates the deployment | Nothing to mutate; the value is an argument to a process about to start. |
| `wait_until_ready()` waits out a rollout / reconcile | Nothing to wait for. |
| `read()` returns **observed** state from the live system | Nothing to observe. A "read" could only echo the value pysweep already holds — exactly the spec-read the `Property` contract forbids, which would make the preflight tautological. |

Modelling them as a third `Plane` would have required a `Property` whose three
methods are all vacuous, quietly weakening the invariant that makes "what was
tested" trustworthy for the knobs that *do* have observable state. So they get
their own adapter instead, and still receive schema validation, product expansion,
result-file naming, and a recorded snapshot.

**They are required, unlike deployment knobs.** An omitted deployment knob means
"leave it alone" — a meaningful, observable state. An omitted client knob has no
such reading: the load test would run anyway on `benchmark.py`'s defaults (`-c 32`,
`-d 30m`, `-x 0`) and the result CSV records none of it. Requiring them keeps every
result attributable to a configuration someone wrote down, and makes the sweep
config the single source of truth — a client flag passed after `--` that the config
already declares is **rejected**, rather than silently overriding a sweep
dimension.

`server` (`-s`) and `bearer_file` (`-b`) are optional exceptions: pysweep
auto-provisions both (see "Running a sweep"), and they are single-valued, because
one endpoint serves the whole sweep.

**There is no `tokenizer` key, though `benchmark.py` has `-m`.** It is not a knob:
[the benchmark README](../README.md) requires it to be the eRAG LLM model, because
it builds the tokenizer that produces the `input_tokens` / `output_tokens` columns
and pads questions to `client.input_tokens`. Those counts only describe what the
deployment tokenized if they come from that model's own tokenizer — any other one
still yields numbers, just on a basis nobody chose, and tokens/s then compares
against nothing. So the value is fully determined by the model being served, and
the only thing declaring it could express is a mismatch.

pysweep fills it in instead: after each deployment combination converges,
`resolve_tokenizer()` takes the **observed** model — from the combination's snapshot
when `llms.model` is swept, otherwise `LlmModel.read()` against the running
pipeline pods — and maps it to a HuggingFace repo id through the model catalog
(`model_catalog.tokenizer_for()`), because the two names differ: the pipeline
serves the catalog name `llama3-8b-awq`, while `AutoTokenizer.from_pretrained`
needs `casperhansen/llama-3-8b-instruct-awq`. The result is appended as `-m` for
every load test in that combination's inner loop, resolved once per combination
since the deployment is held still throughout it. Declaring `client.tokenizer` is
an unknown-key error, and `-m` after `--` is refused. If the served model cannot be
read, or cannot be mapped to a repo id, the sweep aborts rather than falling back
to `benchmark.py`'s default (an embedding model's tokenizer).

This also removes the one constraint a config author had to maintain by hand: a
sweep may now vary `llms.model` across models with *different* tokenizers, and each
combination is counted with its own.

### `llms.model` — the model the pipeline generates with

`LlmModel` is the one knob whose value is owned by two systems at once, and the
property's job is to drive both and then observe the result:

| Who owns it | What it decides |
|---|---|
| `model-manager` (ns `llm-inference`) | Whether the model is **served** at all: the `LLMInferenceService` CR, weight download, CPU/NUMA sizing, engine args, gateway route. |
| The GMC operator (`gmc-config` in ns `system`) | Whether the **pipeline points at it**: `LLM_MODEL_NAME` in the rendered `llm-usvc-config`, which `llm-usvc` sends as `model=` on every request. |

**Values are catalog names, not repo ids** — `llama3-8b-awq`, never
`casperhansen/llama-3-8b-instruct-awq`. The short name is what model-manager
passes as `--served-model-name`, so it is what the gateway routes on and what
`LLM_MODEL_NAME` must contain. The repo id is derived from it, for the benchmark's
tokenizer only (see the `client:` section above). A name that is not in the
catalog is rejected *before* anything is written, listing the names the deployment
does know.

**Three states, not two.** A model swap passes through *declared* (`gmc-config`)
→ *rendered* (`llm-usvc-config`, written by the operator) → *observed* (the env of
the running `llm-usvc` processes). `read()` returns the last of these — it execs
into every live replica for its effective `LLM_MODEL_NAME` and additionally
requires that model to be a **Ready** `LLMInferenceService`. Both halves matter: a
pipeline pointed at an unserved model answers nothing, and a served but
unreferenced model changes no benchmark. Replicas that disagree mean a rollout is
in flight and raise `PropertyReadError`; `wait_until_ready` is what tolerates that
while it is transient. `declared()` exposes the `gmc-config` side separately,
which is what keeps `apply()` idempotent when it is re-entered mid-swap.

**`apply()` has a mandatory order.** The model must be answering on the gateway
*before* the pipeline is repointed, because the replacement `llm-usvc` pods' init
container refuses to become Ready until it is:

1. `model-manager deploy <name> --wait` — serves the model, carrying the outgoing
   model's replica count over so both models are benchmarked at the same scale.
2. Rewrite `LLM_MODEL_NAME` in `gmc-config`'s `llm-usvc.yaml` (a line-precise text
   edit, not a YAML round-trip). Never `llm-usvc-config` directly: the operator
   re-applies its rendered output on every reconcile, so a direct edit is reverted
   — and the `gmc-config` write is itself the trigger that makes it reconcile.
3. Wait for the operator to render it (`RECONCILE_TIMEOUT`, 300s — if it has not
   by then, `gmc-controller` is not running and waiting longer will not help),
   then **delete the `llm-usvc` pods**.

Step 3's deletion is not a workaround for slowness. The pod template consumes the
ConfigMap through `envFrom` and carries **no checksum annotation**, so the
operator's ConfigMap update leaves the template byte-identical and Kubernetes
starts no rollout at all; `envFrom` is resolved once at container start, so the
running containers would keep the old model indefinitely. The replacements' init
container then gates readiness on the new model actually answering, which makes
pod readiness a real signal here.

**Why delegate to `model-manager` rather than patch the CRs.** It is the
deployment team's own actively-developed CLI and the tool the cluster was built
with, so it already encodes the coupled per-model decisions (runtime, engine
version, CPU/memory, NUMA and balloon placement, gateway wiring) that a
name-substitution patch would get wrong. Its `deploy` is idempotent
(`kubectl diff --server-side`), and `--dry-run` is side-effect free, which is what
lets the e2e test verify the whole path without changing the deployment.
`properties/sub/model_manager.py` is the only module that knows where the
executable lives (`PYSWEEP_MODEL_MANAGER`, then the deployment's wrapper, then
`$PATH`) and how its arguments are spelled.

**It is slow, so it is the outermost loop.** `wait_until_ready` polls the observed
state with a **3h** timeout (`CONVERGE_TIMEOUT`; the inherited 300s default would
expire during a weight download) and tolerates the normal mid-swap states —
replicas disagreeing, a replica unreachable, the old model still being served.
`deployment_combinations()` sorts K8s-plane knobs first, so every `max_new_tokens` and `k`
value runs per model and the model changes as few times as possible.

## Iteration order

The two products are **nested**, not merged into one:

```
for each deployment combination:      # outer — costs a rollout
    apply / wait / verify  (once)
    tokenizer = observed served model  # -m, derived once per combination
    for each client combination:      # inner — costs one more load test
        run benchmark.py
        append a row to benchmark_summary_results.csv  # knobs + TTFT/TPOT KPIs
```

Within the deployment product, `deployment_combinations()` sorts K8s-plane knobs first so
they vary slowest (`itertools.product` varies its leftmost dimension slowest);
runtime-API knobs vary faster. Client knobs are innermost by construction — the
nesting puts them there, so the ordering is structural rather than something a
sort key has to keep getting right.

The cost gradient is the reason: an `llms.model` change is a sequential
multi-replica rollout with a weight download (budget hours), `max_new_tokens` is
one API call, and a concurrency change is free. So each deployment configuration
is established and verified **once**, then held still while every load scenario
runs against it — which is also what makes those results comparable to each other.
The example config's `2 max_new_tokens x 3 concurrency` is 6 load tests behind
2 applies, not 6.

## Running a sweep

`pysweep_bench.py` runs the ChatQNA benchmark once per (deployment, client)
combination pair. For each deployment combination it applies the declared
properties, waits for convergence, verifies the **observed** state, records the
snapshot of what was tested, then reuses `benchmark.py` unchanged to run one load
test per client combination.

There are two separate configs:

| Flag | File | Declares |
|---|---|---|
| `--sweep` (required) | `examples/sweep.yaml` | **What to vary** — the knobs and their values; expands into combinations. |
| `--config` (optional) | `examples/config.yaml` | **How to run** — harness behaviour; never expands into combinations. |

`--summaries` (optional, default `./results`) is the **directory** a sweep writes
into — see "Where the results go" below. It is cleared of the previous sweep's
artifacts before this one starts.

**No preparation steps are required.** A sweep needs only `--sweep` — the load
parameters are declared in it, so nothing after `--` is necessary:

```bash
PYTHONPATH=src python -m tests.e2e.benchmarks.chatqna.pysweep.pysweep_bench \
    --sweep src/tests/e2e/benchmarks/chatqna/pysweep/examples/sweep.yaml \
    --config src/tests/e2e/benchmarks/chatqna/pysweep/examples/config.yaml
```

Run it from anywhere — the repo root above, or any other directory. Paths *inside*
a sweep config are resolved against `pysweep_bench.py`'s directory rather than the
working directory, so a config means the same thing regardless of where it is
launched from. `--sweep` / `--config` / `--summaries` are ordinary command-line
paths and behave normally (relative to the CWD).

Arguments after `--` are still forwarded to `benchmark.py`, but a flag the sweep
config already declares raises a `SweepError` before anything is applied, and `-m`
is refused outright (it is derived from the served model). `-s` / `-b` are the
usable case: passing them bypasses auto-provisioning when the config leaves them
out.

pysweep provisions the things the raw `benchmark.py` makes you prepare by hand:

| Previously manual | Now |
|---|---|
| Look up the ChatQNA URL and pass `-s` | With no `-s`, targets `https://<base_domain_name>/api/v1/chatqna`, the domain read from the installed environment's build config. |
| Generate a UAT and pass `-b uat.txt`, or every worker logs `no bearer, quitting` | With no `-b`, mints one distinct Keycloak user token per worker (for the widest concurrency in the sweep) into a temporary bearer file. Tokens are minted once per sweep and not refreshed, so a sweep longer than the realm's access-token lifetime (5 min on the dev deployment) fails its later runs with 401s. |
| Results of each combination overwrote one another (same timestamped CSV) | Each run writes `per_case_results/bench_<knob>-<value>_..._<clientknob>-<value>.result.csv`. |
| An unreachable endpoint still exited `0` with an empty CSV | A run that records **zero** result rows is reported as a failure (non-zero exit). |
| Load parameters lived only in whatever shell command produced the CSV | Declared in `--sweep` and logged per run alongside the observed deployment state. |
| Comparing runs meant `parse.py` over a directory of headerless CSVs, with the configuration recoverable only from filenames | One `benchmark_summary_results.csv`: a row per run, a column per declared knob (dotted paths), TTFT/TPOT mean and p90. |
| `-m` had to be kept in step with the served model by hand, and defaulted to an embedding model's tokenizer if forgotten | Derived per combination from the model the LLM engine is observed to serve, so token counts always come from that model's tokenizer. |

Supplying `-s` / `-b` (on the command line or as `client.server` /
`client.bearer_file`) disables the matching auto-provisioning.

Result filenames carry the knobs that **vary** in this sweep, not every declared
knob: naming them all can overflow the 255-byte filename limit once an HF repo id
and a question-file path are in there. The full configuration of each run — the
observed deployment snapshot, the resolved tokenizer, and the load parameters — is
logged instead.

### Where the results go (`--summaries`)

Everything a sweep writes goes under one directory, `--summaries` (default
`./results`, relative to the CWD like any other command-line path):

```
results/
  benchmark_summary_results.csv    # the sweep's table: one row per run
  per_case_results/                # one per-request CSV per run
    bench_max_new_tokens-128_concurrency-1_input_tokens-128.result.csv
    bench_max_new_tokens-128_concurrency-16_input_tokens-128.result.csv
    ...
```

The directory is created if missing (parents included), and **the previous
sweep's artifacts in it are deleted before this one starts**: the summary file is
removed and `per_case_results/` is emptied. A sweep's outputs only mean anything
as a set — the summary indexes the per-case CSVs, and a per-case CSV means nothing
without the row saying what produced it — so merging two sweeps into one directory
would leave stale CSVs that read exactly like results of the current run. Nothing
else under the directory is touched, so pointing `--summaries` at a directory that
also holds a copy of the sweep YAML or someone's plots loses none of it. The wipe
happens after the config is validated and the preflight has passed, so a typo in
the YAML or a model that is not in the catalog does not cost you the previous
results.

### The summary CSV

Each `bench_*.result.csv` is one row per *request* with no header and no record
of the configuration, so a sweep also writes a single summary table
(`benchmark_summary_results.csv`, `results.py`):

```
llms.max_new_tokens,llms.model,client.concurrency,client.duration,...,requests,mean_ttft,p90_ttft,mean_tpot,p90_tpot,served_model,error,result_file
128,llama3-8b-awq,1,10m,...,412,0.284192,0.41133,0.021975,0.02698,casperhansen/llama-3-8b-instruct-awq,0,per_case_results/bench_max_new_tokens-128_concurrency-1_input_tokens-128.result.csv
128,llama3-8b-awq,16,10m,...,3907,1.10442,1.83206,0.0344,0.04102,casperhansen/llama-3-8b-instruct-awq,0,per_case_results/bench_max_new_tokens-128_concurrency-16_input_tokens-128.result.csv
```

One row per run. The knob columns come first, named by the **dotted config path**
the sweep YAML used (`client.input_tokens` for `client: input_tokens:`), so a
column maps back to the config with no lookup table. *Every* declared knob gets a
column, not only the varying ones — a summary that dropped the constants would not
say what the numbers were measured under, and including them makes two sweeps'
summaries concatenable. Deployment values are the **observed** read-backs; client
values are the declared ones (there is nothing to observe — they are arguments to
a process). `server` / `bearer_file` are excluded as connection plumbing.

Then the KPIs, in seconds:

| Column | Meaning |
|---|---|
| `requests` | Successful requests the statistics are over. A p90 over four requests is a number, not a measurement — this is what licenses reading the rest. |
| `mean_ttft`, `p90_ttft` | **Time to first token**: `first_chunk` as `benchmark.py` records it — request issued to first streamed chunk. Prefill plus all queueing ahead of it, so this is the KPI that moves with concurrency. |
| `mean_tpot`, `p90_tpot` | **Time per output token**: `(end_to_end - ttft) / (output_tokens - 1)`, the mean inter-token gap during decode. |
| `served_model`, `error`, `result_file` | The **tokenizer repo id** the tokens were counted with (the repo id, where the `llms.model` column holds the catalog name it was resolved from), the run's exit code, and the pointer back to the per-request CSV — relative to the summary itself, so the whole directory can be copied off the benchmark host with its pointers intact. |

TPOT excludes the first token from **both** the time and the token count, because
its cost is TTFT. `parse.py`'s `next_token_lat` divides by `output_tokens`
instead, which charges decode for a token it did not generate — under a percent at
128 output tokens, material at 8. p90 is linearly interpolated (numpy/pandas'
default method), so it agrees with `parse.py`'s figures.

The file is rewritten after every run, so a sweep killed at hour three — or a
model rollout that fails midway — still leaves a complete, valid table of what it
got through. The per-request CSVs are untouched: anything the four KPIs do not
answer (distribution shape, drift within a run, the tail past p90) is still there,
and `result_file` is how you get to it.

### Run config (`--config`)

Validated by `config.load_run_config()`, with the same strictness as the sweep
config — an unknown key or wrong type raises `ConfigError`. Omitting `--config`
uses the defaults.

```yaml
display_kubernetes_client_logs: false   # default: false
```

- **`display_kubernetes_client_logs`** — whether the kr8s/httpx Kubernetes
  client's own output is shown. The port-forward carrying the load test produces
  two kinds of noise: one `101 Switching Protocols` line *per benchmark request*
  (a websocket per forwarded connection), and a multi-frame
  `ConnectionResetError` traceback on teardown (a race inside kr8s — the
  requests themselves still return 200). Both are hidden by default; genuine
  warnings and errors still surface. Set `true` to debug cluster connectivity.

`benchmark.py`'s own output (its per-worker log lines and parsed-args dump) is
always shown — it is the benchmark's primary output and pysweep does not filter
it.

A deployment combination whose observed state does not match the desired value
aborts with a `SweepError` rather than benchmarking the wrong configuration —
including all the client combinations that would have run under it. The
fingerprint transport is chosen by `PYSWEEP_FINGERPRINT_TRANSPORT` (see below).

### What still has to be true

These are environment facts, not steps to run:

- **A kubeconfig** the process can find (`~/.kube/config` or `KUBECONFIG`), with
  the RBAC in `REQUIRED_ACCESS`. Everything else is derived from the cluster.
- **`PYTHONPATH=src`** so the `tests.e2e...` absolute imports resolve.
- **The served model's tokenizer** must be loadable, since pysweep counts tokens
  with it (see above). Gated repos (e.g. `meta-llama/Llama-3.1-8B-Instruct`) need
  `HF_TOKEN` or a warm HF cache — so a sweep over a gated model needs the token even
  though pysweep itself never downloads weights.
- **A model catalog**, whenever a model name has to be resolved — which includes
  sweeps that do *not* declare `llms.model`, since the tokenizer is derived from
  the observed one. Found via `PYSWEEP_MODEL_CATALOG`, else `MM_CONFIG` (which the
  deployment's `model-manager` wrapper exports), else this repo's shipped
  `deployment/models.yaml`. Prefer the deployment's own file: the two drift, and
  only the former lists the names the cluster was built from.
- **The `model-manager` CLI**, only if the sweep declares `llms.model`. Found via
  `PYSWEEP_MODEL_MANAGER`, else the deployment's wrapper
  (`<deploy-root>/model-manager`, which is what sets `MM_CONFIG`), else the
  `model_manager/model-manager` script under the checked-out inference repo, else
  `$PATH`. It shells out to `kubectl`, so it needs the same kubeconfig.
- Running **off-cluster** additionally needs the `no_proxy` / kubeconfig
  `server:` handling described under "Environment gotchas".

### Preflight checks

Preconditions that the schema cannot express are checked by `preflight_check()`, which
raises `PreflightError` (a `SweepError`) naming the knob, the declared value, and
the path or catalog it was checked against. It runs **once**, before any property
is applied, any rollout triggered, or any port-forward opened: a fault that would
doom every run is then reported while stopping costs nothing. Without it, a
mistyped corpus path surfaces *after* a model swap has spent an hour rolling out
replicas.

Once is enough because every check is about the config and its environment, not
about a run: nothing the sweep does between two load tests can change the answer.
That also keeps the cost where it belongs — the model dry run shells out to
`kubectl`, and re-running it per load test would spend cluster round trips to
re-learn a fact that cannot have changed.

Each check is its own named function, called from `preflight_check()`, so what is being
asserted is named rather than inlined and a new precondition has one place to go.
In increasing cost order:

| Check | What it asserts |
|---|---|
| `question_file_verify()` | Every declared `client.question_file` (the knob is sweepable, so a config may name several) resolves to a readable file. A stat per corpus. |
| `model_verify()` | Per declared `llms.model`: the name is in the catalog (the schema can only check it is a non-empty string — the valid set is a fact about the environment's catalog file), then `model-manager deploy --dry-run` accepts it. Skipped entirely when the sweep declares no model. |

The dry run is the same code path the apply takes minus the side effects — it
renders the manifest and runs model-manager's own NUMA/balloons capacity and
node-fit preflight — so a model the cluster cannot host fails here rather than
after the first combination has already been benchmarked.

## Requirements

Use the validation suite's requirements — no separate set:

```
src/tests/e2e/validation/requirements.txt
```

Key pins pysweep relies on: `kr8s==0.20.15` (Kubernetes client + port-forward),
`pydantic==2.11.7` (config validation), `requests==2.33.1`, `pytest==9.0.3`,
`allure-pytest==2.15.3`.

## Running the tests

pysweep's tests run **as part of the e2e validation suite**, the normal way:

```bash
cd src/tests
tox -e e2e -- -k pysweep
```

`-k pysweep` selects just the pysweep tests; drop it to run the whole suite. The
shared conftest's two options both default to a co-located deployment
(`--build-config-dir`, default `../../../../env/local`, from which it reads
`config.yaml` — it also drives the `fingerprint.enabled` skip — and
`--credentials-file`), so on the deployment host no flags are needed. Elsewhere:

```bash
tox -e e2e -- -k pysweep \
  --build-config-dir=<path>/env/local \
  --credentials-file=<path>/default_credentials.txt
```

The unit tests — every `*_unit.py` module, which is the majority of them — need
neither a cluster nor a deployment config, and have their own environment:

```bash
cd src
tox -e pysweep_unit_tests
```

Without tox (venv with the requirements installed, from the repo root):

```bash
PYTHONPATH=src python -m pytest \
  --build-config-dir=<path>/env/local \
  --credentials-file=<path>/default_credentials.txt \
  src/tests/e2e/validation/pysweep
```

## Environment gotchas

Running against a remote cluster **from a local machine** (no SSH per run):

- **Kubeconfig / `KUBECONFIG`.** Point at a kubeconfig whose `server:` is the
  node's routable API endpoint, not `127.0.0.1`. On the benchmark cluster the
  node IP `10.211.187.170:6443` is directly routable and its serving cert SANs
  include that IP, so no SSH tunnel is needed — just rewrite the server URL.
- **`no_proxy` needs the ERAG domain too** when the corporate proxy is set, including
  on the cluster node itself: the load test and the Keycloak token requests go to
  `https://<base_domain_name>` and its `keycloak.` subdomain, and the proxy answers
  those with `504`. Add `solutions.ai,.solutions.ai` (your `base_domain_name`) to
  `no_proxy` / `NO_PROXY`.
- **`no_proxy` needs the literal IP.** The corporate proxy
  (`proxy-dmz.intel.com:912`) intercepts cluster traffic, and `httpx`/`kr8s` do
  **not** honour a CIDR in `no_proxy` (`10.211.0.0/16` is ignored). Add the
  literal node IP: `export no_proxy="$no_proxy,10.211.187.170"` (and
  `NO_PROXY` likewise). Without this you get `403`s from the proxy.
- **`PYTHONPATH=src`** so the absolute imports (`tests.e2e...`) resolve, matching
  the rest of the suite.
- **`ERAG_DOMAIN_NAME`** (default `erag.com`) is used by the token provider /
  ingress transport.

### Fingerprint API transport

The `fingerprint_client` fixture prefers the suite's ingress-based
`FingerprintApiHelper`. On deployments where the `system_fingerprint` API has
**no ingress route** (the benchmark cluster returns `404` on `erag.com`), it
automatically falls back to a **kr8s port-forward** to `fingerprint-svc:6012` in
the `fingerprint` namespace. Pod-locally the gateway's auth is bypassed, so the
port-forward path sends no token. If neither path reaches the API the test skips.
