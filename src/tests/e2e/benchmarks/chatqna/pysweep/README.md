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
| `configs/sweep.yaml` | Sweep config (`--sweep`): what to vary. Keys are described in [`configs/README.md`](./configs/README.md). |
| `configs/general.yaml` | Run config (`--config`): how to run. |
| `authentication.py` | `KubernetesAccess` (side-effect-free RBAC preflight via `SelfSubjectAccessReview`), `load_build_config()` (the installed environment's config), `keycloak_helper()` (the maintained `KeycloakHelper`, for user tokens) and `client_token_lifespan()` (a sweep-long access-token lifespan for the UI client, restored afterwards). |
| `runtime_api.py` | `system_fingerprint` client. `HelperFingerprintClient` adapts the suite's `FingerprintApiHelper`; `PortForwardTransport` / `IngressTransport` cover deployments the helper can't reach. |
| `properties/max_tokens.py` | First property: `MaxNewTokens` (LLM `max_new_tokens`, Runtime-API plane). |
| `served_model.py` | `ServedModel`: reads the model the pipeline is **observed** to generate with, out of the running `llm-usvc` pods. Not a knob — the runner uses it to pick the benchmark's tokenizer. |
| `model_catalog.py` | Reader for the model-manager catalog (`models-rag.yaml` / `models.yaml`): maps a served model name to the HuggingFace repo id the benchmark's tokenizer needs. |
| `paths.py` | Where the installed environment's files are looked for (deploy root derived from the checkout's location). |

Only the modules directly under `properties/` define knobs. A file there means a
`Property` subclass bound to a YAML leaf in `config.py`.

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
`resolve_tokenizer()` takes the **observed** model — `ServedModel.read()` against
the running pipeline pods — and maps it to a HuggingFace repo id through the model catalog
(`model_catalog.tokenizer_for()`), because the two names differ: the pipeline
serves the catalog name `llama3-8b-awq`, while `AutoTokenizer.from_pretrained`
needs `casperhansen/llama-3-8b-instruct-awq`. The result is appended as `-m` for
every load test in that combination's inner loop, resolved once per combination
since the deployment is held still throughout it. Declaring `client.tokenizer` is
an unknown-key error, and `-m` after `--` is refused. If the served model cannot be
read, or cannot be mapped to a repo id, the sweep aborts rather than falling back
to `benchmark.py`'s default (an embedding model's tokenizer).

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

The cost gradient is the reason: a deployment knob reconfigures the running
pipeline (`max_new_tokens` is one API call; a K8s-plane knob is a rollout), while a
concurrency change is free. So each deployment configuration
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
| `--sweep` (required) | `configs/sweep.yaml` | **What to vary** — the knobs and their values; expands into combinations. |
| `--config` (optional) | `configs/general.yaml` | **How to run** — harness behaviour; never expands into combinations. |

`--summaries` (optional, default `./results`) is the **directory** a sweep writes
into — see "Where the results go" below. It is cleared of the previous sweep's
artifacts before this one starts.

**No preparation steps are required.** A sweep needs only `--sweep` — the load
parameters are declared in it, so nothing after `--` is necessary:

```bash
PYTHONPATH=src python -m tests.e2e.benchmarks.chatqna.pysweep.pysweep_bench \
    --sweep src/tests/e2e/benchmarks/chatqna/pysweep/configs/sweep.yaml \
    --config src/tests/e2e/benchmarks/chatqna/pysweep/configs/general.yaml
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
| Generate a UAT and pass `-b uat.txt`, or every worker logs `no bearer, quitting` | With no `-b`, mints one distinct Keycloak user token per worker (for the widest concurrency in the sweep) into a temporary bearer file. The tokens are re-minted before every run, and for the duration of the sweep the UI client's access-token lifespan is set to 1h (`SWEEP_TOKEN_LIFESPAN_S`, an arbitrary value that only has to outlast one run; the realm default of 5 min does not), then restored. A supplied `-b` is never re-minted, so its tokens must outlive the whole sweep. |
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
the YAML or a missing question file does not cost you the previous results.

### The summary CSV

Each `bench_*.result.csv` is one row per *request* with no header and no record
of the configuration, so a sweep also writes a single summary table
(`benchmark_summary_results.csv`, `results.py`):

```
llms.max_new_tokens,client.concurrency,client.duration,...,requests,mean_ttft,p90_ttft,mean_tpot,p90_tpot,served_model,error,result_file
128,1,10m,...,412,0.284192,0.41133,0.021975,0.02698,casperhansen/llama-3-8b-instruct-awq,0,per_case_results/bench_max_new_tokens-128_concurrency-1_input_tokens-128.result.csv
128,16,10m,...,3907,1.10442,1.83206,0.0344,0.04102,casperhansen/llama-3-8b-instruct-awq,0,per_case_results/bench_max_new_tokens-128_concurrency-16_input_tokens-128.result.csv
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
| `served_model`, `error`, `result_file` | The **tokenizer repo id** the tokens were counted with (resolved from the observed served model's catalog name), the run's exit code, and the pointer back to the per-request CSV — relative to the summary itself, so the whole directory can be copied off the benchmark host with its pointers intact. |

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
config. Omitting `--config` uses the defaults; the keys are listed in
[`configs/README.md`](./configs/README.md).

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
- **A model catalog**, to map the served model to its tokenizer. Found via
  `PYSWEEP_MODEL_CATALOG`, else `MM_CONFIG` (which the deployment's
  `model-manager` wrapper exports), else the installed environment's
  `env/local/models-rag.yaml` (the RAG pipeline's catalog), then
  `env/local/models.yaml`, then this repo's shipped `deployment/models.yaml`.
  Prefer the deployment's own file: they drift, and only it lists the names the
  cluster was built from.
- Running **off-cluster** additionally needs the `no_proxy` / kubeconfig
  `server:` handling described under "Environment gotchas".

### Preflight checks

Preconditions that the schema cannot express are checked by `preflight_check()`, which
raises `PreflightError` (a `SweepError`) naming the knob, the declared value, and
the path or catalog it was checked against. It runs **once**, before any property
is applied, any rollout triggered, or any port-forward opened: a fault that would
doom every run is then reported while stopping costs nothing. Without it, a
mistyped corpus path surfaces only after the first deployment combination has
been applied.

Once is enough because every check is about the config and its environment, not
about a run: nothing the sweep does between two load tests can change the answer.

Each check is its own named function, called from `preflight_check()`, so what is being
asserted is named rather than inlined and a new precondition has one place to go.
In increasing cost order:

| Check | What it asserts |
|---|---|
| `question_file_verify()` | Every declared `client.question_file` (the knob is sweepable, so a config may name several) resolves to a readable file. A stat per corpus. |

## Requirements

Use the validation suite's requirements — no separate set:

```
src/tests/e2e/validation/requirements.txt
```

Key pins pysweep relies on: `kr8s==0.20.15` (Kubernetes client + port-forward),
`pydantic==2.13.4` (config validation), `requests==2.33.1`, `pytest==9.0.3`,
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
`global_config.yaml`, `config.erag.yaml` and `config.inference.yaml` — they also
drive the `fingerprint.enabled` skip — and
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
