<!-- Copyright (C) 2026 Intel Corporation -->
<!-- SPDX-License-Identifier: Apache-2.0 -->

# pysweep configs

| File | Flag | Declares |
|---|---|---|
| `sweep.yaml` | `--sweep` (required) | What to vary. |
| `general.yaml` | `--config` (optional) | How the sweep runs. Never expanded into combinations. |

Both are validated before the sweep starts: an unknown key, a wrong type or an
out-of-range value is a `ConfigError`.

## `sweep.yaml`

**`pre_bench`** prepares what every run shares, once, before the first run.
Single values only; it is not part of the product. Each value is recorded in
every run as `settings.<path>` (e.g. `settings.edp.vector_db.num_vectors`).

| Key | Meaning |
|---|---|
| `pre_bench.edp.vector_db.num_vectors` | Load the vector DB through EDP until it holds at least this many vectors: first the documents in `../documents.csv`, then Simple English Wikipedia, split by `../wikiparse.py`, two files at a time. Files already in EDP are not uploaded again; downloads are cached in `$PYSWEEP_SEED_CACHE` (default `~/.cache/pysweep/seed`). Fails if the data runs out first. It never removes vectors: use `../prepare_cleanup_vectors.sh` to start from an empty DB. |

Any other knob takes a single value (fixed for the whole sweep) or a list (one
dimension of the product). Runs = deployment combinations × client combinations;
the file here is 2 × (3 × 2) = 12 runs.

**Deployment knobs** change the running RAG. Each is optional: omit it to leave
the deployment as it is. They are the outer loop, applied and verified once per
combination.

| Key | Meaning |
|---|---|
| `llms.max_new_tokens` | Tokens the LLM may generate per request. |

**Client knobs** are `benchmark.py`'s options. All are required, so no run
silently falls back to the tool's defaults. They are the inner loop: each value
is one more load test, and nothing on the cluster changes.

| Key | Flag | Meaning |
|---|---|---|
| `question_file` | `-f` | Question corpus. A relative path is resolved against `pysweep_bench.py`'s directory, not the working directory. Checked before anything is applied. |
| `concurrency` | `-c` | Concurrent users. |
| `duration` | `-d` | Length of each run (`<number><s\|m\|h>`). |
| `worker_startup_delay` | `-u` | Stagger between worker starts; `0s` starts them all at once. |
| `failures_allowed` | `-e` | Failed requests tolerated before the run gives up. |
| `input_tokens` | `-x` | Pad/trim every question to this many tokens; `0` leaves them as written. |
| `warmup_requests` | `-w` | Requests per worker discarded as warmup. |
| `server` | `-s` | Optional. Default: the deployment's own ingress, `https://<base_domain_name>/api/v1/chatqna`. |
| `bearer_file` | `-b` | Optional. Default: tokens minted by pysweep and re-minted before every run. A supplied file is never refreshed. |

There is no `tokenizer` key: `-m` is derived from the model the deployment
serves, and declaring it is an error.

Every run adds a record to `results.json` and a row to
`benchmark_summary_results.csv` in `--summaries` (default `./results`); the
previous sweep's results there are deleted first.

## `general.yaml`

| Key | Default | Meaning |
|---|---|---|
| `display_kubernetes_client_logs` | `false` | Show the Kubernetes client's per-request and port-forward teardown log noise; for debugging connectivity. |
