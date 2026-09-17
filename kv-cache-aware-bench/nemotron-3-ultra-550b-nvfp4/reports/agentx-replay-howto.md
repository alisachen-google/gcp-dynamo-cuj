# Run the actual AgentX replay against DynoSim

The adapter is [`dynosim_aiperf_replay.py`](../../scripts/dynosim_aiperf_replay.py).
It runs aiperf's CLI, loader, session manager, trajectory sampler, dependency
barriers, branch lifecycle, and metrics exporter. Its in-process transport uses
the aggregated or disaggregated DynoSim engine. An accelerated asyncio clock advances
both aiperf and simulated serving; there is no live GPU endpoint.

The two reference cells are **6 × TP4 / 24 GB300 GPUs: RR96 and default KV192**.
The [hardware targets](../sim-results/agentx_calibration_targets.json) supply the
actual resolved configuration, seed, and benchmark ID. Retaining the benchmark
ID reproduces the original cache-bust markers as well as their token lengths.
Use separate artifact directories for each invocation.

## Environment and data

AIPerf already supplies the open-source replay implementation and dataset
loader. See its [AgentX tutorial](https://github.com/ai-dynamo/aiperf/blob/v0.12.0/docs/tutorials/agentx-mvp.md),
[Weka loader tutorial](https://github.com/ai-dynamo/aiperf/blob/v0.12.0/docs/tutorials/weka-trace.md),
and the [public dataset](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k).
The simulations below use that implementation directly; the adapter provides
the simulated serving transport, local dataset source, service harness, and
accelerated clock. It does not substitute a handwritten AgentX scheduler.

The current workspace already has everything staged under `/tmp/n3u-*`. From
`kv-cache-aware-bench`, rerun either cell with the command below. A fresh setup
needs Python 3.12, the pinned dependencies, the matching aiperf source checkout
(its component-test service harness), and the tokenizer/dataset files.

```bash
REPLAY_WORK=/tmp/n3u-replay
mkdir -p "$REPLAY_WORK"
git clone https://github.com/ai-dynamo/aiperf.git "$REPLAY_WORK/aiperf"
git -C "$REPLAY_WORK/aiperf" checkout be53bf2953d30e46c500e6a80fc1f8b6f84bc718
uv venv --python 3.12 "$REPLAY_WORK/venv"
uv pip install --python "$REPLAY_WORK/venv/bin/python" \
  -r scripts/requirements-agentx-replay.txt \
  "$REPLAY_WORK/aiperf/tests/aiperf_mock_server"

mkdir -p "$REPLAY_WORK/data/arrow" "$REPLAY_WORK/data/tokenizer"
gcloud storage cp \
  'gs://alisachen-models/hf-cache/datasets/semianalysisai___cc-traces-weka-062126-256k/default/0.0.0/8fecd2fc56694469f758f0afbbb6335ad3043740/*.arrow' \
  "$REPLAY_WORK/data/arrow/"
for name in tokenizer.json tokenizer_config.json special_tokens_map.json chat_template.jinja config.json; do
  gcloud storage cp \
    "gs://alisachen-models/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4/$name" \
    "$REPLAY_WORK/data/tokenizer/"
done
```

The two Arrow files contain **393 roots**. Their SHA-256 hashes are recorded in
each simulation's provenance. They are the processed cache for dataset snapshot
`8fecd2fc56694469f758f0afbbb6335ad3043740`; the public-dataset loader still performs
its normal reconstruction into **9,602 conversations / 68,266 turns**. Only its
download source is redirected to these local Arrow files.

The Weka loader creates an approximately 2.15 GB mmap dataset. Stage artifacts
on a filesystem with enough space. The selected reference runs use `/tmp`.

## Run

### Direct AIPerf against an existing real endpoint

No custom replay code is needed for a hardware run. With the pinned environment
and tokenizer above, set a reachable existing endpoint and run:

```bash
AGENTX_ENDPOINT=http://localhost:8000
AGENTX_MODEL=alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4
AGENTX_CLIENTS=192
/tmp/n3u-faithful-venv/bin/aiperf profile \
  --scenario inferencex-agentx-mvp \
  --url "$AGENTX_ENDPOINT" --endpoint-type chat \
  --model "$AGENTX_MODEL" \
  --tokenizer /tmp/n3u-faithful-data/tokenizer --tokenizer-trust-remote-code \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --num-dataset-entries 393 --max-context-length 262144 \
  --concurrency "$AGENTX_CLIENTS" --random-seed 42 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --system-idle-gap-cap-seconds 10 --cache-bust first_turn_prefix \
  --benchmark-duration 3600 --benchmark-grace-period 60 \
  --request-timeout-seconds 1200 \
  --streaming --use-server-token-count --extra-inputs ignore_eos:true \
  --workers-max 200 --record-processors 8 --profile-export-level raw \
  --goodput 'time_to_first_token:5000 inter_token_latency:10' \
  --slice-duration 1.0 --ui simple \
  --artifact-dir "/tmp/agentx-real-c${AGENTX_CLIENTS}-rerun"
```

Set `AGENTX_CLIENTS=384` for that point. The endpoint's deployment selects agg
or disagg and KV or RR; AIPerf uses the same client replay in each case. This
command sends real inference traffic. The simulation command below stays local.
The upstream tutorial uses different example sampling bounds and duration;
the explicit values here preserve this study's actual hardware configuration.
A new ordinary AIPerf invocation creates a new benchmark ID/cache-bust literal;
the simulation adapter pins the saved ID when auditing an existing run.

### AIPerf with the simulated serving transport

These commands use the environment staged in this workspace. Set `POINT` to
`agg6-kv-c192` for the other cell. The default profiling duration is 3,600
simulated seconds, after the actual trajectory warmup, with 60 seconds of grace.

```bash
POINT=agg6-rr-c96
/tmp/n3u-faithful-venv/bin/python scripts/dynosim_aiperf_replay.py \
  --aiperf-source /tmp/n3u-aiperf-source \
  --arrow-dir /tmp/n3u-faithful-data/arrow \
  --tokenizer /tmp/n3u-faithful-data/tokenizer \
  --targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_calibration_targets.json \
  --point "$POINT" \
  --artifact-dir "/tmp/n3u-faithful-runs/$POINT-rerun"
```

For the fresh environment above, substitute the corresponding paths under
`$REPLAY_WORK`. A short harness smoke test can add `--duration 30`; such a run is
explicitly marked unsafe by aiperf and is not a performance result.

Outputs include aiperf's normal `profile_export_aiperf.json`, per-request
`profile_export.jsonl`, `simulation-dispatch.jsonl`, source/data fingerprints,
the resolved replay configuration, and simulation statistics. Throughput and
latency comparisons use the same aiperf summary statistics as hardware.

The tokenizer cache reuses exact message segments at explicit special-token
boundaries. It verifies full token arrays for the first 100 requests and every
100th request thereafter. This saves CPU time spent encoding repeated history;
it does not alter the simulated serving model.

## Audit

The original hardware logs, request records, and summaries are staged in
`/tmp/n3u-faithful-data` as `real-rr96.*` / `real-kv192.*` and
`real-rr96-summary.json` / `real-kv192-summary.json`. Their GCS summary URIs are
in the targets file; `profile_export.jsonl` and `logs/aiperf.log` are siblings.

```bash
/tmp/n3u-faithful-venv/bin/python scripts/audit_aiperf_replay.py \
  --targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_calibration_targets.json \
  --real-dir /tmp/n3u-faithful-data \
  --run-root /tmp/n3u-faithful-runs \
  --run-suffix=-verified \
  --output nemotron-3-ultra-550b-nvfp4/sim-results/agentx_aiperf_replay_20260917/audit.json
```

The audit asserts exact initial lane snapshots, dependency fingerprints,
warmup source/turn selection, and warmup input/output counts. It also reports
token-count agreement for matching requests from the initially warmed trees.
It treats the number and mix of completed requests as performance outcomes:
AgentX is closed loop, so changing service latency changes the workload reached
within the measurement window.

## RR sweep: agg and disagg

The saved [RR targets](../sim-results/agentx_rr_sweep_targets.json) contain agg6
at 48/96/192/384 clients and disagg12:6 at 48/96/192/384/480/768. Agg targets use
their own hardware configurations and benchmark IDs. Disagg targets use the
actual disagg KV192 **client configuration**, changing concurrency and the
simulated routing policy. They have no RR hardware performance reference.

The GitHub [resume recipe](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/96a019d8922002de91f4f9acc68005c7aac4ea21/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/resume_126_after_capacity.sh#L18)
specifies `rr:192 rr:96 rr:384`, through `agentx_runner.sh` and the
[12:6 manifest](../../sglang/manifests/n3u-mnnvl-126.yaml). These three
concurrencies are marked `recipe_point: true`; the others extend the curve.
Do not execute the hardware resume script to run a local simulation: it
deploys and modifies cluster workloads.

From `kv-cache-aware-bench`, run all ten points locally:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/sweep_aiperf_replay.py \
  --aiperf-source /tmp/n3u-aiperf-source \
  --arrow-dir /tmp/n3u-faithful-data/arrow \
  --tokenizer /tmp/n3u-faithful-data/tokenizer \
  --targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_rr_sweep_targets.json \
  --run-root /tmp/n3u-rr-sweep-rerun \
  --jobs 4
```

To run only the disagg recipe points, append
`--points disagg12-6-rr-c192,disagg12-6-rr-c96,disagg12-6-rr-c384`.
Use a new run directory. Parallel processes have separate event loops, caches,
and aiperf services; each point receives the full profiling window.

The disagg engine explicitly uses independent round-robin counters on the
prefill and decode tiers. The historical engine default remains
`least_inflight` for decode; passing `policy=rr` alone to the legacy engine
does **not** select RR on both tiers. The replay targets set
`serving.decode_routing=round_robin` to reproduce the recipe's routing mode.

Audit and archive a completed sweep:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/collect_aiperf_sweep.py \
  --targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_rr_sweep_targets.json \
  --run-root /tmp/n3u-rr-sweep-rerun \
  --real-dir /tmp/n3u-rr-sweep-real \
  --output /tmp/n3u-rr-sweep-rerun-archive
```

The staged real-data directory has one subdirectory per agg point with
`summary.json`, `records.jsonl`, and `aiperf.log`; fetch these from each target's
`hardware_comparison_source` directory. It also contains
`disagg12-6-kv-c192/` from the disagg `replay_config_source`. That KV source is
used only to validate the C192 replay inputs, never as a real RR throughput
measurement. `--run-root` may be repeated if the sweep was split into groups.

The collector verifies the 3,600-second send window, trajectory warmup,
dependency graph, tokenizer checks, request timeouts, worker IDs, and balanced
RR assignment counts on both tiers, including warmup and cancelled requests.
For agg, it also runs the cross-hardware source/turn and token-count audit.

## 64-GPU topology sweep

The [64-GPU study](agentx-64gpu-topology.md) holds TP4 fixed and searches P/D allocation and concurrency with both policies. The [saved targets](../sim-results/agentx_64gpu_targets.json) record every selected point and the real disagg client-config source. Each has `hardware_comparison_source: null`.

Generate the initial ladder from the existing disagg replay target, then execute locally:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/make_aiperf_topology_targets.py \
  --source-targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_rr_sweep_targets.json \
  --splits 4:12,8:8,12:4 --clients 16,48,96,192,384,768 \
  --output /tmp/agentx-64gpu-targets.json

/tmp/n3u-faithful-venv/bin/python scripts/sweep_aiperf_replay.py \
  --aiperf-source /tmp/n3u-aiperf-source \
  --arrow-dir /tmp/n3u-faithful-data/arrow \
  --tokenizer /tmp/n3u-faithful-data/tokenizer \
  --targets /tmp/agentx-64gpu-targets.json \
  --cache-capacity-tokens 51801984 --jobs 4 \
  --run-root /tmp/n3u-64gpu-rerun
```

To reproduce the final adaptive set, use the saved repository targets instead of generating the initial ladder. The cache value is an explicit proxy from disagg model metadata, not a measured prefill/Mamba cache capacity. KV targets set `decode_routing=active_blocks`; RR targets set `round_robin`.

Audit and draw the curves:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/collect_aiperf_sweep.py \
  --targets /tmp/agentx-64gpu-targets.json \
  --run-root /tmp/n3u-64gpu-rerun \
  --real-dir /tmp/n3u-rr-sweep-real \
  --output /tmp/n3u-64gpu-audit --compact

/tmp/n3u-faithful-venv/bin/python scripts/plot_aiperf_topologies.py \
  --audit /tmp/n3u-64gpu-audit/audit.json \
  --output /tmp/n3u-64gpu-curves
```

`--compact` retains summaries, configuration, source provenance, and full-record hashes after auditing the raw records locally. Omit it to archive compressed per-request records too. The actual disagg KV192 source files are needed only for C192 input-replay checks; no 72-GPU performance is compared with 64-GPU predictions. Parallel runs can consume several GB to tens of GB each; choose `--jobs` for available memory and use a separate artifact directory for every run.

## What disagg correctness requires

Replay fidelity and serving-model fidelity need separate evidence:

| Layer | Required reference | Current status |
|---|---|---|
| Workload | aiperf 0.12.0, same Arrow snapshot and chat tokenizer, seed 42, 393 roots, full branches and joins, sampled starts, trajectory warmup, recycling | Actual aiperf code and source/token audits |
| Timing contract | 3,600 s profiling, 60 s grace, 1,200 s timeout, 10 s global idle cap; concurrency counts live trees | Actual aiperf code and exported phase checks |
| Resources | 12 P + 6 D workers, TP4/EP4; normalize over all 72 GPUs | Worker counts and normalization matched; EP communication cost remains implicit in fitted rates |
| Routing | Frontend RR on both P and D; separate candidate sets/counters and stable-worker ordering | Both-tier RR implemented; live eligibility, overload rejection, and discovery-order effects are not modeled |
| Prefill scheduling | 16,384-token chunks; max-running-requests 8 per P worker | Not implemented: fixed-rate serial FCFS approximation |
| Decode scheduling | Max-running-requests 64 per D worker; admission, memory pressure, batching as requests enter/leave | Not implemented: static TPOT based on admitted request count, including requests still in prefill |
| KV transfer | Mooncake over MNNVL; bootstrap/handoff, bytes, buffers, bandwidth contention, cancellation | Not implemented: zero transfer time |
| Cache | 64-token pages plus hybrid/Mamba reuse rules, finite capacity, actual insertion/eviction events | Idealized large prefix cache; insert at admission; hybrid checkpoints not modeled |
| Software | Recipe requests Dynamo 1.4.2, SGLang image `v0.5.19-cu130-runtime`, FlashInfer 0.6.18 | Recipe captured; actual installed package versions and image digest should be checked in future hardware artifacts |
| Initialization | Actual completed warmups and cache lifetime across points | Trajectory warmup reproduced; no invented 900 s prewarm; prior persistent GPU cache unknown |

Dynamo's [prefill activation](https://github.com/ai-dynamo/dynamo/blob/2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a/lib/llm/src/kv_router/prefill_router/activation.rs#L261)
uses the frontend's mode for its simple prefill router. The
[backend routing builder](https://github.com/ai-dynamo/dynamo/blob/2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a/lib/llm/src/entrypoint/input/common.rs#L291)
also passes that mode to the decode-side router. The
[RR implementation](https://github.com/ai-dynamo/dynamo/blob/2ecbdfdf192c69c02c6d21e931d20d3b4a0bb64a/lib/runtime/src/pipeline/network/egress/push_router.rs#L693)
rotates over eligible workers, so a fixed worker set is an explicit simulation
assumption.

Before treating disagg predictions as calibrated performance, compare real
RR96 and RR384 (or the recipe's RR192 first) with the simulation using the same
client configuration. Check request/token identity, P/D assignment, cache hit
fraction, prefill queue/service time, transfer time, ITL distribution, and error
counts. Fit transfer and scheduler behavior using those separate measurements,
then validate a held-out concurrency. Matching total throughput alone does not
establish correctness.

## Scope

This establishes replay fidelity. The existing engine remains an approximation:
FCFS prefill at 19,700 uncached tokens/s per worker; agg TPOT `8.9 + 1.73 × admitted requests` ms or disagg TPOT `5.97 + 0.4 × admitted requests` ms;
64-token prefix blocks; an LRU cache with explicit optional capacity; and simplified KV routing. Cache
availability at admission, fixed per-request decode speed, hybrid/Mamba cache
behavior, networking, and real prefill/decode interference still require engine
calibration. The transport emits first and final streaming chunks, sufficient
for TTFT and per-request average ITL, not token-level jitter or chunk metrics.

The separate 900-second prewarm in the hardware job template failed before
running (duplicated flags); both selected artifact directories are empty and
their main runs began only 16–17 seconds after the job epoch. The successful
one-token AgentX trajectory warmup is reproduced. Persistent GPU cache state
left by earlier jobs is not captured.
