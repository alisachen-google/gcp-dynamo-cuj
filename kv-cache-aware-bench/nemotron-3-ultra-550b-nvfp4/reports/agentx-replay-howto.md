# Run the actual AgentX replay against DynoSim

The adapter is [`dynosim_aiperf_replay.py`](../../scripts/dynosim_aiperf_replay.py).
It runs aiperf's CLI, loader, session manager, trajectory sampler, dependency
barriers, branch lifecycle, and metrics exporter. Its in-process transport uses
the existing aggregated DynoSim engine. An accelerated asyncio clock advances
both aiperf and simulated serving; there is no live GPU endpoint.

The two reference cells are **6 × TP4 / 24 GB300 GPUs: RR96 and default KV192**.
The [hardware targets](../sim-results/agentx_calibration_targets.json) supply the
actual resolved configuration, seed, and benchmark ID. Retaining the benchmark
ID reproduces the original cache-bust markers as well as their token lengths.
Use separate artifact directories for each invocation.

## Environment and data

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

## Scope

This establishes replay fidelity. The existing engine remains an approximation:
FCFS prefill at 19,700 uncached tokens/s per worker; TPOT `8.9 + 1.73 × batch` ms;
64-token prefix blocks; a large LRU cache; and simplified KV routing. Cache
availability at admission, fixed per-request decode speed, hybrid/Mamba cache
behavior, networking, and real prefill/decode interference still require engine
calibration. The transport emits first and final streaming chunks, sufficient
for TTFT and per-request average ITL, not token-level jitter or chunk metrics.

The separate 900-second prewarm in the hardware job template failed before
running (duplicated flags); both selected artifact directories are empty and
their main runs began only 16–17 seconds after the job epoch. The successful
one-token AgentX trajectory warmup is reproduced. Persistent GPU cache state
left by earlier jobs is not captured.
