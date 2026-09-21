# Eight matched D88 native simulations

This study applies the agg replay and frozen V10 AIC timing methodology to eight
existing **64-GPU D88 hardware points**, using the native disaggregation extension.
It compares default KV with RR. It does not fit the timing model to D88 or sweep
router flags. See [the comparison report](../../reports/agentx-serving-perf-report.md#42-d88-native-forecasts-alongside-the-measured-kvrr-curves).

| Policy | Session concurrency | Reason for selection |
| --- | --- | --- |
| Default KV | 192, 480, 768, 1152 | Low-load reference, sampled TTFT-SLO choice, throughput peak, overload |
| RR | 72, 96, 192, 384 | Last sampled TTFT pass, first sampled failure, throughput peak, overload |

The point selection, hardware run IDs, benchmark IDs and frozen input hashes are
in [plan.json](plan.json). Each point starts new frontend and worker processes
with empty caches. The standard trajectory warmup then establishes cache state.

## Completed results

All eight one-hour profiles completed on September 21, 2026, with 136,044
successful profiling requests and 128 errors. All 3,084 warmup requests succeeded
and matched their hardware inputs. The errors occur at RR C384; its failure
durations differ from hardware and remain an explicit model limitation.

| Policy | Mean absolute throughput error | Mean absolute TTFT p95 error | Mean absolute I90 error | TTFT / combined-SLO agreement |
| --- | ---: | ---: | ---: | ---: |
| Default KV | 3.6% | 16.5% | 14.1% | 4/4 |
| RR | 1.2% | 1.6% | 3.1% | 3/4 |

Every throughput prediction is within 8.8% of hardware. RR C72 changes from a
hardware TTFT pass at 9.91 seconds to a native failure at 10.31 seconds. KV C768
has a larger latency gap: native TTFT is 39.0% higher and I90 37.4% lower.
The timing model was not refit after observing these results. The report contains
all eight paired rows, curves, sampled peaks, SLO brackets and a summary CSV.

## Method and configuration

- AIPerf 0.12.0, `inferencex-agentx-mvp`, 393-root Weka 256k corpus, seed 42.
- Same hardware benchmark ID, tokenizer and replay configuration per pair;
  trajectory start ratio 0.25–0.75, recorded delays, first-turn-prefix cache bust.
- 3,600-second profile, 60-second drain, 1,200-second request timeout.
- Actual Dynamo 1.4.2 KV/RR routing, native SGLang scheduler and finite hybrid cache.
- Eight TP4/EP4 prefill workers and eight TP4/EP4 decode workers: 64 modeled GPUs.
- P running limit 8; D running limit 64. KV, recurrent-state and token/chunk budgets
  are unchanged from the prior native D88 configuration.
- AIC 0.11.0 GB300 / SGLang 0.5.14 timing tables and the frozen agg V10 coefficients.
  Both speedup ratios remain 1.0; replay runs on the normal wall clock.
- Default KV: temperature 0, prefill load scale 1, overlap credit 1, decay 0, FCFS.
  RR uses native round-robin routing.
- Numeric per-request JSONL exports remain enabled. Optional duplicate raw request
  and response payload export is disabled to bound disk usage. This changes saved
  artifacts, not request construction, recorded delays or responses.

Exact [prefill](configs/prefill-engine.json), [decode](configs/decode-engine.json),
[topology](configs/p8d8.json) and [calibration](configs/matrix_v10_throughput.json)
inputs are retained. The copied topology file contains an older prose note about
pending validation; its structured worker counts and linked engine files define
the launch. Section 4.2 of the report supplies the new validation results.

## Handoff queue correction

The original native disaggregation build used `max_num_seqs` for both scheduled
batch size and the transport's handoff-session capacity. A burst could therefore
be rejected before the scheduler had a chance to queue it. This explained the
earlier `mocker handoff session limit reached` failures.

[The patch](patches/d88-handoff-queue.patch) adds optional
`handoff_max_sessions`. Its default preserves previous behavior. These eight
runs set it to **4096 per worker**, a bounded protocol queue separate from engine
admission. This is an explicit modeling choice, not a measured production limit.
It does not increase the P8/D64 running limits, cache capacities, Mamba state
slots or forward-pass batch size. The original 300-second handoff timeout remains.

| Native binary | `_core.abi3.so` SHA-256 |
| --- | --- |
| V11 disaggregation base, frozen V10 timing | `39e41354104490981f35f76850bd17973369fc8222dab9f7813e25f50146e36e` |
| This study: base plus bounded queue correction | `cf4b4e65c22c993934ad45642dc21e89d110c88148ec2ee7c78d0f67f499d1c6` |

This is a local patched native build, not an NVIDIA release named V10 or V11.
Earlier extension patches are preserved in
[native_disagg_flags/patches](../../scripts/native_disagg_flags/patches/).

Before the sweep, three configuration regression tests, 81 SGLang scheduler tests,
14 handoff tests and `cargo clippy -p dynamo-mocker --lib -- -D warnings` passed.
The modified Rust file also passed rustfmt. Unrelated formatting differences in
the frozen base were left intact. Build and test logs are in [patches](patches/).

The [live burst check](preflight.json) submitted 32 requests to one P and one D,
each limited to two running requests. The old binary completed 2/32; the queued
binary completed 32/32. Half requested one output token and half 32. The queued
run's maximum decode batch stayed at two, it executed exactly 496 decode tokens,
and the decode worker performed no second prefill. This verifies protocol and
admission behavior; its latency is not a calibration point.

## Run and regenerate

The local prepared environment has the preserved Dynamo/AIC build, AIPerf client,
tokenizer, offline corpus cache, NATS and etcd. This workflow provisions no GPUs.
The controller checks native/configuration hashes before launching and records
every exact command in its `jobs.json` and per-run provenance.

From the `nemotron-3-ultra-550b-nvfp4` directory:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/native_disagg_flags/run_d88_matched.py \
  --study "$PWD/sim-results/agentx_d88_matched_20260920" \
  --server-python /tmp/n3u-path1-d88-queue-venv/bin/python \
  --max-parallel 8
```

The controller resumes its recorded run directories without overwriting them.
Its memory, disk and CPU gates limit actual parallelism; the long warmups start
first. Warmup is paced by the replay schedule and is additional to the one-hour
profile. A failed warmup is retained as a failure, not a performance point.
The memory gate defaults to 70 GiB and can be set with
`--minimum-free-memory-gib`; this campaign used 65 GiB for its final, smaller
C72 launch after measuring the remaining clients' memory footprint.

After all eight runs complete:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/import_d88_matched.py
/tmp/n3u-faithful-venv/bin/python scripts/gen_agentx_serving_report.py
```

The importer checks original hardware-export hashes and exact warmup
source/turn/input-token identities. It preserves numeric request projections,
summaries, input configurations, native log evidence and their hashes in
[the matched data directory](../../reports/agentx-native-d88-matched-data/).
The generator recomputes request-level TTFT and E2E percentiles and checks request
counts, error counts, replay controls, native binary and frozen calibration.

## Interpretation limits

All eight are holdouts of the agg timing fit, with one run per point. Identical
warmup inputs and replay controls do not imply identical completed profiling
requests: this is a closed-loop, time-limited replay. The report retains request
counts, context lengths, cache reuse and errors beside the performance deltas.

TTFT uses **p95 <10 seconds**. The second criterion is
**I90 = 1 / P90(E2E seconds / output tokens) ≥20 output tokens/s/user**, calculated
per successful profiling request before aggregation. Throughput agreement alone
does not validate either SLO. Errored requests are disclosed; failed warmups have
no plotted throughput or latency point.

Prefill cache sizing is still inherited from agg, Mamba state allocation remains
partly assumed, transfer bandwidth is 64 GB/s per rank, shared-link contention is
not modeled, and the AIC SGLang table version differs from the hardware framework.
The new paired measurements can expose discrepancies but cannot identify which
assumption caused them without further instrumentation. No tuning winner or
optimal P:D ratio is established by this eight-point sweep.
