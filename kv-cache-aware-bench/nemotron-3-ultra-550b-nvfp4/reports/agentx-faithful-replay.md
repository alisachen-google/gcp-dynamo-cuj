# AgentX replay fidelity and executed agg calibration

Completed 2026-09-17. Study base commit:
`96a019d8922002de91f4f9acc68005c7aac4ea21`.

**The previous simulation did not replay the real AgentX workload faithfully.**
I ran both selected agg cells through aiperf 0.12.0 itself, with simulated
serving and an accelerated clock. Both completed a full 3,600-second profiling
window after the actual trajectory warmup. The current DynoSim engine constants
were held fixed; no engine parameters were fitted to these results.

The replay checks pass. This establishes agreement on the audited replay
inputs and rules, not identical hardware timing or a calibrated GPU model.
AgentX is closed loop: response latency and failures change when dependent
requests become eligible, which trees recycle, and what completes in an hour.

## What was wrong in the previous replay

The latest saved v5 results are disaggregated 12:6 KV, not an agg v5 sweep.
The published agg reference is the v3 CSV; later engine-substitution results
are separate calibration experiments. See the [historical comparison](agentx-replay-calibration.md).

| Replay component | Previous approximation | Actual aiperf behavior retained in the new run |
|---|---|---|
| Dataset construction | v3 uses a 4k-request slice; v5 reconstructs stream trees separately | Public Weka loader, including flattened chains, subagents, seams, and context synthesis |
| Initial root selection | Random draws with replacement in v5 | Sequential root sampler, seed 42; sampled timestamps between 25% and 75% |
| Dependencies and initial dispatch | Simplified parent/child joins and stream delays | Full interval barriers, future child spawning, joins, and sampled-boundary offsets |
| Warmup | Instantaneous cache priming plus 900 seconds of replay in v5 | Spread one-token trajectory priming, followed by a barrier before profiling |
| Recycling | v5b resamples a position; v5d starts at turn 0 but draws roots randomly | A drained whole tree releases its lane; the shared sampler supplies the next root at turn 0 |
| Prompt identity and sizes | Trace block IDs and approximate input sizes | Actual synthesized messages, tokenizer/chat template, cache-bust markers, and output caps |
| Metrics | Custom windows; some old reports compare mean TPOT with median ITL | aiperf's own summary/export pipeline on both simulation and hardware |

The new adapter uses the installed aiperf 0.12.0 implementation. Seven core
loader/replay files were checked byte-for-byte against upstream tag
`be53bf2953d30e46c500e6a80fc1f8b6f84bc718`. Its dataset comes from the actual
processed Arrow cache, snapshot `8fecd2fc56694469f758f0afbbb6335ad3043740`.
Provenance retains the dataset, tokenizer, replay-source, and engine-source
SHA-256 hashes. Hardware benchmark IDs are reused to reproduce cache-bust text.

## Parity results

| Check | RR96 | KV192 |
|---|---:|---:|
| Logged initial lane snapshots matching hardware | 96 / 96 | 192 / 192 |
| Warmup conversation and turn selections matching | 91 / 91 | 185 / 185 |
| Warmup input/output token counts matching | 91 / 91 | 185 / 185 |
| Common successful requests from initially warmed trees with matching input/output counts | 1,798 / 1,798 | 3,956 / 3,956 |
| Actual profiling send window | 3,600.000 s | 3,600.000 s |
| Full token-array checks of cached vs full prompt encoding | 163 passed | 174 passed |

Both loaded **393 roots → 9,602 conversations → 68,266 turns**. Both reproduced
the same **34,015 gated turns**, including the complete join-width histogram
from the hardware logs. They preserve tree-wide cache-bust identity, seed 42,
the 10-second whole-system idle cap, 60-second grace period, and 1,200-second
request timeout. The audit asserts those checks and rejects unexpected errors.

The 5,754 common profiling requests above are an explicitly matched subset,
identified through their initially warmed trees. They are not a claim that
every request or dispatch timestamp in two latency-dependent runs must match.
Full prompt token arrays from hardware were not exported in the downloaded
per-request metric records; the direct hardware comparison is of source/turn
identity and token counts, supported by the pinned reconstruction code.

**Prewarm correction:** the separate 900-second prewarm in the hardware job
template did not complete for either cell. Both GCS prewarm directories contain
only empty directory markers. The main profiling directory timestamps are
16 seconds (RR96) and 17 seconds (KV192) after the job epoch. The
[runner](../scripts/agentx_runner.sh) uses
[`sgl-d72-agentx.yaml`](../../manifests/perf/sgl-d72-agentx.yaml), whose prewarm
repeats `--no-fixed-schedule`. Reproducing it with aiperf 0.12.0 fails immediately
with `Parameter --no-fixed-schedule specified multiple times.` The successful
one-token trajectory warmup is reproduced; no fictitious 900-second prewarm is
added. Persistent GPU cache left by earlier jobs remains outside this model.

## Performance: 6 × TP4, 24 GB300 GPUs

Throughput is **input plus output tokens/s/GPU**. Error is
`100 × (simulation / hardware − 1)`. Values use the same aiperf summary metrics,
including its measurement/grace accounting; these are not active-throughput
or independently normalized record estimates.

| Cell | Published v3 sim | New faithful replay | Hardware | New throughput error |
|---|---:|---:|---:|---:|
| RR, 96 clients | 2,924 | **6,106** | 6,137 | **−0.5%** |
| Default KV, 192 clients | 3,715 | **6,983** | 9,655 | **−27.7%** |

| Metric | RR96 sim / real | Error | KV192 sim / real | Error |
|---|---:|---:|---:|---:|
| TTFT p95, seconds | 7.53 / 12.56 | −40.0% | 1.79 / 11.66 | −84.7% |
| ITL mean, ms | 20.41 / 19.47 | +4.8% | 51.42 / 30.61 | +68.0% |
| ITL p50, ms | 19.28 / 12.12 | +59.1% | 33.12 / 23.85 | +38.8% |
| Requests/s | 1.743 / 1.747 | −0.2% | 1.982 / 2.594 | −23.6% |
| Mean input tokens | 83,143 / 83,372 | −0.3% | 83,706 / 88,441 | −5.4% |
| Mean output tokens | 913 / 922 | −1.0% | 844 / 896 | −5.8% |
| Cached input fraction | 73.94% / 69.73% | +4.21 pp | 93.15% / 74.38% | +18.77 pp |

RR now closely reproduces aggregate throughput and request composition, while
still missing latency distribution. KV remains substantially wrong despite
the replay checks passing. It predicts excessive cache reuse, very cheap
prefill, and slow decoding. This is a useful separation: changing the dataset
replay again to force throughput agreement would conceal the serving-model
problem.

Successful profiling request composition was:

| Cell | Main sim / real | Subagent sim / real | Flat sim / real |
|---|---:|---:|---:|
| RR96 | 2,262 / 2,294 | 3,609 / 3,588 | 527 / 531 |
| KV192 | 2,596 / 3,699 | 4,041 / 5,013 | 638 / 807 |

The new KV simulation reported six request timeouts versus three hardware
timeout records. RR had no request errors. End-of-window cancellations are
handled separately by aiperf's grace-period logic.

## Concrete next calibration issue: KV worker load

The existing Python engine's routing score adds **queued prefill blocks** to
the overlap-adjusted request cost. It does not charge active decode load.
Dynamo 1.4.2's selector includes decode-load cost in its worker score, as shown
in the [pinned upstream implementation](https://github.com/ai-dynamo/dynamo/blob/v1.4.2/lib/kv-router/src/scheduling/selector.rs#L217).
The hardware runner installs `ai-dynamo==1.4.2`; the old simulator's comment
claiming an exact router formula is therefore insufficient evidence of parity.

The simulation's worker placement demonstrates the resulting calibration concern:

| KV worker | Warmup requests | Successful profile requests | Mean simulated ITL, ms |
|---:|---:|---:|---:|
| 0 | 110 | 2,101 | 109.2 |
| 1 | 23 | 1,844 | 39.0 |
| 2 | 16 | 1,436 | 26.4 |
| 3 | 16 | 744 | 18.9 |
| 4 | 12 | 490 | 15.2 |
| 5 | 8 | 660 | 20.4 |

Worker 0 gets 59.5% of warmup requests and becomes a strong cache-affinity
destination. The resulting imbalance is consistent with excessive locality
and inadequate decode-load accounting. It is not a measurement of hardware
worker placement, and does not prove how much of the whole gap each mechanism
causes. The cache model also makes prefixes available at admission, assumes
64-token reuse with a large LRU pool, and does not reproduce hybrid/Mamba cache
checkpoint behavior. Decode speed is fixed for each request at admission;
actual batch changes and prefill/decode interference are not modeled.

Use RR96 to constrain engine latency/cache behavior, then KV192 to constrain
placement and load accounting. Preserve these replay rules during that work.
Check RR192 and KV96 without fitting to them before interpreting KV-vs-RR gains
at equal concurrency. The two cells here deliberately serve different
calibration purposes and have different client counts.

## Artifacts and reproduction

- [Complete audit and comparison](../sim-results/agentx_aiperf_replay_20260917/audit.json)
- [CSV metrics](../sim-results/agentx_aiperf_replay_20260917/comparison.csv)
- [Archived summaries, logs, dispatch records, and provenance](../sim-results/agentx_aiperf_replay_20260917/README.md)
- [Hardware targets and source artifact URIs](../sim-results/agentx_calibration_targets.json)
- [Runnable setup, simulation, and audit commands](agentx-replay-howto.md)
- [Replay adapter](../../scripts/dynosim_aiperf_replay.py) and [audit script](../../scripts/audit_aiperf_replay.py)

The verified runs took approximately 338 seconds (RR96) and 553 seconds (KV192)
of local wall time. Full live artifacts remain under
`/tmp/n3u-faithful-runs/agg6-{rr-c96,kv-c192}-verified/`; compressed request
records and summaries are retained in the study directory. Ruff, compilation,
110 standalone token-array comparisons, 337 in-run token-array checks, and the
cross-hardware replay audit passed. No live serving workload was launched.
