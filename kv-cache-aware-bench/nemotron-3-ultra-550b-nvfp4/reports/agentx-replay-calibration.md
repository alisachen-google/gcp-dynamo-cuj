# AgentX replay parity and two agg calibration points

Inspected 2026-09-17 at study commit `96a019d8922002de91f4f9acc68005c7aac4ea21`.
The executed replay and validation are documented in [the faithful replay results](agentx-faithful-replay.md), with [rerun instructions](agentx-replay-howto.md). The tables below preserve the historical simulation comparisons and calibration-point selection.

The latest saved v5 runs are **12:6 disaggregated KV, 72 GPUs**. There is no saved v5 agg sweep. For agg, distinguish the published v3 sweep from the later engine-substitution experiments.

**Where replay starts.** A sampled position is a timestamp drawn between 25% and 75% of a session's recorded duration, followed by a snapshot of its root and child conversations. It is not a uniform draw over request indices. Turn 0 is the first request of a newly selected root session. Both v5b and v5d initially use sampled positions; v5b samples again on recycling, while v5d recycles at turn 0.

The hardware exports confirm `aiperf_version: 0.12.0`, `scenario: inferencex-agentx-mvp`, and initial ratios 0.25/0.75. The corresponding upstream tag is `be53bf2953d30e46c500e6a80fc1f8b6f84bc718`. Its [`_dispatch_recycled_on_lane`](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/strategies/agentic_replay.py#L1432) selects a new root from the shared dataset sampler and dispatches turn 0 with a fresh cache-bust marker. **v5d matches this recycling rule.** The fact that v5b matches one throughput number better is not evidence for using its recycling rule.

**Saved disagg v5 results.** Throughput counts input plus output tokens per second per GPU. Percentages use `(simulation / hardware - 1) × 100`.

| Clients | Variant | Total tok/s/GPU | Error | TTFT p95, s | Reported decode time, ms | P90 interactivity, tok/s/user |
|---:|---|---:|---:|---:|---:|---:|
| 192 | Hardware | 4,510 | — | 1.77 | 10.0 | 89 |
| 192 | v5b | 2,981 | −33.9% | 1.32 | 7.4 | 120 |
| 192 | v5d | 6,062 | +34.4% | 1.49 | 9.1 | 90 |
| 768 | Hardware | 15,004 | — | 7.00 | 20.9 | 45 |
| 768 | v5b | 15,558 | +3.7% | 3.16 | 34.5 | 17 |
| 768 | v5d | 18,193 | +21.3% | 3.80 | 59.1 | 13 |

Sources: [v5 output](../sim-results/agentx_v5_sim.txt), [disagg analysis](../AGENTX_D72_RESULTS.md). The existing report compares simulation mean TPOT with hardware ITL p50. Hardware means, checked in the GCS exports, are 9.899 and 20.839 ms; use these for mean-to-mean comparisons. The difference is small on disagg but material on agg.

The saved v5 variants have 71–81% subagent requests versus approximately 51% measured. Throughput agreement at 768 therefore does not establish replay or latency agreement.

**Existing agg differences.** The published [v3 CSV](../sim-results/dynosim_n3u_agentx_v3.csv) and [measured summary series](../sim-results/measured_agentx.json) give:

| Clients | KV sim / real total tok/s/GPU | KV sim error | RR sim / real total tok/s/GPU | RR sim error |
|---:|---:|---:|---:|---:|
| 48 | 1,474 / 3,334 | −55.8% | 1,792 / 3,250 | −44.9% |
| 96 | 2,382 / 6,844 | −65.2% | 2,924 / 6,137 | −52.4% |
| 192 | 3,715 / 9,655 | −61.5% | 3,889 / 6,802 | −42.8% |
| 384 | 4,706 / 8,257 | −43.0% | 4,156 / 5,076 | −18.1% |

At 192, the published simulation predicts KV **4.5% behind RR**, whereas hardware has KV **41.9% ahead**. Hardware KV TTFT p95 is 11.68 s versus RR 60.10 s. Both hardware arms are beyond their useful knee at 384. Tuned KV at 192 reaches 11,012 total tok/s/GPU and 6.33 s TTFT p95; the corresponding published simulation is 2,740 total tok/s/GPU, a 75.1% underestimate.

The latest saved agg [KV](../sim-results/agentx_decomp_agg.txt) and [RR](../sim-results/agentx_decomp_agg_rr.txt) engine-substitution experiments improve the 192-client prediction to **6,008 / 9,741 (−38.3%)** and **4,673 / 6,827 (−31.6%)**, respectively. Their hardware denominators are derived from profiling records and differ from the summary series above; do not mix denominators between tables. These substitutions use measured output-length scaling, a fixed scheduling floor, and measured decode lines. They are calibration experiments, not an independent v5 validation.

**How to replay the actual aiperf scenario in simulation.**

1. Pin **aiperf 0.12.0**, the same dataset snapshot, tokenizer/chat template, resolved configuration, and random seed used by hardware. The [saved calibration targets](../sim-results/agentx_calibration_targets.json) retain the actual exported configurations and source artifact URIs. Match the effective dataset sampler; the resolved dataset default is sequential, and the production trajectory source reuses that sampler for recycling. v5 instead draws roots with replacement.
2. Use aiperf's actual [Weka loader](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/loader/weka_trace.py), including [agent-chain detection](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/dataset/loader/weka_agent_chains.py), [TrajectorySource](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/src/aiperf/timing/trajectory_source.py), AgenticReplayStrategy, branch orchestrator, and SessionTreeRegistry. Keep initial snapshot selection, parent/child dependencies, end-to-start delays, shared tree cache-busting, and whole-tree recycling in that code.
3. For the first reference implementation, run the real CLI against a local OpenAI-compatible **simulated serving endpoint**. The endpoint supplies model discovery and streaming chat completions; a serving-model adapter schedules first-token and subsequent-token events, emits consistent token usage, and maintains per-worker cache and queue state. Preserve the actual synthesized prompt/token identities and output lengths. Completion timing must feed back into aiperf so the next turn waits for the simulated response plus its recorded delay. A fixed replay of hardware request timestamps cannot preserve this feedback when the routing policy changes.
4. Match warmup exactly. The selected hardware runs completed the AgentX invocation's one-token trajectory warmup (91 requests for RR96, 185 for KV192). **Correction after auditing the artifacts:** the separate 900-second prewarm in the job template failed before running because `--no-fixed-schedule` is duplicated. Both prewarm directories are empty, and their profiling directories were created only 16–17 seconds after the job epoch. v5 instead primes cache instantaneously and then advances its AgentX replay for 900 seconds before measuring. Reuse the actual trajectory warmup and its barrier.
5. First establish a wall-clock reference with the unmodified scenario. For faster sweeps, a later adapter can retain the same loader/strategy/branch logic and provide a virtual scheduler and simulated credit completions. That adapter must advance all scheduling, completion, phase, idle-cap, and grace-period clocks consistently. The upstream [component test](https://github.com/ai-dynamo/aiperf/blob/be53bf2953d30e46c500e6a80fc1f8b6f84bc718/tests/component_integration/test_agentic_replay_e2e.py) demonstrates the real loader-to-strategy path and provides a useful integration starting point.

The profiling invocation, **after the simulated endpoint exists**, should preserve these hardware flags (set `C` to 96 for RR, then 192 for KV; select the routing policy on the endpoint):

```bash
aiperf profile \
  -m "$MODEL" --tokenizer "$TOKENIZER" --tokenizer-trust-remote-code \
  --url http://127.0.0.1:8000 --endpoint-type chat --streaming \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --scenario inferencex-agentx-mvp \
  --cache-bust first_turn_prefix --system-idle-gap-cap-seconds 10 \
  --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 \
  --use-server-token-count --extra-inputs ignore_eos:true \
  --num-dataset-entries 393 --max-context-length 262144 \
  --concurrency "$C" --random-seed 42 \
  --benchmark-duration 3600 --benchmark-grace-period 60 \
  --request-timeout-seconds 1200 \
  --workers-max 200 --record-processors 8 \
  --slice-duration 1.0 --profile-export-level raw \
  --artifact-dir "$ARTIFACT_DIR"
```

**The two selected agg cells: 6 × TP4, 24 GB300 GPUs.**

| Cell | Purpose | Real total tok/s/GPU | Req/s | Mean input / output tokens | TTFT p95 | ITL mean / p50 |
|---|---|---:|---:|---:|---:|---:|
| RR, 96 clients | Establish replay and engine behavior below saturation | 6,137 | 1.747 | 83,372 / 922 | 12.557 s | 19.466 / 12.118 ms |
| Default KV, 192 clients | Exercise cache placement and queueing with engine parameters held fixed | 9,655 | 2.59 | 88,441 / 896 | 11.657 s | 30.61 / 23.85 ms |

These are direct summary-export values; the historical decomposition rounds KV TTFT p95 to 11.68 s. Compare the same error-adjustment choice on both sides. Reported frontend cached-token fractions are approximately 0.697 for RR96 and 0.743 for KV192.

Use RR192 and KV96 as validation cells without fitting to them; tuned KV192 is a further routing check. If only one cell is available initially, use RR96, then add KV192 before drawing conclusions about KV-vs-RR gain. Two cells cannot identify a general cache, queue, prefill, and decode model: constrain engine parameters with the measured counters and per-request distributions, rather than fitting throughput alone.

Before engine fitting, compare the loader's conversation graph, sampled initial states, per-kind input/output distributions, child spawning, joins, recycle sequence, and recorded delay distributions. Under controlled identical completion times, the adapter and real aiperf should produce the same dispatch sequence. Against hardware, compare request rates by kind jointly with service latency because replay is closed loop. Then compare same-window total/output throughput, TTFT p50/p95, ITL mean/p50/p90, cache-hit fraction, worker occupancy, and queue stationarity. Keep the 51% subagent share as a check on the measured disagg workload, not a hard-coded target for every agg cell.

Two reporting issues matter for this calibration. First, the historical agg decomposition compares **mean simulated TPOT to median hardware ITL**: at KV192 the calibrated simulation is 33.1 ms, while hardware mean ITL is 30.61 ms, not 23.85 ms. That comparison alone does not establish a 1.4× decode slowdown. Second, the agg narrative describes a piecewise decode cliff, but the inspected `apply_n3u_constants()` currently installs the linear function `8.9 + 1.73 * batch`. Save source commit, trace fingerprint, effective engine/router parameters, and metric-window definitions with every new simulation; do not assume the old narrative identifies the current code path.
