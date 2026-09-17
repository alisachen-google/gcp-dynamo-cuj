# Nemotron-3-Ultra 550B — disaggregated serving under the AgentX concurrency definition

Current simulation source: [AIPerf replay](reports/agentx-aiperf-results.md). Hardware measurements remain in section iii; the old simulation curves and gap interpretation have moved to the historical archive.

72 GPU (GB300 NVL72), TP4/EP4, KV over NVLink (MNNVL + mooncake), SGLang 0.5.16 / Dynamo 1.4.2 /
FlashInfer 0.6.18, Weka 256K Claude-Code trace. Companion to D72_RESULTS.md (busy-stream results) and
AGENTX_COMPARISON.md (methodology). Status 2026-09-16 07:30 UTC: 2 of 6 KV points measured, ladder running.

Pages: [AgentX-concurrency curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html)
(y = total tok/s per GPU by default; toggles for output, TTFT p50/p95; RR on/off) · run index: RUN_INDEX.md.

## Links

| what | link |
|---|---|
| AgentX scenario definition (aiperf fork, pinned) | [inferencex_agentx_mvp.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/inferencex_agentx_mvp.py) · [registry.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/registry.py) · [agentic_replay.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/agentic_replay.py) · [session_tree.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/session_tree.py) · [trajectory_source.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/trajectory_source.py) |
| InferenceX (pinned 2f4201c) | [nvidia-master.yaml agentic entry](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/configs/nvidia-master.yaml#L6676) · [benchmark_lib.sh (aiperf command)](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/benchmark_lib.sh) · [agentic_srt.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/multi_node/agentic_srt.sh) · [launch_gb300-nv.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/runners/launch_gb300-nv.sh) · [DSV4 GB300 AgentX recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/multi_node/srt-slurm-recipes/dsv4/sglang/gb300-fp4/agentx/disagg-1p1d-dep8-dep16-c480-mtp-kvoffload.yaml) · [srt-slurm @984180e](https://github.com/NVIDIA/srt-slurm/tree/984180e5b8755aef85e9995048b5a16cb5336bce) |
| Our AgentX bench template + runner | [sgl-d72-agentx.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-agentx.yaml) · [agentx_runner.sh](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/agentx_runner.sh) · [busy-stream template sgl-d72-flagsweep.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml) |
| Fleet manifests | [n3u-mnnvl-99.yaml (9:9 disagg)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-99.yaml) · [n3u-mnnvl-full.yaml (6:12)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-full.yaml) · [n3u-agg-newstack.yaml (24-GPU agg)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-agg-newstack.yaml) |
| Current simulation + audits | [AIPerf replay adapter](../scripts/dynosim_aiperf_replay.py) · [results](reports/agentx-aiperf-results.md) · [reproduction](reports/agentx-replay-howto.md) |
| Pages | [AgentX-concurrency curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html) ([source](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html), [generator](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_agentx_curve.py)) · [busy-stream disagg curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-disagg-curve.html) · [frontier](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-frontier.html) · [Pareto](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-pareto.html) |
| Companion reports | [D72_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/D72_RESULTS.md) · [AGG24_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGG24_RESULTS.md) · [AGENTX_COMPARISON.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_COMPARISON.md) · [PARETO_E2E_REPORT.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/PARETO_E2E_REPORT.md) · [RUN_INDEX.md (every job artifact)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/RUN_INDEX.md) |
| Model, dataset, stack | [Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4) · [cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k) · [aiperf 0.12.0](https://github.com/ai-dynamo/aiperf/releases/tag/v0.12.0) · [SGLang v0.5.16](https://github.com/sgl-project/sglang/releases/tag/v0.5.16) · [Dynamo v1.4.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.4.2) · [FlashInfer v0.6.18](https://github.com/flashinfer-ai/flashinfer/releases/tag/v0.6.18) · [Mooncake](https://github.com/kvcache-ai/Mooncake) · [aiconfigurator (AIC)](https://github.com/ai-dynamo/aiconfigurator) · [NVIDIA DynoSim blog](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/) · [MNNVL / IMEX](https://docs.nvidia.com/multi-node-nvlink-systems/imex-guide/overview.html) |

## i. Concurrency: what we measured before, what AgentX measures, and where we misused the word

**Our previous definition (every point in D72_RESULTS.md / AGG24_RESULTS.md).** aiperf was launched with
[`--concurrency C --no-fixed-schedule --ignore-trace-delays`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/fixed_schedule.py). That is C independent request streams
that each fire the next request the instant the previous response finishes: think-time in the trace is
discarded, so C is a constant offered load, not a user count. Measured from the per-request records it
is even more than C requests in flight — the Weka sessions spawn subagent requests that run alongside
the parent turn, so the busy-stream 9:9 kv:48 run averaged **94 requests in flight (peak 425)**. It is
the right axis for finding a fleet's capacity knee (TTFT stationarity), and every knee, KV-vs-RR gain
and agg-vs-disagg ratio we published is on that axis. It is the wrong axis to compare with [InferenceX](https://github.com/SemiAnalysisAI/InferenceX)'s
numbers, and calling it "concurrency" next to theirs was the misuse: same word, ~6× different load
per unit.

**The AgentX definition** ([`--scenario inferencex-agentx-mvp`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/registry.py), [the only scenario registered in aiperf](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/registry.py);
definition in [`src/aiperf/common/scenario/inferencex_agentx_mvp.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/inferencex_agentx_mvp.py); trace in [AGENTX_COMPARISON.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_COMPARISON.md) §5e).
C is the number of live Claude-Code [session trees](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/session_tree.py) (root + subagents share one slot). Each lane snapshots
a session at a random point, primes its prefix in warm-up, then replays forward honouring the recorded
think-time between turns (10 s cap on whole-system idle only), [recycling a fresh session when the tree](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/agentic_replay.py)
drains. Most sessions are idle at any instant. Measured: **48 clients ≈ 5.6 requests in flight, 96 ≈ 16**
— about one sixth of a busy-stream slot per client. Their 480–1,920 therefore maps to roughly our
busy-stream c40–c200, the band where our knees sit.

| run | definition | configured C | mean in-flight | peak | req/s | mean latency |
|---|---|---|---|---|---|---|
| 9:9 KV kv:48 | busy-stream (ours) | 48 streams | 93.8 | 425 | 7.68 | 12.2 s |
| 9:9 KV 48 clients | AgentX | 48 sessions | 5.6 | 17 | 0.79 | 7.0 s |
| 9:9 KV 96 clients | AgentX | 96 sessions | 16.1 | 40 | 2.08 | 7.7 s |

([`scripts/inflight_from_records.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/inflight_from_records.py) on [`profile_export.jsonl`](https://console.cloud.google.com/storage/browser/alisachen-models/perf).) What we changed to adopt their definition:
one template ([`manifests/perf/sgl-d72-agentx.yaml`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-agentx.yaml)) that runs the scenario with [their launcher flags](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/benchmark_lib.sh)
([`--trajectory-start-min/max-ratio 0.25/0.75 --use-server-token-count --cache-bust first_turn_prefix`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/benchmark_lib.sh),
3600 s window), the same [aiperf 0.12.0](https://github.com/ai-dynamo/aiperf/releases/tag/v0.12.0), same trace, same fleet manifests. Only [`--warmup-requests-per-lane`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/config.py#L362)
(their fork) is absent. Bench pods run on arm64 GPU nodes from 192 clients up (aiperf needs > 60 GB there).

## ii. Current simulation performance through AIPerf

The handwritten v3/v5 predictions are superseded. Use [the actual AIPerf replay results](reports/agentx-disagg-rr-predictions.md) and [current result index](reports/agentx-aiperf-results.md). The [historical curves and analysis](reports/agentx-disagg72-legacy-analysis.md) retain the old predictions and point-selection rationale; their topology, knee and gap claims are not current evidence.

## iii. Real runs: performance, curve, and the KV-vs-RR points

| arm | clients | output tok/s (/GPU) | total tok/s/GPU | TTFT p50 / p95 / p99 | ITL p50 / p90 | in-flight | knee | guard |
|---|---|---|---|---|---|---|---|---|
| 9:9 KV | 48 | 786 (10.9) | 1,128 | 0.32 / 1.5 / 3.8 s | 6.4 / 6.8 ms | 5.6 | stationary | PASS |
| 9:9 KV | 96 | 2,019 (28.0) | 2,497 | 0.31 / 1.42 / 4.3 s | 7.4 / 8.4 ms | 16.1 | stationary | PASS |
| 9:9 KV | 192 | 3,267 (45.4) | 4,600 | 0.36 / 2.03 / 6.1 s | 8.5 / 9.9 ms | 30.0 (peak 60) | stationary | PASS (third attempt; the first was evicted at 59 GB on a system node, the second invalidated by a transfer failure during an operator-caused ComputeDomain deletion) |
| 12:6 KV | 96 | 1,980 (27.5) | 2,471 | 0.30 / 1.37 / 3.9 s | 8.2 / 9.6 ms | 17.2 (peak 37) | stationary | PASS (first point on the sim-optimal split; 11:54 UTC) |
| 12:6 KV | 192 | 3,217 (44.7) | 4,510 | 0.36 / 1.77 / 5.1 s | 10.0 / 11.2 ms | 33.5 (peak 63) | stationary (q1 0.33 → q4 0.37 s) | PASS (mooncake TE peak 765 MB/s, no transfer failures; 13:40 UTC) |
| 12:6 KV | 384 | 6,253 (86.8) | 8,637 | 0.43 / 2.58 / 7.1 s | 13.7 / 15.6 ms | 91.1 (peak 142) | stationary (q1 0.44 → q4 0.45 s) | PASS (mooncake TE peak 1,075 MB/s, no transfer failures; 15:48 UTC) |
| 12:6 KV | 480 | 7,544 (104.8) | 10,434 | 0.50 / 3.52 / 9.6 s | 16.4 / 17.5 ms | 129.4 (peak 196) | stationary (q1 0.51 → q4 0.52 s) | PASS (mooncake TE peak 998 MB/s, no transfer failures; 18:10 UTC) |
| 12:6 KV | 768 | 10,354 (143.8) | 15,004 | 0.80 / 7.00 / 14.6 s | 20.9 / 22.5 ms | 236.7 (peak 341) | stationary at p50 (q1 0.83 → q4 0.79 s); p95 drifts 6.0 → 7.8 s | PASS (mooncake TE peak 1,392 MB/s, no transfer failures; 20:57 UTC) |
| 12:6 KV | 1440 | **interrupted** — the cluster's GPU node pools (np-1, np-3, np-4) were resized to 0 by a GKE operation at 23:15–23:16 UTC while this cell was ~5 min into its profiling window; re-run when capacity returns | | | | | | |
| 12:6 RR | 192 / 96 / 384 | queued behind 12:6 KV | | | | | | |

Throughput scaled 2.57× from 48 to 96 clients and 1.84× from 96 to 192 with TTFT p50 flat at 0.31–0.36 s
(p95 2.0 s at 192), so 9:9 is still pre-knee at 192 with 30 requests in flight on average; the sim's 9:9 cell at 192
(2,733 total/GPU, TTFT p50 0.3 s, P90 93) is 0.59× real on total tokens and within 10% on latency and interactivity.
**12:6 vs 9:9 at 96 clients, measured: 2,471 vs 2,497 total tok/s per GPU (0.99×), TTFT p95 1.37 vs 1.42 s, P90 104 vs 119** — the
two splits are indistinguishable below the knee, exactly as the sim said (1,389 vs 1,395). **At 192 clients the same holds:
12:6 4,510 vs 9:9 4,600 total tok/s per GPU (0.98×), TTFT p95 1.77 vs 2.03 s (12:6 better on the tail, as its four extra
prefill workers predict), P90 89 vs 101 (9:9 better on decode, with its three extra decode workers), in flight 33.5 vs 30.0.**
The sim's same-config 192 cell for 12:6 KV is 2,679, so silicon is 1.68× above it, the same ratio band as the 9:9 cells.
**At 384 clients 12:6 KV is still stationary: 8,637 total tok/s per GPU (+92% over 192), TTFT p95 2.58 s, P90 64
tok/s/user, 91 in flight.** This is already above the sim's disagg *ceiling* (6,187 at 1440 clients) and 2.0× the sim's
own 384 cell, and it is the load at which the agg fleet is fully saturated (agg KV 384: TTFT p50 83 s and falling
throughput). Per GPU, disagg at 384 clients (5.3 clients/GPU) is now 0.89× of agg's best stationary cell (9,655 at
192 clients = 8 clients/GPU) while carrying 2× the sessions; the load-normalised comparison in AGENTX_DISAGG_VS_AGG.md
§3 is where the two meet. **At 480 clients (the sim's same-SLO cell for disagg KV) 12:6 is still stationary: 10,434 total tok/s per GPU
(+21% over 384), TTFT p95 3.5 s, P90 57, 129 in flight, 8.0 req/s.** The sim's cell is 5,217, so silicon is 2.0× above
it for the third cell running. This is the first disagg cell whose per-GPU total exceeds agg's best stationary cell
(9,655 at 192 clients) — at 6.7 clients per GPU against agg's 8 — and it does so with a TTFT tail 3.3× shorter (3.5 vs
11.7 s p95). **At 768 clients 12:6 KV is still pre-knee: 15,004 total tok/s per GPU (+44% over 480), TTFT p50 0.80 s and p95
7.0 s, P90 44.5 tok/s/user, 237 in flight, 11.3 req/s.** The p50 is flat across the hour (0.83 → 0.79 s) so the queue
is bounded; the p95 creeps from 6.0 to 7.8 s across quarters, the first sign of the prefill tier filling, but the cell
stays well inside the 20 s budget. This is the cell where the sim put the disagg knee (768, TTFT p95 30 s, 5,981
total/GPU): silicon is at 2.5× the sim's throughput with a 4.3× shorter tail, so the sim's knee is at least 2× early on
the load axis — the prefix-hit-rate term from §iv in action. Per GPU the disagg fleet is now 1.55× agg's best
stationary cell (9,655) at 10.7 clients per GPU versus agg's 8. The 1440 cell (running) will locate the real knee. Per-user interactivity is very high in this regime (P90 147 tok/s/user at 48,
119 at 96) because decode batches are tiny. The KV-vs-RR comparison table will be filled from the
96/384 pairs when the RR points land; the busy-stream KV/RR gains (2.02× at c48, 1.94× at c96, both
instance-2) are the reference expectation.

Measured points are overlaid on the curve page as solid markers; the run index links every artifact.

## iv. Current replay and performance-validation status

The [12P:6D RR prediction curve](reports/agentx-disagg-rr-predictions.md) uses actual AIPerf 0.12.0 replay and independent RR on both P and D. Its 96/192/384 points reproduce the GitHub recipe's deployment and client settings. No matching real disagg RR artifacts are available in this comparison, so the measured KV ladder above is not an RR performance reference.

The [64-GPU KV/RR topology search](reports/agentx-64gpu-topology.md) uses the same replay approach with an explicit cache-capacity proxy and KV decode-block routing. Those counterfactual deployments are separate from the 72-GPU measurements here. Follow the [disagg correctness requirements](reports/agentx-replay-howto.md#what-disagg-correctness-requires) before treating their capacity predictions as calibrated performance.

### Reading the measured tail itself (independent of the simulator)

[`scripts/agentx_ttft_tail.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/agentx_ttft_tail.py)
splits the per-request records three ways:

| 12:6 KV | 192 clients | 768 clients |
|---|---|---|
| cold first turns (cache-busted): TTFT = a + ISL / rate | −0.03 s + ISL / **26,200 tok/s** (1,653 turns) | 0.34 s + ISL / **22,200 tok/s** (5,067 turns) |
| cold turns over 100 k tokens: TTFT p50 / p95 | 5.95 / 12.3 s (n = 67) | 6.85 / 14.5 s (n = 223) |
| warm turns, lowest vs highest in-flight quartile: TTFT p95 | 1.77 vs 1.93 s | 7.01 vs 7.50 s |
| p95 tail composition | 19 % cold first turns, median ISL 108 k, in-flight at issue = overall median | 7 % cold, median ISL **136 k**, in-flight at issue = overall median |

Three facts follow. (1) The real uncached prefill rate is 22–26 k tok/s per TP4 worker, so the AIC seed of 19.7 k
used by the sim is 12–33 % slow: a modest, second-order term. (2) The tail is **not queue-driven**: warm turns issued
under the heaviest quarter of the load have the same p95 as those issued under the lightest quarter. (3) The tail is
**size-driven**: it is the long-context turns (median 136 k tokens at 768) whose prefix is not on the worker that gets
them — a cold 130 k turn costs 6–7 s at the median, which is exactly the p95 level. So the way to reproduce the
measured tail in a simulator is not a heavier prefill queue but the right *miss distribution*: which long turns miss,
and how much of them, which is the cache-churn / routing term (hit 0.94 → 0.88 as load rises).

