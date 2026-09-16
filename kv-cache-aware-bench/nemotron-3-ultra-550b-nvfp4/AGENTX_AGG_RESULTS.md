# Nemotron-3-Ultra 550B — aggregated serving (24 GPU) under the AgentX concurrency definition

6 × TP4/EP4 aggregated workers (24 GPU, GB300), SGLang 0.5.16 / Dynamo 1.4.2 / FlashInfer 0.6.18 (the
"new stack"), Weka 256K Claude-Code trace. Companion to AGG24_RESULTS.md (busy-stream results, old and new
stack) and AGENTX_COMPARISON.md (methodology). Status 2026-09-16 07:30 UTC: no AgentX-mode agg point
measured yet — the ladder is queued behind the last busy-stream agg cell (rr:512, running).

Pages: [AgentX-concurrency curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html) (agg series in orange) · run index: RUN_INDEX.md.

## Links

| what | link |
|---|---|
| AgentX scenario definition (aiperf fork, pinned) | [inferencex_agentx_mvp.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/inferencex_agentx_mvp.py) · [registry.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/registry.py) · [agentic_replay.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/agentic_replay.py) · [session_tree.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/session_tree.py) · [trajectory_source.py](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/trajectory_source.py) |
| InferenceX (pinned 2f4201c) | [nvidia-master.yaml agentic entry](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/configs/nvidia-master.yaml#L6676) · [benchmark_lib.sh (aiperf command)](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/benchmark_lib.sh) · [agentic_srt.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/multi_node/agentic_srt.sh) · [launch_gb300-nv.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/runners/launch_gb300-nv.sh) · [DSV4 GB300 AgentX recipe](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/multi_node/srt-slurm-recipes/dsv4/sglang/gb300-fp4/agentx/disagg-1p1d-dep8-dep16-c480-mtp-kvoffload.yaml) · [srt-slurm @984180e](https://github.com/NVIDIA/srt-slurm/tree/984180e5b8755aef85e9995048b5a16cb5336bce) |
| Our AgentX bench template + runner | [sgl-d72-agentx.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-agentx.yaml) · [agentx_runner.sh](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/agentx_runner.sh) · [busy-stream template sgl-d72-flagsweep.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml) |
| Fleet manifests | [n3u-mnnvl-99.yaml (9:9 disagg)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-99.yaml) · [n3u-mnnvl-full.yaml (6:12)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-full.yaml) · [n3u-agg-newstack.yaml (24-GPU agg)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-agg-newstack.yaml) |
| Simulator + sweeps | [dynosim_agentx.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx.py) · [dynosim_pd.py (engine model)](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_pd.py) · [dynosim_n3u_agentx_v2.csv](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_agentx_v2.csv) · [measured_agentx.json](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/measured_agentx.json) · [decomposition dynosim_agentx_decomp.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_decomp.py) · [inflight_from_records.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/inflight_from_records.py) |
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

For agg the busy-stream reference is the new-stack bounded peak **KV c32 = 1,854 tok/s = 77.3/GPU,
13,320 total tok/s/GPU** (AGG24 §4.5); on that axis KV knees between c32 and c64 and RR between c32 and
c64. Under the AgentX definition those knees move to hundreds of clients.

## ii. Simulated curve under the AgentX definition, and the selected points

Same simulator and sweep as the disagg report ([`sim-results/dynosim_n3u_agentx_v2.csv`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_agentx_v2.csv), arm [`agg6`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx.py) =
6 × TP4 workers, 24 GPU). Total tok/s per GPU (output tok/s per GPU) · TTFT p50 · P90 interactivity:

Cell = **total tok/s per GPU** (output tok/s per GPU) · TTFT p50 · P90 interactivity (tok/s/user = 1000 / per-request TPOT p90). Sim v3 (`sim-results/dynosim_n3u_agentx_v3.csv`): input tokens counted exactly per simulated request (len(hash_ids)×64), the InferenceX total-token convention.

| clients | agg6 KV | agg6 RR |
|---|---|---|
| 48 | **1,474** (24) · 0.13 s · 22 | **1,792** (30) · 0.89 s · 48 |
| 96 | **2,382** (37) · 0.14 s · 13 | **2,924** (48) · 2.21 s · 29 |
| 192 | **3,715** (58) · 0.16 s · 9 | **3,889** (60) · 4.43 s · 15 |
| 384 | **4,706** (64) · 0.22 s · 6 | **4,156** (55) · 14.43 s · 8 |
| 480 | **4,863** (64) · 0.23 s · 5 | **4,091** (51) · 24.04 s · 6 |
| 768 | **5,107** (58) · 0.38 s · 4 | **3,712** (41) · 78.95 s · 4 |
| 960 | **5,138** (56) · 0.42 s · 3 | **3,404** (35) · 117.82 s · 3 |
| 1440 | **4,756** (46) · 0.80 s · 2 | **2,381** (23) · 238.70 s · 2 |
| 1536 | **4,632** (44) · 0.98 s · 2 | **2,198** (21) · 259.49 s · 2 |
| 1920 | **4,165** (37) · 2.03 s · 2 | **1,351** (15) · 369.09 s · 2 |

Reading (total-token axis): agg KV peaks at **5,138 total/GPU at 960 clients (TTFT p50 0.4 s, 56 output/GPU)** — the peak sits later than
on the output axis (960 vs 384–480 clients) because input tokens keep accruing while output per GPU decays,
and TTFT p50 is still 0.4 s there (no prefill hand-off, every worker keeps its own prefix cache). RR peaks at
4,156 total/GPU at 384 clients (TTFT p50 14.4 s, 55 output/GPU) and falls away after 480 as prefix reuse across six workers collapses (hit 0.4 vs 0.8).
Against disagg 9:9 KV (5,501 total/GPU at 480 clients (TTFT p50 4.1 s, 92 output/GPU)) and 12:6 KV (6,187 total/GPU at 1440 clients (TTFT p50 4.7 s, 85 output/GPU)), agg's total-token ceiling is
0.83–0.93× per GPU but its TTFT stays an order of magnitude lower up to ~1,000 clients; on interactivity agg is
the weak arm (P90 45–172 ms ITL → 6–22 tok/s/user at 48–384 clients) because the same workers prefill and
decode.

Selected points (from the v3 sim knees):

| framing | rule | agg KV | agg RR | sim gain (total tok/s/GPU) |
|---|---|---|---|---|
| knees | throughput-slope knee | ≈ 192 clients (TTFT p50 0.2 s; throughput keeps creeping to 960) | ≈ 192 clients (TTFT p50 4.4 s) | — |
| **same config** | both knees coincide at 192 | 192 → 3,715 (p50 0.2 s, P90 9 tok/s/user) | 192 → 3,889 (p50 4.4 s, P90 15) | 0.96× on tokens, KV 22× lower TTFT, RR better interactivity |
| **same SLO (TTFT p95 ≤ 20 s)** | best throughput under the budget | 192 → 3,715 (p95 6 s) | 96 → 2,924 (p95 11 s) | **1.27×** |
| same SLO (TTFT p95 ≤ 60 s) | | 480 → 4,863 | 192 → 3,889 | 1.25× |

Agg is the arm where the sim says KV-aware routing barely pays on tokens: with six independent prefix caches and
no prefill hand-off, RR's cache misses cost latency (4.4 s vs 0.2 s TTFT at 192) more than throughput, and KV's
session affinity concentrates decode batches on fewer workers, which is why its simulated P90 interactivity is
*lower* than RR's (9 vs 15 tok/s/user at 192). The measured agg RR ladder therefore runs the full 48 → 1536
in parallel with KV (second fleet `n3u-agg-ns2`), so both framings can be read off the same client counts:
**192** for same-config and **96 (RR) vs 192 (KV)** for the 20 s TTFT-p95 SLO.

## iii. Real runs: performance, curve, and the KV-vs-RR points

| arm | clients | output tok/s (/GPU) | total tok/s/GPU | TTFT p50 / p95 / p99 | ITL p50 / p90 | in-flight | knee | status |
|---|---|---|---|---|---|---|---|---|
| agg KV | 48 | 775 (32.3) | 3,334 | 0.42 / 3.83 / 7.7 s | 7.3 / 9.8 ms (P90 interactivity 102) | 6.8 (peak 18) | stationary | complete 09:18 UTC |
| agg KV | 96 | 1,803 (75.1) | 6,844 | 0.67 / 5.36 / 12.2 s | 11.9 / 23.5 ms (P90 42.6) | 27.5 (peak 51) | stationary | complete 10:46 UTC |
| agg KV | 192 | 2,323 (96.8) | 9,655 | 1.56 / 11.68 / 17.5 s | 23.9 / 50.6 ms (P90 19.8) | 74.5 (peak 116) | stationary (q1 1.63 → q4 1.13 s) | complete 12:29 UTC |
| agg KV | 384 | 1,866 (77.7) | 8,257 | 82.8 / 119.4 / 134 s | 40.3 / 79.1 ms (P90 12.6) | 255.8 (peak 320) | **POST-KNEE** (TTFT p50 q1 37.9 → q4 99.6 s, growing) | complete 14:29 UTC; 768 / 1536 skipped (see below) |
| agg RR | 48 | 752 (31.3) | 3,250 | 0.79 / 8.27 / 15.3 s | 7.5 / 11.0 ms (P90 90.9) | 8.5 (peak 22) | stationary | complete 10:42 UTC (second fleet `n3u-agg-ns2`) |
| agg RR | 96 | 1,611 (67.1) | 6,137 | 0.81 / 12.56 / 19.8 s | 12.1 / 29.6 ms (P90 33.8) | 33.2 (peak 53) | stationary | complete 12:11 UTC |
| agg RR | 192 | 1,713 (71.4) | 6,802 | 10.87 / 60.1 / 90.1 s | 30.5 / 82.0 ms (P90 12.2) | 102.7 (peak 154) | stationary by the q1/q4 test (9.8 → 10.4 s) but at the throughput knee: +11% tokens for 2× clients | complete 13:53 UTC |
| agg RR | 384 | 1,192 (49.6) | 5,076 | 78.3 / 495 / 600 s | 57.0 / 126 ms (P90 7.9) | 265.6 (peak 366) | **POST-KNEE** (saturated; TTFT p50 74 s in q1) | complete 15:53 UTC; 768 / 1536 skipped |

An earlier agg AgentX smoke at 48 clients (2026-09-15) failed on [`--warmup-requests-per-lane`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/config.py#L362), a flag that
exists only in SemiAnalysis's aiperf fork; the template was corrected and no agg AgentX result predates
this report. First measured agg point vs the sim's agg KV cell at 48 clients: total 3,334 vs 1,474 per GPU (sim 0.44×), TTFT p50 0.42 vs 0.13 s, P90 interactivity 102 vs 22 tok/s/user — the agg simulator is far more pessimistic than the disagg one at low load (its refit decode curve 8.9 + 1.73·bs ms is a busy-stream fit; under replayed think-time the workers run near batch 1 and the real ITL is 7.3 ms). Mean in-flight 6.8 requests (peak 18) from the per-request records. Rows fill as points land; KV-vs-RR uses 192 (same config) and 192-vs-96 (TTFT budget). **Measured KV-vs-RR pairs on agg (same config, all stationary): 48 clients — KV 3,334 vs RR 3,250 total tok/s per GPU
= 1.03× on tokens, TTFT p95 3.83 vs 8.27 s (RR 2.2× worse), P90 interactivity 102 vs 91; 96 clients — KV 6,844 vs RR 6,137
= 1.12×, TTFT p95 5.36 vs 12.56 s (RR 2.3× worse), P90 42.6 vs 33.8.** The KV advantage on tokens grows with load (1.03× →
1.12×) as the sim's shape predicted, while the sim's absolute claim that RR is *ahead* at these loads (0.82×, 0.81×) is wrong.
RR also carries more requests in flight per client (33 vs 27.5 at 96) because its cache-miss prefills lengthen every request.

**KV at 192 clients (the sim's same-config comparison point) is still stationary: 9,655 total tok/s per GPU, TTFT p95
11.7 s, 74.5 requests in flight (peak 116) on 24 GPUs, ITL p50 24 ms (P90 interactivity 19.8 tok/s/user).** That is
2.6× the sim's 3,715 for the same cell and already above the sim's agg *ceiling* (5,138 at 960 clients): the agg fleet
under AgentX load is far from its knee at 192 (TTFT p50 falls from 1.63 s in the first quarter to 1.13 s in the last).
The TTFT p95 budget of 20 s used for the same-SLO selection is met with margin (11.7 s); the same-SLO KV point will
therefore move up the ladder (384 running) rather than sit at 192 as the sim suggested. Interactivity is the axis
that has moved: P90 dropped 102 → 43 → 20 tok/s/user from 48 to 192 clients as per-worker decode batches grew
(in flight per worker 1.1 → 4.6 → 12.4), the trade the sim predicted for KV packing on agg.

**Measured agg KV knee: 192 clients.** At 384 the fleet is saturated: TTFT p50 climbs through the hour (38 → 100 s across
quarters), 256 of 384 sessions are in flight at any moment and total tokens *fall* to 8,257 per GPU (0.86× of the
192-client cell) because the KV cache churns under 43 concurrent contexts per worker and prefix hits are lost. The
384 row is reported for the curve shape but is not a throughput point under the stationarity rule; the agg KV ladder
was stopped there (768 / 1536 would only deepen the same queue) so that the agg flag sweep at the 192-client comparison
point could start on the same fleet. The sim placed the agg KV knee at 1536 clients; silicon puts it between 192 and
384, a 4–8× miss on the load axis — the largest single disagreement in the study (the sim's 4 k-request slice carries
~30% fewer input tokens per turn and no subagent fan-out, so its prefill demand per client is far lighter, §iv).

**Agg RR at 384 collapses harder than KV**: total tokens fall to 5,076 per GPU (0.75× of its 192-client cell, vs KV's
0.86×), TTFT p95 reaches 495 s (KV 119 s) and 266 of 384 sessions sit in flight. Both agg ladders therefore end at
384: the measured agg knees are **KV 192 / RR 192** (RR at its throughput knee, KV still scaling there), and every
cell above is saturated. The remaining agg runs are the flag sweep at 192.

### KV vs RR at 192 clients — the same-config comparison point, measured

| | KV | RR | KV ÷ RR |
|---|---|---|---|
| total tok/s per GPU | **9,655** | 6,802 | **1.42×** |
| output tok/s per GPU | 96.8 | 71.4 | 1.36× |
| TTFT p50 / p95 / p99 | 1.56 / **11.7** / 17.5 s | 10.9 / **60.1** / 90.1 s | RR 5.1× worse at p95 |
| ITL p50 / p90 → P90 interactivity | 23.9 / 50.6 ms → 19.8 | 30.5 / 82.0 ms → 12.2 | 1.62× |
| in flight (mean / peak) of 192 clients | 74.5 / 116 | 102.7 / 154 | RR holds 38% more requests |
| gain 96 → 192 clients | 6,844 → 9,655 (+41%) | 6,137 → 6,802 (+11%) | RR is at its throughput knee, KV is not |

The sim's same-config verdict for this cell (KV 3,715 vs RR 3,889 = 0.96×, RR ahead) is reversed on silicon: **KV
1.42× on tokens, 5× better TTFT tail, 1.6× better interactivity, at the same 192 live sessions.** The mechanism is the
one the sim attributes to RR's early knee, only stronger: at 192 clients RR re-prefills most of each ~90 k-token turn
(prefix hit ≈ 0.2–0.4 across six workers), so its prefill demand saturates the fleet, TTFT p50 sits at 11 s with a
60 s p95, 103 requests are queued or running at any time and total tokens grow only 11% over the 96-client cell.
KV routing lands each session's turns on the worker that holds its prefix, prefill demand stays ~3× lower, and the
fleet keeps scaling (+41%) with 1.6 s p50 TTFT. The trade KV pays is decode-batch depth (P90 19.8 tok/s/user), but RR is
worse on that axis too because its long prefills stall the decode loop.

**Same-SLO comparison (TTFT p95 ≤ 20 s, the budget used to pick the cells):** RR's best cell inside the budget is 96
clients (p95 12.6 s, 6,137 total/GPU; 192 fails at 60 s). KV meets the budget at 192 (11.7 s) with 9,655 → **1.57×**,
and the 384-client KV cell fails the budget outright (p95 119 s), so **KV 192 vs RR 96 = 1.57× is the final same-SLO pair on agg**. Under the interactivity
budget (P90 ≥ 20 tok/s/user) neither policy meets it at 192 on agg; KV at 96 (P90 42.6, 6,844) vs RR at 96 (33.8,
6,137) = 1.12× is the pair inside that budget.

Sim vs silicon at this cell: KV total 3,715 sim vs 9,655 real (0.38×), RR 3,889 vs 6,802 (0.57×); TTFT p95 KV 6 s sim vs
11.7 real (0.51×), RR 34 s sim vs 60 real (0.57×). The sim's RR cell is the closer one because RR's cost is prefill
volume, which the sim models directly; the KV cell is where the decode cliff and over-packing (§iv) bite. The sim had predicted 0.82× on tokens and a 2.3× TTFT p95 gap at 48 clients:
the latency ratio is right, the token ratio is not (measured KV does not lose throughput to affinity at this load). From 48 to
96 clients agg KV scaled 2.05× on total tokens with TTFT p95 5.4 s and in-flight 27.5, i.e. agg carries more requests in flight
per client than 9:9 (16 at 96) because its per-request latency is ~2× longer (14.3 s vs 7.7 s).

## v. KV-router flag sweep at the 192-client comparison point (measured)

The sim (§ii) predicted that the tuned router (`--router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8`)
would *lose* 26% on agg at 192 clients (2,740 vs 3,715) because it spreads a session's turns across workers and gives
up prefix hits. Silicon says the opposite. Variants run on the same fleet (`n3u-agg-newstack.yaml`, 6 × TP4 GB300),
same 192 live-session clients, same 900 s warm + 3,600 s window; runner variants are the `ROUTER` map in
[`scripts/agentx_runner.sh`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/agentx_runner.sh).

| router flags (192 clients, agg) | total tok/s/GPU | output tok/s (/GPU) | TTFT p50 / p95 / p99 | ITL p50 / p90 → P90 | in flight | stationary | artifact |
|---|---|---|---|---|---|---|---|
| kv default (scale 1, credit 0, temp 0, fcfs) | 9,655 | 2,323 (96.8) | 1.56 / 11.7 / 17.5 s | 23.9 / 50.6 → 19.8 | 74.5 | yes | [1789555981](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789555981_alisachen-n3u-agg-ns-agentx-kv-c192) |
| **kvs3c08**: prefill-load scale 3.0, overlap credit 0.8 | **11,012 (+14%)** | 2,613 (108.9) | 0.87 / **6.33** / 12.1 s | 19.6 / 37.3 → **26.8** | 64.1 (peak 107) | yes (q1 0.89 → q4 0.82 s) | [1789569414](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789569414_alisachen-n3u-agg-ns-agentx-kvs3c08-c192) |
| kvs2c08: scale 2.0, credit 0.8 | running (started 16:13) | | | | | | |
| kvt05: temperature 0.5 | 7,816 (−19%) | 1,908 (79.5) | 5.0 / 22.1 / 31.6 s | 33.3 / 77.7 → 12.9 | 93.6 (peak 131) | yes (q1 4.5 → q4 2.5 s) | [1789574468](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789574468_alisachen-n3u-agg-ns2-agentx-kvt05-c192) |
| rr (reference) | 6,802 | 1,713 (71.4) | 10.9 / 60.1 / 90.1 s | 30.5 / 82.0 → 12.2 | 102.7 | at knee | [1789560983](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789560983_alisachen-n3u-agg-ns2-agentx-rr-c192) |

**Reading.** Weighting prefill load 3× and crediting overlap at 0.8 lets the router move a turn off a worker whose
prefill queue is deep even when that worker holds the prefix, so bursts (subagent fan-out, back-to-back tool calls)
stop piling onto one worker: fewer requests in flight (64 vs 75), a TTFT tail cut almost in half (p95 6.3 vs 11.7 s),
faster decode (ITL p50 19.6 vs 23.9 ms, P90 26.8 vs 19.8) and 14% more total tokens. Against RR the tuned router is
**1.62× on tokens with a 9.5× better TTFT p95** at the same 192 sessions. The sim got the sign wrong for the reason
already found in §iv: it charges the KV router a decode cliff for packing and models the load-scale flag as pure
prefix loss, while the live effect is queue relief. The same-SLO agg pair therefore moves to **tuned KV 192 (11,012,
p95 6.3 s) vs RR 96 (6,137, p95 12.6 s) = 1.79×**.

**Temperature 0.5 goes the other way.** Sampling the worker from a softmax over the KV scores instead of taking the
argmax sends roughly a third of a session's turns to a worker without its prefix: throughput drops to 7,816 (0.81× of
default KV, only 1.15× RR), TTFT p95 doubles to 22 s (outside the 20 s budget) and 94 requests sit in flight. On this
workload the prefix is worth far more than the load smoothing a random choice buys, which is the same conclusion the
busy-stream sweep reached for temperature. The remaining cell, scale 2 / credit 0.8 (running), will show whether the
tuned gain is monotone in the load weight.

## iv. Simulation-vs-real gap, with the apple-to-apple decomposition

Method (same ladder as the disagg report and AGG24 §5.2): start from the published simulator, substitute one measured
quantity at a time, and watch which substitution moves the sim/real ratio toward 1.0. Script:
[`scripts/dynosim_agentx_decomp_agg.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_decomp_agg.py)
(agg engine, 6 × TP4, KV policy); table saved as
[`sim-results/agentx_decomp_agg.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_decomp_agg.txt).
Ratios are **sim ÷ measured**; **the TTFT standard is p95**. Measured request rate, output tokens and input tokens per
request come from the profiling-phase per-request records (`inflight_from_records.py` inputs), not from the summary CSV,
so the input side is exact. The decode substitution replaces the sim's piecewise agg curve (7 + 1.6·bs ms up to batch 7,
then 28.4 + 5.68·bs — the busy-stream refit) with a straight line through the two measured AgentX ITL p50 points
(5.8 + 1.33·bs ms; bs = in-flight per worker, 1.1 → 7.3 ms at 48 clients, 4.6 → 11.9 ms at 96).

### At 48 clients (measured: 0.78 req/s · 780 output tok/s = 32.5/GPU · 3,357 total/GPU · TTFT p95 3.83 s (p50 0.42) · ITL p50 7.3 ms · 6.8 in flight)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p95 | TPOT | sim/real: req/s · output · total · TTFT p95 |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 0.50 | 567 (23.6) | 1,466 | 2.85 s | 27.4 ms | 0.65× · 0.73× · **0.44×** · 0.74× |
| + output length = measured (×0.81) | 0.55 | 517 (21.5) | 1,612 | 3.25 s | 24.2 ms | 0.71× · 0.66× · 0.48× · 0.85× |
| + fixed 0.19 s per request (scheduling) | 0.55 | 517 (21.5) | 1,609 | 3.29 s | 24.4 ms | 0.70× · 0.66× · 0.48× · 0.86× |
| + decode = measured ITL line (5.8 + 1.33·bs, no cliff) | 0.64 | 632 (26.3) | 1,880 | 3.29 s | 14.3 ms | 0.82× · 0.81× · 0.56× · 0.86× |
| **residual after all substitutions** | | | | | | **0.82× requests · 0.56× total · TTFT p95 0.86× (slightly light) · TPOT still 2.0× too slow** |

### At 96 clients (measured: 1.92 req/s · 1,811 output tok/s = 75.5/GPU · 6,877 total/GPU · TTFT p95 5.36 s (p50 0.67) · ITL p50 11.9 ms · 27.5 in flight)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p95 | TPOT | sim/real: req/s · output · total · TTFT p95 |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 0.82 | 890 (37.1) | 2,375 | 3.71 s | 39.8 ms | 0.43× · 0.49× · **0.35×** · 0.69× |
| + output length = measured (×0.78) | 1.03 | 939 (39.1) | 3,008 | 3.72 s | 32.0 ms | 0.54× · 0.52× · 0.44× · 0.69× |
| + fixed 0.19 s per request (scheduling) | 0.97 | 869 (36.2) | 2,812 | 3.97 s | 34.1 ms | 0.51× · 0.48× · 0.41× · 0.74× |
| + decode = measured ITL line (5.8 + 1.33·bs, no cliff) | 1.24 | 1,139 (47.5) | 3,614 | 3.79 s | 18.6 ms | 0.64× · 0.63× · 0.53× · 0.71× |
| **residual after all substitutions** | | | | | | **0.64× requests · 0.53× total · TTFT p95 0.71× (light) · TPOT still 1.6× too slow** |

### Where the gap is, in plain words

1. **The decode cliff is the biggest single term on agg** (unlike disagg, where the engine model was within 10%). The
   published sim runs agg decode at 27–40 ms per token against 7–12 ms measured, because its busy-stream refit jumps to
   28.4 + 5.68·bs past batch 7. Removing the cliff lifts total tokens by 17–29% (0.48 → 0.56×, 0.41 → 0.53×) and is the
   reason the sim ranked RR ahead of KV on agg below the knee: the router that packs a session's turns onto one worker
   was being charged a decode penalty the real engine does not pay (measured: KV ahead 1.03× / 1.12×, §iii).
2. **Even with the measured line, the sim's mean TPOT is 1.6–2.0× too slow** (14.3 / 18.6 vs 7.3 / 11.9 ms). The line is
   right, so the sim must be running larger per-worker batches than the live fleet: it lets the KV policy pile many
   sessions' bursts onto the worker that holds their prefix while other workers idle, whereas the live Dynamo router
   (prefill-load scale 1, overlap credit 0) spreads bursts. This is the same over-packing that made the sim predict
   KV interactivity below RR on agg (P90 9 vs 15 at 192 clients); measured P90 is KV 102 vs RR 91 at 48 and 43 vs 34
   at 96.
3. **TTFT p95 is the one axis where the sim is slightly optimistic on agg** (0.71–0.86×), the opposite of disagg
   (2–2.7× pessimistic). On agg the prefill of a 100 k-token turn shares the worker with decode, and the sim's
   agg engine serialises prefill before decode without the chunked-prefill interference the live engine shows
   (measured p95 3.8–5.4 s at 0.4–0.7 s p50; sim tails are shorter because its prefill queue is per worker and
   requests do not lengthen each other's decode).
4. **The residual after the engine substitutions is trace representation, the same term as on disagg**: the sim's
   slice carries ~70 k input tokens per request against 102 k (48) and 85 k (96) measured (0.68× and 0.82×), and its
   sessions issue fewer requests per hour (0.82× and 0.64×) because the 4 k-request slice has no subagent fan-out and
   its sessions cycle through their turns more slowly than the live replay. Requests × input length reproduces the
   residual: 0.82 × 0.68 = 0.56 at 48, 0.64 × 0.82 = 0.53 at 96.

Take-away for using the sim on agg: keep its **rankings across topologies**, but on the **KV-vs-RR ordering below
the knee do not trust it** — the decode cliff and over-packing bias it against KV. For the sim-vs-real overlay (curve
page and `n3u-agentx-sim-vs-real.html`) the agg markers sit 1.8–2.9× above the dashed line on total tokens for these two
reasons in roughly equal parts.
