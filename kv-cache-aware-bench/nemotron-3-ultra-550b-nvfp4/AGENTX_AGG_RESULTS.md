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
| agg KV | 192 / 384 / 768 / 1536 | | | | | | | running (192 started 10:47) |
| agg RR | 48 | 752 (31.3) | 3,250 | 0.79 / 8.27 / 15.3 s | 7.5 / 11.0 ms (P90 90.9) | 8.5 (peak 22) | stationary | complete 10:42 UTC (second fleet `n3u-agg-ns2`) |
| agg RR | 96 / 192 / 384 / 768 / 1536 | | | | | | | running in parallel (96 started 10:42) |

An earlier agg AgentX smoke at 48 clients (2026-09-15) failed on [`--warmup-requests-per-lane`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/config.py#L362), a flag that
exists only in SemiAnalysis's aiperf fork; the template was corrected and no agg AgentX result predates
this report. First measured agg point vs the sim's agg KV cell at 48 clients: total 3,334 vs 1,474 per GPU (sim 0.44×), TTFT p50 0.42 vs 0.13 s, P90 interactivity 102 vs 22 tok/s/user — the agg simulator is far more pessimistic than the disagg one at low load (its refit decode curve 8.9 + 1.73·bs ms is a busy-stream fit; under replayed think-time the workers run near batch 1 and the real ITL is 7.3 ms). Mean in-flight 6.8 requests (peak 18) from the per-request records. Rows fill as points land; KV-vs-RR uses 192 (same config) and 192-vs-96 (TTFT budget). **First measured KV-vs-RR pair on agg
(48 clients, same config, both stationary): KV 3,334 vs RR 3,250 total tok/s per GPU = 1.03× on tokens; TTFT p95 3.83 vs 8.27 s
(RR 2.2× worse); P90 interactivity 102 vs 91.** The sim had predicted 0.82× on tokens and a 2.3× TTFT p95 gap at 48 clients:
the latency ratio is right, the token ratio is not (measured KV does not lose throughput to affinity at this load). From 48 to
96 clients agg KV scaled 2.05× on total tokens with TTFT p95 5.4 s and in-flight 27.5, i.e. agg carries more requests in flight
per client than 9:9 (16 at 96) because its per-request latency is ~2× longer (14.3 s vs 7.7 s).

## iv. Simulation-vs-real gap, with the apple-to-apple decomposition

Pending the first measured agg points. The decomposition ladder ([`scripts/dynosim_agentx_decomp.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_decomp.py))
will be run with the agg engine model: v1 → output length = measured → TTFT floor → decode = measured
(the busy-stream agg decomposition in AGG24 §5.2 found decode to be the whole gap, 8.9 + 1.73·bs ms
vs the AIC 5.26 + 0.277·bs; the AgentX sim already uses the refit constants). For the disagg arm the
same ladder shows the engine model within ~10% and the residual to be trace representation
(AGENTX_D72_RESULTS.md §iv); the agg arm is expected to share that residual since it replays the same
4 k-request slice.
