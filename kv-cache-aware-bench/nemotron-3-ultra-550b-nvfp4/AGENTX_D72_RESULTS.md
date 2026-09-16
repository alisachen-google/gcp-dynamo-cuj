# Nemotron-3-Ultra 550B — disaggregated serving under the AgentX concurrency definition

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

## ii. Simulated curve under the AgentX definition, and the selected points

[`scripts/dynosim_agentx.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx.py) (same engine constants as the busy-stream DynoSim; lanes, warm-up at a random
25–75% start, recorded cadence anchored per lane, 10 s idle cap, per-play cache-bust, 1 h window),
sweep v2 = 5 splits × KV/RR × 10 client counts ([`sim-results/dynosim_n3u_agentx_v2.csv`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_agentx_v2.csv); 12:6 and 15:3
still computing at the time of writing). Total tok/s per GPU (output tok/s per GPU) · TTFT p50 · P90 interactivity:

Cell = **total tok/s per GPU** (output tok/s per GPU) · TTFT p50 · P90 interactivity (tok/s/user = 1000 / per-request TPOT p90). Sim v3 (`sim-results/dynosim_n3u_agentx_v3.csv`): input tokens counted exactly per simulated request (len(hash_ids)×64), the InferenceX total-token convention.

| clients | 9:9 KV | 9:9 RR | 6:12 KV | 6:12 RR | 12:6 KV | 12:6 RR |
|---|---|---|---|---|---|---|
| 48 | **701** (12) · 0.13 s · 148 | **681** (12) · 1.84 s · 148 | **701** (12) · 0.13 s · 148 | **681** (12) · 1.24 s · 148 | **699** (12) · 0.13 s · 139 | **679** (12) · 2.34 s · 132 |
| 96 | **1,395** (24) · 0.16 s · 125 | **1,336** (23) · 3.16 s · 114 | **1,395** (24) · 0.18 s · 132 | **1,296** (22) · 3.95 s · 125 | **1,389** (23) · 0.15 s · 109 | **1,342** (23) · 2.88 s · 100 |
| 192 | **2,733** (47) · 0.26 s · 93 | **2,047** (34) · 12.70 s · 78 | **2,701** (46) · 0.76 s · 100 | **1,631** (27) · 26.86 s · 86 | **2,679** (45) · 0.19 s · 72 | **2,271** (38) · 6.51 s · 63 |
| 384 | **4,937** (83) · 1.83 s · 54 | **2,062** (34) · 63.29 s · 45 | **4,064** (68) · 13.03 s · 63 | **1,577** (26) · 108.29 s · 54 | **4,657** (78) · 0.45 s · 40 | **2,508** (41) · 34.52 s · 34 |
| 480 | **5,501** (92) · 4.09 s · 44 | **2,021** (33) · 80.06 s · 37 | **3,922** (66) · 25.24 s · 50 | **1,544** (26) · 120.45 s · 46 | **5,217** (87) · 0.63 s · 32 | **2,465** (40) · 49.34 s · 27 |
| 768 | **5,469** (89) · 20.16 s · 27 | **1,926** (31) · 142.59 s · 25 | **3,732** (61) · 57.13 s · 33 | **1,405** (23) · 212.09 s · 32 | **5,981** (95) · 1.56 s · 19 | **2,345** (36) · 103.46 s · 18 |
| 960 | **5,380** (85) · 29.00 s · 22 | **1,789** (28) · 195.19 s · 21 | **3,589** (57) · 78.59 s · 27 | **1,446** (22) · 280.64 s · 26 | **6,139** (94) · 2.35 s · 15 | **2,275** (33) · 130.92 s · 14 |
| 1440 | **4,884** (71) · 56.62 s · 15 | **1,720** (23) · 306.84 s · 14 | **3,202** (48) · 126.26 s · 19 | **916** (13) · 452.52 s · 19 | **6,187** (85) · 4.74 s · 10 | **2,087** (28) · 207.30 s · 10 |
| 1536 | **4,884** (71) · 59.58 s · 14 | **1,665** (22) · 327.38 s · 14 | **3,050** (46) · 149.21 s · 18 | **808** (12) · 474.39 s · 17 | **6,144** (82) · 5.58 s · 9 | **2,085** (27) · 230.90 s · 9 |
| 1920 | **4,544** (61) · 83.76 s · 11 | **1,189** (16) · 416.25 s · 11 | **2,916** (41) · 171.26 s · 14 | **523** (8) · 605.61 s · 14 | **5,945** (74) · 12.26 s · 8 | **1,928** (23) · 305.91 s · 7 |

| clients | 3:15 KV | 3:15 RR | 15:3 KV | 15:3 RR |
|---|---|---|---|---|
| 48 | **696** (12) · 0.15 s · 157 | **677** (12) · 0.94 s · 148 | **690** (12) · 0.13 s · 114 | **669** (12) · 2.60 s · 114 |
| 96 | **1,373** (23) · 0.57 s · 132 | **1,166** (20) · 8.45 s · 125 | **1,349** (23) · 0.15 s · 76 | **1,296** (22) · 2.77 s · 72 |
| 192 | **2,276** (38) · 10.10 s · 100 | **1,219** (20) · 60.37 s · 93 | **2,364** (39) · 0.18 s · 41 | **2,158** (36) · 3.66 s · 39 |
| 384 | **2,150** (36) · 56.44 s · 64 | **1,149** (19) · 140.26 s · 63 | **3,218** (51) · 0.20 s · 20 | **2,704** (43) · 7.89 s · 19 |
| 480 | **2,074** (35) · 78.68 s · 54 | **1,027** (17) · 181.55 s · 53 | **3,370** (52) · 0.22 s · 16 | **2,760** (42) · 11.58 s · 15 |
| 768 | **1,892** (31) · 137.77 s · 38 | **1,045** (16) · 285.54 s · 38 | **3,612** (50) · 0.23 s · 10 | **2,721** (37) · 29.16 s · 9 |
| 960 | **1,706** (28) · 182.71 s · 32 | **715** (11) · 418.18 s · 32 | **3,695** (48) · 0.24 s · 8 | **2,637** (34) · 45.87 s · 8 |
| 1440 | **1,646** (23) · 232.11 s · 23 | **229** (4) · 724.60 s · 23 | **3,802** (45) · 0.29 s · 5 | **2,416** (28) · 87.09 s · 5 |
| 1536 | **1,393** (19) · 302.22 s · 21 | **197** (4) · 776.22 s · 21 | **3,809** (44) · 0.29 s · 5 | **2,379** (27) · 96.18 s · 5 |
| 1920 | **629** (9) · 497.30 s · 17 | **124** (3) · 935.90 s · 17 | **3,803** (42) · 0.34 s · 4 | **2,235** (24) · 130.39 s · 4 |

Reading (total-token axis): the prefill-heavier splits win once input tokens are counted, because
every request carries ~70 k input tokens against ~1 k output and the prefill tier is what turns them into
served tokens. The sim's total-token peaks: **12:6 KV 6,187 total/GPU at 1440 clients (TTFT p50 4.7 s, 85 output/GPU)**; 9:9 KV 5,501 total/GPU at 480 clients (TTFT p50 4.1 s, 92 output/GPU);
6:12 KV 4,064 total/GPU at 384 clients (TTFT p50 13.0 s, 68 output/GPU); 3:15 KV 2,276 total/GPU at 192 clients (TTFT p50 10.1 s, 38 output/GPU); 15:3 KV 3,809 total/GPU at 1536 clients (TTFT p50 0.3 s, 44 output/GPU) — 15:3 never queues
(TTFT p50 ≤ 0.3 s to 1,920 clients) because 15 prefill workers absorb the whole trace, but its 3 decode workers
cap interactivity (P90 falls to 4 tok/s/user). RR peaks at 2,062 total/GPU at 384 clients (TTFT p50 63.3 s, 34 output/GPU) on 9:9 and 2,508 total/GPU at 384 clients (TTFT p50 34.5 s, 41 output/GPU) on
12:6, then queues. On the output-token axis (the numbers in parentheses) the ordering reverts to 9:9 ≥ 12:6 >
6:12, the same as the busy-stream silicon. Interactivity: the disagg splits hold P90 interactivity above 100
tok/s/user up to ~200 clients and above 40 up to ~1,000 (decode batches stay small), versus agg's 20–60.

Selected points (from the v3 sim knees; `scripts/dynosim_agentx.py` + the knee/SLO scan below):

| framing | rule | 9:9 KV | 9:9 RR | sim gain (total tok/s/GPU) |
|---|---|---|---|---|
| knees | throughput-slope knee (marginal gain < 25% of the initial slope) | ≈ 480 clients (TTFT p50 4.1 s) | ≈ 192 clients (TTFT p50 12.7 s, RR's cache-miss prefill saturates) | — |
| **same config** | RR's knee, the highest client count where RR still scales | 192 → 2,733 (p50 0.3 s) | 192 → 2,047 (p50 12.7 s) | **1.34×**, and 42× lower TTFT |
| same config, both bounded | last count where both are pre-knee | 96 → 1,395 (0.2 s) | 96 → 1,336 (3.2 s) | 1.04× (parity on tokens; KV wins only TTFT) |
| **same SLO (TTFT p95 ≤ 20 s)** | each policy's best throughput under the budget | 480 → 5,501 (p95 18 s) | 96 → 1,336 (p95 16 s) | **4.1×** |
| same SLO (P90 interactivity ≥ 20 tok/s/user) | each policy's best throughput above the floor | 480 → 5,501 (P90 44) | 384 → 2,062 (P90 45) | 2.7× |

So the measured KV ladder is 48 / 96 / 192 / 384 / 768 / 1536 and the measured RR points are **192** (same-config
at RR's knee), **96** (RR's best under the 20 s TTFT-p95 budget, and the both-bounded config) and **384** (RR's best
under the interactivity floor); the runner is queued in that order. Expectation from the busy-stream silicon
(KV/RR 2.02× at c48, 1.94× at c96) is that the measured same-config gain lands above the sim's 1.34×, because
the sim's RR under-counts the prefix-miss penalty (hit 0.30 vs KV 0.74).

## iii. Real runs: performance, curve, and the KV-vs-RR points

| arm | clients | output tok/s (/GPU) | total tok/s/GPU | TTFT p50 / p95 / p99 | ITL p50 / p90 | in-flight | knee | guard |
|---|---|---|---|---|---|---|---|---|
| 9:9 KV | 48 | 786 (10.9) | 1,128 | 0.32 / 1.5 / 3.8 s | 6.4 / 6.8 ms | 5.6 | stationary | PASS |
| 9:9 KV | 96 | 2,019 (28.0) | 2,497 | 0.31 / 1.42 / 4.3 s | 7.4 / 8.4 ms | 16.1 | stationary | PASS |
| 9:9 KV | 192 | running (07:20 UTC redeploy; first attempt evicted at 59 GB, second invalidated by a transfer failure during an operator-caused ComputeDomain deletion window) | | | | | | |
| 9:9 KV | 384 / 768 / 1536 | queued | | | | | | |
| 9:9 RR | 96 / 384 | queued behind the KV ladder | | | | | | |

Throughput scaled 2.57× for 2× clients between 48 and 96 with TTFT flat at 0.31 s, i.e. the fleet is
still in its linear region. Per-user interactivity is very high in this regime (P90 147 tok/s/user at 48,
119 at 96) because decode batches are tiny. The KV-vs-RR comparison table will be filled from the
96/384 pairs when the RR points land; the busy-stream KV/RR gains (2.02× at c48, 1.94× at c96, both
instance-2) are the reference expectation.

Measured points are overlaid on the curve page as solid markers; the run index links every artifact.

## iv. Simulation-vs-real gap, with the apple-to-apple decomposition

Method (same as AGG24 §5.2): start from the published simulator, substitute one measured quantity at a time,
and watch which substitution moves the sim/real ratio toward 1.0. Script: `scripts/dynosim_agentx_decomp.py`.
Ratios are **sim ÷ measured**; 1.00× is exact, below 1 means the sim under-predicts. Each concurrency is
decomposed separately so the two ladders are apples to apples at their own load.

### At 48 clients (measured: 0.80 req/s · 786 output tok/s = 10.9/GPU · 1,128 total/GPU · TTFT p50 0.32 s · ITL p50 6.4 ms)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p50 | TPOT | sim/real: req/s · output · total · TTFT |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 0.71 | 866 (12.0) | 691 | 0.13 s | 6.6 ms | 0.89× · **1.10×** · 0.61× · 0.42× |
| + output length = measured (×0.81) | 0.72 | 720 (10.0) | 701 | 0.13 s | 6.5 ms | 0.90× · 0.92× · 0.62× · 0.42× |
| + fixed 0.19 s per request (KV hand-off + scheduling) | 0.72 | 719 (10.0) | 700 | 0.32 s | 6.5 ms | 0.90× · 0.91× · 0.62× · **1.01×** |
| + decode = measured ITL curve (5.9 + 0.8·bs ms) | 0.72 | 713 (9.9) | 695 | 0.32 s | 7.1 ms | 0.90× · 0.91× · 0.62× · 1.00× |
| **residual after all substitutions** | | | | | | **0.90× requests · 0.62× total tokens** |

### At 96 clients (measured: 2.08 req/s · 2,019 output tok/s = 28.0/GPU · 2,497 total/GPU · TTFT p50 0.31 s · ITL p50 7.4 ms)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p50 | TPOT | sim/real: req/s · output · total · TTFT |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 1.41 | 1,699 (23.6) | 1,373 | 0.16 s | 7.2 ms | 0.68× · 0.84× · 0.55× · 0.52× |
| + output length = measured (×0.81) | 1.43 | 1,402 (19.5) | 1,393 | 0.16 s | 7.1 ms | 0.69× · 0.69× · 0.56× · 0.52× |
| + fixed 0.19 s per request | 1.43 | 1,400 (19.4) | 1,390 | 0.35 s | 7.1 ms | 0.69× · 0.69× · 0.56× · 1.14× |
| + decode = measured ITL curve | 1.42 | 1,381 (19.2) | 1,375 | 0.35 s | 8.3 ms | 0.68× · 0.68× · 0.55× · 1.13× |
| **residual after all substitutions** | | | | | | **0.69× requests · 0.55× total tokens** |

### Where the gap is, in plain words

1. **The engine model is not the gap.** Three things describe the engine: how fast it prefills, how fast it
   decodes, and a fixed cost per request. Prefill rate needs no correction (once the fixed cost is added the
   TTFT matches to 1%). Decode needs no correction (swapping in the measured ITL curve changes throughput by
   under 1%). The only engine-side omission is a **constant 0.19 s per request** — the KV hand-off over NVLink
   plus router and scheduler latency — which the sim did not model and which explains the whole TTFT gap.
2. **The apparent 1.10× "over-prediction" at 48 clients was an artefact of output length.** The sim replays the
   trace's recorded outputs (about 1,210 tokens per request); the real engine stops at end-of-sequence after
   about 980. Correcting for that, the sim is *under*, not over, silicon on output tokens at both loads.
3. **What remains is how the workload is represented, and it is the whole residual.** After every engine
   substitution the sim still issues only 90% (48 clients) and 69% (96 clients) of the requests per second
   that aiperf actually sends, and each simulated request carries about 69k input tokens where aiperf sends
   85k to 101k. Multiplying those two shortfalls gives the total-token ratio of 0.62× and 0.55×. The cause is
   the trace the sim reads: a 4,000-request slice (874 sessions) replayed one session per lane with a
   per-lane cadence, whereas aiperf reconstructs all 393 sessions with their subagent streams and issues
   subagent requests in parallel with the parent turn. Fan-out is why the shortfall grows with clients.
4. **Direction and consequence.** The residual makes the sim *pessimistic* about absolute total throughput
   (by 1.6 to 1.8×) but leaves the ordering of arms and policies intact, because every arm is fed the same
   under-sized trace. Use the sim for ranking topologies and routers; use the measured markers for level.
5. **The fix is data, not modelling.** Sim v4 will build its trace from aiperf's own reconstruction of a real
   run (per-request input length and timestamps from `profile_export.jsonl`, subagent requests included), which
   removes items 2 and 3 at once. The 0.19 s per-request cost becomes a constant in `dynosim_pd.py`.

