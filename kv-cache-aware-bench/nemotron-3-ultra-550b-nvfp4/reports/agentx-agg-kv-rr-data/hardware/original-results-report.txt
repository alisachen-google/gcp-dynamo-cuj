# Nemotron-3-Ultra 550B — aggregated serving (24 GPU) under the AgentX concurrency definition

Current simulation source: [AIPerf replay](reports/agentx-aiperf-results.md). Hardware measurements remain in section iii; the old simulation curves and gap interpretation have moved to the historical archive.

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

For agg the busy-stream reference is the new-stack bounded peak **KV c32 = 1,854 tok/s = 77.3/GPU,
13,320 total tok/s/GPU** (AGG24 §4.5); on that axis KV knees between c32 and c64 and RR between c32 and
c64. Under the AgentX definition those knees move to hundreds of clients.

## ii. Current simulation performance through AIPerf

The handwritten v3/v5 predictions are superseded. Use [the actual AIPerf replay results](reports/agentx-faithful-replay.md) and [current result index](reports/agentx-aiperf-results.md). The [historical curves and analysis](reports/agentx-agg-legacy-analysis.md) retain the old predictions and point-selection rationale; their topology, knee and gap claims are not current evidence.

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
| kvs2c08: scale 2.0, credit 0.8 | 10,241 (+6%) | 2,458 (102.4) | 1.10 / 8.11 / 14.5 s | 22.0 / 44.8 → 22.3 | 70.0 (peak 111) | yes (q1 1.06 → q4 1.02 s) | [1789575514](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789575514_alisachen-n3u-agg-ns-agentx-kvs2c08-c192) |
| kvt05: temperature 0.5 | 7,816 (−19%) | 1,908 (79.5) | 5.0 / 22.1 / 31.6 s | 33.3 / 77.7 → 12.9 | 93.6 (peak 131) | yes (q1 4.5 → q4 2.5 s) | [1789574468](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789574468_alisachen-n3u-agg-ns2-agentx-kvt05-c192) |
| **kvs3c08 at 96 clients** (below the knee; fleet 2) | **7,045 (+3% vs KV 96 = 6,844)** | 1,880 (78.3) | 0.47 / **3.11** / 6.7 s | 10.9 / 18.3 → **54.6** | 24.1 (peak 50) | yes (q1 0.45 → q4 0.48 s) | [1789582550](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789582550_alisachen-n3u-agg-ns2-agentx-kvs3c08-c96) |
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
busy-stream sweep reached for temperature. Scale 2 / credit 0.8 sits between the two: +6% total (10,241), TTFT p95 8.1 s, P90 22.3, 70 in flight — so the gain is
monotone in the prefill-load weight over the range tried (scale 1 → 2 → 3 = 9,655 → 10,241 → 11,012; p95 11.7 → 8.1 →
6.3 s). A scale-4 or scale-5 cell would tell whether it has peaked; it is not queued. Adopted setting for the agg
recipe under AgentX load: **scale 3, credit 0.8, temperature 0**.

**Below the knee the tuned router still does not lose.** At 96 clients (4 per GPU, where the sim predicted −13 to
−26% for the tuned router on agg) it delivers 7,045 total/GPU vs 6,844 default (+3%, inside run-to-run noise), with
TTFT p95 3.1 vs 5.4 s (−42%), P90 interactivity 54.6 vs 42.6 (+28%) and 24 vs 27.5 in flight. So the flag is a
latency and interactivity win at every load tried and a throughput win once the prefill queues start to form; there is
no regime on agg where the default beats it. Against RR at 96 (6,137, p95 12.6 s) the tuned router is 1.15× on tokens
with a 4× shorter tail, and the same-SLO pair under the P90 ≥ 20 budget becomes tuned KV 96 (7,045, P90 55) vs RR 96
(6,137, P90 34) = 1.15×.

## iv. Current replay versus real performance

The [audited gap report](reports/agentx-faithful-replay.md) compares the actual AIPerf workload with matching real agg runs. RR throughput errors are −3.5%, −0.4%, +29.0% and +118.2% at 48, 96, 192 and 384 clients. Matching low-load throughput does not validate the serving model near saturation.

For KV192, adding the projected decode-block load term reduces the throughput error from −27.7% to −5.9%; latency and cache gaps remain. This one-point calibration needs validation on another concurrency. The RR pool-capacity ablation leaves large high-load errors, pointing to further cache and scheduling model work. See the linked report for exact metrics, model variants and provenance.
