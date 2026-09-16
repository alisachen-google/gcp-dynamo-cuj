# Nemotron-3-Ultra 550B — disaggregated serving under the AgentX concurrency definition

72 GPU (GB300 NVL72), TP4/EP4, KV over NVLink (MNNVL + mooncake), SGLang 0.5.16 / Dynamo 1.4.2 /
FlashInfer 0.6.18, Weka 256K Claude-Code trace. Companion to D72_RESULTS.md (busy-stream results) and
AGENTX_COMPARISON.md (methodology). Status 2026-09-16 07:30 UTC: 2 of 6 KV points measured, ladder running.

Pages: [AgentX-concurrency curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html)
(y = total tok/s per GPU by default; toggles for output, TTFT p50/p95; RR on/off) · run index: RUN_INDEX.md.

## i. Concurrency: what we measured before, what AgentX measures, and where we misused the word

**Our previous definition (every point in D72_RESULTS.md / AGG24_RESULTS.md).** aiperf was launched with
`--concurrency C --no-fixed-schedule --ignore-trace-delays`. That is C independent request streams
that each fire the next request the instant the previous response finishes: think-time in the trace is
discarded, so C is a constant offered load, not a user count. Measured from the per-request records it
is even more than C requests in flight — the Weka sessions spawn subagent requests that run alongside
the parent turn, so the busy-stream 9:9 kv:48 run averaged **94 requests in flight (peak 425)**. It is
the right axis for finding a fleet's capacity knee (TTFT stationarity), and every knee, KV-vs-RR gain
and agg-vs-disagg ratio we published is on that axis. It is the wrong axis to compare with InferenceX's
numbers, and calling it "concurrency" next to theirs was the misuse: same word, ~6× different load
per unit.

**The AgentX definition** (`--scenario inferencex-agentx-mvp`, the only scenario registered in aiperf;
definition in `src/aiperf/common/scenario/inferencex_agentx_mvp.py`; trace in AGENTX_COMPARISON.md §5e).
C is the number of live Claude-Code session trees (root + subagents share one slot). Each lane snapshots
a session at a random point, primes its prefix in warm-up, then replays forward honouring the recorded
think-time between turns (10 s cap on whole-system idle only), recycling a fresh session when the tree
drains. Most sessions are idle at any instant. Measured: **48 clients ≈ 5.6 requests in flight, 96 ≈ 16**
— about one sixth of a busy-stream slot per client. Their 480–1,920 therefore maps to roughly our
busy-stream c40–c200, the band where our knees sit.

| run | definition | configured C | mean in-flight | peak | req/s | mean latency |
|---|---|---|---|---|---|---|
| 9:9 KV kv:48 | busy-stream (ours) | 48 streams | 93.8 | 425 | 7.68 | 12.2 s |
| 9:9 KV 48 clients | AgentX | 48 sessions | 5.6 | 17 | 0.79 | 7.0 s |
| 9:9 KV 96 clients | AgentX | 96 sessions | 16.1 | 40 | 2.08 | 7.7 s |

(`scripts/inflight_from_records.py` on `profile_export.jsonl`.) What we changed to adopt their definition:
one template (`manifests/perf/sgl-d72-agentx.yaml`) that runs the scenario with their launcher flags
(`--trajectory-start-min/max-ratio 0.25/0.75 --use-server-token-count --cache-bust first_turn_prefix`,
3600 s window), the same aiperf 0.12.0, same trace, same fleet manifests. Only `--warmup-requests-per-lane`
(their fork) is absent. Bench pods run on arm64 GPU nodes from 192 clients up (aiperf needs > 60 GB there).

## ii. Simulated curve under the AgentX definition, and the selected points

`scripts/dynosim_agentx.py` (same engine constants as the busy-stream DynoSim; lanes, warm-up at a random
25–75% start, recorded cadence anchored per lane, 10 s idle cap, per-play cache-bust, 1 h window),
sweep v2 = 5 splits × KV/RR × 10 client counts (`sim-results/dynosim_n3u_agentx_v2.csv`; 12:6 and 15:3
still computing at the time of writing). Output tok/s per GPU · TTFT p50:

| clients | 9:9 KV tok/s/GPU · TTFT p50 | 9:9 RR tok/s/GPU · TTFT p50 | 6:12 KV tok/s/GPU · TTFT p50 | 6:12 RR tok/s/GPU · TTFT p50 | 3:15 KV tok/s/GPU · TTFT p50 | 3:15 RR tok/s/GPU · TTFT p50 |
|---|---|---|---|---|---|---|
| 48 | 12.0 · 0.13 s | 11.8 · 1.84 s | 12.0 · 0.13 s | 11.8 · 1.24 s | 12.0 · 0.15 s | 11.7 · 0.94 s |
| 96 | 23.6 · 0.16 s | 22.6 · 3.16 s | 23.6 · 0.18 s | 21.9 · 3.95 s | 23.2 · 0.57 s | 19.5 · 8.45 s |
| 192 | 46.5 · 0.26 s | 34.4 · 12.70 s | 46.0 · 0.76 s | 27.1 · 26.86 s | 38.4 · 10.10 s | 20.5 · 60.37 s |
| 384 | 83.3 · 1.83 s | 34.0 · 63.29 s | 68.4 · 13.03 s | 26.2 · 108.29 s | 36.0 · 56.44 s | 19.0 · 140.26 s |
| 480 | 92.1 · 4.09 s | 33.1 · 80.06 s | 65.6 · 25.24 s | 25.5 · 120.45 s | 34.5 · 78.68 s | 17.0 · 181.55 s |
| 768 | 88.6 · 20.16 s | 30.6 · 142.59 s | 60.9 · 57.13 s | 23.4 · 212.09 s | 30.6 · 137.77 s | 16.3 · 285.54 s |
| 960 | 85.1 · 29.00 s | 27.9 · 195.19 s | 57.3 · 78.59 s | 22.4 · 280.64 s | 27.9 · 182.71 s | 10.8 · 418.18 s |
| 1440 | 71.1 · 56.62 s | 23.4 · 306.84 s | 48.2 · 126.26 s | 13.4 · 452.52 s | 22.8 · 232.11 s | 4.5 · 724.60 s |
| 1536 | 70.6 · 59.58 s | 22.0 · 327.38 s | 45.8 · 149.21 s | 11.8 · 474.39 s | 18.7 · 302.22 s | 4.2 · 776.22 s |
| 1920 | 61.5 · 83.76 s | 16.0 · 416.25 s | 41.3 · 171.26 s | 8.5 · 605.61 s | 9.5 · 497.30 s | 3.4 · 935.90 s |

Reading: under replayed think-time the fleet is nowhere near its knee below ~200 clients; the sim puts
the 9:9 KV peak at **92/GPU around 480 clients** (TTFT p50 4 s) and 6:12 at 68/GPU around 384; RR peaks
at ~34/GPU (9:9) / ~27 (6:12) near 192–384 and then queues. The sim also says 9:9 > 6:12 > 3:15 on this
axis, the same ordering the busy-stream silicon gave.

Selected points (why): the measured KV ladder is 48 / 96 / 192 / 384 / 768 / 1536 (the sim's rise, peak
and decline, plus the two we could compare with their 480–1,920 range); RR is measured at **96 and 384**
only — 96 is the last client count where the sim has RR still stationary, 384 is where KV peaks and RR
is already deep in its queue, so the pair brackets the KV-vs-RR gap without spending 8 h on a full RR
ladder.

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

Headline (9:9 KV): on **output tokens** the sim is 1.10× real at 48 clients and 0.84× at 96; on
**total tokens per GPU** it is 0.61× / 0.55×; on **request rate** 0.89× / 0.68×; TTFT p50 is
under-predicted 2.4× (0.13 vs 0.32 s). Substituting measured quantities one at a time
(`scripts/dynosim_agentx_decomp.py`, same method as AGG24 §5.2):

| sim variant | clients | req/s | output tok/s (/GPU) | total/GPU | TTFT p50 | TPOT | sim/real: req/s · output · total · TTFT |
|---|---|---|---|---|---|---|---|
| v1 (AIC-seeded, as published) | 48 | 0.71 | 866 (12.0) | 691 | 0.13 s | 6.6 ms | 0.89× · **1.10×** · 0.61× · 0.42× |
| v1 | 96 | 1.41 | 1,699 (23.6) | 1,373 | 0.16 s | 7.2 ms | 0.68× · 0.84× · 0.55× · 0.52× |
| + output length = measured (×0.81) | 48 | 0.72 | 720 (10.0) | 701 | 0.13 s | 6.5 ms | 0.90× · 0.92× · 0.62× · 0.42× |
| + output length = measured | 96 | 1.43 | 1,402 (19.5) | 1,393 | 0.16 s | 7.1 ms | 0.69× · 0.69× · 0.56× · 0.52× |
| + TTFT floor 0.19 s (transfer + scheduling) | 48 | 0.72 | 719 (10.0) | 700 | 0.32 s | 6.5 ms | 0.90× · 0.91× · 0.62× · **1.01×** |
| + TTFT floor 0.19 s | 96 | 1.43 | 1,400 (19.4) | 1,390 | 0.35 s | 7.1 ms | 0.69× · 0.69× · 0.56× · 1.14× |
| + decode = measured ITL (5.9 + 0.8·bs ms) | 48 | 0.72 | 713 (9.9) | 695 | 0.32 s | 7.1 ms | 0.90× · 0.91× · 0.62× · 1.00× |
| + decode = measured ITL | 96 | 1.42 | 1,381 (19.2) | 1,375 | 0.35 s | 8.3 ms | 0.68× · 0.68× · 0.55× · 1.13× |
| **measured** | 48 | 0.80 | 786 (10.9) | 1,128 | 0.32 s | 6.4 ms | |
| **measured** | 96 | 2.08 | 2,019 (28.0) | 2,497 | 0.31 s | 7.4 ms | |

What the ladder says, step by step:
1. **Output length.** The sim's requests emit 1,205–1,218 output tokens on average, the engine emits
   971–995 (it stops at EOS earlier than the recorded length). Scaling the sim's outputs to the measured
   mean removes the apparent 1.10× "over-prediction" at 48 clients — it was longer outputs, not a faster
   engine — and leaves a clean 0.92× / 0.69×.
2. **TTFT.** A fixed 0.19 s per request closes the TTFT gap exactly at 48 clients (1.01×) and to 1.14× at
   96. The engine's prefill *rate* is right; what the sim lacks is a constant per-request cost
   (KV hand-off over NVLink plus router/scheduler latency, ≈ the 0.13 s TTFT floor aiperf sees on every
   run plus transfer). This is a model omission, not an engine surprise.
3. **Decode.** Substituting the measured ITL curve changes nothing material (TPOT 6.5 → 7.1 ms); the
   decode model is already right at these batch sizes.
4. **What remains is the workload representation, not the engine.** After all three substitutions the sim
   still issues only 0.90× (48) / 0.69× (96) of the measured *requests per second*, and its requests carry
   ~69 k input tokens versus the ~85–101 k aiperf actually sends, which is why the total-token ratio stays
   at 0.55–0.62×. The sim replays a 4,000-request slice of the trace (874 sessions) with a per-lane cadence
   anchored at the lane start; aiperf reconstructs all 393 sessions with their subagent streams and issues
   parallel subagent requests inside a turn. Both effects grow with clients (0.90× → 0.69×), which is
   consistent with subagent fan-out being the missing term. Fix queued as sim v3: build the sim trace from
   aiperf's own reconstruction (`inputs.json` / per-request ISL and timestamps of a real run) instead of the
   4 k slice.

Bottom line: the engine-side model (prefill rate, decode curve) is within ~10% under the AgentX definition
once a 0.19 s per-request floor is added; the remaining 30–45% gap on requests and total tokens is the
trace representation, and it under-predicts silicon rather than flattering it.
