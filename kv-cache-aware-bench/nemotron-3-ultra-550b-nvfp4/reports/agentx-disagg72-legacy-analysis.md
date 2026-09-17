# Historical disagg72 AgentX simulation curves and analysis

Historical material superseded on 2026-09-17. These results used handwritten AgentX workload models and must not be used as current AIPerf predictions. Statements about parity, topology rankings, and calibration below record the earlier interpretation; the [current replay audit](agentx-faithful-replay.md) revises it.

[Current simulation results](agentx-aiperf-results.md)

---

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

**Topology for the measured ladder: 12:6**, the sim's optimum on total tokens per GPU (the 9:9 points at 48 / 96 / 192 were
run before that result and stay as a cross-check). On 12:6 the same rules give: same config **192** (sim KV 2,679 vs RR
2,271 = 1.18×, TTFT 0.2 vs 6.5 s), both-bounded 96 (1.04×), same SLO TTFT p95 ≤ 20 s **KV 480 (5,217) vs RR 96 (1,342) = 3.9×**,
P90 ≥ 20 tok/s/user KV 480 vs RR 384 (2,508) = 2.1×. The measured 12:6 KV ladder is 96 / 192 / 384 / 480 / 768 / 1440
(480 and 1440 are the sim's same-SLO and peak cells) and the 12:6 RR points are **192, 96, 384**, queued in that order
behind the last 9:9 point. Re-analysis with tuned KV (KNEE_ANALYSIS.md, AgentX section) confirms these cells and adds
that tuned KV's best cell under the same 20 s budget is 768 (6,210, +19% over KV 480) — an optional extra run. Expectation from the busy-stream silicon
(KV/RR 2.02× at c48, 1.94× at c96) is that the measured same-config gain lands above the sim's 1.34×, because
the sim's RR under-counts the prefix-miss penalty (hit 0.30 vs KV 0.74).

## iv. Simulation-vs-real gap, with the apple-to-apple decomposition

Method (same as AGG24 §5.2): start from the published simulator, substitute one measured quantity at a time,
and watch which substitution moves the sim/real ratio toward 1.0. Script: `scripts/dynosim_agentx_decomp.py`.
Ratios are **sim ÷ measured**; 1.00× is exact, below 1 means the sim under-predicts. **The TTFT standard for
sim-vs-real is p95** (the tail a user sees), reported alongside the throughput terms. Each client count is
decomposed separately so the two are apples to apples at their own load.

### At 48 clients (measured: 0.80 req/s · 786 output tok/s = 10.9/GPU · 1,128 total/GPU · TTFT p95 1.50 s (p50 0.32) · ITL p50 6.4 ms)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p95 | TPOT | sim/real: req/s · output · total · TTFT p95 |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 0.71 | 866 (12.0) | 691 | 2.92 s | 6.6 ms | 0.89× · **1.10×** · 0.61× · **1.95×** |
| + output length = measured (×0.81) | 0.72 | 720 (10.0) | 701 | 2.91 s | 6.5 ms | 0.90× · 0.92× · 0.62× · 1.94× |
| + fixed 0.19 s per request (KV hand-off + scheduling) | 0.72 | 719 (10.0) | 700 | 2.96 s | 6.5 ms | 0.90× · 0.91× · 0.62× · 1.97× |
| + decode = measured ITL curve (5.9 + 0.8·bs ms) | 0.72 | 713 (9.9) | 695 | 3.13 s | 7.1 ms | 0.90× · 0.91× · 0.62× · 2.09× |
| **residual after all substitutions** | | | | | | **0.90× requests · 0.62× total · TTFT p95 2.1× too pessimistic** |

### At 96 clients (measured: 2.08 req/s · 2,019 output tok/s = 28.0/GPU · 2,497 total/GPU · TTFT p95 1.42 s (p50 0.31) · ITL p50 7.4 ms)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p95 | TPOT | sim/real: req/s · output · total · TTFT p95 |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 1.41 | 1,699 (23.6) | 1,373 | 3.52 s | 7.2 ms | 0.68× · 0.84× · 0.55× · 2.48× |
| + output length = measured (×0.81) | 1.43 | 1,402 (19.5) | 1,393 | 3.76 s | 7.1 ms | 0.69× · 0.69× · 0.56× · 2.65× |
| + fixed 0.19 s per request | 1.43 | 1,400 (19.4) | 1,390 | 3.90 s | 7.1 ms | 0.69× · 0.69× · 0.56× · 2.75× |
| + decode = measured ITL curve | 1.42 | 1,381 (19.2) | 1,375 | 3.76 s | 8.3 ms | 0.68× · 0.68× · 0.55× · 2.65× |
| **residual after all substitutions** | | | | | | **0.69× requests · 0.55× total · TTFT p95 2.7× too pessimistic** |

### At 192 clients (measured: 3.50 req/s · 3,267 output tok/s = 45.4/GPU · 4,600 total/GPU · TTFT p95 2.03 s (p50 0.36) · ITL p50 8.5 ms · 30 in flight)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p95 | TPOT | sim/real: req/s · output · total · TTFT p95 |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 2.76 | 3,349 (46.5) | 2,694 | 5.38 s | 8.7 ms | 0.79× · **1.03×** · 0.59× · **2.65×** |
| + output length = measured (×0.78) | 2.87 | 2,717 (37.7) | 2,792 | 5.75 s | 8.3 ms | 0.82× · 0.83× · 0.61× · 2.83× |
| + fixed 0.19 s per request | 2.85 | 2,706 (37.6) | 2,780 | 6.32 s | 8.4 ms | 0.82× · 0.83× · 0.60× · 3.11× |
| + decode = measured ITL curve | 2.75 | 2,597 (36.1) | 2,674 | 5.52 s | 11.6 ms | 0.79× · 0.79× · 0.58× · 2.72× |
| **residual after all substitutions** | | | | | | **0.79× requests · 0.58× total · TTFT p95 2.7× too pessimistic** |

### The 12:6 ladder (96 / 192 / 384 / 480 clients), same substitutions, run 2026-09-16 19:05 UTC

Script [`scripts/dynosim_agentx_decomp_126.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_decomp_126.py);
table [`sim-results/agentx_decomp_126.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_decomp_126.txt).
Measured inputs come from the profiling-phase per-request records (request rate, input and output tokens per request);
the decode line 6.9 + 0.44·bs ms is fitted through the four measured ITL p50 points (bs = in flight per decode worker).

| sim variant → sim ÷ real (req/s · output · total · TTFT p95) | 96 | 192 | 384 | 480 |
|---|---|---|---|---|
| v1, AIC-seeded (as published) | 0.68 · 0.85 · **0.55** · **2.54** | 0.78 · 1.01 · **0.58** · **2.38** | 0.71 · 0.90 · **0.53** · **2.13** | 0.67 · 0.82 · **0.49** · **1.95** |
| + output length = measured (×0.79–0.81) | 0.70 · 0.70 · 0.56 · 2.53 | 0.82 · 0.83 · 0.61 · 2.66 | 0.78 · 0.79 · 0.58 · 2.29 | 0.75 · 0.75 · 0.56 · 1.93 |
| + fixed 0.19 s per request (KV hand-off + scheduling) | 0.70 · 0.70 · 0.56 · 2.66 | 0.82 · 0.83 · 0.61 · 2.68 | 0.78 · 0.78 · 0.58 · 2.30 | 0.75 · 0.75 · 0.56 · 2.26 |
| + decode = measured ITL line (6.9 + 0.44·bs ms) | 0.69 · 0.69 · 0.55 · 2.71 | 0.80 · 0.81 · 0.59 · 2.46 | 0.75 · 0.75 · 0.56 · 2.37 | 0.71 · 0.70 · 0.53 · 2.31 |
| **residual** | **0.69× requests · 0.55× total · p95 2.7× heavy** | **0.80× · 0.59× · 2.5×** | **0.75× · 0.56× · 2.4×** | **0.71× · 0.53× · 2.3×** |
| sim TPOT after the decode substitution vs measured ITL p50 | 8.9 vs 8.2 ms | 11.5 vs 10.0 | 19.8 vs 13.7 | 25.5 vs 16.4 |

Measured cells: 2.05 / 3.45 / 6.63 / 7.98 req/s · 1,998 / 3,231 / 6,276 / 7,566 output tok/s · 2,493 / 4,530 / 8,669 /
10,464 total/GPU · TTFT p95 1.37 / 1.77 / 2.58 / 3.52 s · ITL p50 8.2 / 10.0 / 13.7 / 16.4 ms.

**The engine substitutions barely move the 12:6 ratios** (total 0.55 → 0.55, 0.58 → 0.59, 0.53 → 0.56, 0.49 → 0.53), so
on this topology the engine constants are not the gap. Two terms that the ladder cannot substitute are:

1. **Prefix hit rate.** The sim's 12:6 cells run at hit 0.78 → 0.73 (96 → 768 clients), i.e. ~15–18 k uncached tokens
   per turn. The engine's own counter (`dynamo_frontend_cached_tokens` ÷ `input_sequence_tokens`, scraped by aiperf
   during the window; [`sim-results/agentx_server_metrics.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_server_metrics.txt))
   says **0.94 / 0.94 / 0.92 / 0.91 / 0.88 at 96 / 192 / 384 / 480 / 768**, i.e. 5–11 k uncached tokens per turn (the
   earlier TTFT-based estimate of 0.93 → 0.83 was right at low load and too low at high load, where queueing was being
   counted as prefill). Per request the sim therefore does **3× the prefill work at low load and 1.6× at 768**. The
   router's estimated KV hand-off latency is 0.19–0.21 s up to 192 clients and grows to 0.26 / 0.37 s at 480 / 768, which
   is the fixed per-request term the ladder adds and the reason it should scale with load rather than stay constant. It is why the sim's disagg knee lands at 768 and its ceiling at 6,187 total/GPU (silicon is at 15,004 at 768 and
   still pre-knee). The TTFT-tail term (2.3–2.7× after every other substitution) turned out to be the warm-up-storm
   artifact plus the slice rather than the cache model itself — see the re-simulation below, where the tail lands at
   0.9× of silicon once both are fixed.
2. **Trace representation.** Input tokens per request 70 k (sim) vs 86–94 k (measured) = 0.75–0.81×, and request rate
   0.69–0.80× (the 4 k-request slice cuts sessions before their longest turns and carries no subagent fan-out).
   Requests × input length reproduces the residual: 0.80 × 0.75 = 0.60 at 192 (observed 0.59), 0.71 × 0.79 = 0.56 at
   480 (observed 0.53).

Calibration order for the disagg sim, by payoff: replay the full trace (fixes term 2 and part of term 1), calibrate
the cache model to the measured 0.88–0.94 hit rate (fixes the tail and the knee), then halve the decode slope. Until
then, read the disagg sim as: total tokens ≈ 0.5–0.6× silicon, TTFT p95 ≈ 2–2.7× silicon, knee and ceiling not
predictive; topology ranking and cell selection are still sound (12:6 vs 9:9 came out in the sim's order on silicon).

### Finding the cache gap, and re-simulating on the full trace (2026-09-17)

**How the gap was found.** (1) The engine's own counter, `dynamo_frontend_cached_tokens ÷ input_sequence_tokens`
scraped by aiperf during every window, gives the measured hit rate: **0.94 / 0.94 / 0.92 / 0.91 / 0.88** at 96 / 192 /
384 / 480 / 768 on 12:6 ([`sim-results/agentx_server_metrics.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_server_metrics.txt)).
The sim's `hit_rate` column says 0.78 → 0.73. (2) [`scripts/trace_hit_ceiling.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/trace_hit_ceiling.py)
computes the best hit rate a trace can give with an infinite cache and perfect routing (same prefix-stop rule as the
sim's worker and the engine counter, sessions grouped as the sim groups them):

| trace | requests | sessions | turns per session (mean / median) | cold first-turn token share | **hit ceiling** | mean ISL |
|---|---|---|---|---|---|---|
| 4 k slice the sim replays (`weka_256k_bench4k.jsonl`) | 4,000 | 874 | 4.6 / 1 | 8.8 % | **0.846** | 70,114 |
| full trace the silicon runs replay (`weka_256k_aiperf.jsonl`) | 28,444 | 1,558 | 18.3 / 1 | 0.8 % | **0.969** | 137,232 |

The silicon value (0.94) is above the slice's ceiling, so no cache model could match it on the slice: the slice
truncates sessions (median one turn), and hit rate is a property of session length. (3)
[`scripts/dynosim_agentx_hitrate_attrib.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_hitrate_attrib.py)
then re-ran the sim substituting one step at a time
([`sim-results/agentx_hitrate_attrib.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_hitrate_attrib.txt)):

| 12:6 KV variant | 192: hit · req/s · total/GPU · TTFT p95 | 768: hit · req/s · total/GPU · TTFT p95 |
|---|---|---|---|
| A. as published (4 k slice, 100 M-token cache) | 0.763 · 2.70 · 2,679 · 4.21 s | 0.725 · 6.22 · 5,981 · 30.0 s |
| B. A + infinite cache (no eviction) | 0.763 · 2.70 · 2,679 · 4.21 s | 0.725 · 6.22 · 5,981 · 30.0 s |
| C. A + full trace | **0.950** · 1.83 · 3,609 · 5.08 s | **0.926** · 3.71 · 6,013 · 45.0 s |
| silicon (engine counter) | 0.937 · 3.45 · 4,510 · 1.77 s | 0.880 · 11.27 · 15,004 · 7.0 s |

Eviction changes nothing (the sim's LRU only ever drops blocks of finished replays); routing loses 2 points against
the ceiling, the same as the real router's 3. **The trace slice is the whole cause of the hit-rate gap.**

**Why fixing the hit rate did not fix the tail at first.** Variant C still had TTFT p95 5–45 s. The sim opened its
measurement window at the first measurable request, while every lane was still replaying its back-to-back warm-up
turns (with 18-turn sessions that storm lasts minutes), so the window contained a queueing burst the real scenario
never measures (aiperf ramps the lanes and waits 900 s). The sim's prefill tier was only 16 % utilised in that
variant, which ruled out compute. `simulate_agentx` now takes `warm_s` (0 keeps the published behaviour) and reports
the TTFT split into prefill-queue wait and prefill service
([`scripts/dynosim_agentx_fulltrace.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_fulltrace.py),
[`sim-results/agentx_fulltrace_sim.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_fulltrace_sim.txt)):

| 12:6 KV | clients | hit | req/s | total/GPU | TTFT p50 / p95 | queue wait p50 / p95 | prefill service p50 / p95 | uncached tok/req |
|---|---|---|---|---|---|---|---|---|
| 4 k slice, no warm-up (published) | 192 | 0.763 | 2.70 | 2,679 | 0.19 / 4.21 s | 0.00 / **2.87 s** | 0.13 / 2.56 s | 9,024 |
| **full trace + 900 s warm-up** | 192 | 0.953 | 1.64 | 3,674 | 0.09 / **1.58 s** | 0.00 / 0.45 s | 0.08 / 0.64 s | 4,177 |
| silicon | 192 | 0.937 | 3.45 | 4,510 | 0.36 / **1.77 s** | | | ≈ 5,900 |
| **full trace + 900 s warm-up** | 768 | 0.931 | 4.07 | 8,264 | 0.10 / **2.37 s** | 0.00 / 1.40 s | 0.09 / 0.83 s | 4,805 |
| silicon | 768 | 0.880 | 11.27 | 15,004 | 0.80 / **7.0 s** | | | ≈ 11,400 |

**Result of the re-simulation.** With the full trace and a warm-up the sim's TTFT p95 lands at 1.58 s against 1.77 s
measured at 192 clients (0.9×, from 2.4× too heavy), and its hit rate matches the engine within a point at both
cells. The published tail pessimism was therefore two artifacts, the slice and the warm-up storm, not the prefill
service model; the queue-wait p95 fell from 2.9 s to 0.45 s. Total tokens per GPU improve from 0.59× to 0.81× of
silicon at 192 and from 0.40× to 0.55× at 768. **What remains is the request rate: the sim issues 1.64 / 4.07 req/s
where silicon issues 3.45 / 11.27** (0.48× / 0.36×), i.e. its lanes replay the recorded think-time about twice as
slowly as aiperf's agentic-replay mode does, and its full-trace ISL (137 k) overshoots the 93 k aiperf actually sent
because the real lanes start part-way through sessions and cap idle gaps. Those two are the next calibration targets
(the lane cadence rule and the lane start/idle semantics of `inferencex_agentx_mvp`), and they are workload-replay
terms, not engine terms. At 768 the sim's decode (TPOT 39 ms vs 21 ms measured) also re-enters because its slower
request cycle keeps more requests in flight per decode worker.

### Workload-parity check and the v4 trajectory-replay attempt (2026-09-17)

Parity of the five workload dimensions between the sim and the real replay (real = 12:6 KV 192-client records and
engine counters; stream kinds from `source_kind`):

| dimension | real replay | v3 sim (published) | v4 sim | parity |
|---|---|---|---|---|
| input length per request | 93.6 k mean (root streams 134 k, subagent 59 k, flat 114 k) | 70 k (4 k slice) | 149 k | no, both ways |
| reusable prefix (hit rate) | 0.94 | 0.76 | 0.95 | v4 yes |
| output length per request | 936 counted by the engine (root 1,240, subagent 694) | 1,200 (trace field) | 1,211 | no: trace field is 1.3× the engine's count |
| think time between turns | replayed gaps: root p50 2.4 s / mean 35 s, subagent p50 1.7 s / mean 7 s | raw timestamps: p50 13 s / mean 618 s | measured all-stream CDF (mean 22 s) | v4 close, kinds mixed |
| session position at start | uniform over the session, previous turn primed (turn index p50 12) | 25–75 %, history replayed into the window | uniform, previous turn primed | v4 yes |
| lane structure | root stream + subagent chains; 51 % of requests are subagent turns (5 turns, 59 k each) | one sequential session | S identical streams | no |

v4 ([`scripts/dynosim_agentx_v4.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_v4.py),
gap CDF [`sim-results/agentx_gap_cdf.json`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_gap_cdf.json),
results [`sim-results/agentx_v4_sim.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_v4_sim.txt)),
12:6 KV, engine model unchanged:

| variant | clients | hit | req/s | ISL/req | total/GPU | TTFT p50 / p95 | TPOT | P90 |
|---|---|---|---|---|---|---|---|---|
| v4, 1 stream per lane | 192 | 0.954 | 5.73 | 149 k | 11,943 | 0.11 / 3.69 s | 11.8 ms | 78 |
| silicon | 192 | 0.937 | 3.45 | 94 k | 4,510 | 0.36 / 1.77 s | 10.0 ms | 89 |
| v4, 1 stream per lane | 768 | 0.941 | 10.69 | 145 k | 21,741 | 0.14 / 5.41 s | 43.7 ms | 22 |
| silicon | 768 | 0.880 | 11.27 | 95 k | 15,004 | 0.80 / 7.0 s | 20.9 ms | 45 |

Reading: with the measured gap distribution the request rate is right at 768 (10.7 vs 11.3) and 1.7× high at 192,
but every request is 1.5× too large (149 k vs 94 k) because the sim samples whole root-like sessions while half of the
real requests are short-context subagent turns. Fixing three dimensions without stream-kind parity therefore
overshoots total tokens (2.6× at 192, 1.45× at 768). **The remaining gap is stream-kind parity**, and the right input
is the one aiperf replays: the loader's stream-level trace (root and subagent chains as separate streams with their
own turns, hash ids, recorded `delay_ms`, and the engine-counted output length), replayed as trajectory trees. That
conversion is the next step; the engine terms measured so far (prefill 20–33 % slow, decode line right below 200
clients) are second order next to it.

### v5: replaying the same dataset object aiperf replays (stream-level parity), and where it lands

[`scripts/agentx_stream_trace.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/agentx_stream_trace.py)
converts the raw SemiAnalysis dataset (393 traces; 28,444 root requests and 39,822 subagent requests in 1,697 subagent
entries) into root and subagent streams with aiperf's end-to-start delay rule (`t_k − (t_{k−1} + api_{k−1})`, clamped at 0).
[`scripts/dynosim_agentx_v5.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_v5.py)
replays lanes as trajectory trees the way `agentic_replay` does (t* ∈ 0.25–0.75 as launched, previous turn primed,
first profiled turn after its own delay, children spawned on the parent's completion and joined, whole-system idle
cap 10 s, 900 s warm-up); engine untouched. Four replay rules were tried; results in
[`sim-results/agentx_v5_sim.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_v5_sim.txt):

| 12:6 KV | hit | req/s | ISL/req | OSL/req | subagent share | total/GPU | TTFT p95 | TPOT | P90 |
|---|---|---|---|---|---|---|---|---|---|
| v3 published (4 k slice), 192 | 0.763 | 2.70 | 70 k | 1,200 | — | 2,679 | 4.21 s | 10.5 ms | 72 |
| v5b recycle at t*, 192 | 0.946 | 2.52 | 84 k | 795 | 72 % | 2,981 | 1.32 s | 7.4 ms | 120 |
| **v5d recycle at turn 0, 192** | 0.942 | 4.92 | 88 k | 837 | 71 % | 6,062 | 1.49 s | 9.1 ms | 90 |
| **silicon, 192** | 0.937 | 3.45 | 94 k | 936 | 51 % | 4,510 | 1.77 s | 10.0 ms | 89 |
| v3 published, 768 | 0.725 | 6.22 | 68 k | 1,100 | — | 5,981 | 30 s | 43 ms | 19 |
| v5b recycle at t*, 768 | 0.914 | 12.9 | 86 k | 771 | 80 % | 15,558 | 3.16 s | 34.5 ms | 17 |
| v5d recycle at turn 0, 768 | 0.907 | 16.3 | 79 k | 720 | 81 % | 18,193 | 3.80 s | 59 ms | 13 |
| **silicon, 768** | 0.880 | 11.27 | 95 k | 936 | 51 % | 15,004 | 7.0 s | 20.9 ms | 45 |

Where parity now stands (192 clients, v5d vs silicon): hit rate, input length, output length, TTFT p95, TPOT and P90
interactivity are all within 6–16 %, from 0.4–2.4× in the published sim; total tokens per GPU are within 34 % (from
0.59×). The two recycle rules bracket silicon on request rate (0.73× and 1.43×), and the one dimension that stays off in
every v5 variant is the **stream mix: 71–81 % subagent turns against 51 % measured**, i.e. the sim issues subagent turns
about twice as fast as aiperf does while its root-turn rate is within 15 %. The raw dataset carries 23 subagent turns
per subagent entry; the measured replay shows 5 turns per subagent conversation, so aiperf's nested hash-id chain
detection (which splits each subagent entry into sibling chains and drops those no parent turn can spawn) is the rule
the converter does not yet reproduce. That is a dataset-loader behaviour, not a serving-engine one. Two checks that
came out negative on the way: no request in the 256 k dataset exceeds the 262,144 context limit (no keep/drop
truncation), and gating parent joins on blocking-only children changes nothing (v5c = v5b).

At 768 the engine terms re-enter: with the request rate within 15 % (v5b) and total tokens within 4 %, the sim's TPOT is
1.6–2.8× too slow (34–59 ms vs 20.9) because its decode line (5.97 + 0.4·batch per decode worker) is applied to a deeper
in-flight population than silicon carries (its requests last longer, so more overlap), and its TTFT p95 is 0.45–0.55×
of silicon because the size-driven tail (long turns that miss, see the record analysis below) is not in the cache model.

**Total tokens per GPU, step by step (192 clients; total = req/s × tokens per request ÷ 72).**
[`scripts/dynosim_agentx_v5_decomp.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_v5_decomp.py),
[`sim-results/agentx_v5_decomp.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_v5_decomp.txt):

| | req/s (root · subagent) | tokens/request (root ISL · sub ISL · OSL) | total/GPU | request latency |
|---|---|---|---|---|
| silicon | 3.45 (1.69 · 1.76) | 94.5 k (134 k · 59 k · 936) | 4,510 | 9.7 s |
| published sim (4 k slice) | 2.70 (0.78×) | 71.4 k (0.76×) | 2,679 (0.59×) | — |
| v5b (recycle at t*) | 2.52 (0.71 · 1.81) = 0.73× | 85.3 k (0.90×) | 2,981 (0.66×) | — |
| v5d (recycle at turn 0) | 4.92 (1.44 · 3.48) = 1.43× | 88.8 k (131 k · 70 k · 837) = 0.94× | 6,062 (1.34×) | 7.9 s |
| v5d + **ideal engine** (zero prefill/decode time) | 4.49 (1.52 · 2.97) | 90.7 k | 5,660 | 0.0 s |
| v5d + measured engine constants (24 k tok/s prefill, 6.9 + 0.44·bs decode) | 4.98 (1.43 · 3.55) | 88.2 k | 6,100 | 9.3 s |

The engine control settles the question: replacing the engine by an infinitely fast one moves total tokens by −7 %
(6,062 → 5,660) and replacing it by the measured constants by +1 %, so **at 192 clients the engine accounts for under
10 % of the total-token gap; the rest is dataset replay**, and specifically the request rate. Tokens per request is
already within 6 % (root ISL 131 k vs 134 k measured; subagent 70 k vs 59 k; OSL 837 vs 936). Within the request rate,
root turns are issued at 0.85× of silicon and subagent turns at 1.98×: the converter keeps all 23 inner requests of each
subagent entry where aiperf's chain splitting yields ~5 per conversation, so the sim runs twice the subagent traffic.
Halving the subagent rate would put v5d at ≈ 3.2 req/s and ≈ 4,000 total/GPU, i.e. within 10 % of silicon on every
axis at 192. At 768 the engine re-enters through decode (TPOT 35–59 ms vs 21), which is the in-flight feedback item.

**Calibration status.** Published sim → v5: hit 0.76 → 0.94 (silicon 0.94), TTFT p95 4.2 → 1.5 s (1.8), interactivity
72 → 90 (89), total tokens 0.59× → 1.34× / 0.66× (bracket) at 192, and 0.40× → 1.04–1.21× at 768. The next step is
the subagent chain-splitting rule in the converter; after it, the remaining disagreement should be the decode
in-flight feedback at high load, which is an engine-model item.

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

### Where the gap is, in plain words

1. **Output tokens: the engine model is right once output length is corrected.** The sim replays the trace's recorded
   output lengths (~1,200 tokens per request); the engine stops at end-of-sequence after ~930–995. Corrected, the
   sim's output tok/s sits at 0.79–0.92× of silicon at every load — the apparent 1.03–1.10× "over-prediction" of
   v1 was longer outputs, not a faster engine. Substituting the measured decode curve moves nothing material.
2. **TTFT: the sim's distribution has the wrong shape, not the wrong level.** Real TTFT is tight: p50 0.31–0.36 s,
   p95 1.4–2.0 s across 48–192 clients. The sim's p50 is *lower* than real (0.13–0.30 s: it lacks a constant
   ~0.19 s per request for the NVLink KV hand-off plus router and scheduler latency) while its **p95 is 2–2.7×
   *higher*** than real (2.9–5.4 s). Adding the constant fixes the median and makes the tail worse. The heavy
   simulated tail comes from the prefill model: one FCFS queue per prefill worker serving whole requests at a
   fixed token rate, so a burst of long-context arrivals on one worker stalls everything behind it. The real
   engine chunks prefill (16 k-token chunks interleaved across requests) and the KV router's load term steers
   arrivals away from a busy worker, which flattens the tail. On the p95 standard the sim is therefore
   **pessimistic** on latency, which matters for the same-SLO framings (a sim-chosen SLO point is conservative).
3. **Request rate and input tokens: workload representation, the whole throughput residual.** After every engine
   substitution the sim still issues 90% (48), 69% (96) and 79% (192) of the requests per second that aiperf
   sends, and each simulated request carries ~69 k input tokens against 85–101 k real; the product is the
   0.55–0.62× total-token ratio. The sim reads a 4,000-request slice (874 sessions) with a per-lane cadence;
   aiperf reconstructs all 393 sessions and fires subagent requests in parallel with the parent turn.
4. **Direction and consequence.** On total throughput the sim is pessimistic by 1.6–1.8×; on TTFT p95 it is
   pessimistic by 2–2.7×; on output tokens and interactivity it is within 10–20%. Every arm is fed the same
   under-sized trace and the same queue model, so rankings of topologies and routers hold; absolute levels and
   SLO thresholds must be read from the measured markers.
5. **The fix.** Sim v4 builds its trace from aiperf's own per-request records of a real run (input length,
   timestamps, subagent requests) — closes item 3 — and models prefill as chunked, interleaved service with the
   router's load term — closes the tail in item 2; the 0.19 s constant becomes a parameter in `dynosim_pd.py`.

