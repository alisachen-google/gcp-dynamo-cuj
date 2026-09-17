# Historical agg AgentX simulation curves and analysis

Historical material superseded on 2026-09-17. These results used handwritten AgentX workload models and must not be used as current AIPerf predictions. Statements about parity, topology rankings, and calibration below record the earlier interpretation; the [current replay audit](agentx-faithful-replay.md) revises it.

[Current simulation results](agentx-aiperf-results.md)

---

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

### At 192 clients (measured: 2.62 req/s · 2,344 output tok/s = 97.7/GPU · 9,741 total/GPU · TTFT p95 11.68 s (p50 1.56) · ITL p50 23.9 ms · 74.5 in flight)

| sim variant | req/s | output tok/s (/GPU) | total tok/s/GPU | TTFT p95 | TPOT | sim/real: req/s · output · total · TTFT p95 |
|---|---|---|---|---|---|---|
| v1, AIC-seeded (as published) | 1.30 | 1,388 (57.8) | 3,728 | 5.89 s | 60.9 ms | 0.50× · 0.59× · **0.38×** · 0.50× |
| + output length = measured (×0.75) | 1.69 | 1,395 (58.1) | 4,863 | 5.79 s | 52.3 ms | 0.64× · 0.59× · 0.50× · 0.50× |
| + fixed 0.19 s per request (scheduling) | 1.66 | 1,355 (56.5) | 4,771 | 6.06 s | 53.5 ms | 0.63× · 0.58× · 0.49× · 0.52× |
| + decode = measured ITL line (5.8 + 1.33·bs, no cliff) | 2.08 | 1,797 (74.9) | 6,008 | 6.97 s | 33.1 ms | 0.79× · 0.77× · **0.62×** · 0.60× |
| **residual after all substitutions** | | | | | | **0.79× requests · 0.62× total · TTFT p95 0.60× (light) · TPOT still 1.4× too slow** |

At 192 the decode substitution alone lifts the sim from 0.38× to 0.62× of silicon on total tokens (+63%), the largest
single move anywhere in the study; the remaining 0.62× is 0.79 (request rate) × 0.76 (input tokens per request, 67 k
sim vs 88 k measured) = 0.60, i.e. the trace residual, with a small remainder from the sim's still-deeper per-worker
batches (TPOT 33 vs 24 ms).

### The round-robin ladder (agg RR, 48 / 96 / 192): the engine is fine when the router is simple

Script [`scripts/dynosim_agentx_decomp_agg_rr.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_decomp_agg_rr.py),
table [`sim-results/agentx_decomp_agg_rr.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_decomp_agg_rr.txt);
decode line 5.5 + 1.45·bs ms through the measured RR ITL p50 points.

| sim variant → sim ÷ real (req/s · output · total · TTFT p95) | 48 | 96 | 192 |
|---|---|---|---|
| v1, AIC-seeded (as published) | 0.79 · 0.95 · **0.54** · 0.82 | 0.57 · 0.71 · **0.47** · 0.91 | 0.69 · 0.84 · **0.57** · 0.57 |
| + output length = measured | 0.84 · 0.83 · 0.57 · 0.88 | 0.64 · 0.63 · 0.53 · 1.26 | 0.79 · 0.73 · 0.65 · 0.73 |
| + fixed 0.19 s per request | 0.83 · 0.82 · 0.57 · 0.89 | 0.63 · 0.62 · 0.53 · 1.13 | 0.79 · 0.73 · 0.65 · 0.76 |
| + decode = measured ITL line | 0.89 · 0.90 · 0.60 · 0.97 | 0.69 · 0.69 · 0.58 · 1.36 | 0.82 · 0.79 · 0.68 · 0.77 |
| **residual** | **0.89× requests · 0.60× total · p95 0.97×** | **0.69× · 0.58× · 1.36×** | **0.82× · 0.68× · 0.77×** |

Measured RR cells: 0.76 / 1.76 / 1.97 req/s · 756 / 1,620 / 1,713 output tok/s · 3,269 / 6,174 / 6,827 total/GPU · TTFT
p95 8.27 / 12.56 / 60.1 s · ITL p50 7.5 / 12.1 / 30.5 ms. For RR the sim is within 10–20% on request rate and on the
TTFT tail once the engine constants are measured; the 0.6–0.7× on total tokens is the input-length term of the trace
slice (67–70 k vs 82–102 k measured) and nothing else. **The sim models the workload and the agg engine acceptably; what
it models badly is the KV-aware router** (compare the KV residual's 1.4× TPOT and the raw 0.25–0.28× on tuned KV).

### Compute, network and overheads on agg RR, from the engine counters

With round-robin there is no routing intelligence and no prefill→decode transfer, so the agg RR cells are the cleanest
test of the sim's *engine* model. The scraped frontend metrics
([`sim-results/agentx_server_metrics.txt`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_server_metrics.txt))
separate the terms:

| term | measured (RR 48 / 96 / 192) | sim | verdict |
|---|---|---|---|
| network: request-plane send frontend→worker | 1.5–2.1 ms p50, 4–5 ms p95 | 0 | negligible; no transfer on agg |
| request-plane queue | ≤ 1 ms p95 | 0 | negligible |
| tokenizer (frontend CPU, 80–100 k-token prompts) | 4–6 ms p50, **63–153 ms p95** | 0 | the only non-compute term that matters, and it is client-side; grows with load (212 ms p95 at 768 on disagg) |
| prefix hit rate (engine counter) | **0.74 / 0.70 / 0.56** | 0.40 | sim prefills 1.2–1.7× more tokens per turn than silicon: with 6 workers the radix cache on each worker still holds most of every session's context, because KV capacity per worker is far larger than the working set |
| prefix hit rate, KV routing | 0.86 / 0.79 / 0.74 | 0.80 / 0.79 / 0.77 | sim right for KV; the 0.19–0.20 loss from 48 → 192 is real (cache churn under 12 sessions per worker) |
| prefix hit rate, tuned KV | 0.86 / 0.82 (96 / 192) | 0.83 / 0.82 | sim right |
| decode (ITL p50 vs batch) | 5.5 + 1.45·bs ms | 7 + 1.6·bs, then 28 + 5.7·bs past batch 7 | cliff wrong; slope right below 7 |
| prefill/decode contention on one GPU | present (chunked prefill stretches decode; TTFT p95 8 → 60 s from 48 → 192) | absent | the sim's agg tail is *lighter* than silicon despite doing more prefill work |

Reading: on agg RR the sim carries two errors of opposite sign — it prefills 1.2–1.7× too many tokens (hit 0.40 vs
0.56–0.74) but never pays for prefill and decode sharing a GPU — and they roughly cancel on the TTFT tail (0.8–0.9×),
which is why RR looks like the "well-modelled" arm. Network is not a factor on agg (≈ 3 ms per request end to end);
the tokenizer is the one non-GPU cost worth putting in the sim (≈ 0.1 s at p95 at 192 clients, 0.2 s at 768). So
"compare compute and network" reduces on agg to: compute model needs the measured decode line, a cache model sized to
the real KV capacity (RR hit 0.56–0.74, not 0.40), and prefill/decode contention; network can be left at zero.

### Where the gap is, in plain words

1. **The decode cliff is the biggest single term on agg** (unlike disagg, where the engine model was within 10%). The
   published sim runs agg decode at 27–61 ms per token against 7–24 ms measured, because its busy-stream refit jumps to
   28.4 + 5.68·bs past batch 7. Removing the cliff lifts total tokens by 17–63% (0.48 → 0.56×, 0.41 → 0.53×, 0.38 →
   0.62× at 48 / 96 / 192) and is the reason the sim ranked RR ahead of KV on agg below the knee and predicted a 26% loss
   for the tuned router: the router that packs a session's turns onto one worker was being charged a decode penalty the
   real engine does not pay (measured: KV ahead 1.03× / 1.12× / 1.42×, tuned +14%, §iii, §v). The RR ladder is the
   control: with a router that never packs, the same engine lands within 10–20% of silicon on requests and tail.
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
