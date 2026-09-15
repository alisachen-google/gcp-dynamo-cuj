# Pareto-frontier simulation → benchmarking, end to end (Nemotron-3-Ultra disagg on GB300)

Method: the four-step loop in NVIDIA's *DynoSim: Simulating the Pareto Frontier* — **(1) sweep
broadly in simulation, (2) shortlist Pareto-optimal candidates, (3) verify on the real cluster,
(4) calibrate from measurements.** Their DynoSim is a Rust discrete-event twin of the Dynamo
stack seeded with AIC forward-pass timing; ours (`scripts/dynosim_pd.py`) is the same
construction at smaller scope — a trace-driven DES of the Dynamo KV router (exact
`worker_logit` formula) over prefill/decode worker tiers, seeded from AIC 0.11.0 SILICON
timing for this model — so the loop transfers directly. The official runnable analogue is
Dynamo's *mocker* trace replay; not used here.

Pages: [throughput vs concurrency](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-disagg-curve.html) · [TTFT vs concurrency](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-ttft-curve.html) · [throughput/chip vs P90 interactivity](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-frontier.html) · [Pareto: sim frontier + measured](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-pareto.html) · [run index](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/RUN_INDEX.md)

## 1. Sweep broadly in simulation
Grid: 5 P:D splits on 72 GPU (3:15, 6:12, 9:9, 12:6, 15:3; TP4/EP4 workers) × 11 router policies
(rr, ll, kv-defaults, 8 KV flag variants) × 11 concurrencies (12 → 768) = **605 cells**, replaying
the 4,000-request Weka 256k trace closed-loop. Outputs per cell: throughput, TTFT p50/p95/p99,
TPOT, cache hit-rate (`sim-results/dynosim_n3u_disagg72_v1*.csv`; 118–352 s per sub-sweep).

## 2. Shortlist Pareto-optimal candidates
Dominance filter: maximise throughput/GPU, minimise TTFT p95 (the blog's throughput-vs-latency
axis pair); a second pass on throughput/GPU vs TPOT (interactivity).

### KV router, defaults — non-dominated cells (11 of 55)
| split | policy | conc | tok/s/GPU | TTFT p50 / p95 s | interactivity (1/TPOT) | hit |
|---|---|---|---|---|---|---|
| 6:12 | kv-nvda | 12 | 24.3 | 0.12 / 1.04 | 157 | 0.84 |
| 12:6 | kv-nvda | 24 | 37.9 | 0.12 / 1.07 | 132 | 0.84 |
| 12:6 | kv-nvda | 48 | 54.7 | 0.12 / 1.09 | 109 | 0.83 |
| 12:6 | kv-nvda | 96 | 60.9 | 0.13 / 1.20 | 81 | 0.80 |
| 9:9 | kv-nvda | 48 | 62.0 | 0.13 / 1.20 | 123 | 0.82 |
| 9:9 | kv-nvda | 96 | 72.6 | 0.14 / 1.63 | 98 | 0.79 |
| 9:9 | kv-nvda | 144 | 73.0 | 0.17 / 2.77 | 81 | 0.78 |
| 9:9 | kv-nvda | 120 | 73.9 | 0.18 / 3.02 | 88 | 0.78 |
| 6:12 | kv-nvda | 96 | 80.7 | 0.19 / 3.35 | 109 | 0.79 |
| 6:12 | kv-nvda | 144 | 82.2 | 0.42 / 5.18 | 93 | 0.78 |
| 3:15 | kv-nvda | 96 | 82.3 | 1.67 / 7.45 | 117 | 0.80 |

### Round-robin — non-dominated (5 of 55)
| split | policy | conc | tok/s/GPU | TTFT p50 / p95 s | interactivity (1/TPOT) | hit |
|---|---|---|---|---|---|---|
| 3:15 | rr | 12 | 23.0 | 0.32 / 3.73 | 157 | 0.62 |
| 3:15 | rr | 24 | 39.4 | 0.44 / 5.85 | 150 | 0.62 |
| 12:6 | rr | 48 | 47.0 | 2.27 / 10.52 | 109 | 0.28 |
| 3:15 | rr | 48 | 56.3 | 1.57 / 12.46 | 138 | 0.62 |
| 3:15 | rr | 96 | 62.3 | 8.34 / 25.79 | 117 | 0.62 |

### KV, any flag variant — non-dominated (24 of 495); the flag surface adds ≤2% throughput but halves p95
(see D72 "KV-router flag configuration" for the per-cell table)

**Shortlist for silicon (KV defaults, one cell per split/regime):** 6:12 c12 (latency floor),
12:6 c24/c48/c96 (low-latency mid-throughput), 9:9 c48/c96/c120/c144 (balanced), 6:12 c96/c144
(sim's high-throughput end), 3:15 c96 (sim's ceiling). The sim predicts the frontier's
high-throughput end belongs to **6:12 and 3:15**, with 9:9 dominated above c96.

## 3. Verify on the real cluster
Already measured (fleet instance 2, MNNVL, transport-gated, knee-checked):

| cell | sim tok/s/GPU · p95 | **measured** tok/s/GPU · p95 · knee | real / sim |
|---|---|---|---|
| 6:12 KV c12 | 24.3 · 1.0 | 24.1 · 2.8 · AT/PRE | 0.99× |
| 6:12 KV c48 | 65.6 · 1.9 | 65.1 · 10.0 · AT/PRE | 0.99× |
| 6:12 KV c96 | 80.7 · 3.4 | 66.8 · 33.6 · AT/PRE | 0.83× |
| 6:12 KV c120 | 81.6 · 5.5 | 67.8 · 38.6 · AT/PRE (peak) | 0.83× |
| 6:12 KV c144 | 82.2 · 5.2 | 63.4 · 49.1 · POST | 0.77× |
| 9:9 KV c48 | 62.0 · 1.2 | 63.3 · 3.1 · AT/PRE | 1.02× |
| **9:9 KV c96** | 72.6 · 1.6 | **91.1 · 8.5 · AT/PRE** | **1.25×** |
| 9:9 KV c144 | 73.0 · 2.8 | running | — |
| 12:6 KV c48 / c96 | 54.7 / 60.9 | queued (Pareto-verify runner) | — |
| 3:15 KV c96 | 82.3 · 7.5 | queued | — |
| 9:9 KV c120 | 73.9 · 3.0 | queued | — |

## 4. Calibrate
Two findings already force sim changes: (a) **the split ranking above c96 is inverted** — the sim
puts 6:12 (and 3:15) ahead of 9:9, silicon puts 9:9 1.36× ahead at c96 and above agg; the
sim's prefill-tier service model does not capture the tier becoming the binding constraint
(measured: 92% busy, 10-deep queue at 6:12 kv:144); (b) **TTFT is under-predicted 20–50× on KV
from c48 up** while throughput is within 0.77–0.99× — the queueing term is missing, not the
rates. Both point at the same v3 correction: a prefill-tier queue model with the measured
service time and chunking, which will move the sim's frontier toward prefill-heavier splits.
Hit-rate (89% measured vs 78–84% sim) and decode TPOT are *not* the drift sources for disagg.

This document is updated as the queued verification points land.
