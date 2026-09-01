# Nemotron-3-Ultra 550B: 24-GPU Aggregated KV-vs-RR — Silicon Results

First silicon comparison for the study's second model (2026-09-01). 6 × TP4
sglang 0.5.14 workers (np-3), `modelopt_fp4`, native Weka-trace replay, single
fleet with per-point frontend router swap; per point: fresh frontend + 300 s
settle + 900 s trace warmup + 1800 s measured window. Zero request errors at
every point. Knee verdicts from per-request timestamp stationarity
(`knee_check.py`), no latency SLO (methodology rev 3).

## Results (output tok/s; 24 GPUs)

| conc | KV tok/s (/GPU) | KV TTFT p50/p95 | RR tok/s (/GPU) | RR TTFT p50/p95 | gain | knee verdict |
|---|---|---|---|---|---|---|
| 16 | 1,246 (51.9) | 0.56 / 4.0 s | 882 (36.7) | 1.10 / 9.8 s | **1.41×** | **both stationary** |
| 32 | 1,657 (69.0) | 0.62 / 4.3 s | 993 (41.4) | 1.45 / 16.8 s | **1.67×** | **both stationary** |
| 64 | 2,045 (85.2) | 1.09 / 21.5 s | 1,141 (47.5) | 5.13 / 29.4 s | 1.79× | both growing (post) |
| 128 | 1,922 (80.1) | 28.8 / 106 s | 1,096 (45.7) | 34.3 / 129 s | 1.75× | both saturated |

ITL p50 at the bounded cells: 13.5–18.0 ms (KV) / 12.4–19.3 ms (RR).

## Headline

- **Framing 1 (both-bounded, same cell): conc 16 and conc 32 both qualify on
  silicon.** At conc 32 — the strongest fair cell — **KV = 1.67× RR throughput
  with 3.9× better TTFT p95** (4.3 vs 16.8 s), both queues stationary across
  the window.
- **Framing 2 collapses into framing 1 for this model**: both policies knee
  between conc 32 and 64, so each policy's best bounded cell is the *same*
  cell (32) — the honest headline is simply **1.67× at equal, healthy load**.
  (Contrast Kimi, where the story was the knee *shift*; here KV wins on
  throughput at identical operating points.)
- **The hybrid-architecture hypothesis is confirmed on silicon**: with an
  effectively unbounded cache (nothing evicts at these ISLs), the KV gain can
  only be placement. Silicon gains (1.41–1.79×) exceed the sim's placement-only
  prediction (1.19–1.46×) at every cell.

## Sim-vs-silicon verdicts (n3u-sim v1)

| Sim claim | Silicon verdict |
|---|---|
| Same-cell gains 1.19→1.46× rising with conc | Direction confirmed; **silicon larger** (1.41→1.79×) |
| RR knee ≈ conc 32 | Confirmed (c32 stationary, c64 growing) |
| KV knee ≈ conc 128 | **Refuted — KV knees at ~32–48** (c64 already growing) |
| Absolute KV throughput 5,127 @c64 | 2.5× over-predicted (measured 2,045) |
| Both-bounded cells at 16 & 32 | Confirmed exactly |

Calibration v2 items: silicon prefill/decode effective rates ~2.5× below the
AIC-seeded constants (same over-prediction class as Kimi's 1.5–2.7×; AIC agg
warm said 197 tok/s/GPU vs 69 measured at the bounded cell) — refit from these
8 points; the *ratio* and *RR-knee* fidelity is what the sim earns its keep on,
and both held.

## Reproduction

Arms `sglang/manifests/n3u-agg-kv.yaml` (router swapped per point);
sequencer `scripts/sweep_n3u_agg.sh`; jobs sed-derived from the Kimi flagsweep
template (N3U model/tokenizer staging swaps); artifacts
`gs://alisachen-models/perf/17882*_alisachen-n3u-agg-{kv,rr}-c{16,32,64,128}/`;
summaries fetched per point; knee evidence in each `profile_export.jsonl`.
