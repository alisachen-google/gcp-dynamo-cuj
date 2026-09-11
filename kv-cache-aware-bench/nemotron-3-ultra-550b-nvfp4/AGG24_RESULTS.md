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

## KV vs RR under a fixed SLO (framing 3, agg)

Fix a TTFT p95 budget; compare each policy's best measured throughput among
compliant, queue-stationary runs (concurrency free per policy; compliance from
measured p95 only; boundaries refined by measured bisection — details in
`SLO_COMPARISON.md`):

| TTFT p95 SLO | KV max compliant | RR max compliant | KV impact |
|---|---|---|---|
| **≤ 5 s** | **1,657 tok/s @ conc 32** (4.3 s) | **none** — RR misses 5 s even at conc 8 (7.8 s; recompute tail, not queueing) | **servable vs not servable** |
| ≤ 10 s | 1,853 @ conc 48 (7.8 s) | 882 @ conc 16 (9.8 s) | **2.1× throughput, 3× concurrency** |
| ≤ 30 s | 2,045 @ conc 64 | 1,141 @ conc 64 | 1.8× |

Reading: at a strict interactive SLO the router flag is binary — KV-routed agg
is the only Nemotron configuration (any topology) that serves this workload at
all. At relaxed SLOs, KV converts the same latency budget into ~2× the tokens
and ~3× the concurrent users. Same placement mechanism as the framing-1/2
results; the SLO lens prices it in deployment terms.

## Real-job sweep methodology: what we sweep, and how

**Swept on silicon** (each cell = one gated, warmed run):

| axis | values | how varied |
|---|---|---|
| Router policy | kv (defaults + temp 0) vs round-robin | frontend-only arg swap; **fresh frontend per point** (router-state isolation); workers untouched |
| Concurrency | ladder 16/32/64/128 (sim-bracketed) + measured-bisection in-fills 8, 48 (SLO boundaries) | `CONCURRENCIES` env of the bench job |
| Router flags | credit / prefill-load-scale / temperature | **sim-swept (9 variants per cell); deliberately not swept live on agg** — the grid showed defaults tie within 0.1% at bounded cells and tuning pays only post-knee (+12.6% @ c128), so the live flag-sweep budget went to disagg where sensitivity existed |

**Held fixed** so every delta is attributable to a swept axis: worker shape
(6×TP4/EP4 from the AIC SILICON solve), engine flags, dataset + replay mode
(native deterministic loader, 393 sessions, seed 42), and the per-point
protocol — 300 s settle, **900 s trace-replay warmup under the measured
point's own router config** (each policy measured on cache placement it
produced), 1800 s window, zero-error requirement.

**Per-point verification on every cell**: KV-routing liveness
(`kv_hit_rate` histogram + engine `cached_tokens` — excludes silent
load-routing fallback), knee verdict from per-request timestamp stationarity
(`knee_check.py`), artifacts on GCS with summaries in `results/silicon/`.
Reported cells are then selected from measured verdicts only: both-bounded
cells (framing 1), measured knees (framing 2), measured p95 (framing 3).

## Drift analysis: where the 1.9–2.8× sim-vs-silicon gap lives (apple-to-apple)

Method: substitute measured component rates into DynoSim one step at a time at
the c16/c32 cells and observe which substitution closes the throughput gap.

| Sim variant | KV c16 pred/meas | RR c16 | KV c32 | RR c32 |
|---|---|---|---|---|
| v1 (AIC-seeded) | 1.89× | 2.24× | 2.31× | 2.82× |
| **decode substituted** (measured ITL line 8.9 + 1.73 ms/seq) | **0.80×** | **1.06×** | **0.86×** | **1.31×** |
| + prefill ×0.5 | 0.81× | 0.99× | 0.82× | 1.15× |
| + prefill ×0.25 | 0.77× | 0.78× | 0.82× | 0.86× |

**Verdict: the e2e drift is essentially one step — decode.** The AIC gb300
sglang-0.5.14 DB models NemotronH decode as 5.26 + 0.277·bs ms; silicon
measures 13.5 ms @ per-worker bs 2.7 → 18.0 ms @ 5.3, i.e. **8.9 + 1.73
ms/seq — a 6× steeper batch slope**. Substituting decode alone moves every
cell to within ~±30%; prefill scaling then barely moves throughput
(decode-bound regime) and stock prefill already reproduces RR's TTFT
(0.74–1.23 s predicted vs 1.10–1.44 s measured), so the prefill rate is
approximately right. The secondary residual is **hit-rate drift**: engine
telemetry shows KV median cached ≈ 52k/83k ≈ 63% vs the sim's 84% (the sim's
block-aligned reuse is optimistic about hybrid-cache checkpoint granularity),
which explains RR's remaining +31% at c32 and KV's TTFT p50 (0.56 s vs 0.13 s
predicted). Root-cause attribution for the DB: NemotronH decode is
Mamba-state + LatentMoE bound, and the DB's batch-scaling row for this
architecture appears interpolated from too-sparse measurements — worth
reporting upstream to aiconfigurator.

**n3u-sim v2 constants** (applied): agg TPOT 8.9 + 1.73·bs (measured), prefill
unchanged 19.7k, reuse model pending a hybrid-checkpoint granularity term.
With v2 the sim is mildly conservative for KV (0.80–0.86×) — acceptable
polarity (under-promise) for prediction use.

## Reproduction

Arms `sglang/manifests/n3u-agg-kv.yaml` (router swapped per point);
sequencer `scripts/sweep_n3u_agg.sh`; jobs sed-derived from the Kimi flagsweep
template (N3U model/tokenizer staging swaps); artifacts
`gs://alisachen-models/perf/17882*_alisachen-n3u-agg-{kv,rr}-c{16,32,64,128}/`;
summaries fetched per point; knee evidence in each `profile_export.jsonl`.
