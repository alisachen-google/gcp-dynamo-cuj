# Nemotron-3-Ultra 550B: 72-GPU Disaggregated KV-vs-RR — Silicon Results + Drift

Disagg comparison following the Kimi pattern (2026-09-02/03): 6P+12D TP4 (6:12,
sim-best bounded split), single fleet with per-point frontend router swap,
fresh frontend + 300 s settle + 900 s trace warmup + 1800 s measured window per
point, RDMA transport gate PASS on every point (host-staged rc_mlx5 verbs —
first-ever NemotronH disagg serving: NIXL carried attention-KV + ~200 MB Mamba
SSM state per request). Zero request errors anywhere. Knee verdicts from
per-request timestamp stationarity.

## Results (output tok/s; 72 GPUs)

| conc | KV tok/s (/GPU) | KV TTFT p50/p95 | KV knee | RR tok/s (/GPU) | RR TTFT p50/p95 | RR knee | gain |
|---|---|---|---|---|---|---|---|
| **12** | **1,403 (19.5)** | **2.67 / 6.5 s** | **BOUNDED** (stationary) | 1,112 (15.4) | 3.17 / 18.7 s | post-knee (2.2→6.4 s growing) | 1.26× |
| 24 | 2,005 (27.8) | 4.05 / 10.9 s | post | 1,534 (21.3) | 5.56 / 28.6 s | post | 1.31× |
| 48 | 2,844 (39.5) | 7.5 / 22.7 s | post | 1,750 (24.3) | 13.7 / 43.3 s | post | 1.63× |
| 96 | 3,412 (47.4) | 18.5 / 53.9 s | post | 1,743 (24.2) | 33.8 / 93.5 s | post (saturated) | **1.96×** |

ITL p50 7.8–10.4 ms at every point — the decode tier idles throughout.

**Per-GPU accounting note**: "/GPU" columns divide by ALL 72 GPUs (the
deployment pays for the prefill tier whether or not it emits output tokens).
The alternative decode-GPU normalization (÷48) answers the narrower
decode-tier-efficiency question: e.g. KV c96 = 71.1 tok/s/decode-GPU ≈ agg's
69/GPU — i.e. the decode workers are as productive as agg workers when fed;
disagg's per-GPU loss is the oversized prefill tier + transfer, not decode
inefficiency. KV-vs-RR ratios are identical under either accounting. Cross-
topology comparisons in this study always use total-GPU.

## Headlines

1. **The router flag determines whether the deployment is bounded at all.** At
   the identical conc-12 cell, KV is stationary (2.67 s p50, 2.40 req/s) while
   RR diverges — RR has no bounded operating point above conc 12 (its knee is
   below any load that makes sense on 72 GPUs). This is the Kimi disagg
   finding reproduced on a second architecture.
2. **Routing doubles the saturation ceiling**: RR plateaus at ~1,750 tok/s
   from c48; KV is still climbing at c96 (3,412) — 1.96× at the top of the
   measured ladder, from placement alone.
3. **Agg-beats-disagg confirmed by measurement**: best bounded disagg (KV c12,
   19.5 tok/s/GPU) vs best bounded agg (KV c32, 69.0/GPU) — **agg is 3.5×
   better per GPU at the bounded operating points** (sim predicted ~2.6× at
   ideal cells). For this architecture, disaggregation is a measured negative.
4. **The Kimi disagg ceiling was transfer-bound**: N3U disagg reaches 3.3–6.0
   req/s where Kimi hard-ceilinged at ~1.5–1.7 req/s on the same host-staged
   path — request ceiling scales with per-request transfer volume (~0.8 GB vs
   ~3.4 GB). Sharpens the GPUDirect-driver escalation: the broken driver is
   costing disagg deployments their viability.

## Dataset completion (2026-09-04): boundary, floor, and flag points

| point | tok/s | TTFT p50/p95 | knee verdict |
|---|---|---|---|
| KV c8 | 1,007 | 2.22 / 6.4 s | bounded |
| **KV c16** | **1,614** | 3.19 / 8.6 s | **bounded — KV's refined best bounded cell** |
| RR c8 | 818 | 2.42 / 14.6 s | post-knee (1.8→5.2 s growing) |
| **RR c4** | **504** | 2.04 / 8.1 s | **bounded — RR's floor** (7 tok/s/GPU on 72 GPUs) |
| KV-scale2 c12 | 1,420 | 2.50 / 6.3 s | bounded (+1.2% vs defaults) |
| KV-credit0.8 c12 | 1,444 | 2.66 / 6.3 s | bounded (+2.9% — noise) |
| KV-temp0.5 c12 | 1,177 | 3.59 / 12.5 s | **POST-KNEE** (−16%; temperature destabilizes the bounded cell) |

**Refined framing-2 headline: KV 1,614 tok/s @ conc 16 vs RR 504 @ conc 4 —
3.2× throughput at 4× concurrency**, each policy at its best bounded cell.
There is no both-bounded same-conc cell above c4: KV's bounded range
(≤16) and RR's (≤4) barely overlap, and KV was not run at c4 (RR's floor is
below any operationally meaningful load).

**Flag sweep verdict (matches sim + Kimi)**: score-shaping flags (scale,
credit) are within noise at the bounded cell; **router temperature 0.5 is
actively harmful** — it randomizes placement enough to tip the same cell from
bounded to post-knee. Defaults + temperature 0 remain correct.

**Transport verification (per user directive)**: per-point UCX log sampling
found `rc_mlx5` RDMA endpoints (e.g. 393 on the decode tier) and **zero
`cuda_ipc`/MNNVL lines** at every point where the log window was fresh; the
kv-transport-guard gate passed all 15 points. All disagg traffic ran RoCE v2
RDMA (host-staged).

## Ceiling & topology verification (2026-09-08; all gates PASS)

| point | tok/s | /GPU-total | /GPU-decode | TTFT p50/p95 |
|---|---|---|---|---|
| 6:12 KV c144 | 3,307 | 45.9 | 68.9 | 36.7 / 61 s |
| 6:12 KV c192 | 3,263 | 45.3 | 68.0 | 51.4 / 84 s |
| 6:12 RR c144 | 1,690 | 23.5 | 35.2 | 63.9 / 205 s |
| 3:15 KV c96 | 1,828 | 25.4 | 30.5 | 38.0 / 86 s |
| 3:15 KV c144 | 1,679 | 23.3 | 28.0 | 71.8 / 140 s |

- **Ceilings confirmed**: KV saturates at ~3,400 (c96; flat-to-declining
  through c192); RR's ~1,750 plateau re-confirmed at c144. The 2× saturation
  gap holds across the extended range.
- **The decode-heaviest split (3:15) — disagg's only arithmetic path to agg
  per-GPU parity — performs ~2× WORSE than 6:12**: with the host-staged
  transfer tax, 3 prefill workers starve 15 decode workers (decode-GPU
  productivity collapses 68.9 → 30.5). The prefill+transfer tier is the
  binding constraint; enlarging decode cannot help. First on-hardware
  validation of the sim's split-ranking direction.
- **Closes the agg-vs-disagg question by measurement**: best disagg under any
  measured topology/concurrency = 47.4 tok/s/GPU (post-knee) vs agg 69
  bounded / 85 post-knee. Aggregated serving wins for this architecture under
  every consistent accounting, at every measured load, on every measured
  split.

## Certified-transport re-sweep (2026-09-10): reproduction + transport closure

Per directive, the full ladder (kv/rr × 12/24/48/96) was re-run on a
**certified transport**: probe re-confirmed GPUDirect still faults (0/20,
112 UCX errors), certified config = host-staged RDMA with **TCP removed from
UCX_TLS entirely** (`cuda_copy,rc_x`; probe 20/20 clean), adopted permanently
in the arm generator. Positive per-fleet evidence captured at startup: UCX's
protocol table routes every size class to `rc_mlx5` (zero-copy striped across
two NICs), zero tcp/cuda_ipc lines.

| point | certified tok/s | original | Δ |
|---|---|---|---|
| kv 12/24/48/96 | 1,394 / 2,043 / 2,853 / 3,423 | 1,403 / 2,005 / 2,844 / 3,412 | −0.6…+1.9% |
| rr 12/24/48/96 | 1,135 / 1,500 / 1,646 / 1,746 | 1,112 / 1,534 / 1,750 / 1,743 | −5.9…+2.1% |

**Verdicts**: (1) the original dataset is *reproduced* within run-to-run noise
(|Δ| ≤ 6%, no systematic direction) — every knee verdict identical (kv:12
bounded/stationary, everything else post-knee); the study's disagg
conclusions are robust. (2) Removing TCP from the transport changed nothing
measurable — confirming the baseline's tcp was wireup-only and the data path
was always RDMA, now guaranteed by construction. (3) GPUDirect remains the
open lever: still broken as of 2026-09-09 (fresh escalation evidence).

## Drift analysis (apple-to-apple, mirroring the agg method)

Substituting measured component rates into DynoSim at c12/48/96 for both
policies:

| Sim variant | KV drift (c12/48/96) | RR drift (c12/48/96) |
|---|---|---|
| v1 (AIC prefill 19.7k tok/s/worker) | 1.25 / 1.66 / 1.70× | 1.35 / 1.83 / 2.06× |
| prefill ×0.5 (≈9.9k effective) | 1.20 / 1.53 / 1.51× | **1.10 / 1.15 / 1.21×** |
| prefill ×0.33 | 1.16 / 1.34 / 1.23× | 0.88 / 0.80 / 0.83× |

Step-by-step attribution:
- **Decode: no drift** (measured ITL 7.8–10.4 ms ≈ the model's base; the tier
  is idle — unlike agg, where decode carried the whole gap).
- **RR fits at prefill ×0.5**: the prefill tier's *effective* rate is ~half
  the AIC compute-only rate. For RR traffic (mostly uncached, work ∝ ISL) the
  host-staged transfer + staging overhead scales with ISL exactly like
  compute, so it absorbs into a rate — **the transfer tax ≈ doubles
  prefill-tier cost**.
- **KV's residual is an unmodeled per-request transfer FLOOR**: no prefill
  rate fits KV (still 1.2–1.5× over at ×0.5) because cached requests shrink
  their *compute* with hit rate but still ship the full ~0.8 GB KV+state to
  decode — a fixed ~2–2.5 s/request cost the sim charges nothing for
  (KV c12: measured p50 2.67 s vs 0.25 s simulated at the fitted rate).
  Secondary: hit-rate optimism (engine ≈63% vs sim 84%, as measured on agg).
- **Knee drift explained by the same two terms**: sim said KV bounded to
  c144; with the transfer floor + halved tier rate, the real knee lands at
  ~12 — consistent, not anomalous.

**n3u-sim v3 items**: add a per-request transfer term (floor + BW), halve the
disagg prefill-tier rate under host-staging, add hybrid-checkpoint hit-rate
granularity. Polarity check: with RR fitted, the sim's KV *gain* prediction
becomes conservative — acceptable.

## Reproduction

Arm `manifests/n3u-d72.yaml`; sequencer `scripts/sweep_n3u_d72.sh` (3
revisions — ladder re-anchored after silicon knees came in far below sim;
halts preserved in DIARY.md); artifacts
`gs://alisachen-models/perf/17883*_alisachen-n3u-d72-{kv,rr}-c{12,24,48,96}/`;
summaries in `results/silicon/` (`d72-*.json`).
