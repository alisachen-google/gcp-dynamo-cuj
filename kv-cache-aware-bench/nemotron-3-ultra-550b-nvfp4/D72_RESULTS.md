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
