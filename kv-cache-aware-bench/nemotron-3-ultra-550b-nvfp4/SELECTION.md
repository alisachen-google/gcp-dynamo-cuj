# N3U — Sim-Selected Operating Points (agg first; disagg72 pending sweep)

Selection per methodology revision 3: boundedness by queue-drain stationarity
(cross-conc latency slope on sim data; latency reported, never gated). Source:
`sim-results/dynosim_n3u_agg_v1.csv` (n3u-sim v1 constants, AIC 0.11.0 NVFP4 fit).

## Aggregated 24 GPU (6 × TP4) — the primary comparison for this model
AIC picked agg over disagg at both scales (1.09–1.63×), so agg carries the
N3U headline. Sim table (KV defaults vs RR):

| conc | KV tok/s · p50/p95 · hit | RR tok/s · p50/p95 · hit | gain | knee state |
|---|---|---|---|---|
| 16 | 2,356 · 0.13/1.18 s · 83% | 1,972 · 0.74/8.2 s · 43% | 1.19× | both drain |
| 32 | 3,825 · 0.15/1.44 s · 80% | 2,797 · 1.23/20.6 s · 43% | 1.37× | KV drains; **RR at-knee** (p50 slope goes super-linear at 48) |
| 64 | 5,127 · 0.19/2.40 s · 79% | 3,502 · 3.97/40.9 s · 43% | 1.46× | KV drains; RR post |
| 128 | 5,185 · 0.34/4.88 s · 79% | 3,590 · 13.4/79 s · 43% | 1.44× | **KV at-knee** (thr rollover 128→192); RR deep post |

**Silicon ladder selected: conc 16 / 32 / 64 / 128, both arms** — brackets both
knees; framing-1 cells at 16 & 32; framing-2 prediction **KV 5,185@128 vs RR
2,797@32 → 1.85×, 4× conc**. Router flags: KV defaults + temp 0 (flag grid at
these cells: within noise, as for Kimi agg).

**Mechanism note (the study hypothesis, now sim-confirmed):** cache is
unbounded — nothing evicts — yet RR pins at 43% hit at every conc (rotation
scatters session turns) vs KV 79–84%. The entire gap is *placement*. The
no-cliff Mamba decode moves the whole operating range 4–8× deeper in
concurrency than Kimi agg (knee 128 vs 16).

## Disaggregated 72 GPU — swept; confirms the agg verdict, no silicon cell selected

`dynosim_n3u_disagg72_v1.csv` (5 splits × 8 conc × 11 policies). Best KV cells
per split: 3:15 → 5,927 tok/s @c96 (p95 7.5 s); 6:12 → 5,920 @c144 (5.2 s);
9:9 → 5,259 @c144; 12:6 → 4,383 @c96; 15:3 → 2,932 @c48. RR knees by c48–96
on every split.

**The verdict**: best disagg = **82 tok/s/GPU** on 72 GPUs vs agg's **216
tok/s/GPU** on 24 — DynoSim independently reproduces AIC's agg-wins call:
**disaggregation costs ~2.6× per-GPU for this architecture** (cheap linear
prefill + tiny KV remove the separation benefit; the decode tier idles).
Reported as the cross-model topology finding — the study's silicon budget goes
to agg (ladder above). A silicon disagg arm would only be revisited if the
GPUDirect fix lands *and* the cross-model story needs a measured confirmation
of the negative.
