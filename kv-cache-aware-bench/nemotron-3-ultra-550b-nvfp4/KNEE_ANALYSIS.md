# Knee points — simulation vs silicon, KV vs RR, aggregated vs disaggregated

**Knee definition (both worlds, same rule).** Silicon: last concurrency whose
queue is stationary across the 1800 s window (`knee_check.py`, per-request
timestamps; AT/PRE-KNEE vs POST-KNEE). Sim (DynoSim v1): last concurrency before
TTFT p50 growth turns super-linear in concurrency (p50 ratio > conc ratio on the
ladder). No latency SLO in either. "→" brackets mean the true knee lies between
two measured ladder points; pinning it needs a bisection cell.

## Knee table

| arm | policy | sim knee (DynoSim v1) | silicon knee | sim / real | tok/s at knee: sim → real |
|---|---|---|---|---|---|
| agg 24 GPU | KV | c96 → 128 (p50 0.25 → 0.34 s) | **c48** (c48 stationary, c64 growing) | **2–2.7× high** | 5,110–5,185 → 1,853 (2.8×) |
| agg 24 GPU | RR | c24 (super-linear from c32) | **c32** (c32 stationary, c64 growing) | ~matched (one step low) | 2,473 → 993 (2.5×) |
| disagg 6:12 MNNVL | KV | c96 (super-linear from c144) | **c48 → 96** (c48 AT/PRE, c96 POST) | up to 2× high | 5,813 → 3,573 (1.6×) |
| disagg 6:12 MNNVL | RR | c48 (super-linear from c96) | **c24 → 48** (c24 AT/PRE, c48 POST) | up to 2× high | 3,196 → 1,980 (1.6×) |
| disagg 6:12 host-staged | KV | (same sim) | ~c16 → 24 | 4–6× high | — |

## Findings

1. **The sim over-places every KV knee, by ~2× on disagg and 2–2.7× on agg.**
   Both KV knees sit one to two ladder steps to the right of silicon. This is
   the same physics as the throughput drift: the AIC decode batch-slope is ~6×
   too flat (5.26 + 0.277·bs vs measured 8.9 + 1.73·bs ms), so the modelled
   decode tier absorbs far more concurrency before its queue backs up. A second
   contributor is hit-rate optimism (sim 84% vs measured ~63% cached tokens):
   less modelled prefill load ⇒ KV backs up later in the sim than on hardware.
2. **RR knees are much closer to the sim** — agg RR matches within one step
   (sim c24 vs real c32; the sim is slightly *conservative*), disagg RR is ≤2×.
   RR gets no reuse benefit, so the hit-rate error barely touches it; only the
   decode-slope error remains. This asymmetry is itself evidence that the
   KV-specific drift is the reuse/hit-rate model, not just decode.
3. **Transport moves the knee more than the ceiling.** Host-staged disagg KV
   kneed at ~c16–24; MNNVL moved it to c48–96 (**3–6×**), while the throughput
   ceiling rose only 8–31%. The per-request transfer floor (~2.4 s) was a
   *queueing* tax: it made every prefill hand-off block the decode tier, so the
   queue stopped draining at very low concurrency. NVLink removes the floor and
   the knee jumps; the ceiling is then set by compute, which transport can't
   raise.
4. **Knee order across arms (silicon).** RR always knees before KV, in both
   arms (agg: 32 vs 48; disagg: 24–48 vs 48–96). Disagg's KV knee is deeper in
   concurrency than agg's (48–96 vs 48) because 12 decode workers hold 3× the
   in-flight capacity of 6 agg workers — but at 3× the GPUs, so the *per-GPU*
   verdict still favours agg (69 vs 49.6 tok/s/GPU at each arm's knee).
5. **Throughput at the knee is over-predicted more on agg (2.5–2.8×) than on
   disagg (1.6×).** Agg's real knee comes so early (c48) that the sim's
   flat-decode curve is furthest from measured there; disagg's later real knee
   sits where the sim curve has already begun to bend.

## What would pin the bracketed knees (optional bisection cells)
- disagg KV: c64 and c72 (between 48 and 96); disagg RR: c32 and c40.
- These are ~45 min each; not required for the arm-level verdicts, which hold on
  the brackets, but they would tighten the sim/real knee ratio from "up to 2×"
  to a point estimate.

## Sim correction (n3u-sim v2 → v3)
v2 already replaced the agg decode line with the measured 8.9 + 1.73·bs; with it
the agg KV knee prediction moves from c96–128 to ~c48 (matches). The pending v3
recalibration from measured 0.5.16 silicon (agg re-sweep + MNNVL disagg) will
add the hybrid-cache hit-rate term, which is what the remaining KV-knee error
points at.
