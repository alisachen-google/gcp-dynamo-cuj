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

## Knees under the AgentX concurrency definition (2026-09-16; sim v3, measured ladder in progress)

Axis = live session clients (`--scenario inferencex-agentx-mvp`), not busy streams; see AGENTX_COMPARISON.md §2/§5e.
Knee rule for the sim: **throughput-slope knee** — the last client count before the marginal total-token gain per
added client drops below 25% of the initial slope. The TTFT-stationarity rule used for silicon does not transfer to
the sim's RR arms (their TTFT is set by prefix misses even at 48 clients), so the slope rule is used for both
policies and the TTFT p50 ≤ 1 s column is reported alongside as the latency-bounded proxy.

| arm | policy | throughput knee (clients) | total tok/s/GPU at knee | TTFT p50 at knee | last count with TTFT p50 ≤ 1 s | peak total (clients) |
|---|---|---|---|---|---|---|
| disagg 9:9 | KV | 480 | 5,501 | 4.1 s | 192 | 5,501 (480) |
| disagg 9:9 | KV tuned (s3/c0.8) | 768 | 7,598 | 2.2 s | 480 | 7,947 (960) |
| disagg 9:9 | RR | 192 | 2,047 | 12.7 s | < 48 | 2,062 (384) |
| disagg 12:6 | KV | 480 | 5,217 | 0.6 s | 480 | 6,187 (1440) |
| disagg 12:6 | KV tuned (s3/c0.8) | 480 | 5,320 | 0.4 s | 1920 | 6,796 (1920) |
| disagg 12:6 | RR | 192 | 2,271 | 6.5 s | < 48 | 2,508 (384) |
| disagg 6:12 | KV | 384 | 4,064 | 13.0 s | 192 | 4,064 (384) |
| disagg 6:12 | KV tuned (s3/c0.8) | 480 | 5,686 | 5.4 s | 192 | 5,686 (480) |
| disagg 6:12 | RR | 96 | 1,296 | 4.0 s | < 48 | 1,631 (192) |
| disagg 3:15 | KV | 192 | 2,276 | 10.1 s | 96 | 2,276 (192) |
| disagg 3:15 | KV tuned (s3/c0.8) | 192 | 2,574 | 2.7 s | 96 | 2,847 (384) |
| disagg 3:15 | RR | 96 | 1,166 | 8.4 s | 48 | 1,219 (192) |
| disagg 15:3 | KV | 384 | 3,218 | 0.2 s | 1920 | 3,809 (1536) |
| disagg 15:3 | KV tuned (s3/c0.8) | 384 | 3,224 | 0.2 s | 1920 | 3,887 (1920) |
| disagg 15:3 | RR | 192 | 2,158 | 3.7 s | < 48 | 2,760 (480) |
| agg 24-GPU | KV | 192 | 3,715 | 0.2 s | 1536 | 5,138 (960) |
| agg 24-GPU | KV tuned (s3/c0.8) | 192 | 2,740 | 0.1 s | 1920 | 4,947 (960) |
| agg 24-GPU | RR | 192 | 3,889 | 4.4 s | 48 | 4,156 (384) |

Measured so far (`knee_check.py`, stationarity of TTFT p50 across quarters): disagg 9:9 KV **48 and 96 clients
stationary** (TTFT p50 0.32 / 0.31 s, in-flight 5.6 / 16); 192 running; agg KV 48 running; RR ladders queued.
The measured knee will be reported as the last stationary client count once the ladders complete.

### KV-vs-RR comparison points chosen from these knees

| arm | same config (RR's knee) | both-bounded reference | same SLO, TTFT p95 ≤ 20 s | same SLO, P90 interactivity ≥ 20 tok/s/user |
|---|---|---|---|---|
| disagg 9:9 | **192 clients**: sim KV 2,733 vs RR 2,047 total/GPU (1.34×), TTFT 0.3 vs 12.7 s | 96: 1,395 vs 1,336 (1.04×) | **KV 480 (5,501) vs RR 96 (1,336): 4.1×** | KV 480 vs RR 384 (2,062): 2.7× |
| disagg 12:6 (measured ladder) | **192 clients**: sim KV 2,679 vs RR 2,271 (1.18×), TTFT 0.2 vs 6.5 s | 96: 1,389 vs 1,342 (1.04×) | **KV 480 (5,217) vs RR 96 (1,342): 3.9×** | KV 480 vs RR 384 (2,508): 2.1× |
| agg 24-GPU | **192 clients**: sim KV 3,715 vs RR 3,889 (0.96×), TTFT 0.2 vs 4.4 s | 96: 2,382 vs 2,924 (0.81×) | **KV 192 (3,715) vs RR 96 (2,924): 1.27×** | RR 96 (2,924) vs KV 48 (1,474): KV loses on interactivity (session affinity → bigger decode batches) |

Runners: disagg measured ladder = **12:6** (KV 96/192/384/480/768/1440, then RR 192/96/384) after the 9:9 192-client
point; 9:9 48/96/192 stay as a cross-check; agg RR = full 48 → 1536 on the second fleet, in parallel with agg KV. Rationale and the full sim tables: AGENTX_D72_RESULTS.md §ii, AGENTX_AGG_RESULTS.md §ii.
