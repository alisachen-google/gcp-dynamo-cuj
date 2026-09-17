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

Measured so far (`knee_check.py`, stationarity of TTFT p50 across quarters; updated 12:30 UTC): every finished cell is
**stationary** — disagg 9:9 KV 48 / 96 / 192 (TTFT p95 1.50 / 1.42 / 2.03 s, in-flight 5.6 / 16 / 30), disagg 12:6 KV 96 / 192 / 384 / 480 / 768
(p95 1.37 / 1.77 / 2.58 / 3.52 / 7.00 s, in-flight 17 / 33.5 / 91 / 129 / 237; 15,004 total/GPU at 768, the cell where the sim put the
disagg knee, still pre-knee on silicon at 2.5× the sim's throughput), agg KV 48 / 96 (p95 3.83 / 5.36 s, in-flight 6.8 / 27.5), agg RR 48 / 96 (p95 8.27 / 12.56 s, in-flight
8.5 / 33.2), agg KV 192 (p95 11.68 s, in-flight 74.5, TTFT p50 falling across quarters). Measured knees so far: agg RR at 192, agg KV at 192 (both 384 cells saturated; agg ladders stopped there). The agg programme is complete
(both ladders, flag sweep at 192, tuned at 96). The 12:6 KV 1440 cell, the 12:6 RR pairs and the 12:6 flag sweep are blocked: the GPU node pools were resized to 0 at 23:15 UTC (2026-09-16). Agg KV at 192 is at 9,655 total tok/s per GPU,
already above the sim's agg ceiling (5,138), so the sim's agg knee (1536) is the cell to watch rather than 192.
The measured knee will be reported as the last stationary client count once the ladders complete.

Measured KV-vs-RR pairs available so far (agg, same config, total tok/s per GPU, TTFT p95):

| clients | KV | RR | KV/RR measured | KV/RR sim v3 | TTFT p95 KV vs RR |
|---|---|---|---|---|---|
| 48 | 3,334 | 3,250 | **1.03×** | 0.82× | 3.83 vs 8.27 s (RR 2.2× worse) |
| 96 | 6,844 | 6,137 | **1.12×** | 0.81× | 5.36 vs 12.56 s (RR 2.3× worse) |
| **192** (same-config point) | **9,655** | 6,802 | **1.42×** | 0.96× | 11.7 vs 60.1 s (RR 5.1× worse) |
| 96, tuned KV (scale 3, credit 0.8) | 7,045 | 6,137 | 1.15× | 0.67× | 3.1 vs 12.6 s (RR 4.0× worse) |
| 192, tuned KV (scale 3, credit 0.8) | **11,012** | 6,802 | **1.62×** | 0.70× | 6.3 vs 60.1 s (RR 9.5× worse) |

The sim's ordering on agg below the knee (RR ahead on tokens because KV packs sessions onto one worker) is not
reproduced: measured KV is ahead on every axis and the gap widens with load. The sim's agg decode cliff (TPOT jumps
from 7 + 1.6·bs to 28 + 5.7·bs ms past batch 7) penalises the KV router's larger per-worker batches; the measured ITL
curve has no such cliff (p50 7.3 → 11.9 ms from 1.1 → 4.6 in flight per worker). At 192 the sim's same-config verdict (0.96×, RR ahead) is reversed
on silicon (1.42×, KV ahead), and the **first measured knee is in: agg RR reaches its throughput knee at 192 clients**
(+11% total tokens for 2× clients, TTFT p50 11 s, 103 of 192 sessions in flight; stationary by the q1/q4 test, so it
is a knee, not a collapse). Agg KV is still scaling at 192 (+41%). Same-SLO pair (TTFT p95 ≤ 20 s): KV 192 (9,655,
11.7 s) vs RR 96 (6,137, 12.6 s) = **1.57×**, final: the KV 384 cell is POST-KNEE (TTFT p50 83 s, growing; total tokens fall to 8,257), so **the measured agg KV knee is 192**
against the sim's 1536, and the agg KV ladder was stopped at 384 (768 / 1536 skipped as post-knee) to free the fleet for the flag sweep.

### KV-vs-RR comparison points chosen from these knees (re-analysed 2026-09-16 09:20 UTC, sim v3 incl. tuned KV)

Knee = throughput-slope knee (marginal total-token gain per added client < 25% of the initial slope); TTFT in these tables is **p95** (the busy-stream reports use p95 budgets too). RR's knee is
earlier on every arm because its prefix hit rate is ~0.2–0.4 (each request re-prefills most of its 70 k-token context)
versus ~0.75 for KV routing; the fewer prefill workers an arm has, the higher RR's hit rate (agg 0.40, 6:12 0.40,
9:9 0.30, 12:6 0.23) because a session lands on the same worker more often by chance.

| arm | KV knee (TTFT p95) | tuned-KV knee (TTFT p95) | RR knee (TTFT p95) | RR peak (TTFT p95) | last count with KV TTFT p95 ≤ 5 s |
|---|---|---|---|---|---|
| disagg 12:6 (measured ladder) | 480 (7 s) | 480 (5 s) | 192 (32 s) | 2,508 at 384 (89 s) | 192 |
| disagg 9:9 | 480 (18 s) | 768 (14 s) | 192 (52 s) | 2,062 at 384 (128 s) | 96 |
| disagg 6:12 | 384 (32 s) | 480 (22 s) | 96 (25 s) | 1,631 at 192 (91 s) | 96 |
| agg 24-GPU | 192 (6 s) | 192 (8 s) | 192 (34 s) | 4,156 at 384 (83 s) | 96 |

Chosen points (total tok/s per GPU; gain = KV ÷ RR; all TTFT figures are **p95** from the sim):

| arm | same config at RR's knee (TTFT p95 KV vs RR) | both pre-knee | same SLO, TTFT p95 ≤ 20 s | same SLO, P90 interactivity ≥ 20 tok/s/user | where tuned KV would land (same 20 s p95 budget) |
|---|---|---|---|---|---|
| **disagg 12:6 (measured ladder)** | **192**: KV 2,679 vs RR 2,271 = **1.18×**, TTFT p95 4 vs 32 s | 96: 1,389 vs 1,342 (1.04×), p95 3 vs 10 s | **KV 480 → 5,217 (p95 7 s) vs RR 96 → 1,342 (p95 10 s) = 3.89×** | KV 480 → 5,217 (p95 7 s) vs RR 384 → 2,508 (p95 89 s) = 2.08× | tuned 768 → 6,210 (p95 8 s) (+19% over KV) |
| disagg 9:9 (cross-check) | **192**: KV 2,733 vs RR 2,047 = **1.34×**, TTFT p95 5 vs 52 s | 96: 1,395 vs 1,336 (1.04×), p95 4 vs 16 s | **KV 480 → 5,501 (p95 18 s) vs RR 96 → 1,336 (p95 16 s) = 4.12×** | KV 480 → 5,501 (p95 18 s) vs RR 384 → 2,062 (p95 128 s) = 2.67× | tuned 768 → 7,598 (p95 14 s) (+38% over KV) |
| **agg 24-GPU** | **192**: KV 3,715 vs RR 3,889 = **0.96×**, TTFT p95 6 vs 34 s | 96: 2,382 vs 2,924 (0.81×), p95 4 vs 11 s | **KV 192 → 3,715 (p95 6 s) vs RR 96 → 2,924 (p95 11 s) = 1.27×** | KV 48 → 1,474 (p95 3 s) vs RR 96 → 2,924 (p95 11 s) = 0.50× | tuned 192 → 2,740 (p95 8 s) (-26% over KV) |

What changed versus the first pass: nothing in the chosen client counts — the finer analysis confirms 192 as the
same-config point on every arm and 480-vs-96 (disagg) / 192-vs-96 (agg) for the TTFT-p95 budget — but two findings
sharpen the expectation for the measured runs: (1) on agg, KV-aware routing trades interactivity for TTFT (it packs
a session's turns onto one worker, so decode batches grow: P90 9 vs RR's 15 tok/s/user at 192), so the agg KV-vs-RR
result will look different on the throughput and interactivity axes; (2) the tuned router only pays on disagg past
the prefill knee, so a tuned measured point belongs on 12:6 at 768, not on agg.

Runners: disagg = 12:6 KV 96 / 192 / 384 / 480 / 768 / 1440 then RR 192 / 96 / 384 (np-3, after the 9:9 192-client
cross-check); agg = KV 48 → 1536 and RR 48 → 1536 in parallel on two fleets (np-1). Every chosen cell above is inside
those ladders. A 12:6 tuned-KV point at 768 (`--router-prefill-load-scale 3 --router-kv-overlap-score-credit 0.8`) is
the one extra run the analysis suggests; it is not queued.

## 64-GPU disagg under AgentX (np-2 footprint: 16 × TP4 workers) — topology, knees and KV-vs-RR points (sim, 2026-09-17)

Simulator: the stream-level trajectory-tree replay ([`scripts/dynosim_agentx_v5.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_v5.py),
rule v5b, which tracked 12:6 silicon within 4 % on total tokens at 768 clients) with the engine constants measured on
silicon (prefill 24 k tok/s per TP4 worker, decode 6.9 + 0.44·batch ms); sweep driver
[`scripts/dynosim_agentx_v5_sweep.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_v5_sweep.py),
rows in [`sim-results/dynosim_n3u_agentx_d64_v5.csv`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_agentx_d64_v5.csv).
Cells are total tok/s per GPU · TTFT p95 · P90 interactivity (tok/s/user). Caveats: single seed (load-limited cells
below the knee vary ±20 % with the sampled sessions), and this sim issues subagent turns ~2× too fast, so read client
counts at the knee as ±40 %; rankings of splits and policies are the reliable output.

**Topology scan, KV routing**

| split P:D | 192 | 576 | 768 | 1,152 |
|---|---|---|---|---|
| 12:4 | 3,614 · 1.1 s · 82 | 11,495 · 1.9 s · **14** | | 13,113 · 1.9 s · 4.6 |
| 11:5 | 3,483 · 1.0 s · 96 | 12,177 · 2.1 s · 19.6 | | 16,507 · 2.4 s · 5.8 |
| 10:6 | 3,407 · 1.1 s · 105 | 12,313 · 2.4 s · 25 | | rerunning |
| **8:8** | 3,329 · 1.1 s · 110 | 12,148 · 2.9 s · **36** | running | **24,663** · 12.8 s · 9.3 |
| 7:9 | | 13,256 · 4.7 s · 34 | 19,946 · 7.7 s · 18.4 | 23,036 · **47.7 s** · 9.6 |
| 6:10 | | 13,891 · 6.9 s · 33 | 20,497 · **21.8 s** · 19 | 17,247 · 89 s · 10 (collapsed) |

**Selected split: 8:8.** With a 90–95 % prefix hit rate the prefill tier is lightly loaded under KV routing, so
prefill-heavy splits starve decode (12:4 and 11:5 drop under 20 tok/s/user already at 576 clients and cap at 13–16 k per
GPU), while decode-heavy splits run out of prefill (6:10 crosses TTFT p95 20 s at 768, 7:9 at 1,152). 8:8 holds both
budgets longest and has the highest ceiling (24.7 k at 1,152 with p95 12.8 s). This reverses the prefill-heavy 12:6
choice made from the 4 k-slice sim: that sim's 0.74 hit rate overstated prefill demand 2–3×.

**Policies on 8:8**

| clients | default KV | tuned KV (scale 3, credit 0.8) | RR |
|---|---|---|---|
| 96 | 2,131 · 0.7 s · 116 | | 2,322 · 5.4 s · 110 |
| 192 | 3,329 · 1.1 s · 110 | 3,325 · 1.7 s · 116 | 4,083 · 8.6 s · 96 |
| 384 | 8,379 · 2.9 s · 70 | 8,124 · 2.4 s · 72 | **9,079 · 82 s** · 30 (RR peak) |
| 576 | 12,148 · 2.9 s · 36 | 12,311 · 2.3 s · 36 | 7,075 · 223 s · 18 (collapsing) |
| 768 | 19,034 · 4.4 s · **17.9** | **18,004 · 2.3 s · 20.2** | 5,769 · 368 s · 13 |
| 1,152 | 24,663 · 12.8 s · 9.3 | 24,834 · 3.2 s · 9.5 | 4,694 · 532 s · 7.8 |

**Knees.** RR: throughput knee and peak at 384 clients (6 per GPU), with the TTFT tail already at 82 s there; it leaves
the 20 s budget between 192 and 384. KV (default and tuned): no throughput knee up to 1,152 clients; the binding limit
is interactivity, which crosses 20 tok/s/user at ≈ 768 clients (12 per GPU). Tuned KV's benefit on 8:8 is the tail at
high load (p95 3.2 vs 12.8 s at 1,152), not tokens.

**KV-vs-RR comparison points for the 64-GPU run**

| rule | cell | expectation from the sim |
|---|---|---|
| same config | **384 clients** (RR's peak) | tokens ≈ equal (both load-limited: KV 8,379 vs RR 9,079, within seed noise); TTFT p95 2.9 s vs 82 s; P90 70 vs 30 |
| same config, both inside the TTFT budget | 192 clients | tokens ≈ equal; TTFT p95 1.1 s vs 8.6 s |
| same SLO (TTFT p95 ≤ 20 s and P90 ≥ 20) | **tuned KV 768 vs RR 192** | 18,004 vs 4,083 = **4.4×**; default KV falls just under the interactivity budget at 768 (P90 17.9), so its cell is 576 → 12,148 = 3.0× — the router flags decide the same-SLO point |
| flag sweep | 768 (same-SLO cell) and 384 (same-config cell) | scale 3 / credit 0.8, scale 2 / credit 0.8, temperature 0.5 at 768; scale 3 at 384 |

**Recipes (chain launched 2026-09-17 06:04 UTC; stage 1 is waiting at its capacity gate — np-2's 16 nodes belong to other tenants today).** Fleet manifest
[`sglang/manifests/n3u-mnnvl-88.yaml`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-88.yaml)
(8 prefill + 8 decode TP4 workers, one 16-node MNNVL ComputeDomain, pinned to np-2); chain
[`scripts/run_agentx_88_np2.sh`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/run_agentx_88_np2.sh):
KV 192 / 384 / 576 / 768 / 960 → RR 192 / 384 / 96 → measured point selection
([`scripts/select_agentx_points.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/select_agentx_points.py): same config = RR's peak cell; same SLO = best stationary cell per policy with TTFT p95 ≤ 20 s and P90 ≥ 20; validated on the agg ladder) → flag sweep at the selected cells (sim's 768 / 384 as fallback), each stage gated on the previous DONE marker, first point a
smoke, MNNVL transport guard after every cell (`MNNVL_GUARD=1`), aiperf pod co-located on np-2 (`BENCH_POOL=np-2`). The
KV runner waits until 16 np-2 nodes are free and never deletes anything it did not create. Wall time ≈ 24 h for the 12 cells.
