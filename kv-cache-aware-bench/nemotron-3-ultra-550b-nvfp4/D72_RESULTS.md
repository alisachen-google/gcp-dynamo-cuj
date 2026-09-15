# Nemotron-3-Ultra 550B: 72-GPU Disaggregated KV-vs-RR — Silicon Results + Drift

## Links (reports, curves, evidence)

| What | Link |
|---|---|
| **Disagg real-perf curve** (interactive: MNNVL/host-staged × KV/RR, sim dashed, agg rule, knee markers) | [rendered](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-disagg-curve.html) · [artifact](https://claude.ai/code/artifact/5342de51-ce44-4a08-b434-51178e0030ab) |
| DynoSim pareto curves (agg 24-GPU + disagg 72-GPU 6:12) | [rendered](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-curves.html) |
| Agg silicon curves (KV vs RR) | [rendered](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/agg-silicon-curves.html) |
| Aggregated technical report | [AGG24_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGG24_RESULTS.md) |
| Knee-point analysis (sim vs silicon, KV/RR, agg/disagg) | [KNEE_ANALYSIS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/KNEE_ANALYSIS.md) |
| GPUDirect regression + filed bug | [GPUDIRECT_REGRESSION.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/GPUDIRECT_REGRESSION.md) · [BUG_GPUDIRECT.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/BUG_GPUDIRECT.md) |
| InferenceX AgentX (dsv4/GB300) vs our study — concurrency semantics, knee, **P90 interactivity** | [AGENTX_COMPARISON.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_COMPARISON.md) |
| Profiled agg-vs-disagg gap analysis | `profiles/GAP_ANALYSIS.md` (generated when the profiled comparison lands) |


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


## Deep-saturation extension (conc 615 / 1024; user-directed)

| point | tok/s | req/s | TTFT p50/p95 |
|---|---|---|---|
| KV c615 | 2,683 | 5.13 | 165 / 263 s |
| KV c1024 | 3,324 | 6.52 | 232 / 309 s |
| RR c615 | 1,557 | 3.13 | 305 / 471 s |
| RR c1024 | 1,571 | 3.33 | 452 / 593 s |

(AIPERF_HTTP_CONNECTION_LIMIT raised 200→1100 for these points — the template
default would have silently capped effective concurrency.) Verdicts: KV's
~3.3–3.4k ceiling holds to conc 1024 with no deep-saturation collapse (mild
sag at 615 within run variance); RR's plateau sags to ~1.56–1.57k; the **~2×
KV-over-RR saturation gap persists to conc 1024** (2.1× at the top). TTFT is
pure queue arithmetic at these depths (Little's-law forecast ~3/~7 min vs
measured 3.9/7.5 min p50) — confirming, as argued pre-run, that beyond-ceiling
concurrency buys only queue depth.

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


## MNNVL+mooncake re-run (2026-09-12/13; COMPLETE) — transport, disagg-vs-agg, sim drift

Motivation: GPUDirect RDMA is broken on the rebuilt node image (see
`GPUDIRECT_REGRESSION.md`), so the host-staged disagg numbers above carry a
~2.4 s/request transfer floor. MNNVL routes KV over NVLink within the NVL72
domain, sidestepping the fault. Stack: **sglang 0.5.16 + dynamo 1.4.2 +
flashinfer 0.6.18** (newest runnable pair; base image `lmsysorg/sglang:v0.5.19-cu130-runtime`;
the old set was 0.5.14+1.3.1), `--disaggregation-transfer-backend mooncake`,
`MC_FORCE_MNNVL=1`, ComputeDomain channel per worker, no mrdma/NET_DEVICES.
**Every point transport-gated and PASSED** (no transfer failures / fallback on the data
path; see "Transport verification — what the evidence actually is" below for the
corrected evidence basis). Knee verdicts from
per-request timestamp stationarity (`knee_check.py`), zero request errors.
NOTE: MNNVL-vs-host-staged folds in a minor version bump (0.5.14→0.5.16); the
clean transport isolation is the same-stack reproducer A/B (0.34 GB/s staged
vs NVLink cuda_ipc, `scripts/gdr-reproducer/`). The new-stack agg re-sweep
(in progress) isolates the pure engine-version delta.

### Results (output tok/s; 72 GPU, 6:12 TP4/EP4; all 8 points MNNVL-gate PASS)

| conc | MNNVL KV (/GPU) · TTFT p50 · knee | host-staged KV | MNNVL/HS | DynoSim v1 KV | sim/MNNVL |
|---|---|---|---|---|---|
| 12 | 1,734 (24.1) · 0.37 s · AT/PRE | 1,394 | 1.24× | (grid starts c48) | — |
| 24 | 2,672 (37.1) · 0.79 s · AT/PRE | 2,043 | **1.31×** | — | — |
| 48 | **3,573 (49.6) · 3.23 s · AT/PRE** | 2,853 | **1.25×** | 4,725 | **1.32×** |
| 96 | 3,691 (51.3) · 16.2 s · POST | 3,423 | 1.08× | 5,813 | **1.58×** |

| conc | MNNVL RR (/GPU) · TTFT p50 · knee | host-staged RR | MNNVL/HS | DynoSim v1 RR | sim/MNNVL |
|---|---|---|---|---|---|
| 12 | 1,381 (19.2) · 1.32 s · AT/PRE | 1,135 | 1.22× | (c48+) | — |
| 24 | **1,980 (27.5) · 3.10 s · AT/PRE** | 1,500 | 1.32× | — | — |
| 48 | 2,286 (31.8) · 9.03 s · POST | 1,646 | 1.39× | 3,196 | 1.40× |
| 96 | 2,460 (34.2) · 23.9 s · POST (sat) | 1,746 | 1.41× | 3,595 | 1.46× |

**KV peak bounded = 3,573 tok/s @c48 (49.6/GPU)**; ceiling ~3,691 post-knee.
**RR peak bounded = 1,980 @c24 (27.5/GPU)**. KV/RR: same-cell both-bounded
1.26× (c12) → 1.35× (c24); peak-bounded per-GPU **1.80×**.


### Extended concurrency (2026-09-14; c144–512 on the same fleet/stack; RR 384/512 not run)

| conc | MNNVL KV (/GPU) · TTFT p50 · knee · errors | MNNVL RR (/GPU) · TTFT p50 · knee · errors | DynoSim v1 KV | sim/real KV |
|---|---|---|---|---|
| 144 | **6,221 (86.4) · 11.5 s · AT/PRE · 0%** — *unreproduced spike, see note* | 2,078 (28.9) · 49.7 s · POST · **0.8%** | 5,920 | 0.95× |
| 192 | 2,707 (37.6) · 63.7 s · POST · 0% | 1,931 (26.8) · 83.4 s · POST · 0% | 5,613 | 2.07× |
| 288 | 2,489 (34.6) · 97.5 s · POST · 0% | ~~1,754~~ **INVALID** — 83/5,782 requests failed (1.4%): `Decode transfer failed` burst at 16:31 (mooncake transfer timeouts under 78 s p50 / 294 s p99 queueing); runner halted per transport policy | 4,947 | 1.99× |
| 384 / 512 | KV pending (queued after reproduction) | not run — RR already fails transfers at c288 | 4,397 / 3,703 | — |

**Reproduction (2026-09-15, same fleet instance as c144–288): instance 1 was degraded, not c144.**

| point | fleet instance 1 (2026-09-12/13) | fleet instance 2 (2026-09-14/15) |
|---|---|---|
| kv:48 | 3,573 (49.6/GPU) · AT/PRE · 3.2 s · 10,976 req | **4,684 (65.1/GPU) · TTFT p50 1.0 s · 14,823 req · 0 err** (re-verified 2026-09-15) |
| kv:96 | 3,691 · POST-knee · TTFT 16 s growing · 11,551 req | **4,807 (66.8/GPU) · AT/PRE · 9.6 s stationary · 15,331 req · 0 err** |
| kv:120 | — | **4,878 (67.7/GPU) · AT/PRE · 13.9 s stationary · 15,642 req · 0 err** |
| kv:144 | 6,221 (86.4/GPU) · AT/PRE · 11.5 s — **not reproduced** | **4,562 (63.4/GPU) · POST-knee · 21.5 s (stationary, saturated) · 14,699 req · 1 err** |
| kv:384 | — | 3,082 (42.8/GPU) · POST-knee · 108 s · 10,256 req · 0 err |
| kv:512 | — | 3,495 (48.5/GPU) · POST-knee · 120 s · 11,792 req · 0 err |

**Outcome.** Instance 1 was low across the ladder (−31% at c48, −25% at c96) and high once
(+36% at c144); instance 2 gives a smooth, nearly flat curve: **4,684 (c48) → 4,807 (c96)
→ 4,878 (c120, peak bounded) → 4,562 (c144, post-knee)** → 3,082 / 3,495 (c384 / c512,
saturated). The 6,221 point does not reproduce and is disregarded. **Operating-point
note:** KV delivers 65.1/GPU already at c48 with TTFT p50 **1.0 s**, vs 67.7/GPU at the
c120 peak with p50 13.9 s — 96% of peak throughput at 14× lower latency; c48 is the
sensible deployment cell.

**Revised disagg-vs-agg verdict: parity at the bounded operating point.** KV disagg
peak-bounded **67.7 tok/s/GPU (c120) vs agg 69.0 (c32) = 0.98×**. Post-knee ceilings
still favour agg (85 vs ~68/GPU, 1.25×). Neither the earlier "agg wins 1.39×" (built on
instance-1 cells) nor "disagg wins" (the unreproduced 6,221) survives. Remaining
caveat: c12–48 and all RR cells are still instance-1 measurements — kv:48 / rr:48 /
rr:96 are being re-run on instance 2 before the KV/RR ratios are re-issued. Sim drift
at c96–144 is 0.77–0.83× (real/sim), from 0.63× on instance 1. The protocol now
includes a cross-instance repeat of one anchor cell; the cause of instance 1's
variance was not isolated (cache warmth excluded — both instances had 8+ prior points
and 900 s warm-ups per point).

**RR at deep saturation fails, not just slows.** From c144 on, RR shows request
errors (0.8% at c144, 1.4% at c288) from KV-transfer timeouts while queued behind
50–80 s TTFTs; KV at the same concurrencies has 0%. This is a second, qualitative KV
advantage: placement keeps the prefill queue short enough that hand-offs do not time
out. RR 384/512 were therefore not run.

### Profiling at kv:144 (read-only live capture over the measurement window) — where the disagg drift comes from

| tier | GPUs | mean util | ≥90% busy | idle (<10%) | scheduler state (SGLang step lines) |
|---|---|---|---|---|---|
| prefill | 24 | **92.3%** | 83.2% | 0.1% | 1 batch in flight (running-req 0), **queue-req 10.3/worker**, 2.2 new-seq/step, 14.6k new tok/step, **cached-token share 89.1%** |
| decode | 48 | **99.6%** | 99.4% | 0.0% | **running-req 9.7/worker of 64 slots** (max 61), 815 gen tok/s/worker, queue 0 |

Reading: **the prefill tier is the binding constraint** — every prefill worker has
~10 requests queued while running one batch at a time, so decode workers sit at
~15% slot occupancy (9.7/64) waiting to be fed. Decode GPUs read "99% busy" because
Mamba/MoE decode kernels saturate the SM even at small batch — busy is not
productive; 48 GPUs generate at low batch efficiency. This is the mechanism behind
the per-GPU gap: the 24 prefill GPUs cap the rate at which 48 decode GPUs receive
work, and aggregated workers avoid the hand-off entirely.

Drift attribution for disagg, revised with this evidence:
- **Hit rate is not the disagg drift source.** Measured cached-token share is 89%
  at c144 — *above* the sim's 84%, the opposite of agg (63%). KV routing across 6
  prefill workers concentrates each trace slice well. So the reuse model
  over-predicts nothing here.
- **The drift is the prefill-tier queueing the sim does not model.** With 89%
  reuse, only ~11% of tokens are new, yet the prefill tier still runs 92% busy with
  a 10-deep queue — chunked 16k prefills of 256K-context requests are expensive
  even when mostly cached (the sim's near-linear prefill rate is too optimistic
  at this ISL), and the sim treats hand-off as free. That is why real throughput
  sits 1.3–2× under the sim at c48–288 while, at the one point where both tiers
  happen to be fully fed (c144), real ≈ sim.
- The AIC decode batch-slope error (agg finding) matters less here: decode runs at
  batch ~10 where the slope difference is small; decode is starved, not slow.


### Transport verification — what the evidence actually is (corrected 2026-09-15)

**The MNNVL KV path is mooncake, not UCX.** `--disaggregation-transfer-backend mooncake`
with `MC_FORCE_MNNVL=1`; the pods contain **0 UCX and 0 NIXL log lines** and 2,812
mooncake transfer-engine lines. The `UCX_*` variables in the manifest are inert
carry-overs from the dsv4 recipe. Guard v1's positive check counted `cuda_ipc|mnnvl`
lines, and the `mnnvl` half matched the deployment *name* (`n3u-mnnvl-full`) inside
ordinary dynamo request logs — so the "cuda_ipc evidence = 34k–71k" figures reported
earlier were not transport evidence and are withdrawn. v1's *negative* check was
sound (it caught rr:288's real `Decode transfer failed` burst). Guard v2
(`scripts/mnnvl_transport_guard.sh`) gates on evidence that is valid post-run:

| check | evidence (live 6:12 fleet, 2026-09-15) |
|---|---|
| no transfer failures / fallback in the run window | 0 on all clean points; 83 on rr:288 (caught) |
| mooncake Transfer Engine moved KV | `Transfer Engine Stats … Throughput` peak 2.0 GB/s per prefill worker |
| RDMA path physically absent | `/dev/infiniband` **missing** in the pod (no mrdma claim) — RDMA/UCX cannot carry KV |
| mooncake pinned to MNNVL | `MC_FORCE_MNNVL=1` in the pod env |

Live physical proof during a running point (`scripts/mnnvl_live_probe.sh`, decode
GPU0, 10 s): **NVLink Rx ≈ 3.65 GB, Tx ≈ 0.96 GB** per link (Rx ≫ Tx = a decode GPU
*receiving* KV), **eth0 ≈ 2 MiB** (the NIC is idle — KV is not on TCP), IMEX
`channel0` present (the cross-node cuda_ipc prerequisite), GPU Fabric GUIDs
populated. Bytes leave prefill and arrive at decode over NVLink with the NIC idle
and no RDMA device in the pod — that, not a log-line count, is the transport proof.

### 1. The transport win (MNNVL vs host-staged)
NVLink beats host-staging at every point. KV TTFT p50 **2.67 → 0.37 s (7×)** at
c12. KV throughput +24–31% through c48, shrinking to +8% at c96 — because KV is
**compute-saturated** there and transport is no longer the bottleneck. RR's gain
*grows* with load (1.22 → 1.41×): RR's poorer placement keeps it
transfer-limited longer. Transfer tax removal is real but bounded.


**Measured MNNVL KV transfer rate (mooncake transfer-engine metrics, `MC_TE_METRIC=true`,
5-s windows, all 6 prefill workers, 6 h of the c96–c384 runs, 15,890 samples):**
per prefill worker mean **0.44 GB/s**, p50 0.36, p95 **1.06**, p99 1.39, **peak 2.00 GB/s**;
fleet aggregate ≈ 6.4 GB/s at p95, ≈ 12 GB/s at peak. These are *observed* send rates —
demand-limited by how much KV is handed off per window — so the 2.0 GB/s peak is a
lower bound on the cuda_ipc/NVLink path's capacity, not its ceiling (NVLink-5 per-GPU
bandwidth is three orders of magnitude higher). Per request: mean ISL ≈ 80k tokens ×
~6 KB/token ≈ 0.48 GB attention KV + ~0.2 GB Mamba state ≈ **0.7 GB per hand-off**,
i.e. ~0.35–0.5 s at the p95 rate vs ~2.4 s on host-staged (0.28–0.34 GB/s, reproducer
+ TTFT floor). Transfer is therefore no longer the binding constraint on MNNVL — the
prefill tier's compute queue is (see profiling at kv:144).

### 2. Disagg-vs-agg — revised verdict: PARITY on tok/s/GPU at the bounded point (0.98×); agg wins post-knee (1.25×); **disagg wins on interactivity (P90 ~33 vs 19–24 tok/s/user, see AGENTX_COMPARISON.md §4)**
agg bounded reference **69.0 tok/s/GPU** (KV c32, 24 GPU); agg post-knee 85.
- KV disagg peak bounded **49.6/GPU** vs agg **69.0** → **agg wins 1.39×**.
- Post-knee ceilings: disagg 51.3 vs agg 85 → agg wins 1.66×.
Removing the transfer tax narrowed the gap (host-staged best was 47.4/GPU
post-knee) but did not close it. **Aggregated serving wins per-GPU for this
architecture on both transports.** The remaining gap is not transfer — it is
tier imbalance: cheap linear prefill (Mamba + 12/108 attention) leaves the
24-GPU prefill tier bursty while 48 decode GPUs wait; agg keeps every GPU
doing both phases. (Profiling comparison to attribute this in detail —
in progress; see §"Profiling gap analysis" when landed.)

### 3. DynoSim drift — narrowed, not closed; widens with load
sim/MNNVL: **1.32× (c48) → 1.58× (c96)** (was 1.66×/1.70× vs host-staged).
Decomposition of the sim − host-staged gap:

| conc | sim − HS | transfer tax recovered (MNNVL − HS) | residual model optimism (sim − MNNVL) |
|---|---|---|---|
| 48 | 1,872 | 720 (**38%**) | 1,152 (**62%**) |
| 96 | 2,390 | 268 (**11%**) | 2,122 (**89%**) |

Transfer was the *minority* of the drift; the majority is genuine model
optimism, and it dominates at saturation because real KV plateaus ~3.7k while
the sim projects 5.8k. Two sources, both now identified: (a) the AIC decode
batch-slope (6× too shallow, as on agg), and (b) **the AIC seed was labelled
0.5.14 but AIC 0.11.0's gb300 SGLang DB tops out at 0.5.12** — the seed was an
extrapolation above the DB's newest real point. No AIC release ships a
0.5.14/0.5.16 DB (0.11.0 is the latest on PyPI), so the sim is re-anchored by
**DynoSim v3 recalibration from measured 0.5.16 silicon** (agg + disagg), not
by an AIC re-solve.

### 4. Topology
**9:9 on MNNVL, kv:48 (2026-09-15): 4,555 tok/s (63.3/GPU), TTFT p50 0.56 s, 14,286 req,
0 err — vs 6:12 kv:48 4,684 (65.1/GPU), p50 1.0 s.** At c48 the two splits are within 3%
on throughput, with 9:9 lower-latency (more prefill workers → shorter prefill queue).
9:9 kv:96 / kv:144 (6:12's peak region) are queued to decide the split at the peak.

6:12 is the optimal split among all silicon-measured splits (6:12 ≫ 3:15 on
host-staged; 6:12 complete on MNNVL). The sim-preferred 9:9 is being verified
on MNNVL (KV c48/96/144, peak-bounded tok/s/GPU vs 6:12's 49.6) — first
attempt failed at 0/9 prefill pods (three stale ComputeDomains held the
nodes' IMEX channels; a node belongs to one CD); re-run with CD cleanup +
domain-availability wait in progress.



## KV vs RR — real jobs vs DynoSim v1: throughput and TTFT p50 / p99 (disagg 6:12 MNNVL)

Real = aiperf summaries (instance-2 reproduced runs from c96 up; c12–48 and RR ≤c96 are
instance-1 pending re-verification). Sim = DynoSim v1 `kv-nvda` / `rr`, 6:12 grid
(starts at c48). TTFT in seconds. "—" = not in the sim grid. rr:288 shown but INVALID
(83 transfer failures); rr:144 had 56 errors.

| conc | **Real KV** tok/s · p50 · p99 | **Sim KV** tok/s · p50 · p99 | **Real RR** tok/s · p50 · p99 | **Sim RR** tok/s · p50 · p99 | KV/RR thr gain real · sim |
|---|---|---|---|---|---|
| 12 | 1,734 · 0.37 · 7.4 | — | 1,381 · 1.32 · 19.8 | — | 1.26× · — |
| 24 | 2,672 · 0.79 · 16.3 | — | 1,980 · 3.10 · 30.7 | — | 1.35× · — |
| 48 | 3,573 · 3.23 · 33.2 | 4,725 · 0.14 · 4.1 | 2,286 · 9.03 · 62.9 | 3,196 · 2.46 · 41.8 | 1.56× · 1.48× |
| 96 | 4,807 · 9.57 · 47.3 | 5,813 · 0.19 · 6.1 | 2,460 · 23.9 · 132 | 3,595 · 8.48 · 75.5 | 1.95× · 1.62× |
| 120 | **4,878 · 13.9 · 51.5** (KV peak bounded) | ~5,870 (interp.) | — | — | — |
| 144 | 4,562 · 21.5 · 67.6 (post-knee) | 5,920 · 0.42 · 6.9 | 2,078 · 49.7 · 262 (56 err) | 3,501 · 18.8 · 105 | 2.20× · 1.69× |
| 192 | 2,707 · 63.7 · 141 | 5,613 · 1.93 · 13.4 | 1,931 · 83.4 · 232 | 3,363 · 30.6 · 134 | 1.40× · 1.67× |
| 288 | 2,489 · 97.5 · 247 | 4,947 · 4.07 · 20.6 | ~~1,754 · 132 · 327~~ INVALID | 3,065 · 51.1 · 187 | — · 1.61× |

### What the table says

1. **Throughput drift is moderate; TTFT drift is enormous, and it is KV-specific.**
   Real/sim throughput: KV 0.76–0.83× (c48–144), RR 0.59–0.72×. But real KV TTFT p50 is
   **23× (c48), 50× (c96), 51× (c144)** the sim's; p99 is 8–10×. For RR the p50 gap is
   only 2.6–3.7× and p99 1.5–2.5×. The sim's KV TTFT (0.14–0.42 s through c144) is
   essentially "prefill of the uncached suffix with no waiting" — it models the
   placement benefit (hit rate) but not the **prefill-tier queue** that the kv:144
   profiling measured directly (10 requests queued per prefill worker, tier 92% busy).
   RR's TTFT is dominated by recompute, which the sim does model, so RR drifts less.
2. **The sim therefore overstates KV's *latency* advantage and understates its
   *throughput* advantage.** Sim RR/KV p50 ratio: 18× (c48) → 45× (c96–144); real: 2.8×
   → 2.3×. Sim KV/RR throughput gain 1.5–1.7×; real 1.56× → **1.95× (c96) → 2.20×
   (c144)**, growing with load as RR loses more to recompute and, from c144, to
   transfer failures.
3. **Real KV tail latency at the bounded peak is high**: p99 51 s at c120 (p50 13.9 s).
   That is the price of queue-stationary operation deep in the concurrency range —
   the study reports it and does not gate on it (no SLO gate); a deployment with a
   TTFT SLO would pick a lower cell (c24: p50 0.8 s / p99 16 s at 2,672 tok/s).
4. **Knee locations agree better than latencies**: sim KV p50 turns super-linear at
   c144–192, real KV knees at c120→144 — matched within one step (see
   `KNEE_ANALYSIS.md`), even though the absolute TTFT is off by 20–50×. The sim's
   *shape* is right; its *queueing constant* is missing.

For the sim: the v3 recalibration (from measured 0.5.16 silicon) needs a prefill-tier
queueing term — a per-worker service model with the measured ~10-deep queue at c144 —
not just the decode-slope and hit-rate corrections identified on agg.

## Knee points — simulation vs silicon (KV / RR, agg / disagg)

Full analysis: [KNEE_ANALYSIS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/KNEE_ANALYSIS.md). Same rule both worlds —
silicon knee = last concurrency whose queue stays stationary over the window
(`knee_check.py`); sim knee = last concurrency before TTFT p50 growth turns
super-linear in concurrency. No SLO gate either side.

| arm | policy | sim knee (v1) | silicon knee | sim / real |
|---|---|---|---|---|
| agg 24 GPU | KV | c96 → 128 | **c48** | 2–2.7× high |
| agg 24 GPU | RR | c24 | **c32** | ~matched |
| disagg 6:12 MNNVL | KV | c96 | **c48 → 96** | up to 2× high |
| disagg 6:12 MNNVL | RR | c48 | **c24 → 48** | up to 2× high |
| disagg 6:12 host-staged | KV | (same) | ~c16 → 24 | 4–6× high |

- **The sim over-places every KV knee (~2×)** — the flat AIC decode slope lets
  the modelled decode tier absorb more concurrency before backing up, and the
  sim's 84% hit rate (vs ~63% measured) under-models prefill load so KV backs
  up later. **RR knees are near-matched** because RR gets no reuse benefit —
  the asymmetry pins the KV-specific error on the reuse/hit-rate model.
- **Transport moves the knee 3–6× but the ceiling only 8–31%**: the host-staged
  transfer floor was a queueing tax that blocked the decode tier at low load;
  NVLink removes it and the knee jumps, but the ceiling is compute-set.
- RR always knees before KV; disagg's KV knee is deeper than agg's (48–96 vs
  48) because 12 decode workers hold 3× the in-flight capacity — at 3× the
  GPUs, so per-GPU still favours agg.

## Reproduction

Arm `manifests/n3u-d72.yaml`; sequencer `scripts/sweep_n3u_d72.sh` (3
revisions — ladder re-anchored after silicon knees came in far below sim;
halts preserved in DIARY.md); artifacts
`gs://alisachen-models/perf/17883*_alisachen-n3u-d72-{kv,rr}-c{12,24,48,96}/`;
summaries in `results/silicon/` (`d72-*.json`).
