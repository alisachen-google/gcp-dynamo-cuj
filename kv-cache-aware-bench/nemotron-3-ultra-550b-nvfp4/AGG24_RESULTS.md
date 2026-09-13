# Nemotron-3-Ultra-550B-A55B-NVFP4 — 24-GPU Aggregated Serving: KV-Aware vs Round-Robin Routing

**Technical Report — Silicon Results with Simulation Cross-Validation**
Study: KV-cache-aware routing benchmark (NVIDIA Dynamo + SGLang, GB300 NVL72)
First silicon comparison for the study's second model. Data collected 2026-09-01;
live KV-flag sweep 2026-09-11. Cluster: GKE GB300 NVL72, nodepool `np-3`.

---

## 1. Summary

On 24 aggregated GPUs serving the 256K agentic Weka trace, **KV-aware routing
delivers 1.67× the throughput of round-robin at the strongest fair (both-bounded)
operating point (concurrency 32), with 3.9× better TTFT p95** (4.3 s vs 16.8 s).
Under a strict interactive SLO the advantage becomes categorical: KV-routed
aggregation is the *only* Nemotron configuration (any topology) that serves this
workload within a 5 s TTFT p95 budget at all. The gain is pure request
*placement* — at these input lengths the cache never evicts, so routing is the
only free variable. Simulation (AIC + DynoSim) predicted the direction and the
both-bounded cells correctly; it over-predicted absolute throughput ~1.9–2.8×,
a drift traced by component substitution to a single cause: the AIC decode
batch-scaling curve for this hybrid architecture.

---

## 2. Experimental Setup

### 2.1 Model and workload
- **Model:** [Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
  (`nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`) — hybrid Mamba-2 /
  Latent-MoE, 108 layers (48 Mamba + 48 Latent-MoE + 12 attention), NVFP4 (MoE)
  / FP8 (Mamba mixers) mixed quantization, ~6 KB/token attention KV + ~200 MB
  Mamba state/request. Base (BF16) checkpoint:
  [`nvidia/Nemotron-3-Ultra-550B-A55B`](https://huggingface.co/nvidia/Nemotron-3-Ultra-550B-A55B).
- **Workload:** [`semianalysisai/cc-traces-weka-062126-256k`](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k)
  agentic trace, native deterministic replay (393 sessions, seed 42), 256K
  context.

### 2.2 Hardware and software provenance

**These agg results were collected on the study's *pinned* silicon pair
(SGLang 0.5.14 + Dynamo 1.3.1)** — the pair anchored to the AIC simulation
database version. (The N3U *disagg* MNNVL runs use a newer stack — SGLang 0.5.16
+ Dynamo 1.4.2 + FlashInfer 0.6.18 — recorded in `D72_RESULTS.md`; the two are
kept distinct so agg↔sim comparisons stay version-consistent.)

| Component | Version / image | Role |
|---|---|---|
| Serving engine image | `lmsysorg/sglang:v0.5.14-cu130-runtime` | prefill+decode workers (agg) |
| Dynamo runtime glue | `ai-dynamo[sglang]==1.3.1` (pip at pod start) | SGLang↔Dynamo worker bridge |
| Frontend image | `nvcr.io/nvidia/ai-dynamo/tensorrtllm-runtime:1.3.1` | `dynamo.frontend` + KV router |
| Engine config | `--tp-size 4 --ep-size 4 --quantization modelopt_fp4` | 6 × TP4/EP4 workers = 24 GPUs |
| Platform | GKE GB300 NVL72, np-3, driver 580.126.20, kernel 6.12.85+ | — |
| **AIC** | `aiconfigurator==0.11.0`, sglang DB **0.5.14**, `plotext==5.3.2` | analytic solve (§3.1) |
| **DynoSim** | `n3u-sim` constants v1 (AIC-seeded) → v2 (silicon-recalibrated) | trace-replay sim (§3.2) |

Provenance note: AIC 0.10.0 does **not** support NemotronH (2/6 architecture
vote); **0.11.0 does** (5/6), and its SGLang database version is 0.5.14 — which
is why the silicon serving pair is pinned to 0.5.14 / 1.3.1, so the simulation
and the hardware model the *same* engine build.

### 2.3 Methodology
Single fleet, per-point frontend router swap (workers untouched). Per point:
fresh frontend → **300 s settle → 900 s trace-replay warmup under the point's
own router config → 1800 s measured window**. Zero request errors required at
every reported point. Boundedness by per-request timestamp stationarity
(`knee_check.py`); **no latency SLO gate** (methodology rev 3) — latency is
reported, never used to admit/reject a point. Every cell verified for
KV-routing liveness (`kv_hit_rate` histogram + engine `cached_tokens`, to
exclude silent load-routing fallback).

**Swept axes:** router policy (kv vs round-robin, frontend-only swap);
concurrency ladder 16/32/64/128 (sim-bracketed) + bisection in-fills at 8/48.
**Held fixed:** worker shape (6×TP4/EP4 from AIC), engine flags, dataset +
replay mode, per-point protocol — so every delta is attributable to a swept
axis.

---

## 3. Simulation Stage — What AIC and DynoSim Deliver

The study runs a two-stage simulation pipeline *before* silicon, both in
`--database-mode SILICON`. They answer different questions:

### 3.1 AIC (aiconfigurator 0.11.0) — the analytic engine-shape solver
A closed-form performance model over NVIDIA's measured GB300×SGLang database.
**Delivers:**
- **Engine shape:** the worker parallelism to deploy. For N3U it selected
  **TP4 / EP4** (weights ~352 GB don't fit a TP2 worker's HBM at 256K context;
  AIC ranked TP4 over TP2).
- **Per-component rate curves:** prefill throughput (tok/s/worker) and decode
  TPOT-vs-batch — the raw constants DynoSim is seeded from.
- **Concurrency ceilings** and an **agg-vs-disagg ranking**: AIC ranked
  aggregation *over* disaggregation for this architecture (1.09–1.63×), which is
  why agg is the primary comparison and the study's silicon budget centers here.

AIC does **not** model request placement, cache reuse, or queue dynamics — it is
a per-worker capacity model. That is DynoSim's job.

### 3.2 DynoSim (`n3u-sim` constants) — the trace-replay policy simulator
A discrete-event simulator that replays the *actual* Weka trace through the
Dynamo router, seeded from AIC's rate curves. **Delivers, per policy (KV/RR)
across the concurrency ladder:** throughput, TTFT/TPOT percentiles, cache
**hit-rate**, **knee locations**, and the framing-1/2/3 **cell selections** that
tell the silicon sweep which cells to run. It models what AIC cannot: KV-router
placement scoring, radix cache reuse, and queueing.

### 3.3 Simulation predictions (fed into the silicon sweep)

| DynoSim prediction (n3u-sim v1) | Value |
|---|---|
| Same-cell KV/RR gain, rising with conc | 1.19× → 1.46× |
| RR knee | ≈ conc 32 |
| KV knee | ≈ conc 128 |
| Both-bounded cells (framing 1) | conc 16 & 32 |
| Absolute KV throughput @ c64 | 5,127 tok/s |

---

## 4. Silicon Results

### 4.1 Throughput and latency (output tok/s, 24 GPUs)

| conc | KV tok/s (/GPU) | KV TTFT p50/p95 | RR tok/s (/GPU) | RR TTFT p50/p95 | gain | knee verdict |
|---|---|---|---|---|---|---|
| 16 | 1,246 (51.9) | 0.56 / 4.0 s | 882 (36.7) | 1.10 / 9.8 s | **1.41×** | **both stationary** |
| 32 | 1,657 (69.0) | 0.62 / 4.3 s | 993 (41.4) | 1.45 / 16.8 s | **1.67×** | **both stationary** |
| 64 | 2,045 (85.2) | 1.09 / 21.5 s | 1,141 (47.5) | 5.13 / 29.4 s | 1.79× | both growing (post) |
| 128 | 1,922 (80.1) | 28.8 / 106 s | 1,096 (45.7) | 34.3 / 129 s | 1.75× | both saturated |

ITL p50 at the bounded cells: 13.5–18.0 ms (KV) / 12.4–19.3 ms (RR).

### 4.2 Headline (framings 1 & 2)
- **Framing 1 (both-bounded, same cell):** conc 16 and 32 both qualify. At
  conc 32 — the strongest fair cell — **KV = 1.67× RR throughput with 3.9×
  better TTFT p95** (4.3 vs 16.8 s), both queues stationary.
- **Framing 2 collapses into framing 1 for this model:** both policies knee
  between conc 32 and 64, so each policy's best bounded cell is the *same* cell.
  The honest headline is simply **1.67× at equal, healthy load**. (Contrast
  Kimi, where the story was a knee *shift*; here KV wins on throughput at
  identical operating points.)
- **Hybrid-architecture hypothesis confirmed on silicon:** with an effectively
  unbounded cache (nothing evicts at these ISLs), the KV gain can only be
  placement. Silicon gains (1.41–1.79×) *exceed* the sim's placement-only
  prediction (1.19–1.46×) at every cell.

### 4.3 KV vs RR under a fixed SLO (framing 3)
Fix a TTFT p95 budget; compare each policy's best throughput among compliant,
queue-stationary runs (details in `SLO_COMPARISON.md`):

| TTFT p95 SLO | KV max compliant | RR max compliant | KV impact |
|---|---|---|---|
| **≤ 5 s** | **1,657 tok/s @ conc 32** (4.3 s) | **none** — RR misses 5 s even at conc 8 (7.8 s; recompute tail, not queueing) | **servable vs not servable** |
| ≤ 10 s | 1,853 @ conc 48 (7.8 s) | 882 @ conc 16 (9.8 s) | **2.1× throughput, 3× concurrency** |
| ≤ 30 s | 2,045 @ conc 64 | 1,141 @ conc 64 | 1.8× |

At a strict interactive SLO the router flag is binary — KV-routed agg is the
only Nemotron configuration (any topology) that serves this workload at all. At
relaxed SLOs, KV converts the same latency budget into ~2× the tokens and ~3×
the concurrent users.

### 4.4 Live KV-flag sweep at conc 48 (2026-09-11; wspt/decay grid, temp 0 fixed)

| variant | tok/s (/GPU) | TTFT p50/p95 | vs control |
|---|---|---|---|
| **control (fcfs, scale 1.0, defaults)** | **1,860 (77.5)** | 0.75 / 9.7 s | — |
| wspt + scale 1.0 + decay 0.50 | 1,820 (75.8) | 0.70 / 8.2 s | −2.1% |
| wspt + scale 1.0 + decay 0.65 | 1,803 (75.1) | 0.71 / 6.8 s | −3.0% |
| wspt + scale 1.0 + decay 0.85 | 1,788 (74.5) | 0.72 / 9.7 s | −3.9% |
| wspt + scale 0.5 + decay 0.65 | 1,644 (68.5) | 0.72 / 7.2 s | −11.6% |
| control-repeat (drift control) | 1,867 (77.8) | 0.72 / 8.6 s | +0.4% |

All six bounded/stationary; drift control at +0.4% sets the noise floor near
±0.5%, so deltas are real. **Selection rule (max tok/s/GPU): defaults win** —
wspt+decay costs 2–4% at scale 1.0 and 12% at scale 0.5. This closes the last
silicon-unverified corner of the flag question (the sim's defaults-optimal
verdict now holds live on agg). Secondary finding: **wspt buys tail latency**
(p95 9.7 → 6.8 s at decay 0.65) at that throughput cost — a knob for
tail-sensitive deployments, not a throughput optimization.

---

## 5. Simulation vs Silicon — Drift Analysis

### 5.1 Prediction scorecard (n3u-sim v1)

| Sim claim | Silicon verdict |
|---|---|
| Same-cell gains 1.19→1.46× rising with conc | Direction confirmed; **silicon larger** (1.41→1.79×) |
| RR knee ≈ conc 32 | Confirmed (c32 stationary, c64 growing) |
| KV knee ≈ conc 128 | **Refuted — KV knees at ~32–48** (c64 already growing) |
| Absolute KV throughput 5,127 @c64 | 2.5× over-predicted (measured 2,045) |
| Both-bounded cells at 16 & 32 | Confirmed exactly |

### 5.2 Where the 1.9–2.8× absolute gap lives (apple-to-apple decomposition)
Method: substitute measured component rates into DynoSim one step at a time at
c16/c32 and observe which substitution closes the gap.

| Sim variant | KV c16 pred/meas | RR c16 | KV c32 | RR c32 |
|---|---|---|---|---|
| v1 (AIC-seeded) | 1.89× | 2.24× | 2.31× | 2.82× |
| **decode substituted** (measured ITL 8.9 + 1.73 ms/seq) | **0.80×** | **1.06×** | **0.86×** | **1.31×** |
| + prefill ×0.5 | 0.81× | 0.99× | 0.82× | 1.15× |
| + prefill ×0.25 | 0.77× | 0.78× | 0.82× | 0.86× |

**Verdict: the e2e drift is essentially one step — decode.** The AIC gb300
sglang-0.5.14 DB models NemotronH decode as 5.26 + 0.277·bs ms; silicon
measures 13.5 ms @ per-worker bs 2.7 → 18.0 ms @ 5.3, i.e. **8.9 + 1.73 ms/seq
— a 6× steeper batch slope**. Substituting decode alone moves every cell to
within ~±30%; prefill scaling then barely moves throughput (decode-bound
regime), and stock prefill already reproduces RR's TTFT (0.74–1.23 s predicted
vs 1.10–1.44 s measured), so the prefill rate is approximately right. Secondary
residual is **hit-rate drift**: engine telemetry shows KV median cached ≈
52k/83k ≈ 63% vs the sim's 84% (block-aligned reuse is optimistic about
hybrid-cache checkpoint granularity), explaining RR's remaining +31% at c32 and
KV's TTFT p50 (0.56 s vs 0.13 s predicted).

**Root cause for upstream:** NemotronH decode is Mamba-state + Latent-MoE bound,
and the AIC DB's batch-scaling row for this architecture appears interpolated
from too-sparse measurements — worth reporting to aiconfigurator.

**n3u-sim v2 constants** (applied): agg TPOT 8.9 + 1.73·bs (measured), prefill
unchanged 19.7k, reuse model pending a hybrid-checkpoint granularity term. With
v2 the sim is mildly conservative for KV (0.80–0.86×) — acceptable polarity
(under-promise) for prediction use.

---

## 6. Reproduction and Artifacts

- **Manifests:** `sglang/manifests/n3u-agg-kv.yaml` (router swapped per point).
- **Sequencer:** `scripts/sweep_n3u_agg.sh`; jobs sed-derived from the Kimi
  flagsweep template (N3U model/tokenizer staging swaps).
- **Artifacts:** `gs://alisachen-models/perf/17882*_alisachen-n3u-agg-{kv,rr}-c{16,32,64,128}/`;
  per-point summaries fetched; knee evidence in each `profile_export.jsonl`.
- **Simulation:** AIC solves in `aic-n3u-warm/`; DynoSim sweeps in
  `sim-results/dynosim_n3u_agg_v1.csv`; constants in `scripts/dynosim_pd.py`
  (`apply_n3u_constants()`).
