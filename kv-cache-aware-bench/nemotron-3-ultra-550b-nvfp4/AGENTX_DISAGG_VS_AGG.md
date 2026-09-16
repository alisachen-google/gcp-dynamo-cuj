# Nemotron-3-Ultra 550B — disaggregated vs aggregated under the AgentX concurrency definition (simulation, with measured anchors)

Companion to [AGENTX_D72_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_D72_RESULTS.md) and [AGENTX_AGG_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_AGG_RESULTS.md).
Page: [agg vs disagg curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-agg-vs-disagg.html)
(three panels: throughput per GPU, TTFT, P90 interactivity; x = clients or clients per GPU; disagg split selector; KV / tuned KV / RR).
Sim: `scripts/dynosim_agentx.py` v3 ([dynosim_n3u_agentx_v3.csv](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_agentx_v3.csv) + load-normalised cells in `sim-results/agentx_v3/norm_part*.csv`).
Arms: agg = 6 × TP4/EP4 workers on 24 GPUs; disagg = 72 GPUs, P:D split of TP4/EP4 workers, KV over NVLink. Throughput is
**total tokens (input + output) per second per GPU**, the InferenceX convention.

## 0. How to compare disaggregated with aggregated — the method

The two fleets differ in GPU count (72 vs 24), in where a request's work happens (two tiers vs one), and in how
their latency behaves with load. Four framings are defensible; they answer different questions, and only the first
two are how the published benchmarks rank architectures.

| framing | input held equal | what it answers | pitfall |
|---|---|---|---|
| **A. SLA-anchored best throughput** (InferenceX, NVIDIA AIConfigurator) | the SLO — e.g. TTFT p95 ≤ X and P90 interactivity ≥ Y | "for the latency users will accept, which fleet serves more tokens per GPU (and per dollar)?" | needs the SLO from the product; each arm sits at its own client count |
| **B. Pareto frontier overlay** | nothing — full sweep of each arm | the whole trade-off curve; shows where one arm dominates | needs the whole ladder; a curve, not a number |
| **C. Equal clients per GPU** | offered load per GPU | capacity comparison at the same utilisation, how InferenceX scales conc with fleet size | still says nothing about latency; per-GPU scaling assumes agg scales linearly |
| **D. Equal clients (raw)** | the client count | only a per-fleet view ("this many users on this fleet") | per-GPU numbers are diluted by the 3× fleet-size difference — never use it per GPU |

Two derived views make A concrete for capacity planning:
- **Best total tok/s per GPU under the SLO** (the InferenceX/NVIDIA number; §1 and the tables below).
- **Clients per GPU under the SLO**, i.e. GPUs needed per 1,000 concurrent Claude-Code sessions — the inverse
  question ops teams ask. From sim v3 (KV routing; tuned KV in parentheses):

| SLO | agg 24-GPU | disagg 12:6 | disagg 9:9 |
|---|---|---|---|
| TTFT p95 ≤ 5 s | 4.0 clients/GPU → 250 GPUs per 1,000 sessions | 4.0 → 250 (tuned 5.3 → 188) | 2.0 → 500 (tuned 4.0 → 250) |
| TTFT p95 ≤ 10 s | 8.0 → 125 | 6.7 → 150 (tuned 10.7 → 94) | 4.0 → 250 (tuned 8.0 → 125) |
| TTFT p95 ≤ 20 s | 8.0 → 125 | 8.0 → 125 (tuned 10.7 → 94) | 6.7 → 150 (tuned 10.7 → 94) |
| P90 interactivity ≥ 20 tok/s/user | 2.0 → 500 | 8.0 → 125 | 13.3 → 75 |
| TTFT p95 ≤ 10 s **and** P90 ≥ 20 | 2.0 → 500 | 6.7 → 150 (tuned 8.0 → 125) | 4.0 → 250 (tuned 8.0 → 125) |

Reading: on a TTFT-only SLO agg and disagg 12:6 need about the same GPUs per session (agg even edges it at 10 s),
and disagg's advantage is the 1.4–1.5× more tokens it serves per GPU while doing so; the moment the SLO includes
a per-user speed floor, agg needs 3–4× the GPUs of disagg because its shared workers cannot keep decode fast
under load. The tuned router adds 25–60% capacity to disagg and nothing to agg.

Practical recipe used in this report: (1) sweep both arms over clients under the AgentX definition; (2) pick the
product SLO (we use TTFT p95 ≤ 10–20 s and P90 ≥ 20 tok/s/user as the two candidates); (3) report each arm's best
total tok/s per GPU and clients per GPU under it (framing A); (4) show the frontier (framing B) so the reader can
move the SLO; (5) use clients per GPU (framing C) only for the "why is disagg worse at low load" diagnosis; (6) never
quote per-GPU numbers at equal raw clients (framing D); (7) cost = the framing-A throughput × the price per GPU-hour
(the 72-GPU fleet costs 3× per hour, which the per-GPU normalisation already absorbs).

## 1. Verdict

- **Optimal disagg topology under this load model: 12:6** (12 prefill : 6 decode workers) on total tokens per GPU and on
  TTFT; **9:9** is the optimum on output tokens and interactivity (see AGENTX_D72_RESULTS.md §ii). The curve page uses 12:6.
- **At equal client counts below ~400, disagg is worse per GPU than agg** (0.47× at 48 clients, 0.99× at 384). This is
  not an engine deficit: both fleets receive the same offered load, and disagg spreads it over 3× more GPUs. Per fleet,
  disagg serves 1.4–3× more tokens at every client count.
- **Load-normalised (equal clients per GPU, which is how InferenceX scales concurrency with deployment size), disagg
  12:6 beats agg on total tokens per GPU at every load level simulated — 1.39× at 2 clients/GPU, 1.60× at 4, 1.50× at 8,
  1.32× at 16, 1.27× at 20 — and 1.20× at the two arms' peaks**
  (12:6 6,187 at 1440 clients vs agg 5,138 at 960). 9:9 matches 12:6 up to 8 clients/GPU and
  falls to parity with agg by 20/GPU because its prefill tier saturates (TTFT 40–57 s); it peaks at 5,501 (1.07×).
- **Agg keeps two advantages**: TTFT (0.2–0.4 s up to ~1,000 clients, versus 0.6–5 s for 12:6 and 4–60 s for 9:9 past
  their knees) and simplicity (no KV hand-off, no ComputeDomain). **Disagg keeps interactivity**: 30–140 tok/s per user
  P90 across the range versus 2–22 for agg, because agg's workers must batch prefill and decode together.

## 2. Equal client counts (what the raw sweep shows)

| clients | agg KV total/GPU (TTFT p50 · P90) | disagg 12:6 KV | disagg 9:9 KV | 12:6 ÷ agg per GPU | 12:6 ÷ agg per fleet |
|---|---|---|---|---|---|
| 48 | 1,474 (0.1 s · 22) | 699 (0.1 s · 139) | 701 (0.1 s · 148) | **0.47×** | 1.42× |
| 96 | 2,382 (0.1 s · 13) | 1,389 (0.1 s · 109) | 1,395 (0.2 s · 125) | **0.58×** | 1.75× |
| 192 | 3,715 (0.2 s · 9) | 2,679 (0.2 s · 72) | 2,733 (0.3 s · 93) | **0.72×** | 2.16× |
| 384 | 4,706 (0.2 s · 6) | 4,657 (0.4 s · 40) | 4,937 (1.8 s · 54) | **0.99×** | 2.97× |
| 480 | 4,863 (0.2 s · 5) | 5,217 (0.6 s · 32) | 5,501 (4.1 s · 44) | **1.07×** | 3.22× |
| 768 | 5,107 (0.4 s · 4) | 5,981 (1.6 s · 19) | 5,469 (20.2 s · 27) | **1.17×** | 3.51× |
| 960 | 5,138 (0.4 s · 3) | 6,139 (2.3 s · 15) | 5,380 (29.0 s · 22) | **1.19×** | 3.58× |
| 1440 | 4,756 (0.8 s · 2) | 6,187 (4.7 s · 10) | 4,884 (56.6 s · 15) | **1.30×** | 3.90× |
| 1536 | 4,632 (1.0 s · 2) | 6,144 (5.6 s · 9) | 4,884 (59.6 s · 14) | **1.33×** | 3.98× |
| 1920 | 4,165 (2.0 s · 2) | 5,945 (12.3 s · 8) | 4,544 (83.8 s · 11) | **1.43×** | 4.28× |

Reading: per-GPU throughput crosses over at ~400 clients; per-fleet, disagg is ahead everywhere because it has 3× the
GPUs. Agg's TTFT stays under 1 s to 1,536 clients; its P90 interactivity collapses from 22 to 2 tok/s per user as
decode batches on the shared workers grow.

## 3. Load-normalised comparison (equal clients per GPU)

| clients per GPU | agg (clients) total/GPU · TTFT p50 · P90 | disagg 12:6 (clients) | disagg 9:9 (clients) | 12:6 ÷ agg | 9:9 ÷ agg |
|---|---|---|---|---|---|
| 2 | 1,474 (48) · 0.1 s · 22 | 2,053 (144) · 0.2 s · 86 | 2,075 (144) · 0.2 s · 104 | **1.39×** | 1.41× |
| 4 | 2,382 (96) · 0.1 s · 13 | 3,822 (288) · 0.3 s · 51 | 3,978 (288) · 0.7 s · 68 | **1.60×** | 1.67× |
| 8 | 3,715 (192) · 0.2 s · 9 | 5,580 (576) · 1.0 s · 26 | 5,689 (576) · 8.3 s · 36 | **1.50×** | 1.53× |
| 16 | 4,706 (384) · 0.2 s · 6 | 6,219 (1152) · 2.9 s · 13 | 5,155 (1152) · 40.0 s · 18 | **1.32×** | 1.10× |
| 20 | 4,863 (480) · 0.2 s · 5 | 6,187 (1440) · 4.7 s · 10 | 4,884 (1440) · 56.6 s · 15 | **1.27×** | 1.00× |
| 40 | 5,138 (960) · 0.4 s · 3 | not simulated (beyond the 1,920-client sweep) | not simulated (beyond the 1,920-client sweep) | — | — |
| 64 | 4,632 (1536) · 1.0 s · 2 | not simulated (beyond the 1,920-client sweep) | not simulated (beyond the 1,920-client sweep) | — | — |
| 80 | 4,165 (1920) · 2.0 s · 2 | not simulated (beyond the 1,920-client sweep) | not simulated (beyond the 1,920-client sweep) | — | — |

Reading: once the load is normalised the dilution disappears and the prefill-tier advantage shows at every level; the gap is widest around 4–8 clients per GPU, where agg's shared workers begin batching prefill with decode (its P90 interactivity is already 9–13 tok/s per user there) while the disagg decode tier still serves each user at 26–51.

## 4. Why disagg is worse than agg at low load — step by step

1. **Same offered load, three times the GPUs.** Under the AgentX definition the client count fixes the request rate
   (each session replays its recorded think-time). 48 clients generate ~0.7–0.8 requests/s whether they hit 24 GPUs or
   72. Per-GPU throughput is therefore diluted by 72/24 = 3× on disagg until the fleet approaches saturation. At 48
   clients the sim's per-fleet ratio is 1.42× in disagg's favour while the per-GPU ratio is 0.47×; the two numbers
   describe the same run.
2. **Two-thirds of a disagg fleet is idle at low load.** At 48–192 clients the 12:6 fleet's prefill tier is busy for a
   few seconds per request and its 6 decode workers hold 5–16 requests in flight in total — under one request per
   decode worker — so 24 decode GPUs contribute almost nothing to *total* tokens (input tokens are counted at prefill).
   Agg's 24 GPUs all prefill.
3. **The hand-off is a fixed tax that only pays back under contention.** Every disagg request carries a ~0.19 s
   KV transfer plus scheduling cost (measured, AGENTX_D72_RESULTS.md §iv). Below the knee it is pure overhead:
   TTFT 0.3 s on disagg versus 0.2 s on agg at 192 clients, with no throughput benefit because nothing is queueing.
4. **Prefix reuse is equally good on both, so disagg has no cache advantage to offset the dilution.** Simulated hit
   rate is ~0.8 for KV routing on both arms at low load (both keep one radix cache per prefill worker); RR halves it on
   both. The KV router's cache benefit therefore cancels in the ratio.
5. **Where it flips.** From ~8 clients per GPU the agg workers start batching prefill and decode together: per-request
   decode slows (P90 interactivity 9 → 3 tok/s per user between 192 and 960 clients) and total throughput plateaus
   at ~5,100/GPU. The disagg prefill tier, with 2× the prefill GPUs of agg (48 vs 24 at 12:6), keeps converting input
   tokens at full rate and the decode tier keeps ITL flat, so total tokens per GPU keep rising to ~6,200 at 1,440
   clients while TTFT stays under 5 s. On output tokens the same flip happens earlier (9:9: 92 vs agg 64 per GPU).

## 5. The trade in one line

Agg is the right fleet for **latency-first** serving below ~8 clients per GPU (TTFT 0.2 s, no hand-off, 24 GPUs);
disagg 12:6 is the right fleet for **throughput-first** serving from ~8 clients per GPU up (1.2–1.3× total tokens per
GPU at peak, 5–10× the per-user interactivity), and 9:9 if per-user speed is the binding SLO. Under an SLO of
TTFT p95 ≤ 20 s the sim's best cells are agg 3,715 (192 clients) versus 12:6 5,217 (480 clients): 1.40× for disagg.

## 6. Measured anchors and what is still missing

Measured AgentX-mode points so far: disagg 9:9 KV at 48 clients (1,128 total/GPU, TTFT p50 0.32 s) and 96 clients
(2,497, 0.31 s); the 9:9 ladder continues to 1,536 and the agg ladder (48 → 1,536, KV and RR in parallel on two
fleets) started 2026-09-16 08:00 UTC. No 12:6 point exists under this definition yet; a 12:6 run at 480 and 768
clients is the verification that would confirm the simulated optimum and is queued after the 9:9 ladders.
The sim under-predicts absolute totals ~1.6–2× (trace representation; AGENTX_D72_RESULTS.md §iv) on both arms alike,
so the ratios above are the claim, not the absolute levels. Busy-stream (always-busy streams) results for the same
arms are in D72_RESULTS.md §2 (disagg 9:9 beats agg 1.17× on total, 1.36× on output at their bounded peaks).
