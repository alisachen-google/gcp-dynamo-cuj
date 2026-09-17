#!/usr/bin/env python3
"""Writes AGENTX_DISAGG_VS_AGG.md from sim v3 (+ load-normalised cells) and the measured AgentX points."""
import csv,glob,pathlib
R=pathlib.Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(R/"sim-results/dynosim_n3u_agentx_v3.csv")))
for f in glob.glob(str(R/"sim-results/agentx_v3/norm_part*.csv")): rows+=list(csv.DictReader(open(f)))
def c(pd,pol,cl):
    r=[x for x in rows if x['pd']==pd and x['policy']==pol and int(x['clients'])==cl]
    if not r: return None
    r=r[0]; g=24 if pd=="agg6" else 72
    return dict(cl=cl,g=g,tot=float(r['total_tok_s'])/g,out=float(r['throughput_tok_s'])/g,p50=float(r['ttft_p50_s']),p95=float(r['ttft_p95_s']),i90=1000/float(r['tpot_p90_ms']),hit=float(r['hit_rate']),rps=float(r['req_per_s']),fleet=float(r['total_tok_s']))
N="https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4"
# equal clients table
eq="| clients | agg KV total/GPU (TTFT p50 · P90) | disagg 12:6 KV | disagg 9:9 KV | 12:6 ÷ agg per GPU | 12:6 ÷ agg per fleet |\n|---|---|---|---|---|---|\n"
for cl in [48,96,192,384,480,768,960,1440,1536,1920]:
    a,d,n=c("agg6","kv",cl),c("12:6","kv",cl),c("9:9","kv",cl)
    eq+=f"| {cl} | {a['tot']:,.0f} ({a['p50']:.1f} s · {a['i90']:.0f}) | {d['tot']:,.0f} ({d['p50']:.1f} s · {d['i90']:.0f}) | {n['tot']:,.0f} ({n['p50']:.1f} s · {n['i90']:.0f}) | **{d['tot']/a['tot']:.2f}×** | {d['fleet']/a['fleet']:.2f}× |\n"
# load-normalised table: clients per GPU
pairs=[(2,48,144),(4,96,288),(8,192,576),(16,384,1152),(20,480,1440),(40,960,None),(64,1536,None),(80,1920,None)]
nm="| clients per GPU | agg (clients) total/GPU · TTFT p50 · P90 | disagg 12:6 (clients) | disagg 9:9 (clients) | 12:6 ÷ agg | 9:9 ÷ agg |\n|---|---|---|---|---|---|\n"
avail=True
for cpg,ac,dc in pairs:
    a=c("agg6","kv",ac); d=c("12:6","kv",dc) if dc else None; n=c("9:9","kv",dc) if dc else None
    if dc and (d is None or n is None): avail=False
    f=lambda p:(f"{p['tot']:,.0f} ({p['cl']}) · {p['p50']:.1f} s · {p['i90']:.0f}" if p else "not simulated (beyond the 1,920-client sweep)")
    nm+=f"| {cpg} | {f(a)} | {f(d)} | {f(n)} | {('**%.2f×**'%(d['tot']/a['tot'])) if d else '—'} | {('%.2f×'%(n['tot']/a['tot'])) if n else '—'} |\n"
def peak(pd,pol):
    pts=[c(pd,pol,cl) for cl in [48,96,192,384,480,768,960,1440,1536,1920]]; b=max(pts,key=lambda p:p['tot']); return b
pa,p12,p9=peak("agg6","kv"),peak("12:6","kv"),peak("9:9","kv")
doc=f'''# Nemotron-3-Ultra 550B — disaggregated vs aggregated under the AgentX concurrency definition (simulation, with measured anchors)

Companion to [AGENTX_D72_RESULTS.md]({N}/AGENTX_D72_RESULTS.md) and [AGENTX_AGG_RESULTS.md]({N}/AGENTX_AGG_RESULTS.md).
Page: [agg vs disagg curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-agg-vs-disagg.html)
(three panels: throughput per GPU, TTFT, P90 interactivity; x = clients or clients per GPU; disagg split selector; KV / tuned KV / RR).
Sim: `scripts/dynosim_agentx.py` v3 ([dynosim_n3u_agentx_v3.csv]({N}/sim-results/dynosim_n3u_agentx_v3.csv) + load-normalised cells in `sim-results/agentx_v3/norm_part*.csv`).
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
  1.32× at 16, 1.27× at 20 — and {p12['tot']/pa['tot']:.2f}× at the two arms' peaks**
  (12:6 {p12['tot']:,.0f} at {p12['cl']} clients vs agg {pa['tot']:,.0f} at {pa['cl']}). 9:9 matches 12:6 up to 8 clients/GPU and
  falls to parity with agg by 20/GPU because its prefill tier saturates (TTFT 40–57 s); it peaks at {p9['tot']:,.0f} ({p9['tot']/pa['tot']:.2f}×).
- **Agg keeps two advantages**: TTFT (0.2–0.4 s up to ~1,000 clients, versus 0.6–5 s for 12:6 and 4–60 s for 9:9 past
  their knees) and simplicity (no KV hand-off, no ComputeDomain). **Disagg keeps interactivity**: 30–140 tok/s per user
  P90 across the range versus 2–22 for agg, because agg's workers must batch prefill and decode together.

## 2. Equal client counts (what the raw sweep shows)

{eq}
Reading: per-GPU throughput crosses over at ~400 clients; per-fleet, disagg is ahead everywhere because it has 3× the
GPUs. Agg's TTFT stays under 1 s to 1,536 clients; its P90 interactivity collapses from 22 to 2 tok/s per user as
decode batches on the shared workers grow.

## 3. Load-normalised comparison (equal clients per GPU)

{nm}
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
'''
doc = doc.replace("\n\n", "\n\n" + 'Simulation status, 2026-09-17: these v3 curves are historical and superseded by the [actual AIPerf replay results](reports/agentx-aiperf-results.md). Use that index for current performance and topology decisions.\n\n', 1)
(R/"AGENTX_DISAGG_VS_AGG.md").write_text(doc); print("report written; normalised cells complete:",avail)
