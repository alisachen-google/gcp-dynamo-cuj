# Historical AgentX simulation workflow and gap interpretation

Historical material superseded on 2026-09-17. These results used handwritten AgentX workload models and must not be used as current AIPerf predictions. Statements about parity, topology rankings, and calibration below record the earlier interpretation; the [current replay audit](agentx-faithful-replay.md) revises it.

[Current simulation results](agentx-aiperf-results.md)

---

## 3. Simulation, step by step: what AIC gave us and what DynoSim gave us

Two tools, two questions ([AGG24 §3](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGG24_RESULTS.md#3-simulation-stage--what-aic-and-dynosim-deliver) has the
busy-stream detail):

| step | tool | input | output we used |
|---|---|---|---|
| 3.1 engine constants | **aiconfigurator 0.11.0**, `--database-mode SILICON`, gb300 sglang database | model (N3U NVFP4), GPU (GB300), engine, ISL/OSL/prefix of the trace, candidate agg/disagg shapes | per-worker **uncached prefill rate** (19.7 k tok/s per TP4 worker), **decode TPOT vs batch** (5.97 + 0.4·bs ms disagg; 5.26 + 0.277·bs agg, later refit), **KV bytes/token** (6 KB → pool > 100 M tokens per worker), and AIC's own Pareto of shapes (TTFT/TPOT/throughput per config) used to shortlist P:D splits |
| 3.2 busy-stream policy sim | **DynoSim** = [dynosim_pd.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_pd.py), a trace-driven discrete-event simulator seeded with 3.1 | the 4 k-request trace slice, P:D split, router policy + flags, C busy streams | throughput, TTFT p50/p95/p99, TPOT, prefix hit rate, req/s per cell; curves vs C; knees; the Pareto grid over splits × C ([PARETO_E2E_REPORT.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/PARETO_E2E_REPORT.md)) |
| 3.3 AgentX-definition sim | [dynosim_agentx.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx.py), same engine model, AgentX load model (lanes = live sessions, recorded think-time, 25–75 % start, 10 s idle cap, recycling, 1 h window) | splits 3:15 / 6:12 / 9:9 / 12:6 / 15:3 + agg, policies KV / tuned KV / RR, clients 48 → 1,920 | total and output tok/s per GPU, TTFT p50/p90/p95, per-request TPOT p50/p90 (→ P90 interactivity), input tokens per request, hit rate ([dynosim_n3u_agentx_v3.csv](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/dynosim_n3u_agentx_v3.csv)) |
| 3.4 point selection | [KNEE_ANALYSIS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/KNEE_ANALYSIS.md) (AgentX section) | 3.3 cells | knees (throughput-slope rule), same-config and same-SLO KV-vs-RR cells, the disagg split to measure (12:6) |
| 3.5 calibration loop | [dynosim_agentx_decomp.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_agentx_decomp.py) (AgentX) and AGG24 §5.2 (busy-stream) | measured cells | which model term is off (busy-stream: decode slope 6× → refit; AgentX: a 0.19 s per-request hand-off constant, output-length and trace-representation terms) → next sim version |

How to run it (all from `kv-cache-aware-bench/`):
```
# 3.1 constants come from AIC solves (AGG24 §3.1); they live in dynosim_pd.py::apply_n3u_constants()
# 3.2 busy-stream Pareto grid (splits × conc × policies)
python3 scripts/dynosim_pd.py <trace.jsonl> --splits 3:15,6:12,9:9,12:6,15:3 --conc 12,24,48,96,120,144,192,288,384,512,768 --policies kv,rr --out sim-results/dynosim_n3u_disagg72_v1.csv
# 3.3 AgentX-definition sweep (v3 columns incl. tpot_p90_ms, in_tok_per_req, total_tok_s; v4 adds ttft_p90_s)
python3 scripts/dynosim_agentx.py <trace.jsonl> --splits 12:6,9:9 --clients 48,96,192,384,480,768,960,1440,1536,1920 --policies kv,kv-tuned,rr --agg --out sim-results/agentx_v3/part.csv
# 3.4 knees + comparison points (prints the tables in KNEE_ANALYSIS.md)
python3 scripts/gen_agentx_agg_vs_disagg_report.py       # agg vs disagg tables
# 3.5 decomposition at the measured cells
python3 scripts/dynosim_agentx_decomp.py <trace.jsonl>
# pages
python3 scripts/gen_agentx_curve.py all | agg6 | "3:15,6:12,9:9,12:6,15:3"; python3 scripts/gen_agentx_interactivity.py; python3 scripts/gen_agentx_agg_vs_disagg.py
```
What each tool cannot do: AIC has no notion of a trace, routing policy or queueing (it sizes one request shape);
DynoSim has no engine kernels (it trusts the constants) and, in v3, replays a 4 k-request slice without subagent
fan-out, which is why its absolute totals sit 1.6–2× under silicon while its rankings hold (AGENTX_D72 §iv).

## 4. Which curves the simulation delivered, and how the real points were chosen

Pages (all regenerated from the CSVs by the scripts above; measured points overlay automatically):
- [AgentX curve, all arms](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html) — throughput per GPU vs clients and TTFT vs clients, KV / tuned KV / RR, topology toggles ([agg-only](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-agg-curve.html), [disagg-only](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-disagg-curve.html)).
- [Interactivity frontier](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-interactivity.html) — P90 interactivity vs total tok/s per GPU, every cell, topology and policy toggles.
- [Agg vs disagg](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-agg-vs-disagg.html) — three panels, x = clients or clients per GPU, disagg split selector.
- Busy-stream counterparts: [disagg curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-disagg-curve.html), [frontier](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-frontier.html), [Pareto](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-pareto.html), [TTFT](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-ttft-curve.html).

Selection rules (KNEE_ANALYSIS.md, AgentX section):
1. **Topology**: the split with the highest simulated total tok/s per GPU under the AgentX load — **12:6**
   (9:9 on output tokens / interactivity) — becomes the measured disagg ladder; 9:9 keeps three early points as a
   cross-check.
2. **Ladder**: the sim's rise, knee, peak and decline for KV (12:6: 96 / 192 / 384 / 480 / 768 / 1440; agg: 48 → 1536).
3. **KV vs RR, same config**: RR's throughput-slope knee (192 clients on every arm) — the last count where RR still scales.
4. **KV vs RR, same SLO**: each policy's best throughput under a TTFT budget and under an interactivity floor
   (12:6: KV 480 vs RR 96; agg: KV 192 vs RR 96); the RR ladders contain exactly those cells.
5. **Flag sweep** at the chosen KV cells (prefill-load-scale 3 / credit 0.8, scale 2 / credit 0.8, temperature 0.5),
   because the sim says tuned routing pays only on disagg past the prefill knee.

### 5.3 Simulation vs silicon, apple to apple (TTFT p95 standard)

Disagg (12:6 KV, 96 / 192 / 384 / 480, and 9:9 at 48 / 96 / 192): the engine substitutions (output length, 0.19 s
hand-off, measured decode line) barely move the ratios — total tokens stay at 0.53–0.59× and TTFT p95 at 2.3–2.7× too
pessimistic. The gap is (1) **prefix hit rate**: sim 0.73–0.78 vs **0.88–0.94 reported by the engine's cached-token counter** — proven
to be the 4 k trace slice (hit ceiling 0.846 vs 0.969 for the full trace; re-simulating on the full trace gives 0.95 and,
with an aiperf-style warm-up, a TTFT p95 of 1.58 s vs 1.77 s measured at 192; AGENTX_D72_RESULTS.md §iv) — so the
published sim prefills 1.6–3× more tokens per turn, which sets its tail, its early knee (768) and its low ceiling (6,187 vs 10,434
measured and still rising); (2) **trace representation**: 70 k input tokens per request vs 86–94 k measured and a
0.69–0.80× request rate from the 4 k-request slice. Requests × input length reproduces the residual exactly. Agg (KV 48 / 96 / 192 and RR 48 / 96 / 192): the largest
term is the sim's decode cliff past batch 7 (TPOT 27–61 ms simulated vs 7–24 ms measured), which it applies to the deep
per-worker batches its KV policy creates by packing sessions; removing it lifts the KV cells from 0.38–0.44× to
0.53–0.62×, and the rest is the trace residual. The RR ladder is the control: with no packing the same engine is within
10–20% of silicon on requests and TTFT tail, so the agg sim's problem is its KV-router model, not the engine. The engine counters add two calibration facts: the agg RR hit rate is 0.56–0.74 (sim 0.40), and
network is negligible on agg (≈ 3 ms per request), with the frontend tokenizer (0.06–0.2 s p95) the only non-GPU term worth modelling. Full ladders: AGENTX_D72_RESULTS.md §iv, AGENTX_AGG_RESULTS.md §iv; overlay page
`reports/n3u-agentx-sim-vs-real.html`. Rankings across topologies are the sim's reliable output; levels, and the
KV-vs-RR ordering on agg, must come from silicon.

**Calibrated replay (v5, 2026-09-17).** Replaying the raw SemiAnalysis dataset as root + subagent streams with aiperf's
trajectory-tree rules (AGENTX_D72_RESULTS.md §iv, v5) brings the disagg sim to parity within 6–16 % on hit rate,
input/output length, TTFT p95, TPOT and interactivity at 192 clients, and within 4–21 % on total tokens at 768; the
one open workload term is the subagent stream mix (sim 71–81 % subagent turns vs 51 % measured, a loader
chain-splitting rule), and the open engine term is decode in-flight feedback at high load.

**Faithful replay through aiperf itself (2026-09-17).** The hand re-implementations above are superseded for agg by
[`scripts/dynosim_aiperf_replay.py`](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/scripts/dynosim_aiperf_replay.py):
the real aiperf 0.12.0 CLI, Weka loader (9,602 conversations / 68,266 turns from the 393 roots, including its agent-chain
splitting), trajectory sampler, dependency barriers, warm-up and metrics exporter drive the DynoSim engine through an
in-process transport on an accelerated clock, so replay parity holds by construction (96/96 and 192/192 initial lane
snapshots, every warm-up and common profiled request matching hardware token counts). Reports:
[fidelity and results](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/agentx-faithful-replay.md) ·
[calibration plan](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/agentx-replay-calibration.md) ·
[how to rerun](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/agentx-replay-howto.md) ·
[artifacts](https://github.com/alisachen-google/gcp-dynamo-cuj/tree/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/sim-results/agentx_aiperf_replay_20260917).

| agg 6 × TP4 cell | published v3 sim | faithful replay | silicon | total-token error | what is left |
|---|---|---|---|---|---|
| RR, 96 clients | 2,924 | **6,106** | 6,137 | **−0.5 %** | TTFT p95 −40 % (no prefill/decode contention), ITL p50 +59 % |
| default KV, 192 clients | 3,715 | **6,983** | 9,655 | −27.7 % | the sim's KV router overloads one worker (2,101 requests, ITL 109 ms vs 15–39 ms on the others; cached fraction 93 % vs 74 % measured) |

With the replay exact, the RR cell proves the dataset-replay terms were the whole throughput gap (request rate −0.2 %,
input −0.3 %, output −1.0 %), and the KV cell isolates the remaining error in the simulated KV router's load term —
the over-packing diagnosed in AGENTX_AGG_RESULTS.md §iv. Disagg cells are not replayed this way yet.

