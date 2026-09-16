# AgentX-scenario benchmarking of Nemotron-3-Ultra 550B on GB300 NVL72 — the end-to-end guide

How we set the workload up, what a run consists of, how the simulators were used to pick configurations and
points, which curves they delivered, and what silicon then showed. Every artifact is linked. Status
2026-09-16: measured ladders in progress (section 5 fills as points land; the run index is authoritative).

Companion documents: [AGENTX_COMPARISON.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_COMPARISON.md) (methodology and code trace),
[AGENTX_D72_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_D72_RESULTS.md) (disagg), [AGENTX_AGG_RESULTS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_AGG_RESULTS.md) (agg),
[AGENTX_DISAGG_VS_AGG.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_DISAGG_VS_AGG.md), [KNEE_ANALYSIS.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/KNEE_ANALYSIS.md),
[PARETO_E2E_REPORT.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/PARETO_E2E_REPORT.md) (busy-stream Pareto loop), [RUN_INDEX.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/RUN_INDEX.md).

## 1. Dataset and scenario set-up

**Workload.** [semianalysisai/cc-traces-weka-062126-256k](https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126-256k):
393 recorded Claude-Code sessions (multi-turn, subagents, up to 256 K tokens of context, recorded timestamps).
Model: [NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4)
(hybrid Mamba-2 / LatentMoE, NVFP4), served by SGLang 0.5.16 under Dynamo 1.4.2 (FlashInfer 0.6.18).

**Scenario.** `aiperf profile --scenario inferencex-agentx-mvp` — the only scenario registered in aiperf
([registry](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/registry.py), [definition](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/inferencex_agentx_mvp.py)).
It locks: timing mode `AGENTIC_REPLAY` (one lane per client, a lane = a live session tree incl. subagents, recycled
when it drains), streaming + ignore-EOS, no `--ignore-trace-delays` (think-time is replayed), no input truncation,
the Weka loader family, ≥ 900 s windows, cache-bust on the first-turn prefix, a 10 s whole-system idle cap.
"Concurrency C" therefore means **C live sessions**, of which only a fraction has a request in flight at any instant
(measured: 48 clients ≈ 5.6 in flight, 96 ≈ 16; a busy-stream slot ≈ 6 clients — AGENTX_COMPARISON.md §5d).

**How the dataset reaches the bench pod** ([sgl-d72-agentx.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-agentx.yaml), container script):
1. The HF hub snapshot and aiperf's processed cache are staged from the shared PVC into `/tmp/hf` and the pod runs
   with `HF_HUB_OFFLINE=1` (no network dependency, byte-identical dataset every run).
2. `--public-dataset semianalysis_cc_traces_weka_062126_256k --num-dataset-entries 393 --max-context-length 262144`
   selects the loader; aiperf's `WekaTraceLoader` reconstructs all 393 sessions (parallel workers) before any request.
3. Two warm-up passes precede the measured run: a 1-client sanity pass and a 900 s replay at 96 clients
   (`cache-warmup/` artifact) so the fleet's prefix caches are hot; the measured run then does the scenario's own
   per-lane priming (each lane starts 25–75 % into its session with the history pre-filled).
4. The measured run: `--concurrency C --benchmark-duration 3600 --trajectory-start-min-ratio 0.25
   --trajectory-start-max-ratio 0.75 --cache-bust first_turn_prefix --use-server-token-count --random-seed 42`.
   These are InferenceX's launcher flags ([benchmark_lib.sh](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/benchmark_lib.sh)); the one flag we
   lack is `--warmup-requests-per-lane`, which exists only in their aiperf fork.
5. Artifacts (`profile_export_aiperf.csv`, per-request `profile_export.jsonl`, warm-up outputs) upload to
   `gs://alisachen-models/perf/<jobid>_<jobname>/`; every job is a row in [RUN_INDEX.md](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/RUN_INDEX.md).

The busy-stream runs that preceded this (D72_RESULTS.md, AGG24_RESULTS.md) used the same dataset with
`--no-fixed-schedule --ignore-trace-delays` — C always-busy request streams — via
[sgl-d72-flagsweep.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/manifests/perf/sgl-d72-flagsweep.yaml). The difference between the two
definitions, and why we call the earlier use of "concurrency" a misuse when placed next to InferenceX's numbers, is
section i of the two AgentX reports.

## 2. What a configuration and a run look like

**Fleet manifests (Kubernetes, one Dynamo namespace per fleet).**
- Disagg 72 GPU: [n3u-mnnvl-99.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-99.yaml) (9 prefill : 9 decode TP4/EP4 workers,
  KV over NVLink via mooncake with `MC_FORCE_MNNVL=1`, a ComputeDomain + IMEX channel per fleet);
  [n3u-mnnvl-126.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-126.yaml) (12:6), [n3u-mnnvl-full.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-mnnvl-full.yaml) (6:12);
  generator [gen_mnnvl_arms.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_mnnvl_arms.py).
- Agg 24 GPU: [n3u-agg-newstack.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-agg-newstack.yaml) (6 × TP4/EP4 workers, one per
  node); a second namespace-isolated copy [n3u-agg-newstack2.yaml](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/manifests/n3u-agg-newstack2.yaml)
  lets the RR ladder run in parallel with KV on the same node pool.
- Router: the Dynamo frontend's `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs` (KV-aware) or
  `--router-mode round-robin`; flag variants (`--router-prefill-load-scale`, `--router-kv-overlap-score-credit`,
  `--router-temperature`, `--router-queue-policy wspt`) are swept at the selected points.

**The runner** ([agentx_runner.sh](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/agentx_runner.sh)) takes `ARM MANIFEST "pol:C pol:C …" GATE GATE_LOG
DONE_MARKER LOG NEED_NODES POOL JOB_PREFIX WORKER_LABELS`. Per point it patches the frontend to the policy, restarts it,
waits for the model to register, renders the bench template (`sed` of model path, arm, job name, `CONCURRENCIES`,
`BENCHMARK_DURATION=3600`), applies the Job, polls to completion, runs the **transport guard**
([mnnvl_transport_guard.sh](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/mnnvl_transport_guard.sh): no transfer-failure signatures, mooncake
transfer-engine throughput > 0, no RDMA device in the pod, `MC_FORCE_MNNVL=1`) and the **knee check**
([knee_check.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/sglang/scripts/knee_check.py): TTFT p50 stationarity across window quarters from the
per-request records). Runners chain through done-markers in their logs (`GATE_STRICT=1` ignores stale halt lines);
bench pods for ≥ 192 clients run on arm64 GPU nodes (aiperf needs > 60 GB there).
Per-request derived metrics: [inflight_from_records.py](https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/inflight_from_records.py) (mean requests in
flight by Little's law and by direct integration).

**Stop rules.** Any KV transfer failure or non-NVLink path invalidates the point (the guard halts the ladder); no
latency SLO gate is applied to a run — latency is reported, and SLO framings are applied afterwards.

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

## 5. Real results and analysis (updated 17:45 UTC; RUN_INDEX.md is the authoritative job list)

Measured so far (AgentX definition; total = input + output tokens per second per GPU):

| arm | clients | total/GPU | output/GPU | TTFT p50 / p90 / p95 | ITL p50 / p90 | P90 interactivity | in-flight | knee |
|---|---|---|---|---|---|---|---|---|
| disagg 9:9 KV | 48 | 1,128 | 10.9 | 0.32 / 0.9 / 1.5 s | 6.4 / 6.8 ms | 147 | 5.6 | stationary |
| disagg 9:9 KV | 96 | 2,497 | 28.0 | 0.31 / 0.9 / 1.42 s | 7.4 / 8.4 ms | 119 | 16.1 | stationary |
| disagg 9:9 KV | 192 | 4,600 | 45.4 | 0.36 / 1.2 / 2.03 s | 8.5 / 9.9 ms | 101 | 30.0 | stationary |
| disagg 12:6 KV | 96 | 2,471 | 27.5 | 0.30 / 0.8 / 1.37 s | 8.2 / 9.6 ms | 104 | 17.2 | stationary |
| disagg 12:6 KV | 192 | 4,510 | 44.7 | 0.36 / 1.1 / 1.77 s | 10.0 / 11.2 ms | 89 | 33.5 | stationary |
| disagg 12:6 KV | 384 | 8,637 | 86.8 | 0.43 / 1.5 / 2.58 s | 13.7 / 15.6 ms | 64 | 91.1 | stationary |
| disagg 12:6 KV | 480 | 10,434 | 104.8 | 0.50 / 2.0 / 3.52 s | 16.4 / 17.5 ms | 57 | 129.4 | stationary |
| agg KV | 48 | 3,334 | 32.3 | 0.42 / 2.5 / 3.83 s | 7.3 / 9.8 ms | 102 | 6.8 | stationary |
| agg KV | 96 | 6,844 | 75.1 | 0.67 / 3.4 / 5.36 s | 11.9 / 23.5 ms | 43 | 27.5 | stationary |
| agg KV | 192 | 9,655 | 96.8 | 1.56 / 8.4 / 11.68 s | 23.9 / 50.6 ms | 20 | 74.5 | stationary |
| agg KV | 384 | 8,257 | 77.7 | 82.8 / 111.9 / 119.4 s | 40.3 / 79.1 ms | 13 | 255.8 | **post-knee** (saturated; ladder stopped here) |
| agg KV tuned (scale 3, credit 0.8) | 192 | 11,012 | 108.9 | 0.87 / 4.2 / 6.33 s | 19.6 / 37.3 ms | 27 | 64.1 | stationary (+14% total, −46% TTFT p95 vs default KV) |
| agg KV scale 2, credit 0.8 | 192 | 10,241 | 102.4 | 1.10 / 5.8 / 8.11 s | 22.0 / 44.8 ms | 22 | 70.0 | stationary (+6% total, −31% TTFT p95 vs default KV) |
| agg KV temperature 0.5 | 192 | 7,816 | 79.5 | 5.0 / 18.0 / 22.1 s | 33.3 / 77.7 ms | 13 | 93.6 | stationary (−19% total, 1.9× TTFT p95 vs default KV) |
| agg RR | 48 | 3,250 | 31.3 | 0.79 / 4.8 / 8.27 s | 7.5 / 11.0 ms | 91 | 8.5 | stationary |
| agg RR | 96 | 6,137 | 67.1 | 0.81 / 8.8 / 12.56 s | 12.1 / 29.6 ms | 34 | 33.2 | stationary |
| agg RR | 192 | 6,802 | 71.4 | 10.87 / 44.6 / 60.1 s | 30.5 / 82.0 ms | 12 | 102.7 | stationary, at throughput knee |
| agg RR | 384 | 5,076 | 49.6 | 78.3 / 402 / 495 s | 57.0 / 126 ms | 8 | 265.6 | **post-knee** (saturated; ladder stopped here) |

### 5.1 KV-aware vs round-robin, measured (agg 24 GPU; the disagg RR ladder is queued behind 12:6 KV)

Same config (192 live-session clients, the cell the sim picked as the same-config point on every arm):

| | KV default | KV tuned (scale 3, credit 0.8) | RR |
|---|---|---|---|
| total tok/s per GPU | 9,655 (**1.42× RR**) | **11,012 (1.62× RR)** | 6,802 |
| TTFT p50 / p95 | 1.56 / 11.7 s | 0.87 / **6.3 s** | 10.9 / 60.1 s |
| P90 interactivity (tok/s/user) | 19.8 | 26.8 | 12.2 |
| in flight (of 192) | 74.5 | 64.1 | 102.7 |
| knee | scaling (+41% over 96) | scaling | **at throughput knee** (+11% over 96) |

Same SLO (TTFT p95 ≤ 20 s): RR's best cell inside the budget is 96 clients (6,137, p95 12.6 s); KV meets it at 192, so
**KV 192 vs RR 96 = 1.57×** and **tuned KV 192 vs RR 96 = 1.79×**. Under an interactivity budget (P90 ≥ 20) the pair
inside it is KV 96 vs RR 96 = 1.12×. Below the knee the KV advantage grows with load (1.03× at 48, 1.12× at 96, 1.42× at
192) because RR re-prefills most of each ~90 k-token turn while KV lands turns on the worker holding the prefix.
Measured knees: **agg KV 192, agg RR 192** (both 384 cells saturated and the ladders were stopped there); the sim had
placed the agg KV knee at 1536, a 4–8× miss on the load axis (AGENTX_AGG_RESULTS.md §iii, KNEE_ANALYSIS.md).

### 5.2 Router flag sweep at the comparison point (agg, 192 clients)

| flags | total/GPU vs default KV | TTFT p95 | verdict |
|---|---|---|---|
| prefill-load scale 3, overlap credit 0.8 | **+14%** | 6.3 s (−46%) | adopt: relieves per-worker prefill queues without losing the prefix |
| temperature 0.5 | −19% | 22 s (+90%) | reject: random worker choice discards the prefix; behaves like RR |
| prefill-load scale 2, credit 0.8 | +6% | 8.1 s (−31%) | gain is monotone in the load weight (scale 1 → 2 → 3) |
| tuned at 96 clients (fleet 2) | running | | tests whether the gain holds below the knee, where the sim predicted a loss |

The sim predicted −26% for the tuned router on agg at 192; silicon says +14%. The sign flip has the same root as the
agg decomposition (§5.3): the sim charges the KV router a decode-batch penalty for packing sessions and models the
load-scale flag as pure prefix loss, whereas the live effect is queue relief (AGENTX_AGG_RESULTS.md §v).

### 5.3 Simulation vs silicon, apple to apple (TTFT p95 standard)

Disagg (12:6 KV, 96 / 192 / 384 / 480, and 9:9 at 48 / 96 / 192): the engine substitutions (output length, 0.19 s
hand-off, measured decode line) barely move the ratios — total tokens stay at 0.53–0.59× and TTFT p95 at 2.3–2.7× too
pessimistic. The gap is (1) **prefix hit rate**: sim 0.74–0.78 vs 0.83–0.93 backed out of the measured records, so the
sim prefills 2.5–4× more tokens per turn, which sets its tail, its early knee (768) and its low ceiling (6,187 vs 10,434
measured and still rising); (2) **trace representation**: 70 k input tokens per request vs 86–94 k measured and a
0.69–0.80× request rate from the 4 k-request slice. Requests × input length reproduces the residual exactly. Agg (KV, 48 / 96): the largest term is the sim's decode
cliff past batch 7 (TPOT 27–40 ms simulated vs 7–12 ms measured); removing it recovers a third of the 0.35–0.44× gap,
the rest is the same trace residual. Full ladders: AGENTX_D72_RESULTS.md §iv, AGENTX_AGG_RESULTS.md §iv; overlay page
`reports/n3u-agentx-sim-vs-real.html`. Rankings across topologies are the sim's reliable output; levels, and the
KV-vs-RR ordering on agg, must come from silicon.

### 5.4 Disagg or agg, so far

At equal clients the 72-GPU disagg fleet is diluted 3× until it saturates, so the comparison is made at equal load
per GPU or at equal SLO (AGENTX_DISAGG_VS_AGG.md §0). Measured: agg's best stationary cell is KV 192 (9,655 total/GPU,
8 clients/GPU, p95 11.7 s) and its ceiling is there; disagg 12:6 KV at 480 clients (6.7 clients/GPU) is at **10,434**
with p95 3.5 s and still scaling (+21% over 384). **Disagg now beats agg's best per-GPU total at a lower load per GPU
and with a 3.3× shorter TTFT tail; agg wins per GPU only below ~5 live sessions per GPU, where its packed decode batches
are fuller** (agg 96: 6,844 at 4 clients/GPU vs disagg 288-equivalent ≈ 6,500 interpolated). The disagg ceiling is
still unknown (768 / 1440 running). The load-normalised verdict (clients per GPU under SLO → GPUs per 1,000 sessions) is finalised when the 12:6
ladder finds its knee.

Analysis so far (AGENTX_D72 §iv, AGENTX_AGG §iii, AGENTX_DISAGG_VS_AGG)Analysis so far (AGENTX_D72 §iv, AGENTX_AGG §iii, AGENTX_DISAGG_VS_AGG): the disagg engine model is within ~10 %
once a 0.19 s per-request hand-off is added; the agg simulator is pessimistic at low load (sim 1,474 vs 3,334
total/GPU, P90 22 vs 102) because its refit decode curve is a busy-stream fit; at 48 clients agg's per-GPU total is
3× disagg's because the same offered load is spread over 3× fewer GPUs (the dilution effect), which is why the agg-vs-
disagg comparison must be made at equal SLA or equal clients per GPU, not equal clients.

The disagg KV-vs-RR pairs (12:6 RR 192 / 96 / 384) and the 12:6 flag sweep are queued behind the 12:6 KV ladder; the
busy-stream conclusions (KV/RR 2.0× at c48 on 6:12, disagg 9:9 1.17× agg on total tokens at their bounded peaks) are in
D72_RESULTS.md §2 and AGG24_RESULTS.md §4.
