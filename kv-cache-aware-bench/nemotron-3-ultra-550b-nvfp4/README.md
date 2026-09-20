# KV-Aware Routing — NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 (GB300, sglang)

Second model of the KV-cache-aware routing study: agg + disagg KV-vs-RR on
GB300 NVL72, sglang backend, replaying `semianalysisai/cc-traces-weka-062126-256k`
(same dataset as the Kimi-K2.5 study for cross-model comparability).

Latest measured comparison (2026-09-20): [KV-Aware vs. Round-Robin Routing: NVIDIA Dynamo on Google Cloud](reports/agentx-serving-perf-report.md) · [standalone HTML](reports/agentx-serving-perf-report.html). This Google Cloud customer use journey follows coding-agent requests through Dynamo and measures how routing affects throughput, TTFT, and end-to-end responsiveness. Includes **37 hardware runs**, default-KV/RR curves with knee annotations, full tables including tuned KV, and comparisons at equal concurrency and under **TTFT p95 <10 s**. The E2E requirement is shown separately. At the best sampled points meeting both limits, default KV delivers **2.11× agg** and **6.92× disagg** total served tokens/s/GPU relative to RR, at different selected concurrency values; the tables also retain fixed-concurrency comparisons.

Current AgentX simulation performance (2026-09-17) uses the **new AIPerf replay** for agg and disagg; v3/v5 results are historical. Start with the [current result index](reports/agentx-aiperf-results.md): [64-GPU KV/RR topology curves](reports/agentx-64gpu-topology.md), [replay audit and simulation-vs-hardware gaps](reports/agentx-faithful-replay.md), and [reproduction commands](reports/agentx-replay-howto.md).

For the complete **agg KV versus RR report**, use [hardware sweeps, knee/SLO comparisons and simulation methods](reports/agentx-agg-kv-rr-report.md), or the [standalone HTML](reports/agentx-agg-kv-rr-report.html). It includes measured 24-GPU AgentX curves, the separate native DynoSim V10 calibration samples, and the earlier custom-model sweep with its limitations.

Why this model is the interesting second datapoint: hybrid Mamba-2/LatentMoE
with only 12/108 attention layers — 6 KB/token KV (6x smaller than Kimi),
near-linear prefill, and a smaller attention-KV footprint. Usable cache also depends on Mamba-state capacity and checkpoint eligibility; the original assumption of effectively unbounded cache is not sufficient. The study tests
whether KV-aware routing's win survives when recompute is cheap and cache is
abundant (pure placement value) — see `DESIGN.md` for the full walkthrough.

## Layout

| Path | Contents |
|---|---|
| `DESIGN.md` | experiment design walkthrough (stage-0 gates, AIC/DynoSim plan, silicon plan) |
| `stage-weights-job.yaml` | weight staging job (ungated nvidia NVFP4 checkpoint, 352 GB) |
| `sim-results/` | Current audited AIPerf replay archives and targets; historical AIC/DynoSim CSVs |
| `manifests/` | serving arms + bench jobs (generated; GPUDirect bypass baked in) |
| `scripts/` | arm generation, sweep sequencers, verification |
| `results/silicon/` | per-point measured summaries (KV/RR per conc) |
| `reports/` | pareto/knee curve pages, cross-model synthesis |
| `SELECTION.md` (tbd) | sim-selected operating points + hypothesis predictions |
| `AGG24_RESULTS.md` / `D72_RESULTS.md` (tbd) | silicon results records |

Model source: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (ungated; FP8
mamba mixers / FP4 MoE), staged at `/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4`.
Original design pins: aiconfigurator **0.11.0**, SGLang 0.5.14 and ai-dynamo 1.3.1.
The current measured recipes use Dynamo **1.4.2** with SGLang **0.5.16**;
the [setup and simulation sections](reports/agentx-serving-perf-report.md) distinguish
these from the older AIC timing tables and generated templates.

Original methodology: `../SWEEP_METHODOLOGY.md` (queue-drain stationarity),
`../SIMULATION_GUIDE.md`, Kimi baseline `../sglang/AGG24_RESULTS.md`.
The current report separates sampled throughput knees from the TTFT and E2E SLO boundaries.
