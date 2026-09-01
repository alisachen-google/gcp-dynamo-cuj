# KV-Aware Routing — NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 (GB300, sglang)

Second model of the KV-cache-aware routing study: agg + disagg KV-vs-RR on
GB300 NVL72, sglang backend, replaying `semianalysisai/cc-traces-weka-062126-256k`
(same dataset as the Kimi-K2.5 study for cross-model comparability).

Why this model is the interesting second datapoint: hybrid Mamba-2/LatentMoE
with only 12/108 attention layers — 6 KB/token KV (6x smaller than Kimi),
near-linear prefill, effectively unbounded per-worker cache. The study tests
whether KV-aware routing's win survives when recompute is cheap and cache is
abundant (pure placement value) — see `DESIGN.md` for the full walkthrough.

## Layout

| Path | Contents |
|---|---|
| `DESIGN.md` | experiment design walkthrough (stage-0 gates, AIC/DynoSim plan, silicon plan) |
| `stage-weights-job.yaml` | weight staging job (ungated nvidia NVFP4 checkpoint, 352 GB) |
| `sim-results/` | AIC SILICON solves + DynoSim sweep CSVs (`dynosim_n3u_*.csv`) |
| `manifests/` | serving arms + bench jobs (generated; GPUDirect bypass baked in) |
| `scripts/` | arm generation, sweep sequencers, verification |
| `results/silicon/` | per-point measured summaries (KV/RR per conc) |
| `reports/` | pareto/knee curve pages, cross-model synthesis |
| `SELECTION.md` (tbd) | sim-selected operating points + hypothesis predictions |
| `AGG24_RESULTS.md` / `D72_RESULTS.md` (tbd) | silicon results records |

Model source: `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` (ungated; FP8
mamba mixers / FP4 MoE), staged at `/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4`.
Toolchain pins: aiconfigurator **0.11.0** (0.10.0 lacks NemotronH), sglang
0.5.14 + ai-dynamo 1.3.1 (pending NemotronH smoke), dynamo KV router v1.3.1.

Methodology: `../SWEEP_METHODOLOGY.md` (knee = queue-drain stationarity, no
latency SLO), `../SIMULATION_GUIDE.md`, Kimi baseline `../sglang/AGG24_RESULTS.md`.
