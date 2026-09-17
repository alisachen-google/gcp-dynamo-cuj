# Current AgentX simulation performance — AIPerf replay

**Effective 2026-09-17, the actual AIPerf replay replaces handwritten v3/v5 simulation performance for this study.** Use the following audited results for new comparisons, curves and topology decisions.

| Scope | Current report | Audited artifacts |
|---|---|---|
| 64 GPUs, disagg KV and RR, topology/concurrency search | [Topology curves and completion status](agentx-64gpu-topology.md) | [CSV](../sim-results/agentx_64gpu_20260917/comparison.csv), [audit and provenance](../sim-results/agentx_64gpu_20260917/README.md), [target matrix](../sim-results/agentx_64gpu_targets.json) |
| 72 GPUs, disagg 12P:6D RR | [Prediction curve](agentx-disagg-rr-predictions.md) | [RR sweep](../sim-results/agentx_rr_sweep_20260917/README.md) |
| 24 GPUs, agg RR and KV versus real AgentX | [Replay audit, matched gaps and calibration](agentx-faithful-replay.md) | [RR sweep](../sim-results/agentx_rr_sweep_20260917/README.md), [KV routing correction](../sim-results/agentx_kv_decode_cost_20260917/README.md), [RR capacity check](../sim-results/agentx_rr_capacity_20260917/README.md) |

The [reproduction workflow](agentx-replay-howto.md) runs AIPerf 0.12.0's loader, scheduler, warmup, dependency graph, recycling and exporter against a simulated serving transport. Every published performance point has a 3,600-second profile, actual trajectory warmup, 60-second grace and 1,200-second request timeout. Throughput uses AIPerf token counts and the complete deployment's GPU denominator. Hardware comparisons require the same topology, routing policy and client count.

Replay audit success establishes workload semantics and the checked source/token identities. Serving performance is still approximate: hybrid/Mamba cache behavior, chunk scheduling, admission, dynamic batching and disagg transfer require calibration. The KV192 routing correction is an isolated calibration experiment; it is not silently applied to every other result. The 64-GPU archive explicitly identifies its cache-capacity proxy and decode-routing model. There are no matching 64-GPU hardware measurements in this comparison.

The [old workflow and interpretation](agentx-legacy-workflow.md), [old knee/topology tables](knee-analysis-legacy-20260917.md), and `dynosim_n3u_agentx_v*.csv` / `dynosim_n3u_agentx_d64_v5.csv` are historical. Earlier HTML pages display a historical notice. In particular, the old v5 same-SLO gains and its claimed reliable topology/policy ranking are withdrawn as current evidence. Pending AIPerf points are shown as pending; old simulator values do not fill those gaps.
