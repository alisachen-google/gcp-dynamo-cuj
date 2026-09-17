# AgentX knees and topology selection — current AIPerf replay

As of 2026-09-17, simulation performance for this study comes from the actual **AIPerf 0.12.0 AgentX replay** driving the DynoSim serving model. The handwritten v3/v5 workload results are superseded for both aggregated and disaggregated serving.

| Study | Current results and evidence | Interpretation |
|---|---|---|
| 64 GPUs, disagg KV and RR | [Topology curves and selection status](reports/agentx-64gpu-topology.md), [audited CSV](sim-results/agentx_64gpu_20260917/comparison.csv), [targets](sim-results/agentx_64gpu_targets.json) | TP4, 16 workers; start at 16 clients and extend through the throughput knee. High-load points are still running. |
| 72 GPUs, 12P:6D RR | [Prediction curve](reports/agentx-disagg-rr-predictions.md), [audit](sim-results/agentx_rr_sweep_20260917/README.md) | The sampled throughput peak is near 384–480 clients; no matching disagg RR hardware reference is available. |
| 24 GPUs, agg RR and KV calibration | [Replay audit and performance gaps](reports/agentx-faithful-replay.md) | RR throughput matches closely at 48/96 clients but overpredicts at 192/384. The KV192 routing correction is a one-point calibration. |

The 64-GPU curve reports total input plus output tokens/s divided by all 64 GPUs, together with output throughput, TTFT, average ITL, cache fraction, and errors. A sampled knee proxy is the first tested concurrency reaching 90% of that curve's observed peak. A curve still rising at its upper boundary does not establish a saturation knee. First/second-half queue and latency statistics are reported separately. No latency-SLO gate determines the primary topology ranking.

The earlier v5 recommendation of 8P:8D, its 768/384 flag-sweep choices, and its same-SLO gains are **withdrawn as simulation evidence**. The [historical analysis](reports/knee-analysis-legacy-20260917.md) and original CSV remain available for reproduction. At 8P:8D and 384 clients, v5 predicted KV 8,379 versus RR 9,079 total tok/s/GPU; the audited replay predicts KV 10,389 versus RR 6,241. Replay rules, serving constants, cache capacity, and decode routing all changed, so this difference is not a controlled estimate of the replay effect alone.

The existing [8P:8D hardware manifest](../sglang/manifests/n3u-mnnvl-88.yaml) is an available calibration recipe. Its presence does not establish a winning topology. The [hardware chain](scripts/run_agentx_88_np2.sh) runs measured KV/RR ladders and uses [measured point selection](scripts/select_agentx_points.py) for router-flag tests; it stops if selection fails or produces no eligible points. Its inherited ladder is a hardware sampling plan, not a new AIPerf result. Changes to this file do not update an already launched copy of the chain.

[Current result index](reports/agentx-aiperf-results.md) · [Reproduction commands](reports/agentx-replay-howto.md#64-gpu-topology-sweep)
