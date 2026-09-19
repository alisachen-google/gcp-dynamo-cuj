# AgentX disagg KV flag sweep at concurrency 480

Updated: 2026-09-19 05:16 UTC. Completed: **0/12** scheduled runs. Queue: **running**.

64 GPUs, 8 prefill + 8 decode TP4 workers; 480 live AgentX sessions. Native DynoSim V11 disaggregation extension with frozen V10 AIC timing coefficients. These are simulation forecasts; disaggregated TTFT is not calibrated to hardware.

The initial grid uses scale 1, credit {0.6, 0.8, 1.0}, and decay {0, 0.5, 1.0}. Two controls use scales 2 and 3 at credit 0.8 / decay 0. Temperature stays 0 and FCFS stays fixed. Baseline and finalist repeats check reproducibility.

| Run | Scale | Credit | Decay | Total tok/s/GPU | Δ vs baseline | P95 TTFT | E2E I90 | Errors | Quality / both SLOs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| baseline | 1 | 1 | 0 | — | — | — | — | — | failed |
| credit08 | 1 | 0.8 | 0 | — | — | — | — | — | failed |
| scale3-credit08 | 3 | 0.8 | 0 | — | — | — | — | — | failed |
| scale2-credit08 | 2 | 0.8 | 0 | — | — | — | — | — | failed |
| credit10-decay05 | 1 | 1 | 0.5 | — | — | — | — | — | failed |
| credit10-decay10 | 1 | 1 | 1 | — | — | — | — | — | running |
| credit08-decay05 | 1 | 0.8 | 0.5 | — | — | — | — | — | running |
| credit08-decay10 | 1 | 0.8 | 1 | — | — | — | — | — | pending |
| credit06 | 1 | 0.6 | 0 | — | — | — | — | — | pending |
| credit06-decay05 | 1 | 0.6 | 0.5 | — | — | — | — | — | pending |
| credit06-decay10 | 1 | 0.6 | 1 | — | — | — | — | — | pending |
| baseline-repeat | 1 | 1 | 0 | — | — | — | — | — | pending |

## Interpretation and controls

Select the highest observed total input+output throughput/GPU satisfying P95 TTFT ≤10 s and E2E-normalized interactivity I90 ≥20 output tok/s. I90 is computed as `1 / P90(E2E_seconds / output_tokens)` from individual successful profiling requests; it is not inverse TPOT. Pending runs provide no evidence of a win.

Quality requires valid AgentX replay, the frozen native build, matching baseline warmup, valid per-request latency metrics, and error rate ≤0.1%. Errors remain visible. Different completed turn mixes remain possible in a time-bounded, closed-loop replay.

Load-dependent credit is `credit / (1 + decay × excess_prefill_load / request_size)`. Decay 1 halves credit at one request-equivalent of excess backlog. It does not expire cache entries. In this pure-prefill deterministic configuration, multiplying all candidate prefill costs by the same positive scale is expected to preserve their ordering; scale controls test whether another cost or runtime effect matters.

Every arm uses the same 393-root dataset, tokenizer, seed 42, cache-bust namespace, 25–75% trajectory starts, trajectory warmup, 3600-second profile, 60-second grace, 1200-second request timeout, engine configs, and 64 GB/s per-rank transfer assumption. Each starts with new workers and empty caches. No GPU hardware is provisioned.

The native handoff implementation has finite session admission; overload errors must be considered when comparing flags. Transfer contention and prefill cache-capacity uncertainty remain simulation limitations.

[Plan](../sim-results/agentx_disagg_c480_native_20260919/plan.json) · [Results JSON](../sim-results/agentx_disagg_c480_native_20260919/results.json) · [CSV](../sim-results/agentx_disagg_c480_native_20260919/results.csv) · [Native flag propagation check](../sim-results/agentx_disagg_c480_native_20260919/preflight.json)
