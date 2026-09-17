# 64-GPU actual-AIPerf topology sweep, 2026-09-17

**In progress: 33/47 points audited as of 2026-09-17 06:19 UTC.** These are simulation predictions, with no matching 64-GPU hardware performance reference. TP4 is fixed; 16 workers are split between P and D. The replay configuration comes from an actual 72-GPU disagg KV192 AgentX artifact, used for input/config validation only.

- `audit.json`: source/timing/tokenization/placement checks, aiperf metrics, first/second-half queue diagnostics, and per-worker summaries.
- `comparison.csv`: completed points, with throughput normalized over all 64 GPUs.
- Each point directory: aiperf summary, replay config, model/source provenance, statistics, and SHA-256 hashes of the full locally audited request records.
- `hardware-disagg-server-metrics.csv`: complete source server-metric summary used for the 51.8M-token cache-capacity proxy. The metric is decode-facing model metadata; P-worker and Mamba-state capacity remain unmeasured.
- `union-optimization-check.json`: a computational optimization preserves exact P/D selections, service results, cache hits and final state on 2,000 mixed shared-prefix requests/cancellations. It replaces materialized set union with cardinality plus missing-block count; the model formula does not change. Earlier source-hash variants also differ in documentation and release-argument validation. Each run records its own hashes.

Full request records are temporarily available under `/tmp/n3u-64gpu-{coarse,upper,refine,resumed,kv1536}-20260917/`. This compact Git archive does not contain every raw record; the supplied commands reproduce the experiments, and `--compact` can be omitted to retain compressed full records when collecting.

The first session's sweep controllers stopped while active child replays continued. All completed data was retained; nine never-started points resumed in a new directory with detached controllers and file-based logs. This did not restart or mix state between individual replays.

[Results and limitations](../../reports/agentx-64gpu-topology.md) · [Targets](../agentx_64gpu_targets.json) · [Run instructions](../../reports/agentx-replay-howto.md#64-gpu-topology-sweep)
