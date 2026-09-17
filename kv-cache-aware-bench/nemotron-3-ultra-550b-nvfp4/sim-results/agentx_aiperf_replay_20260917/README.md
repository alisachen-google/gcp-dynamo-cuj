# Actual-aiperf AgentX replay, 2026-09-17

Both cells ran a full 3,600-second profiling window, after the actual trajectory
warmup, with 60 seconds of grace and a 1,200-second request timeout. The serving
model uses the current DynoSim aggregated engine without fitting its parameters.

- `audit.json`: replay assertions, performance comparison, and provenance.
- `comparison.csv`: same aiperf summary metrics on simulation and hardware.
- `worker_placement.json`: simulated placement and per-worker decode/cache metrics.
- Each point directory: aiperf summary, configuration, provenance, statistics,
  and compressed complete simulation request records and log.
- `prewarm-validation.log`: reproduces the job template's prewarm parsing error.

The original hardware artifact URIs and configuration are retained in
[`../agentx_calibration_targets.json`](../agentx_calibration_targets.json).
See [analysis](../../reports/agentx-faithful-replay.md) and
[rerun instructions](../../reports/agentx-replay-howto.md).
