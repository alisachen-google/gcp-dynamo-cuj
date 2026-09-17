# KV192 active-block routing diagnostic, 2026-09-17

A full aiperf 0.12.0 AgentX replay with the original hardware configuration,
benchmark ID, dataset, tokenizer, and timing parameters. Only the KV worker
score changes: it adds the projected unique active prompt-block footprint.
The baseline serving rates, idealized cache, and static per-request decode
speed are unchanged. No empirical coefficient was fitted.

- `audit.json`: replay parity and performance against the actual KV192 AgentX run.
- `worker_placement.json`: simulation warmup/profile placement and mean ITL.
- `agg6-kv-c192/`: summary, config, source fingerprints, statistics, compressed
  per-request records and complete aiperf log.

Result: 9,089 vs 9,655 real total tokens/s/GPU (−5.9%), improving on baseline
6,983 (−27.7%). This single-point diagnostic does not establish full calibration.

From `kv-cache-aware-bench`, reproduce with a new artifact directory:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/dynosim_aiperf_replay.py \
  --aiperf-source /tmp/n3u-aiperf-source \
  --arrow-dir /tmp/n3u-faithful-data/arrow \
  --tokenizer /tmp/n3u-faithful-data/tokenizer \
  --targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_calibration_targets.json \
  --point agg6-kv-c192 --kv-decode-block-cost \
  --artifact-dir /tmp/n3u-kv-decode-cost-rerun
```

[Full analysis](../../reports/agentx-faithful-replay.md#controlled-kv192-decode-load-correction)
