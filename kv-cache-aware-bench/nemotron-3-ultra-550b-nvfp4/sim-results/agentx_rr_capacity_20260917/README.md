# RR cache-capacity diagnostic, 2026-09-17

Two full actual-AIPerf AgentX replays change only the per-worker prefix-cache capacity from 100,000,000 to **28,396,608 tokens**. The actual agg RR192 and RR384 frontend metrics both advertise 443,697 blocks per worker, at 64 tokens/block. The saved model metadata also reports max_num_seqs 16 and max_num_batched_tokens 16,384.

Both runs pass replay parity against their own hardware artifacts. Every simulated worker reaches exactly 443,697 cached blocks, verifying the configured limit actually binds. Yet throughput error changes only from +29.0% to +28.6% at 192, and +118.2% to +111.5% at 384. A smaller plain LRU cache does not reproduce the real cache-hit collapse.

- `audit.json`, `comparison.csv`: replay checks and metrics.
- Per-point directories: complete compressed records, summaries, configs, source hashes, and statistics.
- `hardware-rr-c*-selected-server-metrics.json`: selected exported server gauges/counters with large timeseries arrays removed. These are derived from each real run's `server_metrics_export.json`; full originals remain in `/tmp/n3u-rr-sweep-real/agg6-rr-c*/server_metrics.json`.

Reproduce from `kv-cache-aware-bench`, using the environment in the replay how-to:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/sweep_aiperf_replay.py \
  --aiperf-source /tmp/n3u-aiperf-source \
  --arrow-dir /tmp/n3u-faithful-data/arrow \
  --tokenizer /tmp/n3u-faithful-data/tokenizer \
  --targets nemotron-3-ultra-550b-nvfp4/sim-results/agentx_rr_sweep_targets.json \
  --points agg6-rr-c192,agg6-rr-c384 \
  --cache-capacity-tokens 28396608 --jobs 2 \
  --run-root /tmp/n3u-rr-capacity-rerun
```

[Gap analysis](../../reports/agentx-faithful-replay.md#controlled-rr-cache-capacity-check)
