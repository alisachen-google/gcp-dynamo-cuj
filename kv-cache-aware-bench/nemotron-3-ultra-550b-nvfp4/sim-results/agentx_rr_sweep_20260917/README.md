# AgentX RR sweeps, 2026-09-17

Ten completed full aiperf 0.12.0 replays: agg6 at 48/96/192/384 clients and disagg12:6 at 48/96/192/384/480/768. Agg has matching actual AgentX hardware artifacts. Disagg has only a GitHub RR recipe and uses the actual disagg KV192 client configuration; its performance is a prediction, not a comparison against measured RR.

- `audit.json`: all input, timing, tokenization, RR-placement checks, metrics, source hashes, and queue diagnostics.
- `comparison.csv`: aiperf metrics normalized over 24 GPUs for agg and 72 for disagg; hardware columns populated only for matched agg runs.
- `recipe-provenance.json`: pinned GitHub recipe/manifests and the scoped hardware-artifact search.
- Each point directory: config, summary, source provenance, statistics, and compressed full request records and aiperf log.
- `sha256.json`: integrity manifest for this archive.

Original full local runs were in `/tmp/n3u-rr-sweep-20260917` (agg) and `/tmp/n3u-disagg-rr-sweep-20260917` (disagg). The 100M-token legacy cache is preserved in these runs. Later capacity and KV-routing diagnostics have separate archives.

[Analysis and hardware comparison](../../reports/agentx-faithful-replay.md) · [Disagg predictions](../../reports/agentx-disagg-rr-predictions.md) · [Run instructions](../../reports/agentx-replay-howto.md)
