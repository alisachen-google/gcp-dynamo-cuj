# SLO-Fixed Comparison (Framing 3): Max Compliant Throughput, KV vs RR

Methodology: fix a TTFT **p95** SLO, report each policy's maximum measured
throughput among runs meeting it (all values silicon; SLO boundaries refined by
measured bisection — simulation is never used for compliance verdicts, its
latency tails being its least reliable output per the drift analyses).
Complements the queue-drain framings: this is the deployment-facing question
("what can I promise users"), those are capacity-facing ("where does it break").

## Nemotron-3-Ultra 550B (output tok/s; agg = 24 GPU, disagg = 72 GPU 6:12)

| TTFT p95 SLO | agg KV | agg RR | KV/RR | disagg KV | disagg RR | KV/RR |
|---|---|---|---|---|---|---|
| **≤ 5 s** | **1,657** (c32, 4.3 s) | **none** (best: c8 @ 7.8 s) | **∞** | none (best: c8 @ 6.4 s) | none | — |
| **≤ 10 s** | **1,853** (c48, 7.8 s) | 882 (c16, 9.8 s) | **2.1×** | 1,614 (c16, 8.6 s) | 504 (c4, 8.1 s) | **3.2×** |
| **≤ 30 s** | 2,045 (c64) | 1,141 (c64) | 1.8× | 2,844 (c48) | 1,534 (c24) | 1.9× |

The sharpest result of the framing: **at a 5 s p95 SLO, KV-routed aggregated
serving is the ONLY configuration on this model that can serve the workload at
all** — RR misses 5 s even at conc 8 (its p95 is recompute tail, not queueing:
p50 is 0.99 s), and no disagg configuration of either policy complies (the
host-staged transfer floor alone puts p95 above 6 s).

Secondary notes: agg KV's 10 s optimum is conc 48 (1,853 — the boundary point
found by bisection; c64 breaches at 21.5 s). Disagg RR's only compliant point
is conc 4 (72 GPUs serving 4 concurrent users — compliant but absurd:
7 tok/s/GPU).

## Kimi-K2.5 (same framing, from the existing silicon grids; agg = 24 GPU)

| TTFT p95 SLO | agg KV | agg RR | KV/RR | disagg (72 GPU) |
|---|---|---|---|---|
| ≤ 5 s | 2,119 (c32, 2.6 s) | 964 (c8, 4.7 s) | **2.2×** | **no compliant point, either policy** (all runs ≥ 51 s p95) |
| ≤ 10 s | 2,119 (c32) | 1,379 (c24, 7.9 s) | 1.5× | none |
| ≤ 30 s | 2,119 (c32) | 1,398 (c32, 9.5 s) | 1.5× | none |

Kimi's disagg row is itself a finding: on the host-staged transfer path, a
3.4 GB/request-transfer model cannot meet ANY interactive TTFT SLO
disaggregated — the strongest single statement of the transfer-bound ceiling.

## Cross-model summary at the interactive SLOs

- KV-aware routing multiplies SLO-compliant capacity **1.5–3.2×** everywhere a
  comparison exists, and at the strictest SLO it is the difference between
  servable and unservable (both models).
- Aggregated + KV routing is the SLO-optimal topology for BOTH architectures
  on this trace — for Kimi because disagg can't meet any SLO on this transfer
  path, for Nemotron because agg beats disagg per GPU everywhere measured.
- Boundary points, per-point evidence and knee verdicts: `results/silicon/`,
  DIARY.md; disagg transport verified RDMA (rc_mlx5, zero cuda_ipc/MNNVL
  lines) per point where log windows permitted, transport guard PASS on all.
