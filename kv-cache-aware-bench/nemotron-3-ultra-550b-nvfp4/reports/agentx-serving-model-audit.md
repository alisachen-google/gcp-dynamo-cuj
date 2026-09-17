# Which serving simulator produced the AgentX results?

Audited 2026-09-17. **The published AgentX simulations used custom Python study
code, not NVIDIA's upstream DynoSim.** Calling that code “DynoSim” without this
qualification was misleading. The local v3/v5 labels are study revisions, not
NVIDIA release versions.

The execution path is:

```text
AIPerf 0.12.0 AgentX loader / scheduler / exporter
  -> dynosim_aiperf_replay.py simulated transport
  -> dynosim_agentx.py::Engine
  -> dynosim_pd.py cache and fitted timing constants
```

The RR384 archive records `engine: original`. Its saved source hashes match:

| Component | Source used by the reported RR sweep |
|---|---|
| AIPerf | 0.12.0; source `be53bf2953d30e46c500e6a80fc1f8b6f84bc718` |
| `dynosim_agentx.py` | Study commit `dba4c93f39b5`; SHA-256 `d6e5b434ce3535f82f0c997fd0ddb6e86fc907cea3781fbefcf914dd25a9d660` |
| `dynosim_pd.py` | Study commit `64e36f21a40c`; SHA-256 `1a4d61de9cf7d7fdb9950f9bccbe7bdc3327f10c26ba830d0ab94f0ce581095e` |
| NVIDIA DynoSim package | Not invoked by these runs |

[Saved provenance](../sim-results/agentx_rr_sweep_20260917/agg6-rr-c384/simulation-provenance.json)
is the source of this identification. Later source edits do not change the
identity of earlier artifacts.

## Where waiting requests affect decode timing

In [`Engine.serve`](../../scripts/dynosim_agentx.py), `pf` is a future prefill
completion time. The function immediately increments `self.D[d]`, evaluates
`agg_tpot_ms(self.D[d])`, and returns that fixed time per output token. The
adapter then waits for prefill and decoding and calls `Engine.release` only
when the request completes or is cancelled.

Consequently, `self.D[d]` counts all outstanding assigned requests, including
those queued for prefill. It is not the batch currently executing decode.
The aggregate timing formula is `8.9 + 1.73 * count` milliseconds. Its value is
never recalculated for an existing request when other requests enter or leave.

This small example runs the same engine without AIPerf. From `kv-cache-aware-bench`:

```bash
PYTHONPATH=scripts python - <<'PY'
import dynosim_agentx as sim
sim.dp.apply_n3u_constants()
engine = sim.Engine(1, 0, 'rr', agg=True)
a = engine.serve(list(range(1000)), 16, 0.0)
b = engine.serve(list(range(1000, 2000)), 16, 0.0)
print('A prefill done / request done:', a[0], a[2])
print('B prefill done / fixed TPOT ms:', b[0], b[1] * 1000)
engine.release(a[3])
print('Remaining count:', engine.D, 'B TPOT still:', b[1] * 1000)
assert a[2] < b[0]
assert b[1] * 1000 == sim.dp.agg_tpot_ms(2)
PY
```

Both requests arrive at time zero with distinct 64,000-token prompts. A finishes
at about 3.42 seconds. B does not finish prefill until 6.50 seconds, so A is
already gone when B starts decoding. Nevertheless B retains the two-request
rate, **12.36 ms/token**, instead of the formula's one-request rate,
**10.63 ms/token**. These are illustrative model outputs, not hardware timings.
The adapter uses output-length minus one inter-token intervals; that convention
does not alter the counter or fixed-rate issue.

An outstanding-request count can be an empirical load proxy. The flaw is
treating that proxy as a faithful evolving decode batch and extrapolating it
into saturation. Aggregate serving also needs a shared prefill/decode resource
schedule and admission constraints. Merely postponing the increment until
prefill completes would not fix the permanently assigned decode rate.

**This example does not explain the +29%/+118% throughput error by itself.**
Overcounting alone raises modeled decode latency and can lower throughput;
removing it can therefore increase the overprediction. Missing cache eligibility,
cache insertion timing, admission, and prefill/decode contention must be evaluated
together. The [capacity-only experiment](agentx-faithful-replay.md#controlled-rr-cache-capacity-check)
already shows that correcting one cache-size parameter does not close the gap.

## Upstream status

As checked on 2026-09-17, NVIDIA's latest stable Dynamo release is
[v1.4.2](https://github.com/ai-dynamo/dynamo/releases/tag/v1.4.2); inspected `main`
was `1dca16f9b74dcd938496ec6c79be5d227c372c93`. Upstream DynoSim has a distinct
[SGLang scheduler core](https://github.com/ai-dynamo/dynamo/tree/v1.4.2/lib/mocker/src/scheduler/sglang)
with separate waiting/running state and per-step execution. The local Python
finding is not a reproduction of an upstream defect.

The official `ai-dynamo-runtime==1.4.2` wheel was imported and completed a
six-request synthetic SGLang replay as a smoke test during this audit. It has
**not** produced the published AgentX performance points. Its default synthetic
timing model is not a calibrated Nemotron/GB300 model. Replacing the serving
engine still requires preserving actual AgentX closed-loop replay, configuring
the measured worker limits, and validating timing/cache behavior at 192 and
held-out 384 clients.
