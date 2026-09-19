# Native disagg KV flag sweep at 480 sessions

This study keeps the native V11 disaggregation build and the V10 AIC timing
coefficients frozen while changing three real Dynamo router flags. It models
64 GPUs, 8 prefill and 8 decode TP4/EP4 workers. No hardware deployment is made.

The initial grid tests overlap credits 0.6, 0.8 and 1.0 against credit decays
0, 0.5 and 1.0 at prefill-load scale 1. Two scale controls use scales 2 and 3
at credit 0.8 / decay 0. The baseline and best eligible candidate are repeated.
This is 11 distinct configurations and two repeats, not a 27-cell factorial grid.

Scale controls are deliberate: for deterministic pure-prefill routing, the
positive scale multiplies every candidate cost and is expected to preserve the
minimum-cost candidate set. Minimum-cost ties can still be broken randomly.
Credit and backlog-dependent credit decay can change that set. The functional
preflight confirms that the requested values reach the native prefill selector
and that positive decay reduces credit under excess prefill backlog.

The clock remains wall time at speedup 1. Each arm reuses the 393-root AgentX
configuration, tokenizer, seed 42, sampled trajectory starts, one-hour profiling
window, 60-second grace, and 1200-second request timeout. The shared benchmark ID
fixes the cache-bust namespace. New processes give each arm empty caches before
the standard trajectory warmup. Native tie-breaking and completion-dependent
recycling mean this is not byte-identical traffic for the whole profiling hour.

The native core SHA-256 is
`39e41354104490981f35f76850bd17973369fc8222dab9f7813e25f50146e36e`.
The server is Dynamo 1.4.2 plus the preserved native patches, with AIC 0.11.0.
This V11 disaggregation extension is distinct from the V10 aggregate build.
Patch hashes and reconstruction evidence are retained in `patches/` and the
sweep's `configs/`. No timing coefficients are refit during this sweep.

The local run is managed by `controller.py`; it adopts existing owned runs,
enforces memory/storage/parallelism limits, collects finished exports, and
losslessly compresses raw artifacts after round-trip verification. A failed run
is recorded and does not silently stop all later points. Repeated collection
failures stop the controller with an explicit `needs_attention` state. The
process owns only `/tmp/agentx-disagg-c480-runs` and its child process groups.

For the prepared environment:

```bash
/tmp/n3u-faithful-venv/bin/python scripts/native_disagg_flags/controller.py
```

`run_path1_topology.py --help` lists runtime paths and ports. Every run checks the
native build, saves the resolved frontend configuration, records all per-engine
metrics and native forward-pass events, and writes the official AIPerf exports.
The collector checks the engine configs and paired warmup signatures. Generated
requests and raw traces stay in local run artifacts; compact numeric request
metrics and configuration provenance accompany the report.

Selection uses total input+output tok/s/GPU under both P95 TTFT <=10 seconds
and E2E-normalized I90 >=20 output tok/s. I90 is `1/P90(E2E_seconds/OSL)` with
linear percentiles over successful profiling requests. Quality also requires
valid replay, matching warmup/build, valid metrics, and error rate <=0.1%.
The baseline repeat measures observed variation; one repeat pair is not a
statistical confidence interval. If no candidate passes, report no feasible
winner. An exploratory repeat then targets the eligible candidate with the
highest I90, if one exists.

Disaggregated TTFT is not hardware-calibrated. Native handoff admission has a
finite session pool; errors are not suppressed or counted as successes. The
prefill cache allocation is inherited from the prior simulation, transfer speed
is 64 GB/s per rank, and shared transfer contention is not modeled. A predicted
winner requires a hardware spot-check before serving recommendations.

`test_metrics.py` guards normalization order, exclusions, and units. The parser
was also checked against 17,289 successful requests from a previously completed
native C256 run, independently matching NumPy's linear percentile. That parser
check is not a C480 measurement.
