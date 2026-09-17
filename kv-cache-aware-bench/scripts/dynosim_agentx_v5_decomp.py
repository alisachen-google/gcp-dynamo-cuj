#!/usr/bin/env python3
"""Total tok/s/GPU decomposition for the v5 replay: req/s (root, subagent) x tokens/request, plus request latency,
with an IDEAL-ENGINE control (zero prefill/decode time) to separate replay terms from engine terms."""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import dynosim_agentx_v5 as v5, dynosim_agentx as da
dp = da.dp
traces = v5.load_stream_trace(sys.argv[1]); c = int(sys.argv[2])
def row(name, m): print(f"{name:34s} req/s={m['req_per_s']:5.2f} (root {m['root_rps']:4.2f} sub {m['sub_rps']:4.2f}) tok/req={m['in_tok_per_req']+m['out_per_req']:7.0f} (root ISL {m['root_isl']:6.0f} sub ISL {m['sub_isl']:6.0f} OSL {m['out_per_req']:4.0f}) tot/GPU={m['total_tok_s']/72:6.0f} lat={m['lat_mean_s']:5.1f}s ttft95={m['ttft_p95_s']:4.2f} tpot={m['tpot_mean_ms']:4.1f}", flush=True)
dp.apply_n3u_constants(); row("v5d (recycle at turn 0)", v5.simulate_v5(traces, 12, 6, "kv", c))
# ideal engine: prefill infinitely fast, decode 0.001 ms/token -> latency ~ 0; shows the replay-only request rate
dp.apply_n3u_constants(); dp.PREFILL_TOKRATE = 1e12; dp.TPOT_BASE_MS = 0.001; dp.TPOT_SLOPE_MS = 0.0
row("v5d + ideal engine (lat ~ 0)", v5.simulate_v5(traces, 12, 6, "kv", c))
dp.apply_n3u_constants(); dp.TPOT_BASE_MS = 6.9; dp.TPOT_SLOPE_MS = 0.44; dp.PREFILL_TOKRATE = 24000
row("v5d + measured decode/prefill", v5.simulate_v5(traces, 12, 6, "kv", c))
