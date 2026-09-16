# SemiAnalysis InferenceX AgentX (DeepSeek-V4 on GB300) vs our N3U KV study — concurrency, knee, interactivity

Source run: https://github.com/SemiAnalysisAI/InferenceX/actions/runs/33933793466
("Run Sweep — [NV] Refresh GB300 DeepSeek-V4-Pro AgentX with SGLang DSpark6", PR #2623,
merged 2026-09-10, 11 h 50 min). Recipe: `benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic/agg-gb300-tp4-mtp-lowlatency.yaml` (+ `tp8`). Methodology: inferencex.semianalysis.com/agentx, /about.

## 1. What the AgentX run is

| | InferenceX dsv4 AgentX (GB300) | our N3U study |
|---|---|---|
| model | DeepSeek-V4-Pro-0813, FP4, 1.6T MoE | Nemotron-3-Ultra 550B-A55B NVFP4 (hybrid Mamba/MoE) |
| stack | SGLang v0.5.19 image + Dynamo 1.5.0.dev, frontend `router-mode: kv` + nginx **session affinity** (X-Dynamo-Session-ID, TTL 3600 s) | SGLang 0.5.16 + Dynamo 1.4.2, `router-mode kv` (no session affinity) vs round-robin |
| topology | **aggregated** TP4 (1 node) / TP8 (2 nodes), EP1 — the PR *deleted* its disaggregated configs for this workload | agg 6×TP4 (24 GPU) and disagg 6:12 / 9:9 (72 GPU) on MNNVL |
| speculative decoding | **DSpark K=6** (7 draft tokens) — on | MTP **off** on both arms (router-comparison parity) |
| KV memory | HiCache DRAM tiering (`hicache`), page-size 256 | HBM only; page-size 64 |
| scheduler | max-running-requests 32 (TP4), chunked-prefill 8192 | agg 16; prefill 8 / decode 64; chunked-prefill 16384 |
| dataset | **AgentX v1.0 = 393 opt-in Claude Code sessions**, "256k" variant (median input 142k tok/req, median output 444; 44% sessions have subagents) | **the same dataset**: `semianalysisai/cc-traces-weka-062126-256k`, 393 sessions |
| concurrency values | **8, 480, 960, 1440, 1920** | 12 → 512 (disagg), 16 → 512 (agg) |
| per point | 1 h profile after primer + 10 warmup requests per replay lane; per-replay **cache-bust markers** (no cross-session prefix sharing, reuse only inside a session) | 300 s settle + 900 s trace-replay warmup + 1800 s measure; shared warm cache across streams |
| reported | throughput + TTFT + **interactivity** (≈ 1/ITL) and E2E-normalized interactivity (output tok / e2e latency); no SLO gate | throughput, TTFT p50/p95/p99, ITL; knee by queue-drain stationarity; no SLO gate |

## 2. Their "knee" — there isn't one; they trace a frontier

InferenceX does not compute a knee or gate on an SLO. Every configuration is profiled across
the whole concurrency sweep and the dashboard plots **throughput per GPU against
interactivity (tok/s/user ≈ 1/ITL)**; the reader chooses an interactivity floor and reads
off the throughput there. High concurrency exists to trace the *low-interactivity,
high-throughput* end of that frontier — by construction those points are what our
methodology calls post-knee. Their only saturation signal is on the *client* ("treat the
high-CPU warning as a saturation signal"). In our terms, their comparison point is
"max throughput subject to an interactivity floor", which is our framing 3 with
interactivity instead of TTFT.

## 3. Why they can run 480–1920 "concurrency" and we knee at ~120

**The word means different things.** AgentX: *"Concurrency means concurrent agent clients,
not a fixed request batch … the number of live session trees, not the number of in-flight
HTTP requests."* Clients are closed-loop session replays; a live session tree spends most
of its time waiting (tool execution, subagent fan-out, turn boundaries), so 1,920 clients
is far fewer than 1,920 in-flight prefills. Ours (today): the same aiperf `--concurrency C`, but with `--no-fixed-schedule
--ignore-trace-delays` — **C always-busy request streams**, every stream fires its next
request the instant the previous one returns. Our c48 is 48 saturating streams; their
c480 is 480 mostly-idle agents. The two axes are not comparable one-to-one; ours is the
harsher load per unit of "concurrency".

Three further reasons their high end is usable at all:
1. **DSpark speculative decoding (K=6)** keeps per-user interactivity acceptable at large
   decode batch; we run MTP off for router parity.
2. **HiCache DRAM tiering** lets thousands of 142k-token sessions' KV live outside HBM;
   dsv4's KV footprint needs it. (N3U does *not*: ~0.85 GB per session → a 4-GPU decode
   worker holds ~900 sessions in HBM after weights; 1,920 sessions fit in our 12-worker
   decode tier without tiering — a real N3U advantage.)
3. **Session affinity + agg** keeps every turn's KV on the worker that has it with no
   hand-off; our KV router achieves the same placement (89% cached-token share measured)
   but disagg still pays the prefill→decode hop.

## 4. Our points on their axis: P90 interactivity

P90 interactivity = **1 / ITL p90** — the tok/s/user that 90% of users exceed (aiperf
"Output Token Throughput Per User" p10 agrees to within rounding). E2E-normalized =
output tokens / end-to-end latency (median), InferenceX's second definition.

| arm | point | tok/s | ITL p50/p90/p99 ms | interactivity p50 | **P90 interactivity** | E2E-norm. p50 | TTFT p50 / p90 s |
|---|---|---|---|---|---|---|---|
| disagg 6:12 KV | c48 | 4,684 | 11.5 / 30.7 / 59.0 | 86.8 | **32.5** | 55.3 | 1.0 / 6.9 |
| | c96 | 4,807 | 11.8 / 30.3 / 58.4 | 84.8 | **33.0** | 21.6 | 9.6 / 26.7 |
| | c120 (peak) | 4,878 | 11.8 / 30.5 / 56.4 | 84.6 | **32.7** | 17.6 | 13.9 / 32.3 |
| | c144 | 4,562 | 11.6 / 29.7 / 56.3 | 86.1 | 33.7 | 13.1 | 21.5 / 41.9 |
| | c384 | 3,082 | 9.8 / 24.5 / 46.4 | 102.5 | 40.9 | 3.1 | 108 / 146 |
| | c512 | 3,495 | 10.0 / 24.7 / 48.0 | 100.4 | 40.5 | 2.8 | 120 / 176 |
| disagg 6:12 RR | c48 | 2,286 | 9.1 / 22.6 / 43.4 | 110.1 | 44.3 | 25.6 | 9.0 / 33.7 |
| | c96 | 2,460 | 9.5 / 22.8 / 44.0 | 105.6 | 43.9 | 13.3 | 23.9 / 68.0 |
| | c144 | 2,078 | 8.7 / 23.8 / 53.5 | 115.6 | 42.0 | 7.1 | 49.9 / 109 |
| disagg 9:9 KV | c48 | 4,555 | 12.9 / 33.5 / 64.1 | 77.8 | 29.9 | 59.6 | 0.56 / 3.1 |
| agg 24 KV (old stack) | c16 | 1,246 | 13.5 / 42.6 / 93.5 | 73.9 | **23.5** | 54.2 | 0.56 / 2.6 |
| | c32 (bounded peak) | 1,657 | 18.1 / 51.6 / 115 | 55.4 | **19.4** | 43.3 | 0.62 / 2.7 |
| | c64 | 2,045 | 25.9 / 72.8 / 142 | 38.6 | 13.7 | 26.6 | 1.1 / 12.6 |
| | c128 | 1,922 | 31.5 / 87.3 / 181 | 31.8 | 11.5 | 7.1 | 28.8 / 88.1 |
| agg 24 RR (old stack) | c32 | 993 | 19.3 / 85.4 / 238 | 51.8 | 11.7 | 33.2 | 1.4 / 11.7 |
| | c64 | 1,141 | 35.3 / 121 / 316 | 28.3 | 8.2 | 14.8 | 5.1 / 22.2 |

### What the interactivity axis adds to the agg-vs-disagg verdict
- **Per-GPU throughput is parity (67.7 vs 69.0), but P90 interactivity is not: disagg
  ~33 tok/s/user vs agg 19–24.** Aggregated workers interleave 16k chunked prefills with
  decode steps on the same GPU, so decode stalls (ITL p90 52–87 ms); the disagg decode
  tier never runs a prefill (ITL p90 ~30 ms, flat across the whole ladder). On
  InferenceX's throughput-vs-interactivity frontier **disagg dominates agg for N3U**:
  same tok/s/GPU at 1.7–2× the interactivity. The throughput-only verdict in D72 §2
  stands; this is the second axis.
- **Disagg's interactivity is load-invariant** (32.5 → 33.7 from c48 to c144, and *rises*
  to ~40 at c384–512 because fewer requests reach decode at once). What degrades with
  load is TTFT / E2E-normalized interactivity (55 → 3 tok/s/user) — the prefill queue.
  On the InferenceX ITL-based axis our post-knee points would look *better*; on the
  E2E-normalized axis they collapse. Report both, as they do.
- **RR has higher raw interactivity than KV** (44 vs 33) for the same reason — fewer of
  its requests get through prefill — while its E2E-normalized figure is half KV's.
  Interactivity alone rewards a starved decode tier; it must be read with TTFT.
- At a P90-interactivity floor of **≥ 30 tok/s/user**: disagg KV qualifies at every point
  (best 4,878 tok/s @c120); agg KV never does (23.5 at c16). At **≥ 20**: agg KV c16
  (1,246 tok/s, 51.9/GPU) vs disagg KV c120 (4,878, 67.7/GPU) → disagg 1.3× per GPU.

## 4b. Throughput per chip (total tokens), P90 interactivity, tokens per dollar

Throughput per chip is **total tokens (input + output) served per second per GPU**, from aiperf's `Total Token Throughput` — the InferenceX convention. Total ≈ 140–150× output for this trace (ISL/OSL ≈ 150), and with ~89% cached-token share it counts prefix-served input tokens, not recomputed ones. Output tok/s/chip is kept for continuity with the rest of the report. P90 interactivity = 1/ITL p90. Tokens/GPU-hour is exact; **tokens/$ uses an ASSUMED $10/GPU-hour** (no GB300 rate is pinned; rescales linearly).

| arm | point | **total tok/s/chip** (in+out) | output tok/s/chip | P90 interactivity (tok/s/user) | total tokens / GPU-hour | tokens / $1 @ $10/GPU-h |
|---|---|---|---|---|---|---|
| disagg 6:12 KV | c48 | **9,914** | 65.1 | 32.5 | 35,689,850 | 3,568,985 |
| disagg 6:12 KV | c96 | **9,359** | 66.8 | 33.0 | 33,692,150 | 3,369,215 |
| disagg 6:12 KV | c120 (output peak) | **9,338** | 67.8 | 32.7 | 33,615,000 | 3,361,500 |
| disagg 6:12 KV | c144 | **8,379** | 63.4 | 33.7 | 30,163,100 | 3,016,310 |
| disagg 6:12 KV | c384 | **4,602** | 42.8 | 40.9 | 16,566,600 | 1,656,660 |
| disagg 6:12 KV | c512 | **5,075** | 48.5 | 40.5 | 18,269,450 | 1,826,945 |
| disagg 6:12 RR | c48 | **5,001** | 31.8 | 44.3 | 18,001,950 | 1,800,195 |
| disagg 6:12 RR | c96 | **4,301** | 34.2 | 43.9 | 15,484,800 | 1,548,480 |
| disagg 6:12 RR | c144 | **3,214** | 28.9 | 42.0 | 11,570,750 | 1,157,075 |
| disagg 9:9 KV | c48 | **9,571** | 63.3 | 29.9 | 34,454,900 | 3,445,490 |
| agg 24 KV (old stack) | c16 | **9,026** | 51.9 | 23.5 | 32,493,900 | 3,249,390 |
| agg 24 KV (old stack) | c32 (bounded peak) | **11,800** | 69.0 | 19.4 | 42,478,650 | 4,247,865 |
| agg 24 KV (old stack) | c64 (post-knee) | **13,194** | 85.2 | 13.7 | 47,498,100 | 4,749,810 |
| agg 24 KV (old stack) | c128 | **9,849** | 80.1 | 11.5 | 35,457,300 | 3,545,730 |
| agg 24 RR (old stack) | c32 | **6,925** | 41.4 | 11.7 | 24,929,550 | 2,492,955 |
| agg 24 RR (old stack) | c64 | **7,109** | 47.5 | 8.2 | 25,593,300 | 2,559,330 |

Reading: on **total** tokens per chip the bounded points are *not* parity — agg KV c32 (11,800) leads disagg KV c48 (9,914) by **1.19×**, because agg's per-chip request rate is ~10% higher and its completed mix carries longer inputs; on output tokens they are parity (69.0 vs 65.1). Disagg still delivers **~1.7× the P90 interactivity** (33 vs 19 tok/s/user). The cheapest total tokens overall are agg KV c64 (13,194/chip, post-knee, 13.7 tok/s/user, TTFT p90 12.6 s); the cheapest that clear a 30 tok/s/user P90 floor are disagg KV c48 (9,914/chip ≈ 35.7M tokens/GPU-hour). RR is ~half the tokens per dollar of KV in both arms.

## 5. Can we run high concurrency for KV + N3U the way they do?

**We already can, and did: disagg KV to c512.** Throughput 3.1–3.5k tok/s (43–49/GPU),
ITL fine (P90 interactivity 40), TTFT p50 108–120 s, 0 request errors — prefill-queue-bound,
post-knee, not a useful operating point under our load semantics. Two validity caveats on
those points: (a) at C > 393 aiperf has more streams than sessions, so sessions are
replayed concurrently and share prefixes across streams (inflated cache hits) — AgentX's
per-lane cache-bust markers prevent exactly this; (b) RR fails hand-offs (transfer
timeouts) from c144 (0.8%) and at c288 (1.4%), so RR high-conc points are invalid.

**To run high concurrency *meaningfully* (AgentX semantics) we would change the client, not
the server:**
1. **Honor the trace's think-time**: drop `--ignore-trace-delays` / use `--fixed-schedule`
   so a "client" is a live session tree that idles between turns. Then 480–1,920 clients
   maps to a realistic in-flight load and the comparison to InferenceX is like-for-like.
2. **Per-lane cache-bust** (AgentX first-turn markers) so cross-stream prefix sharing
   cannot inflate hit rate above the trace's intrinsic within-session reuse.
3. **MTP on** as a separate arm (router parity was the reason it is off) — it is the
   lever that moves interactivity at high batch, as DSpark does for dsv4.
4. Keep the server as is: N3U's KV fits in HBM at 1,920 sessions (no HiCache needed), and
   the prefill tier — not memory — is the binding constraint, so `--max-running-requests`
   on prefill (8) is the knob to raise if the client load is made realistic.
Proposed arm: agg-KV and disagg-KV, AgentX client semantics, c480/960/1440/1920, MTP on,
1 h per point; report throughput-vs-P90-interactivity and E2E-normalized interactivity.
Cost ≈ 8 points × ~1.3 h ≈ 10 h on 72 GPU + 6 nodes.


## 5b. Why our disagg throughput *falls* with concurrency while AgentX (dsv4) *rises*

### What we measured (6:12 KV, instance 2)

| conc | output tok/s | total tok/s served | req/s completed | mean ISL of completed | TTFT p50 | knee |
|---|---|---|---|---|---|---|
| 48 | 4,684 | 713,797 | 7.9 | 89.5k | 1.0 s | AT/PRE |
| 120 | 4,878 | 672,300 | 8.4 | 79.4k | 13.9 s | AT/PRE (peak) |
| 144 | 4,562 | 603,262 | 7.9 | 75.8k | 21.5 s | POST |
| 384 | 3,082 | 331,332 | 5.5 | 59.9k | 108 s | POST |
| 512 | 3,495 | 365,389 | 6.3 | 57.4k | 120 s | POST |

From the c120 peak to c384 the fleet completes **35% fewer requests per second, of 25% shorter
input**, and serves **51% fewer tokens per second** — a genuine loss of prefill-tier efficiency,
not just a plateau. (c384 < c512 is saturation variance; both are ~50% of peak.)

### Three reasons ours falls — two measured, one hypothesis to verify

1. **Our load model has no throttle (measured).** aiperf `--concurrency C --no-fixed-schedule
   --ignore-trace-delays` = C always-busy streams; each re-issues the instant its previous
   request returns. Offered load ∝ C with no idle time, so once the prefill tier is at capacity
   (c120–144) every extra stream only lengthens the queue: TTFT p50 13.9 s → 108 s. A plateau
   is the best case under this model.
2. **The binding resource is a fixed 24-GPU prefill tier (measured, profiled at kv:144).** 92%
   busy with ~10 requests queued per worker while 48 decode GPUs ran at ~15% slot occupancy.
   Extra concurrency cannot recruit the idle decode capacity. The 9:9 result is the proof: 3
   more prefill workers → +36% at c96, still queue-stationary.
3. **Under a deep queue the prefill tier does more work per token served (hypothesis).** The
   two measured facts above explain a plateau; the *drop* needs the per-request prefill cost
   to rise. N3U keeps two caches for prefix reuse: attention KV in the radix tree (6 KB/token,
   ample capacity — 89% cached-token share measured at c144) and **Mamba SSM-state
   checkpoints** at reuse boundaries (~200 MB each, a far smaller pool). With hundreds of
   sessions interleaved, the SSM-state pool is the first to evict; a request whose attention
   prefix is still cached but whose Mamba checkpoint is gone must recompute the prefix to
   rebuild the state — a full-length prefill counted as a "hit" on the KV side. This is the
   hybrid-cache granularity risk flagged in DESIGN.md. **Verification** (queued for the next
   6:12 fleet): live capture at c384 — prefill `#new-token`/step and cached-token share, plus
   the SSM-state cache metrics; the prediction is cached share stays high while new tokens per
   step rise sharply.

   Secondary contributors: the completed mix shifts to shorter requests (79k → 57k mean ISL),
   whose fixed per-request costs (bootstrap, ~200 MB Mamba-state hand-off, scheduling) are
   amortized over fewer tokens; and at C > 393 aiperf replays sessions on more than one stream
   (no per-lane cache-bust), which distorts reuse either way.

### Why AgentX's throughput keeps rising to 1,920 (from their methodology and recipe)

1. **Concurrency counts agent clients, not busy streams.** "Live session trees, not in-flight
   HTTP requests"; a closed-loop agent client idles between turns (tool execution, subagent
   fan-out), so in-flight prefills grow sub-linearly with client count. 1,920 clients is a
   modest server load that has not reached saturation — the analogue of our c12→c96 region,
   where our throughput also rises monotonically.
2. **DSpark speculative decoding (K=6, 7 draft tokens).** Verified tokens per decode step grow
   with batch size until verification saturates, so larger batches → more tok/s. Our decode
   runs MTP off (router parity) at ~10 running requests per worker, starved by prefill, so batch
   never grows.
3. **Aggregated + session affinity: no fixed prefill tier.** Each TP4 worker does its own
   prefill and decode; more concurrency recruits the whole fleet, and there is no hand-off
   whose queue can dominate. Our disagg is capped by the 24-GPU prefill tier regardless of
   how idle the 48 decode GPUs are.
4. **HiCache DRAM tiering.** dsv4's KV footprint for thousands of 142k-token sessions lives in
   host DRAM and is paged back in, so reuse survives high concurrency; our N3U KV fits in HBM,
   but the Mamba-state pool (hypothesis 3) is the analogous limit we do not tier.
5. **One-hour windows** halve the fixed-window edge effect that our 1,800 s window suffers
   when request latencies reach 2–3 minutes.

**Bottom line:** the two curves are not measuring the same thing. Ours is a saturation curve of
busy streams against a fixed prefill tier; theirs is a client-count sweep of an aggregated,
spec-decoding, cache-tiered server that has not yet saturated. To make our high-concurrency
points meaningful in their sense, change the *client* (honor trace think-time, per-lane
cache-bust) and, for a topology that keeps rising, move prefill capacity with the load (9:9
already shows the direction) — proposals in §5.


## 5c. Aligning our concurrency with AgentX — it is a flag change, not a tool change

**Same tool.** InferenceX drives AgentX with aiperf, exactly as we do: `aiperf profile --scenario
inferencex-agentx-mvp … --concurrency $CONC --benchmark-duration $DURATION` (their
`benchmark_lib.sh` `build_replay_cmd`, lines ~3073–3110). The scenario is built into aiperf and
is present at **v0.12.0 — the version our bench template already pins**; the corpus loader they
use for the 256k variant, `semianalysis_cc_traces_weka_062126_256k`, is the one we use.

**What the scenario changes (verbatim semantics from aiperf's `docs/tutorials/agentx-mvp.md`):**
- *"Exactly `--concurrency` trees stay live at all times."* A slot holds one **session tree** (root
  conversation + every subagent stream it spawns) and recycles only when the whole tree drains.
  In-flight requests can exceed the concurrency at subagent fan-out and fall below it while a
  tree waits.
- *"Each turn's replay delay is the recorded idle gap from the previous response's end to the
  next request's start."* Think-time is **kept**; a **10-second whole-system idle cap** shifts all
  pending timers only when nothing is active or ready. Per-trajectory caps are forbidden.
- `--cache-bust first_turn_prefix`: a unique per-conversation marker on the first user turn of
  every play, shared by that play's warmup and profile turns — reuse survives *within* a session,
  never across plays.
- Warmup lanes start at a sampled point of each trace (InferenceX: 25–75%) with full prefix
  history attached; `--warmup-requests-per-lane 10`; profile ≥ 900 s (InferenceX: 3,600 s).
- `--use-server-token-count`, `--extra-inputs ignore_eos:true`, `--random-seed 42`,
  `--failed-request-threshold` as the validity gate.

**Our invocation today** (`manifests/perf/sgl-d72-flagsweep.yaml`): `--concurrency C
--no-fixed-schedule --ignore-trace-delays --num-dataset-entries 393 --concurrency-ramp-duration 60`
after a 900 s cache-warmup at concurrency 96 — C always-busy streams, no think-time, no per-play
cache-bust, sessions shared across streams above C = 393.

**The diff** (new template `manifests/perf/sgl-d72-agentx.yaml`, everything else identical):

| remove | add |
|---|---|
| `--no-fixed-schedule --ignore-trace-delays` | `--scenario inferencex-agentx-mvp` (keeps end-to-start delays) |
| `--concurrency-ramp-duration 60` | `--system-idle-gap-cap-seconds 10` |
| (implicit shared prefixes across streams) | `--cache-bust first_turn_prefix` |
| our 900 s warm-up at conc 96 | `--warmup-requests-per-lane 10 --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75` |
| — | `--use-server-token-count`; duration 3,600 s to match |

**Status (2026-09-15): adopted.** AgentX-mode runners are queued: 9:9 KV on np-3 after the MTP
runs, agg KV on np-1 after the new-stack sweep — ladder 48/96/192/384/768/1536 clients, 3,600 s
per point, first point as smoke (`scripts/agentx_runner.sh`, template `manifests/perf/sgl-d72-agentx.yaml`).
Results will be reported on the client axis alongside the busy-stream results, never mixed.

**How to read the new axis.** Concurrency becomes *live agent clients*; the server load is whatever
those clients generate. Our current knee at 48–120 busy streams will map to a much larger client
count, so the ladder should be re-bracketed upward — proposed **48 / 96 / 192 / 384 / 768 / 1536
clients** on the 9:9 KV fleet (and agg KV for the same-axis comparison), 1 h per point, knee by
the same queue-drain rule, and the InferenceX-style report of throughput/chip, TTFT and P90
interactivity. Cost ≈ 6 points × 2 arms × ~1.3 h ≈ 16 h. Two things the change does **not**
alter: the server (same fleet, flags, transport gate) and the trace corpus — only the client's
load model, which is what makes the two studies' "concurrency" commensurable.


## 5d. Simulated curve under the AgentX concurrency definition

`scripts/dynosim_agentx.py` re-runs the same engine model with the AgentX load model — C live
session lanes (one session per lane, recycled when it drains), warm-up start at a random
25–75 % of each session with the prefix primed, **recorded think-time replayed** (start cadence
anchored per lane, never before the previous response returns), a 10 s whole-system idle cap,
per-play cache-bust, 1 h profile window. Output tok/s per GPU and TTFT p50 (s):

| clients | 6:12 KV tok/s/GPU · p50 | 9:9 KV tok/s/GPU · p50 | agg KV tok/s/GPU · p50 | 6:12 RR · p50 | 9:9 RR · p50 |
|---|---|---|---|---|---|
| 48 | 12.0 · 0.13 s | 12.0 · 0.13 s | 23.6 · 0.13 s | 11.8 · 1.24 s | 11.8 · 1.84 s |
| 96 | 23.6 · 0.18 s | 23.6 · 0.16 s | 37.1 · 0.14 s | 21.9 · 3.95 s | 22.6 · 3.16 s |
| 192 | 46.0 · 0.76 s | 46.5 · 0.26 s | 57.8 · 0.16 s | 27.1 · 26.86 s | 34.4 · 12.70 s |
| 384 | 68.4 · 13.03 s | 83.3 · 1.83 s | 64.3 · 0.22 s | 26.2 · 108.29 s | 34.0 · 63.29 s |
| 768 | 60.9 · 57.13 s | 88.6 · 20.16 s | 58.4 · 0.38 s | 23.4 · 212.09 s | 30.6 · 142.59 s |
| 1536 | 45.8 · 149.21 s | 70.6 · 59.58 s | 43.8 · 0.98 s | 11.8 · 474.39 s | 22.0 · 327.38 s |

**First measured AgentX-mode point (2026-09-16 02:18 UTC, 9:9 KV, 48 clients, 1 h profile window):**
786 output tok/s = **10.9/GPU**, TTFT p50 0.32 s / p90 0.9 s / p95 1.5 s / p99 3.8 s, ITL p50 6.4 ms /
p90 6.8 ms (P90 interactivity 147 tok/s/user), 2,887 requests, 0 errors, queue-stationary, MNNVL guard
PASS (TE peak 0.50 GB/s). The sim predicted 12.0/GPU · p50 0.13 s → **sim/real 1.10× on throughput**,
TTFT under-predicted 2.5× (the same prefill-queue term as the busy-stream sim, at much lower load).
Confirms the client-vs-stream ratio: 48 clients drew 4.5× less output than 48 busy streams
(4,555 tok/s). **96 clients (05:55 UTC): 2,019 tok/s = 28.0/GPU**, TTFT p50 0.31 s / p95 1.42 s / p99 4.3 s,
ITL p50 7.4 / p90 8.4 ms, 7,538 requests, stationary, guard PASS — sim said 23.6/GPU (sim/real 0.84×; the
sim is now *under*-predicting as load rises, the opposite sign from the busy-stream sim). Throughput
scaled 2.57× for 2× clients, so the fleet is still far from its knee on this axis. Next: 192 (running), 384, 768, 1536.

Bench-pod placement note (2026-09-16): the 192-client point was evicted mid-warm-up — aiperf itself
used 59 GB (192 session lanes × 256K-token contexts plus the record processors) on a 60 GB x86 system
node. From 192 clients up, the AgentX bench pods run on np-1's arm64 GPU nodes (953 GB; the template
already builds aiperf's aarch64 dependency from source) with a 96 GiB request / 512 GiB limit and an
anti-affinity preference away from our worker pods (`manifests/perf/sgl-d72-agentx.yaml`). This is
the same reason InferenceX's AgentX recipes run aiperf inside the SLURM allocation with `mem: "0"`.

Page: [AgentX-concurrency curve](https://htmlpreview.github.io/?https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/reports/n3u-agentx-curve.html)
(measured AgentX-mode points are overlaid as the client-axis ladders land).

What the client axis changes:
- **A client is ~¼ of a busy stream.** 6:12 KV reaches 46/GPU at 192 clients (≈ our measured 65/GPU
  at 48 busy streams) and peaks at **68.4/GPU at 384 clients** — the same ceiling as the busy-stream
  peak (67.8/GPU at c120), reached at ~3× the nominal concurrency. That ratio is the think-time
  duty cycle of the trace and is exactly why AgentX sweeps 480–1,920.
- **9:9 leads again and holds up under overload**: 88.6/GPU at 768 clients and still 70.6 at 1,536,
  where 6:12 has fallen to 45.8. (The sim's split ranking under AgentX load agrees with silicon
  here; its busy-stream ranking did not — the client model's smoother arrivals keep the prefill
  tier from saturating as early.)
- **RR knees an octave earlier on the client axis too** (6:12 RR: 27/GPU at 192 clients with a
  27 s p50) — recompute, not placement, still sets its knee.
- **Agg in this sim is throughput-capped by its TPOT batch model** (KV 64/GPU at 384 clients with
  112 ms TPOT); the measured new-stack agg ladder will replace it. The sim also has agg RR ahead
  of agg KV at ≤192 clients — an artifact of the steep agg TPOT slope penalising KV's placement
  concentration; not a prediction we carry.
- **Post-knee decline is milder than under busy streams** (6:12 KV: −33% from 384→1,536 clients vs
  −49% from c120→c384 streams): idle clients throttle themselves, so the overload is bounded.
Caveat: the corpus records request *starts*, so end-to-start delays are approximated by the start
cadence minus simulated latency; our bench4k excerpt has 874 linear sessions (no subagent
fan-out), so in-flight ≤ clients.

## 5e. Code trace — how `conc` is declared and consumed in the SemiAnalysis agentic benchmark

Traced on InferenceX `2f4201c` (2026-09-15), srt-slurm submodule `984180e5`, aiperf fork submodule `754356e9`. Every link is to the file at that commit.

| step | file | what happens to the number |
|---|---|---|
| 1. declare | [`configs/nvidia-master.yaml`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/configs/nvidia-master.yaml#L6676) | each `search-space` item under `scenarios.agentic-coding` carries `conc-list: [480]` (one value per topology; 960 / 1440 / 1920 are separate items with their own recipe). Validation schema: `infx/matrix/validation.py` field `conc_list` (alias `conc-list`, must be > 0, exclusive with `conc-start`/`conc-end`). |
| 2. expand | [`infx/matrix/generate.py`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/infx/matrix/generate.py) | `chunk_multinode_agentic_concurrencies()` splits the list into chunks of `MAX_MULTINODE_AGENTIC_CONCURRENCIES_PER_ALLOCATION = 1`, so every agentic concurrency becomes its own matrix entry and its own SLURM allocation. The entry's `exp-name` embeds the conc. `trim_conc()` (PR-time mode) keeps only the lowest conc per deployment shape. |
| 3. dispatch | [`.github/workflows/benchmark-multinode-tmpl.yml`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/.github/workflows/benchmark-multinode-tmpl.yml#L130) | workflow inputs `conc-list` and `conc` become env `CONC_LIST` (space-joined) and `CONC` (the first value). Results are later expected as one `${RESULT_FILENAME}_conc<N>.json` per value in `CONC_LIST`. |
| 4. launcher | [`runners/launch_gb300-nv.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/runners/launch_gb300-nv.sh) | does not pass conc to the server; it only reads `CONC_LIST` for the optional power lane (`inject_srt_power_concurrencies.py`) and eval conc. `srtctl apply` inherits the process environment, so `CONC_LIST` / `CONC` / `DURATION` ride into the sbatch job. The recipe's `benchmark.type: custom` runs `benchmarks/multi_node/agentic_srt.sh`. |
| 5. loop | [`benchmarks/multi_node/agentic_srt.sh`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/multi_node/agentic_srt.sh#L43) | `read -r -a CONCURRENCIES <<< "${CONC_LIST:-$CONC}"`; validates each is a positive integer; for each value exports `CONC=<value>`, builds the aiperf command, runs it, then `wait_for_agentic_servers_idle` (three consecutive idle polls of the workers' running/queued gauges) before the next value. |
| 6. aiperf flag | [`benchmarks/benchmark_lib.sh::build_replay_cmd`](https://github.com/SemiAnalysisAI/InferenceX/blob/2f4201cfa05583b80160dc2dccb9a0310837a1ef/benchmarks/benchmark_lib.sh) | `aiperf profile --scenario inferencex-agentx-mvp --concurrency $CONC --benchmark-duration $DURATION --trajectory-start-min-ratio 0.25 --trajectory-start-max-ratio 0.75 --warmup-requests-per-lane 10 --trace-idle-gap-cap-seconds … --warmup-grace-period 1800 --use-server-token-count --num-dataset-entries 393 --random-seed 42`. |
| 7. scenario lock | [`src/aiperf/common/scenario/inferencex_agentx_mvp.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/common/scenario/inferencex_agentx_mvp.py) | `--scenario inferencex-agentx-mvp` sets `timing_mode=AGENTIC_REPLAY`, forces streaming + ignore-eos, forbids `--ignore-trace-delays` and input truncation, requires a weka trace loader, min duration 900 s / default 1800 s, default trajectory start 0–100 %, `system_idle_gap_cap_seconds=10`, cache-bust `first_turn_prefix`, and ≥ 95 % metric coverage. |
| 8. sizing | [`src/aiperf/timing/config.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/config.py#L108) → [`timing/phase_orchestrator.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/phase_orchestrator.py#L145) | `TimingConfig.concurrency` ("required by AGENTIC_REPLAY to size the trajectory list") is copied from the profiling phase's concurrency. The orchestrator raises if it is unset and constructs `TrajectorySource(concurrency=N, …)`. |
| 9. lanes | [`src/aiperf/timing/trajectory_source.py::_build_trajectories`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/trajectory_source.py#L451) | **one trajectory per concurrency lane**: draws N traces from the dataset sampler (wrapping round-robin when N > the 393-trace pool, which `validate_dataset_wrap_policy` allows only because cache-bust is active), and snapshots each at its own wall-clock instant t* inside the start-ratio window. Turns before t* are warm-up history; turns at/after t* are profiled. |
| 10. slot rule | [`src/aiperf/timing/session_tree.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/session_tree.py) + [`credit/issuer.py::acquire_lane_credit`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/credit/issuer.py#L184) | "`--concurrency N` means N such trees live at any instant." A tree = root session + every subagent/sidecar it spawns. Exactly one session slot per live tree, acquired via `ConcurrencyManager.acquire_session_slot`, released once when the tree drains (root sent its terminal turn AND every descendant completed). |
| 11. steady state | [`src/aiperf/timing/strategies/agentic_replay.py`](https://github.com/SemiAnalysisAI/aiperf/blob/754356e9a39acc6cc6afb242d123bb57c3fb6f75/src/aiperf/timing/strategies/agentic_replay.py) | WARMUP replays each lane's last pre-t* turn to prime the cache; PROFILING dispatches all N lanes at once, then each turn waits its recorded inter-turn `delay_ms` (capped by the 10 s idle gap). When a tree drains, `_dispatch_recycled_on_lane` draws the next root from the sampler so occupancy stays "at exactly the configured concurrency". |

What this means for the number itself: in AgentX, `conc = 480` is **480 live Claude-Code session trees**, each mostly idle between turns (the think-time is replayed), never 480 requests in flight. Our `--concurrency C` with `--no-fixed-schedule --ignore-trace-delays` is C always-busy request streams — the scenario lock forbids exactly that flag combination. Both go through the same aiperf `--concurrency` option; the difference is the timing mode the scenario selects (step 7) and the slot rule (step 10).

## 5f. MTP on our side — measured (the analog of their DSpark / synthetic acceptance)

Their AgentX recipes run DeepSeek-V4 with DSpark speculative decoding at a *synthetic* acceptance
length of 3.77. We ran Nemotron-3's native NEXTN draft (1 step, 2 draft tokens) on the 9:9 disagg
split with real verification — three points, all clean (0 restarts, transport guard PASS). Result:
per-stream decode got 1.4–2.0× faster (ITL p50 8.6–9.1 ms; P90 interactivity 43–48 tok/s/user vs
21–30), but output throughput fell 44–67% (2,542 / 2,581 / 2,473 tok/s at c48 / c96 / c144 vs
4,555 / 6,560 / 7,549) and every point went post-knee because the prefill tier saturated
(effective prefill throughput 0.25–0.6× of MTP-off). On this trace (ISL ≈ 87 k, 89% cached) MTP
accelerates the tier that had slack. Full table and mechanism: D72_RESULTS.md §4b.

## 6. Caveats on the comparison
- Different model class (dsv4 attention-heavy MoE vs N3U hybrid), different stack pins,
  spec-decode on vs off, HiCache on vs off, session affinity vs KV-router placement — the
  numbers are not comparable; the *methodologies* are what this note compares.
- InferenceX's absolute GB300 dsv4 AgentX numbers live in the run artifacts and a
  JS-rendered dashboard; not retrievable here without auth. The analysis uses their
  published methodology and recipe.
