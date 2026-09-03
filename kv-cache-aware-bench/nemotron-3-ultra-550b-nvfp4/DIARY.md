# N3U Study Diary — every step and result, newest last

Running log of the Nemotron-3-Ultra 550B KV-aware-routing study. One entry per
action or result, timestamped UTC. Companion to `DESIGN.md` (the plan); this
file records what actually happened.

## 2026-08-19

- **~07:00 — Stage-0 support checks run** (aiconf pod, system pool).
  - Model config obtained via ungated unsloth mirror (nvidia repo names 401'd —
    later found to be wrong-name artifacts). Parsed: NemotronHForCausalLM, 108
    layers = 48 Mamba-2 + 48 LatentMoE + 12 attention ('*'), n_kv=2×128, 512
    experts top-22, latent 2048, context 262144, MTP.
  - **aiconfigurator 0.10.0: NemotronH UNSUPPORTED** (2/6 architecture vote,
    agg+disagg). **0.11.0: SUPPORTED** (5/6 both), sglang DB **0.5.14** — same
    version as our proven Kimi silicon pair. Pin 0.11.0.
- **~07:10 — Derived architecture constants**: KV 6 KB/token (6× smaller than
  Kimi), Mamba SSM state ~200 MB/request, disagg transfer ~0.8 GB @96k ISL (4×
  lighter than Kimi), per-TP4-worker KV pool >100M tokens (trace's 119M unique
  tokens ≈ cacheable per worker). Study hypothesis recorded in DESIGN.md §0.
- **~07:30 — DESIGN.md written, committed** (`bda4805`): full walkthrough —
  stage-0 gates, AIC solve commands, DynoSim physics changes (linear prefill,
  no-eviction cache with Mamba checkpoint boundaries, transfer-cost term),
  sweep grid, silicon plan with all Kimi lessons pre-applied, agg-first
  sequencing (disagg gated on the ~1.5 req/s ceiling root-cause).
- **Weights source resolved**: user proposed unsloth mirror → investigated:
  unsloth = **BF16, 225 shards, 1.12 TB** (unusable for serving; config source
  only). HF search then revealed canonical
  **nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 is UNGATED**: 113 shards,
  **352 GB**, FP8 mamba mixers / FP4 MoE. No HF token needed. (`e2df633`)
- **Weight staging launched**: job `alisachen-n3u-stage-weights` (dynamo-cloud,
  system pool, gcsfuse) → `/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4`.
  Unauthenticated HF rate limits pace it.
- **Workstream dir created** `kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/`
  (renamed from nemotron3u, history preserved): README (rationale + layout),
  scaffolds sim-results/ manifests/ scripts/ results/silicon/ reports/. (`41d1028`)
- **~07:49 — AIC SILICON solves launched** (job 2 of the one-by-one kickoff;
  runs in parallel with staging per standing sim-concurrency policy): 4 brackets
  sequential in aiconf pod — warm24 (prefix 92k) → cold24 → warm72 → cold72,
  all `--isl 96000 --osl 900 --max-seq-len 262144 --enable-chunked-prefill
  --database-mode SILICON`, outputs `/tmp/n3u-solves/<bracket>/`. Watcher armed.
- Staging progress at solve launch: **79 GB / 352 GB**.

- **~07:55 — Cluster access re-verified on user request**: authenticated as the
  user's account via pinned kubeconfig, all nodepools listable, `dynamo-cloud`
  writable (can-i create jobs: yes); staging job + aiconf pod actively running
  on it. Node health snapshot: np-1 17/18, **np-2 16/17 (a np-2 node is now
  also down — new since yesterday's survey)**, np-3 17/18 (dkwr still dead),
  np-4 18/18. Cluster-owner escalation list grows to four node-level faults.

- **08:0x — AIC solve round 1: all 4 brackets crashed at the finish line** —
  the *experiments succeeded* (warm24: agg 535 results + disagg 76 in ~2 min;
  fast because NemotronH's search space is smaller than Kimi's) but AIC's final
  summary calls `plotext.plot_size` (a plotext-4.x API) and pip had installed
  plotext 5.x (renamed `plotsize`) → AttributeError before results were saved.
  **Fix**: pin `plotext==4.2.0` alongside aiconfigurator 0.11.0 (toolchain-pin
  lesson #2 for this model). Round 2 relaunched, watcher armed.

- **08:00–08:07 — solve round 2 failed the same way**: plotext 4.2.0 lacks
  `theme` (5.x API). Root cause found: AIC declares `plotext>=5.3.2`; pip
  resolved 6.0.0 which *removed* `plot_size`. **Correct pin: plotext==5.3.2
  exactly** (has both APIs). Toolchain pins for this model now: aiconfigurator
  0.11.0 + plotext 5.3.2 + sglang 0.5.14 DB.
- **08:09–08:15 — solve round 3: all 4 brackets exit 0**, outputs fetched to
  `sim-results/aic/` (pareto.csv + top-3 deploy configs per bracket). Headline
  (later invalidated, see next): agg beats disagg 1.45–1.94× at both scales;
  warm 63.4 tok/s/GPU @ TTFT 233 ms TPOT 9.5 ms; cold 31.8 @ 25.3 s.
- **08:2x — round-3 results INVALIDATED before use: they model BF16, not
  NVFP4.** Pareto rows show `gemm=bfloat16` throughout — AIC's quant inference
  read our config source, the unsloth **BF16** config (no `quantization_config`
  key) → solver assumed 1.1 TB weights (hence 8-GPU workers). Lesson recorded:
  **the config.json you hand AIC determines the precision it models** — always
  use the target checkpoint's own config. NVFP4 repo's config.json +
  hf_quant_config.json fetched (`quant_algo` present) → **round 4 launched**
  with `--model-path /tmp/n3u-fp4`. BF16 round-3 outputs kept in
  `sim-results/aic/` as a precision-sensitivity reference.

- **~08:40 — WEIGHT STAGING COMPLETE**: all 243 repo files, **113 shards /
  352 GB in 46 min** → `/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4`.
  Job 1 of the campaign done. Silicon smoke gate is now unblocked (pending AIC
  round-4 worker shape).

- **08:31 — AIC NVFP4 round 4: all 4 brackets exit 0** (real nvfp4 kernels
  confirmed in pareto). Warm agg **197.6 tok/s/GPU** @ TTFT 356 ms / TPOT
  9.6 ms; cold 59.5 @ 28.2 s → **reuse worth 3.3× throughput**. Agg beats
  disagg at both scales (1.09× cold, 1.42–1.63× warm). Winning worker TP4-based
  (top-1 = 8-GPU TP4×DP2; plain TP4 on-pareto) → Kimi routing granularity
  (6 workers @24) carries over. Artifacts: `sim-results/aic-nvfp4/`.
- **08:35 — DynoSim n3u constants fitted + committed** (`apply_n3u_constants`,
  `--model n3u`): prefill 19.7k tok/s/TP4-worker (per-GPU ≈ Kimi — LatentMoE
  dominates prefill despite linear attention), TPOT 5.26+0.277·bs **no batch
  cliff to bs≥80** (the Mamba dividend), cache unbounded (100M tok/worker).
- **08:42 — n3u agg-24 DynoSim sweep complete** (`dynosim_n3u_agg_v1.csv`).
  **HYPOTHESIS ANSWERED (sim): the KV win survives the hybrid architecture,
  and it is pure placement.** With zero eviction anywhere, RR converges to 43%
  hit (rotation splits sessions) vs KV 78–84% — same-cell gains **1.18→1.69×**
  rising with conc; KV knees at conc ~64–128 (5,127–5,185 tok/s ≈ 214/GPU,
  p95 2.4 s) vs Kimi's conc-16 peak — the no-cliff decode lets agg run 4–8×
  deeper. RR knee ≈ conc 32 (p50 super-linear from 48). Framing-2 sim
  prediction: **KV 5,127@64 vs RR 2,797@32 → 1.83×, 2× conc**.

- **~09:0x — cluster resurvey on user request**: access OK; np-3 has 10 free
  nodes (new neighbors joeywan-ubench on 7); np-1/2/4 full. **Our 9:8 Kimi
  fleet was deleted by someone else** (sgl-d72-9x9-* deployments gone; other
  fleets at 0 replicas) — manifests regenerate, GCS data safe.
- **~09:1x — N3U SMOKE LAUNCHED** (user directive: smoke now, 24-agg after
  selection): generator parameterized by model (`MODELS` map, per-arm model
  field) → `n3u-smoke.yaml` (1 agg TP4 worker + kv frontend, np-3),
  `n3u-agg-{rr,kv}.yaml` (6×TP4 comparison arms, ready). Smoke worker Running,
  model loading; watcher fails fast on crashloop. Open q's the smoke answers:
  0.5.14 loads NemotronH? `--quantization modelopt_fp4` correct for this
  checkpoint? hybrid radix reuse + kv-events?
- **~09:1x — SELECTION.md (agg) written from the sweep**: silicon ladder
  **conc 16/32/64/128 both arms**; framing-1 cells 16 & 32; framing-2 sim
  prediction KV 5,185@128 vs RR 2,797@32 → **1.85×, 4× conc**. RR knee c32
  (p50 slope super-linear at 48), KV knee c128 (thr rollover). d72 sweep still
  running (unbounded radix = slow sim); its role is quantifying agg-vs-disagg,
  not silicon selection.

- **08:58 — disagg72 sweep complete** (`dynosim_n3u_disagg72_v1.csv`): best
  disagg KV 5,927 tok/s (3:15@c96) = **82 tok/s/GPU vs agg's 216** —
  **DynoSim independently confirms AIC: disagg costs ~2.6× per-GPU on this
  architecture.** RR knees c48–96 on every split. No silicon disagg cell;
  SELECTION.md disagg section finalized as the topology negative-finding.
- **~09:2x — CURVES PUBLISHED** (Kimi-style panel set + 2 cross-model
  frontier panels): `reports/n3u-curves.html`, artifact
  https://claude.ai/code/artifact/7cc21d53-d658-4047-85bc-c403711b568b.
  Simulation + selection stage of the campaign is now COMPLETE; silicon
  next (smoke → 24-agg ladder 16/32/64/128).

- **09:05 — SMOKE PASSES ALL FOUR GATES.** Worker ready in ~14 min (sglang
  0.5.14 loads NemotronH under `--quantization modelopt_fp4`); serve probe
  coherent; **repeated-prompt reuse: `cached_tokens: 6400/6416` (99.75%) — the
  hybrid Mamba radix cache genuinely reuses prefixes**; kv-events flowing
  (router indexer active) and `dynamo_component_router_kv_hit_rate` histogram
  live on the frontend. The routing experiment proceeds as designed.
- **09:2x — 24-AGG COMPARISON LAUNCHED** (`scripts/sweep_n3u_agg.sh`): smoke
  torn down, 6×TP4 fleet deploying on np-3; 8 points kv/rr × conc 16/32/64/128,
  frontend router-swap protocol, per-point 300 s settle + 900 s trace warmup +
  1800 s measure; knee gate halts only where sim predicts bounded (RR 64/128
  are knee-location points, verdict logged). Bench jobs sed-derived from the
  Kimi template with N3U model/tokenizer staging swaps. ETA ~7.5 h.

- **15:38 — 24-AGG SILICON CAMPAIGN COMPLETE: all 8 points, zero errors.**
  (Mid-campaign note: knee_check's run-dir filter was Kimi-named — fixed to
  model-agnostic; artifact roots looked "empty" in plain `ls` due to gcsfuse
  zero-byte dir markers — recursive listing shows all data; summary fetch had
  a double-slash path bug — fixed.)
- **RESULTS** (`AGG24_RESULTS.md`): both-bounded cells at conc 16 AND 32;
  at c32 **KV 1,657 tok/s (69/GPU) vs RR 993 (41.4) = 1.67× with 3.9× better
  p95**, both queues stationary. Both policies knee at ~32–48 → framing 2
  collapses into framing 1 (same best cell). Gains 1.41–1.79× everywhere —
  **larger than sim's placement-only prediction (1.19–1.46×): hypothesis
  confirmed on silicon.** Sim verdicts: ratios + RR knee confirmed; KV knee
  over-predicted (128 vs ~32–48); absolutes 2.5× optimistic (calibration v2).

- **~16:0x — agg flag-grid verification (user q)**: sim's 9-variant grid says
  **defaults tie for best at the bounded cells** (c16/c32 within 0.1%); tuning
  pays only post-knee (scale3.0+credit0.8: +1.4% @c64, +12.6% @c128) — same
  pattern as Kimi. Silicon headline (c32, defaults) stands; live flag probe at
  c32/c64 queued as the one unverified corner (after disagg).
- **~16:1x — N3U 72-GPU DISAGG CAMPAIGN LAUNCHED** (np-3 fully free — all 18
  nodes): 6P+12D TP4 (sim-best bounded split 6:12), single fleet + router-swap,
  ladder kv/rr × conc 24/48/96/144 (anchored low: silicon ~2.5× below sim
  absolutes). RDMA transport gate every point (stop policy — NIXL now carries
  KV + Mamba state); knee gate halts if kv:24/rr:24/kv:48 measure post-knee.
  Dual purpose: measures the sim's agg-beats-disagg negative on silicon AND
  the transfer-ceiling diagnostic (N3U transfers 4× lighter than Kimi — if
  kv:24 saturates like Kimi disagg did, the ceiling isn't transfer-bound).
  First disagg serving of NemotronH (SSM-state transfer over NIXL untested —
  fleet-up + kv:24 double as that smoke). ETA ~8 h.

- **~16:3x — DRIFT ANALYSIS (goal directive): apple-to-apple substitution
  localizes the agg sim-vs-silicon gap (1.9–2.8×) to ONE step — decode.**
  Measured ITL line 8.9 + 1.73 ms/seq vs AIC DB's 5.26 + 0.277 (slope 6× too
  shallow for NemotronH); decode substitution alone brings every cell to
  ±30%. Prefill ≈ correct (stock constants reproduce RR TTFT). Secondary:
  hit-rate drift (engine cached p50 ≈63% vs sim 84% — hybrid-checkpoint
  granularity optimistic in the block-aligned reuse model). Report section
  added to AGG24_RESULTS.md; **n3u-sim v2 decode constants applied** (now
  mildly conservative, 0.80–0.86× on KV — right polarity). Upstream note:
  AIC gb300/sglang DB decode batch-scaling for NemotronH looks
  under-sampled — candidate for an aiconfigurator issue.

- **20:05 (09-02) — d72 first point + designed halt.** kv:24: **2,005 tok/s
  (27.8/GPU), 3.34 req/s, TTFT p50 4.05 s growing (2.6→5.1) → POST-KNEE**,
  gate PASS (first-ever NemotronH disagg serving: NIXL carried KV + Mamba
  state cleanly). Three findings: (1) **agg-beats-disagg confirmed on
  silicon** — 27.8/GPU vs agg's bounded 69/GPU ≈ 2.5× worse, matching sim's
  2.6×; (2) **Kimi-ceiling was transfer-bound** — N3U disagg reaches 3.34
  req/s vs Kimi's ~1.5–1.7 hard ceiling, scaling with the 4× lighter
  transfers → sharpens the GPUDirect escalation; (3) silicon disagg knee <24
  vs sim 144 — decode is idle (ITL 8.7 ms), so the disagg drift lives in
  prefill-tier/transfer (decomposition when ladder completes). Campaign
  resumed with revised ladder: kv/rr:12 expected-bounded anchors,
  24/48/96 knee-location, 144 dropped.

- **00:42 (09-03) — the conc-12 anchor pair delivers the disagg headline:
  at the SAME cell, KV is BOUNDED (stationary, p50 2.67 s, 2.40 req/s, gate
  PASS) while RR DIVERGES (2.2→6.4 s growing) — on 72-GPU disagg silicon the
  router flag alone determines whether the system is bounded.** RR knee <12 =
  no practical bounded RR disagg regime (mirrors Kimi disagg). Gate halted on
  rr:12-expected-bounded as designed; review: boundary evidence banked, not
  chased below 12. Resumed (rev 3) with saturation/knee-location points only:
  rr:24, kv:48, rr:48, kv:96, rr:96 (~4.5 h).

- **04:55 (09-03) — D72 CAMPAIGN COMPLETE: all 8 points (kv/rr × 12/24/48/96),
  zero errors, every RDMA gate PASS.** Results in `D72_RESULTS.md`:
  KV bounded at c12 (1,403 tok/s) vs RR unbounded everywhere; **RR saturates
  at ~1,750 tok/s, KV reaches 3,412 at c96 → routing doubles the saturation
  ceiling (1.96×)**; agg-beats-disagg measured at 3.5×/GPU at bounded points;
  Kimi ceiling confirmed transfer-bound (ceiling scales with transfer volume).
- **Drift decomposition (disagg)**: decode exonerated (ITL ≈ model); **RR fits
  exactly at prefill ×0.5** → host-staged transfer tax ≈ doubles prefill-tier
  cost; **KV residual = unmodeled per-request transfer floor (~2–2.5 s)** —
  cached requests shrink compute but still ship full KV+state; knee drift
  (sim 144 vs real 12) follows from the same terms. n3u-sim v3 items logged.
  **GOAL COMPLETE**: disagg datapoints collected Kimi-pattern; perf in
  reports; drift analyzed apple-to-apple for both agg (decode slope) and
  disagg (transfer floor + tier-rate halving).

## Next planned (in order)

1. AIC solves complete → pull candidate configs + rates → `sim-results/`,
   record worker shape (TP2-vs-TP4 question) and warm/cold ceilings here.
2. DynoSim n3u constants (Stage 2) → policy sweeps agg/disagg24/disagg72
   (Stage 3) → **curves pages as for Kimi** (pareto/knee panel set, committed
   under `reports/`) → `SELECTION.md` with hypothesis predictions.
3. Weights staged → sglang 0.5.14 NemotronH smoke → hybrid radix reuse probe
   (cached_tokens) + kv-events check — the go/no-go gate for the routing
   experiment as designed.
