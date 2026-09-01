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

## Next planned (in order)

1. AIC solves complete → pull candidate configs + rates → `sim-results/`,
   record worker shape (TP2-vs-TP4 question) and warm/cold ceilings here.
2. DynoSim n3u constants (Stage 2) → policy sweeps agg/disagg24/disagg72
   (Stage 3) → **curves pages as for Kimi** (pareto/knee panel set, committed
   under `reports/`) → `SELECTION.md` with hypothesis predictions.
3. Weights staged → sglang 0.5.14 NemotronH smoke → hybrid radix reuse probe
   (cached_tokens) + kv-events check — the go/no-go gate for the routing
   experiment as designed.
