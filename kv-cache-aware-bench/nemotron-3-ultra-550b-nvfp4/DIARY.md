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

## Next planned (in order)

1. AIC solves complete → pull candidate configs + rates → `sim-results/`,
   record worker shape (TP2-vs-TP4 question) and warm/cold ceilings here.
2. DynoSim n3u constants (Stage 2) → policy sweeps agg/disagg24/disagg72
   (Stage 3) → **curves pages as for Kimi** (pareto/knee panel set, committed
   under `reports/`) → `SELECTION.md` with hypothesis predictions.
3. Weights staged → sglang 0.5.14 NemotronH smoke → hybrid radix reuse probe
   (cached_tokens) + kv-events check — the go/no-go gate for the routing
   experiment as designed.
