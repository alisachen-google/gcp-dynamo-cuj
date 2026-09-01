# Nemotron-3-Ultra 550B: KV-Aware Routing Experiment Design (agg + disagg, GB300, sglang)

Experiment design walkthrough for the second model of the routing study, built on
the Kimi-K2.5 pipeline (`SWEEP_METHODOLOGY.md`, `SIMULATION_GUIDE.md`,
`sglang/AGG24_RESULTS.md`) — same cluster, same dataset, same toolchain, with the
deltas the new architecture forces. Everything marked **[verified]** was checked
on 2026-08-19; everything marked **[open]** is a gate before silicon.

## 0. Model facts and why they reshape the experiment

**[verified]** From the model config (`nvidia/Nemotron-3-Ultra-550B-A55B`, HF-gated;
config read via the ungated unsloth mirror): `NemotronHForCausalLM`, 108 layers
in hybrid pattern — **48 Mamba-2 + 48 LatentMoE + only 12 full-attention
layers** ('*' in the pattern), GQA n_kv=2 × head_dim 128, 512 experts top-22
(latent 2048), 55B active / 550B total, NVFP4-native, context 262,144 (matches
our trace cap exactly). Native MTP layers for speculative decoding.

Architecture-derived constants (vs Kimi-K2.5):

| Quantity | Nemotron-3-Ultra | Kimi-K2.5 | Consequence |
|---|---|---|---|
| KV bytes/token (fp8) | **6 KB** (12 attn × 2 kv × 128 × 2) | ~35 KB | 6× smaller |
| Mamba SSM state/request | **~200 MB** fixed (48 × 256 × 64 × 128 × bf16) | — | new transfer + cache object |
| Disagg transfer @96k ISL | ~0.6 GB KV + 0.2 GB state | ~3.4 GB | **4× lighter transfers** |
| KV pool/TP4 worker (post-weights ≈ 870 GB HBM) | **>100M tokens** | ~12M | trace's 119M unique tokens ≈ cacheable per worker |
| Prefill scaling in ISL | near-linear (Mamba+MoE; attn only 12/108) | quadratic-ish | recompute is cheaper → RR's tax shrinks |
| Decode TPOT | Mamba constant-time; weakly batch-sensitive | slope 0.07 ms/seq | knees move to higher conc |

**The hypothesis this experiment actually tests** (write it down before running):
KV-aware routing's win on Kimi came from expensive recompute + scarce cache.
Nemotron inverts both — recompute is cheap(er) and cache is abundant. If KV
routing still wins, it's because *placement* (hitting the worker that already
holds your prefix) matters even when capacity doesn't. If it doesn't win, that's
a publishable negative result about hybrid-Mamba models. Either outcome is a
finding; the sim predicts which before silicon spends a GPU-hour.

## 1. Stage 0 — support gates

| Gate | Status |
|---|---|
| AIC support, gb300 × sglang, agg + disagg | **[verified] PASS on aiconfigurator 0.11.0** (5/6 architecture vote; 0.10.0 FAILS 2/6 — pin 0.11.0). AIC's sglang DB = **0.5.14**, exactly our proven silicon pair with dynamo 1.3.1 |
| HF weights access | **[verified] CLOSED** — `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` is **ungated** (the earlier 401s were wrong repo names missing the `NVIDIA-` prefix): 113 shards, **352 GB**, mixed FP8 (Mamba mixers) / FP4 (MoE) quantization. The unsloth mirror was evaluated and rejected for serving: it is the **BF16** release (225 shards, 1.12 TB — weights alone would fill a TP4 worker's HBM); it remains useful as the ungated config source. Staged to `/model-cache/alisachen/Nemotron-3-Ultra-550B-A55B-NVFP4` via `nemotron3u/stage-weights-job.yaml` (no token needed) |
| sglang 0.5.14 NemotronH serving | **[open]** AIC's support matrix says yes; still smoke-verify: does 0.5.14 serve NemotronH+LatentMoE+MTP? |
| **Hybrid prefix cache** (the critical one) | **[open]** KV-aware routing requires radix *reuse* over the hybrid cache: attention KV per block + Mamba SSM state checkpoints at reuse boundaries. Verify sglang's hybrid/Mamba radix cache exists in 0.5.14 and reports `cached_tokens`; if reuse is attention-only or disabled for hybrid models, the whole experiment changes meaning |
| kv-events for hybrid cache | **[open]** does `--kv-events-config` publish block events for hybrid models? (Kimi lesson: without events the router silently degrades to load-only — verify via `dynamo_component_router_kv_hit_rate` before trusting any KV run) |
| Disagg transfer of Mamba state | **[open]** NIXL must move SSM state + KV; verify sglang disagg supports NemotronH at all |
| MTP policy | decide up front: **OFF for both arms** (router-comparison parity; the Kimi/NVIDIA-Eagle confound lesson). Optional MTP-on appendix arm later |

Any [open] gate that fails downgrades the plan (e.g. agg-only, or attention-KV-
only reuse) — decided then, not discovered mid-campaign.

## 2. Stage 1 — AIC SILICON solves (engine shape)

Same bracket structure as Kimi, run in the cluster pod (VM glibc can't):

```bash
pip install aiconfigurator==0.11.0     # NOT 0.10.0 — NemotronH unsupported there
# warm bracket (trace reuse ~97% within session):
aiconfigurator cli default --model-path <local nemotron config dir> --system gb300 \
  --backend sglang --total-gpus 24 --isl 96000 --osl 900 --prefix 92000 \
  --ttft 5000 --tpot 10 --max-seq-len 262144 --enable-chunked-prefill \
  --database-mode SILICON --save-dir aic-n3u-warm
# cold bracket: drop --prefix, --ttft 30000; repeat both for --total-gpus 72
```

Notes vs Kimi: ISL/OSL encoded from the *same* trace (use measured means ~96k/880,
not Kimi's 137k encoding — that was a different slice convention; fix at solve
time and record which). TTFT/TPOT args are AIC solver targets only — **the study
has no latency SLO** (methodology revision 3); selection is queue-drain based.
Expected qualitative outputs to sanity-check: much higher warm concurrency
ceilings than Kimi, weaker warm/cold TTFT split (cheap recompute), TP2 possibly
viable for weights (~300 GB ≤ 2×288 GB) — let the solver decide TP2 vs TP4; a
TP2 worker halves the fleet's node count per worker and doubles worker count at
equal GPUs, which *changes the router experiment's granularity* — carry both
shapes into DynoSim if AIC ranks them close.

## 3. Stage 2 — DynoSim constant derivation (new physics required)

The simulator's Kimi constants do not transfer. Changes to `dynosim_pd.py`:

1. **Prefill model**: near-linear rate from AIC cold solve (tok/s/worker); the
   AIC-ratio-transfer trick (Kimi trtllm→sglang) is unavailable — seed directly
   from AIC 0.11.0 SILICON and calibrate on smoke (our v1→v5 lineage shows live
   recalibration works; budget for a 30× surprise like Kimi's).
2. **Cache model**: radix over trace hash_ids stays (router-behavior-exact), but
   per-worker capacity becomes ~effectively-infinite (>100M tokens) → eviction
   never binds. Add the **Mamba checkpoint constraint**: reuse resolves to the
   nearest stored state boundary (model as block-aligned, boundary spacing from
   sglang's actual hybrid-cache implementation once verified — [open] above).
3. **Decode model**: constant-dominant TPOT with small slope (fit from AIC decode
   rows); drop Kimi's agg bs≥8 cliff unless AIC shows one for this architecture.
4. **Transfer cost (disagg)**: add per-request transfer term = 0.8 GB / effective
   BW. Under the **host-staged bypass this is potentially the bottleneck** — sim
   both BW cases (GPUDirect fixed vs host-staged) so silicon has a prediction
   for whichever cluster we get.

## 4. Stage 3 — DynoSim policy sweeps (unchanged machinery, new grid)

Same trace (`semianalysisai/cc-traces-weka-062126-256k`, 393-session interleave,
hash-id-exact), same 11-policy grid (rr, ll, kv-defaults, tuned credit/scale
variants), same closed-loop conc ladders:

```bash
python3 scripts/dynosim_pd.py $T --agg      --backend sglang --model n3u --out nemotron3u/dynosim_n3u_agg_v1.csv
python3 scripts/dynosim_pd.py $T --sweep    --backend sglang --model n3u --out nemotron3u/dynosim_n3u_disagg24_v1.csv
python3 scripts/dynosim_pd.py $T --disagg72 --backend sglang --model n3u --out nemotron3u/dynosim_n3u_disagg72_v1.csv
```

Selection uses the **queue-drain criterion only** (revision 3): drain-probe
metrics (idle-arrival fraction, backlog p50/p95, within-window trend — the
instrumented probe from the 9:9 selection) locate knees; latency percentiles are
reported, never gates. Cells to select per scale, exactly as Kimi:
framing 1 (both-bounded same cell — expect it at *much* higher conc than Kimi if
the reuse hypothesis holds), framing 2 (each policy at its own knee), plus the
router-flag grid verdict (expect flags to matter *less*: with no eviction
pressure, overlap-credit tuning loses its lever — a testable sim prediction).

## 5. Stage 4 — silicon plan (all Kimi hard lessons pre-applied)

**Smokes first** (1P+1D + agg-1): serve NemotronH on 0.5.14, verify hybrid
radix reuse live (repeated-prompt `cached_tokens` probe — the Kimi test that
caught 0% wire reuse), kv-events flowing, router hit-rate telemetry nonzero,
NIXL transfer under the DynamoBench UCX env, RDMA guard PASS.

**Arms** via `gen_sglang_arms.py` (new model entry): worker shape from AIC;
image `lmsysorg/sglang:v0.5.14-cu130-runtime` + `ai-dynamo[sglang]==1.3.1`
(re-smoke the pair for NemotronH; if 0.5.14 lacks hybrid-cache pieces, test
newer sglang against dynamo compat explicitly — version-pair drift was Kimi
failure #1). Everything already institutionalized stays on: `--request-plane
nats`, `--host 0.0.0.0`, shlex-quoted `--kv-events-config`, full UCX pins **with
`UCX_IB_GPU_DIRECT_RDMA=n` baked in the generator** (until the driver fault is
fixed), nodepool pinning + pinned private kubeconfig, native
`--public-dataset` loader (never the mooncake conversion), `--num-dataset-entries
393`, dual HF cache staging, 900 s per-point trace warmup under the measured
point's own router config, fresh-frontend-per-point, per-point RDMA gate +
knee gate (halt on violation), quote-agnostic manifest seds, pod-spec env
verification before trusting any run.

**Ladders**: agg 24-GPU first (it was Kimi's cleanest result and this cluster's
disagg path is currently compromised): conc ladder bracketing the sim knee from
both sides. Disagg 72-GPU **only after the ~1.5 req/s ceiling is root-caused** —
Nemotron's 4× lighter transfers may themselves relieve it (that's diagnostic
signal: if Nemotron disagg scales where Kimi choked, the ceiling was
transfer-bound), but do not spend the full grid before one instrumented probe
run confirms the pipeline isn't choked for this model too.

**Verification per KV run** (unchanged 4-layer): config → kv-events → router
`kv_hit_rate` histogram → engine `cached_tokens`, plus drain-probe-style knee
verdicts from per-request timestamps (`knee_check.py`).

## 6. Deliverables and sequencing

1. Stage 0 gates closed (HF token, smoke matrix) — ~1 day incl. weight staging.
2. AIC solves + DynoSim n3u constants + sweeps — 1 day, all off-cluster, run in
   parallel with any remaining Kimi work (standing policy).
3. `nemotron3u/SELECTION.md` (sim cells + hypothesis predictions) → review point.
4. Agg silicon ladder (both arms + verification) → `nemotron3u/AGG24_RESULTS.md`
   mirroring the Kimi doc, incl. sim-vs-silicon verdict table.
5. Disagg silicon (gated on ceiling diagnosis) → `nemotron3u/D72_RESULTS.md`.
6. Cross-model synthesis: the routing-win-vs-architecture story (attention-heavy
   Kimi vs hybrid Nemotron on identical traffic) — the study's headline chart.

## Open items owed to the user

- GPU budget confirmation: same 24-agg / 72-disagg scales, np-3?
- MTP-off-for-parity confirmation (recommended above).
