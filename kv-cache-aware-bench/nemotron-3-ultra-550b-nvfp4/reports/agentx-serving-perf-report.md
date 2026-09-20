# KV-Aware vs. Round-Robin Routing: NVIDIA Dynamo on Google Cloud

*How routing affects throughput, time to first token, and end-to-end responsiveness for coding-agent workloads.*

A coding agent sends much of its conversation history again on each turn. The worker that receives that request determines whether the service can reuse a resident prefix or must recompute more of the prompt. Across a fleet, routing therefore affects both cache reuse and how work is distributed—and ultimately how long the user waits.

This Google Cloud **customer use journey (CUJ)** compares NVIDIA Dynamo's **KV-aware routing** with **round-robin (RR) routing** using recorded AgentX coding sessions and Nemotron-3-Ultra on GB300 GPUs. We follow each turn from the client, through Dynamo's router, to the streamed response. The study covers **aggregated serving**, where a worker handles prefill and decode, and **disaggregated serving**, where separate worker pools handle those stages.

The comparison follows two customer decisions: how the routing policies perform at the **same session concurrency**, and how much traffic each can serve under the **same latency SLO**. We measure TTFT p95 <10 seconds and separately require E2E-normalized interactivity ≥20 output tokens/s at P90. Default KV/RR curves establish the routing impact; the tuning tables show how that impact changes with router settings. The measured results also show why an agg tuning choice must be checked again on disagg.

At each policy's best sampled point meeting **both** latency criteria, default KV delivered **2.11× total served tokens/s/GPU for agg** and **6.92× for disagg** relative to RR. These compare different selected session counts: C96 versus C48 for agg, and C480 versus C72 for disagg. The same-concurrency tables show the routing differences at a fixed population of users.


**Evidence snapshot: 2026-09-20 02:53 UTC · 37 completed hardware jobs (17 agg / 20 disagg)**

Each architecture has its own measured curves, full data table, comparison at a sampled knee, and SLO table. The baseline curves show **default KV and RR only**; tuned settings remain in the data and tuning tables. Total served tokens include cached prompt tokens; output-token throughput is reported alongside them.

[Standalone HTML](agentx-serving-perf-report.html) · [hardware CSV](agentx-serving-perf-report.csv) · [comparison JSON](agentx-serving-perf-report.json) · [configuration provenance](agentx-serving-perf-report-methodology.json) · [validation](agentx-serving-perf-report-validation.json)

## 1. Setup, agentic workload and benchmarking methodology

### 1.1 Hardware and serving recipes

Both fleets serve **NVIDIA Nemotron-3-Ultra-550B-A55B-NVFP4** on **GB300**, with modelopt FP4 quantization and TP4/EP4 per worker. The recipes install Dynamo **1.4.2** with its SGLang dependency **0.5.16**, and FlashInfer **0.6.18**. The container base tag alone is not the final installed-version record; immutable image/tokenizer revisions and complete live package inventories were not captured.

| Setting | Aggregated serving | Disaggregated serving (D88) |
| --- | --- | --- |
| Fleet | 6 workers × 4 GPUs = **24 GPUs** | 8 prefill + 8 decode workers × 4 GPUs = **64 GPUs** |
| Work placement | Each worker does both prefill and decode | Separate prefill and decode pools |
| Parallelism per worker | TP4, EP4 | TP4, EP4 on both stages |
| Context / page size | 262,144 tokens / 64 tokens | 262,144 tokens / 64 tokens |
| Explicit admission limit | 16 running requests per worker | Prefill 8; decode 64 per worker |
| Prefill chunk / static memory fraction | 16,384 tokens / 0.85 | Prefill 16,384 tokens / 0.85 on both stages |
| Request and transfer path | NATS request plane; no P→D handoff | NATS request plane; Mooncake handoff, MNNVL/IMEX domain |
| Speculative decoding | Not enabled in the saved recipe | Not enabled in the saved recipe |
| Sources | [agg manifest](agentx-serving-perf-data/source/n3u-agg-newstack-np2.yaml.txt) | [disagg manifest](agentx-serving-perf-data/source/n3u-mnnvl-88.yaml.txt) |

The original agg ladder was collected on September 16; two fresh references and three decay variants form the later np-2 campaign. The agg recipes use two separate fleets (`n3u-agg-ns` / `n3u-agg-ns2`). The same-concurrency comparison at C192 uses the original matched campaign; new references are shown individually. GPU-normalized comparisons across agg and disagg remain observations from different fleet sizes, not controlled architecture-scaling estimates.

### 1.2 Agentic workload and replay

AIPerf **0.12.0** runs `inferencex-agentx-mvp` on **`semianalysisai/cc-traces-weka-062126-256k`**, containing 393 recorded coding-session roots. A session includes its subagents. Later turns reuse conversation prefixes, while recorded think time, subagent fan-out and joins determine when requests arrive. Prefix reuse is therefore measured from replay, not imposed as a fixed hit percentage.

| Replay control | Value used in the measured jobs |
| --- | --- |
| Concurrency C | Live session trees; not simultaneous requests or decode batch size |
| Seed / dataset | 42 / 393 roots from the named Weka 256K corpus |
| Initial trajectory | Start ratio 0.25–0.75, with trajectory warmup before profiling |
| Timing | Recorded end-to-start delays; whole-system idle-gap cap 10 seconds |
| Measurement | 3,600-second profiling window; 60-second grace; 1,200-second request timeout |
| Prompt/output handling | Model tokenizer; server token counts; streaming; `ignore_eos` preserves recorded output lengths |
| Recycled traces | A fresh first-turn-prefix cache-bust marker per play |
| Validation | Included runs pass the scenario stamp; success and error counts are retained separately |

The run template and exact router variants are preserved in the [benchmark template](agentx-serving-perf-data/source/sgl-d72-agentx.yaml.txt) and [runner](agentx-serving-perf-data/source/agentx_runner_flags.sh.txt). A fixed seed controls sampling, but a faster arm can complete more turns in the same hour. Compare the ISL/OSL, depth and trace mix in the exported data; do not assume identical completed request cohorts. Busy-stream runs that remove think time are a different workload and are excluded.

### 1.3 Fair KV/RR comparison and metric definitions

**Change the router while holding the serving recipe and replay fixed.** RR uses `--router-mode round-robin`; default KV uses `--router-mode kv --router-temperature 0 --router-queue-policy fcfs`. Prefix caching remains enabled in the agg/prefill engines for both policies. RR is not a no-cache control. Tuned KV adds the explicit flags listed in each architecture's table.

| Metric or comparison | Definition |
| --- | --- |
| Total throughput/GPU | AIPerf input + output tokens/s divided by every GPU in the fleet, including both disagg stages. Cached input is included; this is served token volume. |
| Output throughput/GPU | Output tokens/s divided by the same GPU count. Reported separately to expose workload-mix differences. |
| TTFT | p95 over successful profiling requests, in seconds. The requested table uses **strict TTFT p95 <10 s**. No current point is exactly 10 s. |
| E2E interactivity I90 | Per request, compute `r_i = E2E_seconds / output_tokens`; then `I90 = 1 / P90(r_i)` using linear interpolation. The additional SLO is **I90 ≥20 tok/s/user**. |
| Same configuration | Same topology, GPU count, workload and session concurrency; only router settings differ. |
| Same SLO | Each policy selects its highest-throughput sampled point passing TTFT <10 s, excluding points with post-knee queue evidence. Its concurrency may differ. |
| Combined SLO | Apply TTFT <10 s **and** I90 ≥20. This is shown separately from the TTFT-only table. |
| Errors | Failed profiling requests are excluded from latency percentiles and reported explicitly. No new availability threshold is imposed. |

**Knee evidence:** use the throughput slope/decline, TTFT tail, errors and within-run queue progression. A sampled throughput peak is not an exact continuous knee; an SLO crossing is a separate boundary. The existing queue check flags a high sustained TTFT median, a growing first-to-last-quarter TTFT median, or an error rate above 5%. GPU utilization alone does not determine the knee.

Most cells have one trial. The two agg C192 references have repeats across deployment campaigns, but that is not a complete noise distribution. Caches were not explicitly flushed between every hardware cell. Cache busting and successful warmup reduce some biases without establishing identical physical cache state. The client cached-input metric is `overall_usage_prompt_cache_read_pct`, not per-engine KV occupancy. A scenario-valid closed-loop replay result is not an open-loop production arrival-rate guarantee.


## 2. Aggregated serving: measured KV and RR

### 2.1 Default KV versus RR: curves and knee evidence

![Aggregated serving: measured KV and RR: default KV and RR only, with throughput peaks and latency boundaries](agentx-serving-perf-report-agg-curves.png)

[SVG](agentx-serving-perf-report-agg-curves.svg) · [PDF](agentx-serving-perf-report-agg-curves.pdf)


| Policy | Sampled throughput / knee evidence | TTFT <10 s boundary | Additional E2E boundary |
| --- | --- | --- | --- |
| Default KV | Peak at **C192**; C384 loses **14.5%** throughput and TTFT p95 rises **11.66→119.44 s**. Saturation transition lies in 192–384. | C96 passes; C192 fails. Refine **96–192**. | I90 also crosses between 96 and 192. |
| RR | C96→192 adds only **10.8%** throughput; C384 loses **25.4%** versus C192. C192 is the sampled peak; diminishing returns begin over 96–192. | C48 passes; C96 fails. Refine **48–96**. | I90 also crosses between 48 and 96. |

The throughput-knee comparison below uses **C192 for both arms**. It does not claim that C192 meets the latency SLO. A cross marks the fresh default-KV192 reference; the original ladder remains the line so campaigns are not silently combined.

### 2.2 All collected data points, including tuned KV

Every completed agg run is shown, including repeated references. The flags identify the recipe; each linked name opens its hardware artifacts.

| Setting / artifacts | C | Campaign | Total tok/s/GPU | Output tok/s/GPU | TTFT p95 (s) | E2E I90 | TTFT <10 | Both SLOs | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789545801_alisachen-n3u-agg-ns-agentx-kv-c48) | 48 | Sep 16 | 3,334 | 32.27 | 3.83 | 51.8257 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789550601_alisachen-n3u-agg-ns-agentx-kv-c96) | 96 | Sep 16 | 6,844 | 75.11 | 5.36 | 29.5030 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789555981_alisachen-n3u-agg-ns-agentx-kv-c192) | 192 | Sep 16 | 9,655 | 96.81 | 11.66 | 13.5367 | Fail | Fail | 3 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789856222_alisachen-n3u-agg-ns-agentx-kv-c192) | 192 | np-2 | 9,468 | 95.42 | 11.18 | 12.9310 | Fail | Fail | 3 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789562076_alisachen-n3u-agg-ns-agentx-kv-c384) | 384 | Sep 16 | 8,257 | 77.74 | 119.44 | 1.1372 | Fail | Fail | 7 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789550745_alisachen-n3u-agg-ns2-agentx-rr-c48) | 48 | Sep 16 | 3,250 | 31.32 | 8.27 | 40.5349 | Pass | Pass | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789555623_alisachen-n3u-agg-ns2-agentx-rr-c96) | 96 | Sep 16 | 6,137 | 67.11 | 12.56 | 17.7260 | Fail | Fail | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789560983_alisachen-n3u-agg-ns2-agentx-rr-c192) | 192 | Sep 16 | 6,802 | 71.38 | 60.08 | 3.8961 | Fail | Fail | 3 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789567077_alisachen-n3u-agg-ns2-agentx-rr-c384) | 384 | Sep 16 | 5,076 | 49.65 | 494.98 | 0.6059 | Fail | Fail | 14 |
| [KV scale 3 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789582550_alisachen-n3u-agg-ns2-agentx-kvs3c08-c96) | 96 | Sep 16 | 7,045 | 78.35 | 3.11 | 39.4930 | Pass | Pass | 0 |
| [KV scale 3 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789569414_alisachen-n3u-agg-ns-agentx-kvs3c08-c192) | 192 | Sep 16 | 11,012 | 108.87 | 6.33 | 19.7795 | Pass | Fail | 3 |
| [KV scale 3 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789856381_alisachen-n3u-agg-ns2-agentx-kvs3c08-c192) | 192 | np-2 | 10,992 | 110.04 | 6.20 | 19.7385 | Pass | Fail | 2 |
| [KV scale 2 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789575514_alisachen-n3u-agg-ns-agentx-kvs2c08-c192) | 192 | Sep 16 | 10,241 | 102.43 | 8.11 | 16.1477 | Pass | Fail | 3 |
| [KV temperature 0.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789574468_alisachen-n3u-agg-ns2-agentx-kvt05-c192) | 192 | Sep 16 | 7,816 | 79.49 | 22.15 | 7.0315 | Fail | Fail | 3 |
| [KV decay 0.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789831067_alisachen-n3u-agg-ns-agentx-kvd05-c192) | 192 | np-2 | 9,362 | 94.58 | 12.70 | 12.0259 | Fail | Fail | 3 |
| [KV decay 1.0](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789831039_alisachen-n3u-agg-ns2-agentx-kvd10-c192) | 192 | np-2 | 9,233 | 93.40 | 13.85 | 11.4805 | Fail | Fail | 3 |
| [KV scale 3 / credit 0.8 / decay 0.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789862417_alisachen-n3u-agg-ns-agentx-kvs3c08d05-c192) | 192 | np-2 | 10,713 | 106.29 | 6.97 | 18.2395 | Pass | Fail | 3 |

| Recipe | Collected concurrency | Exact router arguments |
| --- | --- | --- |
| Default KV | 48, 96, 192 (2 runs), 384 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs` |
| RR | 48, 96, 192, 384 | `--router-mode round-robin` |
| KV scale 3 / credit 0.8 | 96, 192 (2 runs) | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8` |
| KV scale 2 / credit 0.8 | 192 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 2.0 --router-kv-overlap-score-credit 0.8` |
| KV temperature 0.5 | 192 | `--router-mode kv --router-temperature 0.5 --router-queue-policy fcfs` |
| KV decay 0.5 | 192 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit-decay 0.5` |
| KV decay 1.0 | 192 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit-decay 1.0` |
| KV scale 3 / credit 0.8 / decay 0.5 | 192 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8 --router-kv-overlap-score-credit-decay 0.5` |

### 2.3 Tuned KV versus RR at the same configuration, near the sampled knee

| C192 setting | Total tok/s/GPU | Throughput / RR | TTFT p95 (s) | RR / TTFT | E2E I90 | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789560983_alisachen-n3u-agg-ns2-agentx-rr-c192) | 6,802 | 1.00× | 60.08 | 1.00× | 3.8961 | 3 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789555981_alisachen-n3u-agg-ns-agentx-kv-c192) | 9,655 | 1.42× | 11.66 | 5.15× | 13.5367 | 3 |
| [KV scale 3 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789569414_alisachen-n3u-agg-ns-agentx-kvs3c08-c192) | 11,012 | 1.62× | 6.33 | 9.49× | 19.7795 | 3 |

At C192, the original tuned setting delivers **1.62× RR throughput** and **9.49× shorter TTFT p95**. These are the September 16 comparison cells. The fresh tuned reference is included above and in the current-campaign flag contrasts below; no fresh RR192 run was collected alongside it.

### 2.4 Tuned KV versus RR under the same SLO: TTFT p95 <10 seconds

Select the highest measured total throughput passing the **TTFT-only** limit and queue check. This table does **not** impose I90 ≥20; its last column shows whether the selected point also passes that additional criterion.

| Policy / selected run | C | Total tok/s/GPU | Output tok/s/GPU | TTFT p95 (s) | E2E I90 | Throughput / RR | I90 ≥20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789550745_alisachen-n3u-agg-ns2-agentx-rr-c48) | 48 | 3,250 | 31.32 | 8.27 | 40.5349 | 1.00× | Pass |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789550601_alisachen-n3u-agg-ns-agentx-kv-c96) | 96 | 6,844 | 75.11 | 5.36 | 29.5030 | 2.11× | Pass |
| [KV scale 3 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789569414_alisachen-n3u-agg-ns-agentx-kvs3c08-c192) | 192 | 11,012 | 108.87 | 6.33 | 19.7795 | 3.39× | Fail |

**TTFT-only:** tuned KV192 versus RR48 gives **3.39× total throughput/GPU**. The selected tuned job is the highest-throughput original sample; its fresh C192 repeat gives 10,992 total tok/s/GPU and 6.20 s TTFT, close to the original 11,012 and 6.33 s. Both tuned C192 trials fail the E2E requirement. The TTFT-only selection has three client errors, shown in the complete table.

**If E2E is also required**, apply both SLOs:

| Policy | Selected C | Total tok/s/GPU | TTFT p95 (s) | E2E I90 | Throughput / RR |
| --- | --- | --- | --- | --- | --- |
| RR | 48 | 3,250 | 8.27 | 40.5349 | 1.00× |
| Default KV | 96 | 6,844 | 5.36 | 29.5030 | 2.11× |
| KV scale 3 / credit 0.8 | 96 | 7,045 | 3.11 | 39.4930 | 2.17× |

### 2.5 What the flag sweep establishes

![Measured agg KV routing flags, with TTFT and E2E thresholds](agentx-serving-perf-report-agg-flags.png)

[SVG](agentx-serving-perf-report-agg-flags.svg) · [PDF](agentx-serving-perf-report-agg-flags.pdf)


| np-2 treatment | np-2 reference | Δ total throughput | Δ TTFT p95 | Δ E2E I90 | Δ cached input |
| --- | --- | --- | --- | --- | --- |
| KV scale 3 / credit 0.8 | Default KV | +16.09% | -44.6% | +52.6% | +9.06 pp |
| KV decay 0.5 | Default KV | -1.13% | +13.6% | -7.0% | -0.71 pp |
| KV decay 1.0 | Default KV | -2.48% | +23.9% | -11.2% | -1.49 pp |
| KV scale 3 / credit 0.8 / decay 0.5 | KV scale 3 / credit 0.8 | -2.53% | +12.6% | -7.6% | -1.83 pp |

**Retain scale 3 / credit 0.8 with decay 0.** The fresh references reproduce original throughput within 2%; the fresh tuned result gives I90 **19.7385**, versus **19.7795** originally, so the C192 miss has repeated. Decay 0.5/1.0 alone and decay 0.5 added to the tuned setting all lose in observed throughput and latency. Their small throughput deltas still need repeats for statistical precision. The first decay-only warmups took about 1,733 s, while later controls took about 1,632–1,634 s; this is consistent with a startup transient, not proof that every np-2 run is slower.

**Next useful points:** tuned C144, then C168 if it passes both limits; repeat the selected operating point. For further flag isolation, compare scale 4/credit 0.8 with scale 3/credit 0.8, and separately scale 3/credit 0. The existing comparisons do not isolate scale from credit when both change against default KV.

## 3. Disaggregated serving: measured KV and RR

### 3.1 Default KV versus RR: curves and knee evidence

![Disaggregated serving: measured KV and RR: default KV and RR only, with throughput peaks and latency boundaries](agentx-serving-perf-report-disagg-curves.png)

[SVG](agentx-serving-perf-report-disagg-curves.svg) · [PDF](agentx-serving-perf-report-disagg-curves.pdf)


| Policy | Sampled throughput / knee evidence | TTFT <10 s boundary | Additional E2E boundary |
| --- | --- | --- | --- |
| Default KV | C672→768 adds only **2.4%** throughput while TTFT rises **59%**. C768 is the sampled peak; C1152 then loses **31.2%** and develops growing queues. Plateau begins around **672–768**. | C480 passes; C672 fails. Refine **480–672**. | I90 also crosses between 480 and 672. |
| RR | C192 is the sampled peak; C384 loses **20.7%**, has 68 errors and TTFT p95 **289.39 s**. Overload transition lies in **192–384**. | C72 passes by only **0.0885 s**; C96 fails. Refine **72–96**. | C96 passes I90; C144 fails. |

There is **no shared throughput knee** for KV and RR. The same-concurrency table uses **C192, RR's sampled peak**, where tuned credit 1.5 also has a real measurement. No tuned/RR pair exists at the default-KV plateau of 672–768; that comparison cannot be filled by extrapolation.

### 3.2 All collected data points, including tuned KV

Every completed disagg run is shown, including repeated references. The flags identify the recipe; each linked name opens its hardware artifacts.

| Setting / artifacts | C | Campaign | Total tok/s/GPU | Output tok/s/GPU | TTFT p95 (s) | E2E I90 | TTFT <10 | Both SLOs | Errors |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789808919_alisachen-n3u-mnnvl-88-agentx-kv-c96) | 96 | D88 | 2,810 | 31.52 | 1.56 | 80.9304 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789803115_alisachen-n3u-mnnvl-88-agentx-kv-c144) | 144 | D88 | 3,964 | 44.14 | 1.87 | 71.7317 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789672351_alisachen-n3u-mnnvl-88-agentx-kv-c192) | 192 | D88 | 5,142 | 50.75 | 2.40 | 67.3601 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789678827_alisachen-n3u-mnnvl-88-agentx-kv-c384) | 384 | D88 | 9,993 | 100.04 | 4.83 | 44.7508 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789740024_alisachen-n3u-mnnvl-88-agentx-kv-c480) | 480 | D88 | 12,204 | 123.36 | 7.12 | 33.5225 | Pass | Pass | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789686412_alisachen-n3u-mnnvl-88-agentx-kv-c672) | 672 | D88 | 15,184 | 149.17 | 17.04 | 13.7406 | Fail | Fail | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789695978_alisachen-n3u-mnnvl-88-agentx-kv-c768) | 768 | D88 | 15,543 | 152.20 | 27.14 | 7.4108 | Fail | Fail | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789706033_alisachen-n3u-mnnvl-88-agentx-kv-c1152) | 1152 | D88 | 10,697 | 107.71 | 164.32 | 1.1161 | Fail | Fail | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789773750_alisachen-n3u-mnnvl-88-agentx-rr-c72) | 72 | D88 | 1,763 | 20.47 | 9.91 | 37.3998 | Pass | Pass | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789734386_alisachen-n3u-mnnvl-88-agentx-rr-c96) | 96 | D88 | 2,671 | 29.62 | 13.03 | 28.0935 | Fail | Fail | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789728380_alisachen-n3u-mnnvl-88-agentx-rr-c144) | 144 | D88 | 3,588 | 40.78 | 21.63 | 13.9890 | Fail | Fail | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789721971_alisachen-n3u-mnnvl-88-agentx-rr-c192) | 192 | D88 | 4,419 | 44.48 | 31.45 | 8.3783 | Fail | Fail | 0 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789795836_alisachen-n3u-mnnvl-88-agentx-rr-c384) | 384 | D88 | 3,504 | 36.04 | 289.39 | 1.0990 | Fail | Fail | 68 |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789757358_alisachen-n3u-mnnvl-88-agentx-rr-c480) | 480 | D88 | 3,219 | 34.32 | 387.54 | 0.6525 | Fail | Fail | 39 |
| [KV scale 3 / credit 0.8](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789748912_alisachen-n3u-mnnvl-88-agentx-kvs3c08-c480) | 480 | D88 | 12,018 | 121.56 | 10.48 | 24.1932 | Fail | Fail | 0 |
| [KV credit 1.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789814379_alisachen-n3u-mnnvl-88-agentx-kvc15-c192) | 192 | D88 | 5,138 | 50.72 | 2.30 | 65.3478 | Pass | Pass | 0 |
| [KV credit 1.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789778938_alisachen-n3u-mnnvl-88-agentx-kvc15-c480) | 480 | D88 | 12,239 | 123.83 | 6.02 | 37.9246 | Pass | Pass | 0 |
| [KV credit 1.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789820579_alisachen-n3u-mnnvl-88-agentx-kvc15-c576) | 576 | D88 | 14,024 | 136.13 | 8.75 | 30.2479 | Pass | Pass | 0 |
| [KV credit 2.0](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789787390_alisachen-n3u-mnnvl-88-agentx-kvc20-c480) | 480 | D88 | 12,239 | 123.55 | 6.55 | 36.6851 | Pass | Pass | 0 |
| [KV decay 0.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789765406_alisachen-n3u-mnnvl-88-agentx-kvd05-c480) | 480 | D88 | 11,839 | 119.49 | 13.13 | 19.2392 | Fail | Fail | 0 |

| Recipe | Collected concurrency | Exact router arguments |
| --- | --- | --- |
| Default KV | 96, 144, 192, 384, 480, 672, 768, 1152 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs` |
| RR | 72, 96, 144, 192, 384, 480 | `--router-mode round-robin` |
| KV scale 3 / credit 0.8 | 480 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-prefill-load-scale 3.0 --router-kv-overlap-score-credit 0.8` |
| KV credit 1.5 | 192, 480, 576 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit 1.5` |
| KV credit 2.0 | 480 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit 2.0` |
| KV decay 0.5 | 480 | `--router-mode kv --router-temperature 0.0 --router-queue-policy fcfs --router-kv-overlap-score-credit-decay 0.5` |

### 3.3 Tuned KV versus RR at the same configuration, near the sampled knee

| C192 setting | Total tok/s/GPU | Throughput / RR | TTFT p95 (s) | RR / TTFT | E2E I90 | Errors |
| --- | --- | --- | --- | --- | --- | --- |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789721971_alisachen-n3u-mnnvl-88-agentx-rr-c192) | 4,419 | 1.00× | 31.45 | 1.00× | 8.3783 | 0 |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789672351_alisachen-n3u-mnnvl-88-agentx-kv-c192) | 5,142 | 1.16× | 2.40 | 13.12× | 67.3601 | 0 |
| [KV credit 1.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789814379_alisachen-n3u-mnnvl-88-agentx-kvc15-c192) | 5,138 | 1.16× | 2.30 | 13.68× | 65.3478 | 0 |

At C192, credit 1.5 delivers **1.16× RR throughput** and **13.68× shorter TTFT p95**; default KV has almost the same throughput as tuned KV at this load. At the heavier C480 tuning point, tuned KV/RR is **3.80× on throughput**, but RR is already overloaded. That C480 ratio is not a comparison of two sustainable knees.

### 3.4 Tuned KV versus RR under the same SLO: TTFT p95 <10 seconds

Select the highest measured total throughput passing the **TTFT-only** limit and queue check. This table does **not** impose I90 ≥20; its last column shows whether the selected point also passes that additional criterion.

| Policy / selected run | C | Total tok/s/GPU | Output tok/s/GPU | TTFT p95 (s) | E2E I90 | Throughput / RR | I90 ≥20 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [RR](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789773750_alisachen-n3u-mnnvl-88-agentx-rr-c72) | 72 | 1,763 | 20.47 | 9.91 | 37.3998 | 1.00× | Pass |
| [Default KV](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789740024_alisachen-n3u-mnnvl-88-agentx-kv-c480) | 480 | 12,204 | 123.36 | 7.12 | 33.5225 | 6.92× | Pass |
| [KV credit 1.5](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789820579_alisachen-n3u-mnnvl-88-agentx-kvc15-c576) | 576 | 14,024 | 136.13 | 8.75 | 30.2479 | 7.95× | Pass |

**TTFT-only:** credit-1.5 KV576 versus RR72 gives **7.95× total throughput/GPU**. Both also pass I90 ≥20 and have zero client errors. RR72 has almost no TTFT headroom and needs a repeat. Default KV576 is not yet measured in this snapshot, so the tuned-versus-default capacity gain is not isolated at equal concurrency.

**If E2E is also required**, apply both SLOs:

| Policy | Selected C | Total tok/s/GPU | TTFT p95 (s) | E2E I90 | Throughput / RR |
| --- | --- | --- | --- | --- | --- |
| RR | 72 | 1,763 | 9.91 | 37.3998 | 1.00× |
| Default KV | 480 | 12,204 | 7.12 | 33.5225 | 6.92× |
| KV credit 1.5 | 576 | 14,024 | 8.75 | 30.2479 | 7.95× |

### 3.5 What the flag sweep establishes

![Measured disagg KV routing flags, with TTFT and E2E thresholds](agentx-serving-perf-report-disagg-flags.png)

[SVG](agentx-serving-perf-report-disagg-flags.svg) · [PDF](agentx-serving-perf-report-disagg-flags.pdf)


| C480 KV setting | Total tok/s/GPU | Δ throughput | TTFT p95 (s) | Δ TTFT | E2E I90 | Both SLOs |
| --- | --- | --- | --- | --- | --- | --- |
| Default KV | 12,204 | +0.00% | 7.12 | +0.0% | 33.5225 | Pass |
| KV scale 3 / credit 0.8 | 12,018 | -1.52% | 10.48 | +47.1% | 24.1932 | Fail |
| KV credit 1.5 | 12,239 | +0.29% | 6.02 | -15.4% | 37.9246 | Pass |
| KV credit 2.0 | 12,239 | +0.29% | 6.55 | -8.0% | 36.6851 | Pass |
| KV decay 0.5 | 11,839 | -2.99% | 13.13 | +84.3% | 19.2392 | Fail |

**Credit 1.5 remains the strongest observed C480 latency candidate:** TTFT −15.4%, I90 +13.1%, throughput +0.29% versus default. Credit 2.0 has essentially identical throughput but worse measured TTFT and I90 than 1.5. Scale 3/credit 0.8 misses TTFT; decay 0.5 misses both limits. Do not transfer the agg winner to disagg without measuring it.

**Next useful points:** collect default KV576 and repeat credit 1.5 at C576, then test credit 1.5 at C624/C672 to bracket its latency boundary. The existing follow-up runs temperature 0.5/0.2 at C480 before default KV576; none had a completed summary in this evidence snapshot. RR384 and RR480 have 68 and 39 client errors respectively; the separate 353 server timeout events reported for RR480 are not a client-error count.

### 3.6 Agg and disagg under both SLOs

At their best sampled points satisfying **both** limits, tuned agg uses C96 and tuned D88 uses C576. D88 delivers **1.99× total throughput/GPU** and **1.74× output throughput/GPU**, on 64 versus 24 GPUs. The fleet totals are **897,553 versus 169,089 total tok/s**. Different fleet sizes and completed request mixes prevent interpreting this as a controlled scaling or cost result.

![Best measured agg and disagg operating points passing both SLOs](agentx-serving-perf-report-operating-points.png)

[SVG](agentx-serving-perf-report-operating-points.svg) · [PDF](agentx-serving-perf-report-operating-points.pdf)


## 4. Simulation versus real hardware jobs

### 4.1 Agg: twelve paired Native DynoSim V10 results

These are **12 actual native simulations paired with the original 12 agg hardware jobs**: four calibration points (default KV/RR at C192/C384) and eight holdouts (default KV/RR at C48/C96 and the four original tuned jobs). The simulator build, engine configuration and timing coefficients are shared. Five later agg hardware jobs have no new native execution paired to their run IDs; two are repeats of already simulated settings, and three are decay variants.

![Default KV and RR: original hardware versus Native DynoSim V10 throughput, TTFT and E2E interactivity](agentx-serving-perf-report-simulation-agg.png)

[SVG](agentx-serving-perf-report-simulation-agg.svg) · [PDF](agentx-serving-perf-report-simulation-agg.pdf)



| Setting | C | Role | Total/GPU real / sim | Throughput error | TTFT p95 real / sim (s) | I90 real / sim | Both SLOs real / sim | Errors real / sim |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Default KV | 48 | holdout | 3,334 / 3,321 | -0.4% | 3.83 / 3.52 | 51.8257 / 56.4714 | Pass / Pass | 0 / 0 |
| Default KV | 96 | holdout | 6,844 / 6,842 | -0.0% | 5.36 / 4.70 | 29.5030 / 32.6301 | Pass / Pass | 0 / 0 |
| Default KV | 192 | calibration | 9,655 / 10,182 | +5.5% | 11.66 / 8.71 | 13.5367 / 15.7431 | Fail / Fail | 3 / 3 |
| Default KV | 384 | calibration | 8,257 / 9,433 | +14.2% | 119.44 / 94.32 | 1.1372 / 1.4157 | Fail / Fail | 7 / 4 |
| RR | 48 | holdout | 3,250 / 3,232 | -0.6% | 8.27 / 8.47 | 40.5349 / 38.7997 | Pass / Pass | 0 / 0 |
| RR | 96 | holdout | 6,137 / 6,121 | -0.3% | 12.56 / 12.26 | 17.7260 / 15.5342 | Fail / Fail | 0 / 0 |
| RR | 192 | calibration | 6,802 / 6,830 | +0.4% | 60.08 / 53.96 | 3.8961 / 3.7086 | Fail / Fail | 3 / 3 |
| RR | 384 | calibration | 5,076 / 5,324 | +4.9% | 494.98 / 305.06 | 0.6059 / 0.6069 | Fail / Fail | 14 / 11 |
| KV scale 3 / credit 0.8 | 96 | holdout | 7,045 / 7,021 | -0.3% | 3.11 / 2.95 | 39.4930 / 41.4877 | Pass / Pass | 0 / 0 |
| KV scale 3 / credit 0.8 | 192 | holdout | 11,012 / 11,379 | +3.3% | 6.33 / 5.34 | 19.7795 / 22.5354 | Fail / Pass | 3 / 2 |
| KV scale 2 / credit 0.8 | 192 | holdout | 10,241 / 10,848 | +5.9% | 8.11 / 6.40 | 16.1477 / 19.1037 | Fail / Fail | 3 / 3 |
| KV temperature 0.5 | 192 | holdout | 7,816 / 8,018 | +2.6% | 22.15 / 20.48 | 7.0315 / 7.4468 | Fail / Fail | 3 / 3 |

| Subset | Points | Mean absolute throughput error | Mean absolute TTFT p95 error | Mean absolute I90 error |
| --- | --- | --- | --- | --- |
| calibration | 4 | 6.2% | 23.7% | 11.4% |
| holdout | 8 | 1.7% | 9.4% | 9.9% |

The four-point acceptance gate covered **±20% total throughput**, not TTFT or I90. All eight holdouts also fall within ±20% throughput. The original model reproduces the sampled default-policy throughput decline from C192 to C384 and the direction of the scale-3/credit-0.8 and temperature-0.5 effects, but it underestimates the RR384 TTFT tail by **38.4%**.

**SLO errors matter:** simulated default KV192 passes TTFT (8.71 s) while hardware fails (11.66 s). Simulated tuned KV192 gives I90 **22.5354** while hardware gives **19.7795**; it incorrectly passes the combined SLO and selects C192 where hardware selects C96. There is **one combined-SLO classification disagreement among 12 pairs**. Throughput calibration therefore supports candidate screening, not automatic SLO approval. [Original paired inputs and calibration/holdout provenance](agentx-agg-kv-rr-report.md#31-native-dynosim-v10-current-completed-calibration-samples).


### 4.2 D88: native forecasts alongside the measured KV/RR curves

Six completed native runs cover **8 prefill + 8 decode TP4 workers / 64 GPUs**, with default KV and RR at **C16, C64 and C256**. They use the **V11 disaggregation extension with frozen V10 timing**, the same Weka corpus and replay settings, and the engine configurations in section 5.3. These are existing September 17–18 runs, separate from the failed C480 flag sweep.

**No sampled concurrency values coincide between hardware and native runs.** The graph shows their collected trends; it does not compute accuracy by interpolating an unmeasured hardware or simulation point. No tuned KV points enter these curves.


![D88 default KV and RR: hardware and native results at different concurrency grids, with errored native results marked](agentx-serving-perf-report-simulation-disagg-d88.png)

[SVG](agentx-serving-perf-report-simulation-disagg-d88.svg) · [PDF](agentx-serving-perf-report-simulation-disagg-d88.pdf)


| Native run / summary | C | Total tok/s/GPU | TTFT p95 (s) | E2E I90 | Successes / errors | Error rate | Use |
| --- | --- | --- | --- | --- | --- | --- | --- |
| [Default KV](agentx-native-disagg-data/native/topo64-v11-p8d8-kv-c16/summary.json) | 16 | 275 | 1.90 | 85.6826 | 601 / 0 | 0.000% | Completed forecast |
| [Default KV](agentx-native-disagg-data/native/topo64-v11-p8d8-kv-c64/summary.json) | 64 | 1,615 | 1.35 | 85.8976 | 4,288 / 0 | 0.000% | Completed forecast |
| [Default KV](agentx-native-disagg-data/native/topo64-v11-p8d8-kv-c256/summary.json) | 256 | 7,399 | 3.13 | 58.0122 | 17,289 / 11 | 0.064% | Forecast with errors |
| [RR](agentx-native-disagg-data/native/topo64-v11-p8d8-rr-c16/summary.json) | 16 | 267 | 8.35 | 41.2092 | 584 / 0 | 0.000% | Completed forecast |
| [RR](agentx-native-disagg-data/native/topo64-v11-p8d8-rr-c64/summary.json) | 64 | 1,561 | 9.17 | 39.6434 | 4,065 / 0 | 0.000% | Completed forecast |
| [RR](agentx-native-disagg-data/native/topo64-v11-p8d8-rr-c256/summary.json) | 256 | 4,740 | 32.53 | 6.9169 | 11,161 / 3,710 | 24.948% | Diagnostic only |

**RR256 is an admission-failure diagnostic, not a usable capacity prediction:** 3,710 of 14,871 profiling requests fail (24.95%). Its TTFT and I90 describe successful requests only, so dropping those errors would make the curve misleading. KV256 also has 11 errors (0.064%). Both are crosses outside the zero-error prediction lines. Worker logs contain handoff-session-limit failures; counts and log hashes are preserved with each run. The scenario-valid stamp alone does not validate the serving model.

The lower-concurrency native points show the direction of KV's latency advantage. They do not validate the measured D88 knee, credit-1.5 tuning, or C480/C576 SLO choices. Those require successful native runs at the same hardware concurrency and settings.

### 4.3 Matched disagg comparison: 12P+6D, 72 GPUs, KV only

Two completed native checks **do** have matching real hardware jobs: the earlier **12P+6D TP4 / 72-GPU KV** recipe at **C192 and C384**. These two historical hardware references are additional to the 37 agg/D88 jobs in sections 2–3; they are not substituted for 64-GPU D88 or for RR. Structured pool counts and launch commands establish 72 GPUs; the original topology JSON retains a stale 64-GPU sentence, documented in the source manifest.


![Matched 72-GPU 12P+6D KV hardware and native throughput, TTFT and E2E interactivity at C192 and C384](agentx-serving-perf-report-simulation-disagg-p12d6.png)

[SVG](agentx-serving-perf-report-simulation-disagg-p12d6.svg) · [PDF](agentx-serving-perf-report-simulation-disagg-p12d6.pdf)


| Matched job | Total/GPU real / native | Throughput error | TTFT p95 real / native (s) | TTFT error | I90 real / native | Errors real / native |
| --- | --- | --- | --- | --- | --- | --- |
| [C192 hardware](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789560095_alisachen-n3u-mnnvl-126-agentx-kv-c192) / [native](agentx-native-disagg-data/native/disagg-p12d6-kv-c192-calibrated-v11/summary.json) | 4,511 / 4,461 | -1.10% | 1.768 / 1.778 | +0.54% | 66.6203 / 66.7212 | 0 / 0 |
| [C384 hardware](https://console.cloud.google.com/storage/browser/alisachen-models/perf/1789566460_alisachen-n3u-mnnvl-126-agentx-kv-c384) / [native](agentx-native-disagg-data/native/disagg-p12d6-kv-c384-calibrated-v11/summary.json) | 8,637 / 8,693 | +0.65% | 2.580 / 2.728 | +5.74% | 48.5505 / 51.9892 | 0 / 0 |

The throughput errors are **−1.10% at C192** and **+0.65% at C384**; TTFT p95 errors are **+0.54%** and **+5.74%**. Both native runs complete without profiling errors. Their 185/349 warmup requests match the respective hardware source/turn/input-token identities; the report independently recomputes request-level TTFT and I90. The timing coefficients remain those derived for agg, and transfer bandwidth/prefill cache sizing remain assumptions. Two checks on this earlier topology do not establish general disagg accuracy or a knee.

### 4.4 The remaining C480 tuning gap

All **12 C480 native flag attempts failed during warmup** with `mocker handoff session limit reached`; no full profiling result exists at that load. The grid tested credits 0.6/0.8/1.0 and did not produce a forecast for the hardware credit-1.5 winner. Failed warmups are not plotted as zero throughput or as hardware limits.

Fix native handoff admission/backpressure while preserving the intended batch limits, then collect matched D88 default-KV/RR points at C96/C192 and C480. Only after those complete should the tuning grid and C576 SLO choice be evaluated. Validate transfer contention and the prefill cache allocation alongside that work.

[Native disagg input manifest](agentx-native-disagg-data/manifest.json) · [numeric request provenance](agentx-native-disagg-data/request-metrics/manifest.json) · [C480 failure evidence](agentx-serving-perf-data/source/native-c480-status.json) · [C480 sweep manifest](../sim-results/agentx_disagg_c480_native_20260919/plan.json)


## 5. How we simulate performance: AIC, DynoSim and recipe selection

### 5.1 What each component contributes

| Component | Input and role | Output used here |
| --- | --- | --- |
| AIPerf | Recorded sessions, tokenizer, seed, lane count and replay rules | Actual request timing, branches, warmup, recycling, streaming metrics and request exports |
| AIConfigurator (AIC) | Model, accelerator, backend, precision and parallelism; pass shape or search constraints | Estimated prefill/decode forward-pass duration; candidate worker shapes, batch limits, P:D ratios, memory estimates and generated deployment/benchmark files |
| Native Dynamo / DynoSim mocker | Requests, router flags, worker topology, scheduler rules and cache capacity | Placement, prefix reuse, admission, batch/chunk composition, cache state and time spent waiting around model work |
| Hardware validation | AIPerf against the real serving fleet | Determines whether a candidate's throughput, latency tails, errors and SLO result actually reproduce |

AIC supplies the duration of model work; the scheduler determines what work is in each pass and when it can execute. This division is described in [NVIDIA's DynoSim explanation](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/) and [AIC 0.11.0](https://github.com/ai-dynamo/aiconfigurator/tree/v0.11.0). The configuration tables below describe **our saved experiment**, not every capability of the current upstream projects.

Our native route is:

```text
AIPerf AgentX (normal wall clock, streaming HTTP)
    → Dynamo 1.4.2 frontend and actual KV / RR routing
    → native SGLang mocker scheduler and hybrid cache state
    → AIC 0.11.0 forward-pass timing + shared calibration
    → streaming response timing → AIPerf TTFT / E2E / throughput
```

The native V10 build is a local patched candidate, not an NVIDIA release named V10. It uses speedup **1.0** for prefill and decode. AIPerf still owns the AgentX replay; no chosen cache-hit rate or per-concurrency throughput multiplier substitutes for that replay.

### 5.2 The detailed configurations AIC generated

The repository preserves **SILICON-mode AIC 0.11.0 solves** for 24- and 72-GPU budgets using SGLang **0.5.14**, fixed **ISL 96,000 / OSL 900**, context 262,144 and chunked prefill. The **warm** bracket sets a synthetic 92,000-token prefix (95.83% of ISL) and a 5,000 ms TTFT solver target; the **cold** bracket sets prefix 0 and a 30,000 ms target. Both use a **10 ms TPOT** target. These are fixed-shape solver inputs and predicted point estimates, not AgentX session concurrency, measured cache hit, or p95 SLO results.

The table shows each saved **rank-1 generated recipe**. MoE parallelism is written as tensor-parallel size / expert-parallel size. The predicted throughput column counts **output tokens only** and uses the GPUs active in that candidate; it must not be compared directly with our total-token AgentX throughput.


| AIC case / generated config | GPU budget / used | Worker recipe | MoE TP / EP | Batch per worker | Predicted output tok/s/active GPU | Predicted TTFT s / TPOT ms |
| --- | --- | --- | --- | --- | --- | --- |
| [warm24 agg](../sim-results/aic-nvfp4/warm24/n3u-fp4_gb300_sglang_isl96000_osl900_ttft5000_tpot10_356262/agg/top1/generator_config.yaml) | 24 / 24 | 3 × TP8 | 4 / 2 | 16 | 197.63 | 0.356 / 9.613 |
| [warm24 disagg](../sim-results/aic-nvfp4/warm24/n3u-fp4_gb300_sglang_isl96000_osl900_ttft5000_tpot10_356262/disagg/top1/generator_config.yaml) | 24 / 24 | 1P × TP8 + 2D × TP8 | P 1/8; D 4/2 | P 1; D 18 | 139.31 | 0.167 / 9.917 |
| [cold24 agg](../sim-results/aic-nvfp4/cold24/n3u-fp4_gb300_sglang_isl96000_osl900_ttft30000_tpot10_427557/agg/top1/generator_config.yaml) | 24 / 24 | 12 × TP2 | 1 / 2 | 1 | 59.48 | 28.199 / 8.364 |
| [cold24 disagg](../sim-results/aic-nvfp4/cold24/n3u-fp4_gb300_sglang_isl96000_osl900_ttft30000_tpot10_427557/disagg/top1/generator_config.yaml) | 24 / 24 | 4P × TP4 + 2D × TP4 | P 1/4; D 2/2 | P 1; D 7 | 54.37 | 4.108 / 9.877 |
| [warm72 agg](../sim-results/aic-nvfp4/warm72/n3u-fp4_gb300_sglang_isl96000_osl900_ttft5000_tpot10_73217/agg/top1/generator_config.yaml) | 72 / 72 | 9 × TP8 | 4 / 2 | 16 | 197.63 | 0.356 / 9.613 |
| [warm72 disagg](../sim-results/aic-nvfp4/warm72/n3u-fp4_gb300_sglang_isl96000_osl900_ttft5000_tpot10_73217/disagg/top1/generator_config.yaml) | 72 / 64 | 1P × TP8 + 7D × TP8 | P 1/8; D 8/1 | P 1; D 12 | 136.65 | 0.167 / 8.809 |
| [cold72 agg](../sim-results/aic-nvfp4/cold72/n3u-fp4_gb300_sglang_isl96000_osl900_ttft30000_tpot10_375345/agg/top1/generator_config.yaml) | 72 / 72 | 36 × TP2 | 1 / 2 | 1 | 59.48 | 28.199 / 8.364 |
| [cold72 disagg](../sim-results/aic-nvfp4/cold72/n3u-fp4_gb300_sglang_isl96000_osl900_ttft30000_tpot10_375345/disagg/top1/generator_config.yaml) | 72 / 72 | 11P × TP4 + 7D × TP4 | P 2/2; D 1/4 | P 1; D 6 | 54.57 | 4.081 / 9.750 |

All listed candidates use PP1/DP1, FP8 static GEMM, NVFP4 MoE, FP8 KV/FMHA and half-precision communication in the saved AIC search. The generator emits `generator_config.yaml`, worker launch scripts, Kubernetes deployment/benchmark YAML and benchmark scripts; the ranking CSV and experiment YAML retain the objective and search inputs. Exact configurations, hashes and links are in the [methodology data](agentx-serving-perf-report-methodology.json).

**The proposed and deployed recipes differ.** Warm24 agg's top generated proposal is **3×TP8**, whereas our real agg fleet is **6×TP4**. Warm72 disagg's top proposal is **1P+7D at TP8**, using **64 of the 72 budgeted GPUs**; our tested 64-GPU D88 fleet is **8P+8D at TP4**. Cold and warm searches select very different P:D ratios. AIC therefore supplies starting shapes and timing data; these saved solves do not certify the measured TP4 recipes, router flags, or AgentX knees. The generated templates also contain old runtime defaults and local model paths, so they require reconciliation with the serving manifests rather than direct deployment unchanged.

### 5.3 Exact native agg and disagg simulation inputs

The timing identity below is common: **AIC 0.11.0, GB300, SGLang 0.5.14 tables, TP4, MoE TP1/EP4, attention DP1, FP8 GEMM, NVFP4 MoE, FP8 KV and BF16 FMHA**. The BF16 FMHA choice differs from the historical AIC search's FP8 FMHA. Hardware recipes specify SGLang 0.5.16; database/framework mismatch is part of the remaining modeling gap.


| Native configuration | Agg V10 | Disagg prefill V11 extension | Disagg decode V11 extension |
| --- | --- | --- | --- |
| Workers / total GPUs | 6 agg / 24 | 8 prefill / 32 | 8 decode / 32 |
| Role | aggregated | prefill | decode |
| Maximum sequences | 16 | 8 | 64 |
| Attention blocks × block size | 443,697 × 64 | 443,697 × 64 | 809,406 × 64 |
| Attention token capacity per worker | 28,396,608 | 28,396,608 | 51,801,984 |
| Token budget / chunk size | 16,384 / 16,384 | 16,384 / 16,384 | 16,384 / 16,384 |
| Prefix caching | True | True | False |
| Mamba state slots | 769 | 769 | 64 |
| State cache chunk / tracking interval | 64 / 256 tokens | 64 / 256 tokens | 64 / 256 tokens |
| Transfer bandwidth per rank | No handoff | 64 GB/s assumed | 64 GB/s assumed |
| Transfer payload per rank | No handoff | 3,072 bytes/input token + 101,990,400 state bytes | Same payload |
| Decode tokens reserved at handoff | Not applicable | 0 | 512 |
| Config source | [agg engine JSON](../reports/agentx-agg-kv-rr-data/native-v10/agg-kv-c192-calibrated-v10/engine.json) | [prefill engine JSON](../sim-results/agentx_disagg_c480_native_20260919/configs/prefill-engine.json) | [decode engine JSON](../sim-results/agentx_disagg_c480_native_20260919/configs/decode-engine.json) |

**Cache size provenance:** the agg attention allocation (443,697 pages ×64 =28,396,608 tokens/worker) comes from observed serving metadata. The 769 Mamba slots and checkpoint settings remain assumptions because the live state pool was not captured. Disagg prefill inherits that allocation as an assumption; decode uses 809,406 observed attention pages and 64 state/request slots in the saved model. AIC did not measure these fleet cache allocations. Cache-hit rate emerges from the replay, placement and finite cache state.

**Transfer provenance:** the 64 GB/s value is a per-rank modeling assumption, not measured Mooncake bandwidth. The native handoff moves the full prompt's modeled KV plus recurrent state; independent delays omit shared-link contention. These assumptions need direct transfer/cache measurements and a matched D88 hardware check. The saved input files are linked above; the failed sweep is not evidence that these values are accurate.

### 5.4 Shared timing calibration and what DynoSim tells us

The same coefficients are used for every policy and concurrency:

```text
prefill_ms = 1.7050018888 × AIC_prefill_ms
           + 165.5695513 × batch × new_tokens × (prefix_tokens + new_tokens / 2) / 1e9
decode_ms  = AIC_decode_ms + 0.288 + 0.03145728 × ready_decode_requests
```

The prefill fit uses **93 isolated one-token RR192 hardware warmup requests**. Decode additions model missing recurrent-state work from geometry/bandwidth and launch-overhead assumptions; they are not directly measured Nemotron kernel times. See the [prefill fit](agentx-agg-kv-rr-data/native-v10/prefill_calibration_v2.json), [decode derivation](agentx-agg-kv-rr-data/native-v10/decode_mamba_timing_candidate_v10.json), and [native build reconstruction](agentx-agg-kv-rr-data/native-v10/v10_patch_reconstruction.json).

| Recipe question | What the completed native evidence tells us | What still needs hardware or a simulator fix |
| --- | --- | --- |
| Agg 6×TP4, default KV versus RR | Reproduces the sampled C192 throughput peak and C384 decline, and the direction of the KV advantage. Eight holdouts average 1.7% absolute throughput error. | TTFT tails remain biased; the simulator falsely passes default KV192 under TTFT-only. |
| Agg router tuning | Frozen V10 reproduces the original scale-3/credit-0.8 throughput benefit and temperature-0.5 loss. | Tuned KV192 falsely passes combined SLO in simulation. Decay variants lack native counterparts. Use measured C96 under both limits; measure C144 next. |
| Disagg 8P+8D TP4 | Six completed native runs at C16/C64/C256 show the sampled routing trends, with separate admission/cache budgets and transfer timing. | No native point matches the hardware concurrency grid; RR256 has 24.95% errors. All C480 tuning attempts fail warmup, so no native tuning ranking or high-load SLO capacity is established. |
| Earlier disagg 12P+6D TP4 | Two matched 72-GPU KV checks have throughput errors of −1.10%/+0.65% and TTFT errors of +0.54%/+5.74%. | These are two historical KV points, not validation of D88, RR, or the full latency boundary. Transfer/cache assumptions still need measurement. |
| Choosing TP or the P:D ratio | AIC supplies candidate shapes; a working replay model can compare them under the workload. | The current evidence does not establish that 6×TP4 or 8P+8D is globally optimal. Compare candidates at fixed total GPUs and validate on hardware. |

Fix native disagg admission/backpressure while preserving the intended batch limits, complete one full matched baseline, and only then repeat the flag grid. Validate transfer timing/contended bandwidth, prefill cache capacity and SGLang-version effects. For agg, use the frozen model to prioritize measurements; the real-job SLO remains the decision source.

### 5.5 Artifacts, validation and regeneration

The [hardware manifest](agentx-serving-perf-data/manifest.json) records the collection timestamp, source commit, summaries, source GCS paths and hashes. Numeric per-request projections retain original full-export hashes without prompt/response text. The generator checks the shared replay controls, request/error counts, raw TTFT and E2E percentiles, I90 normalization and the native build/engine identity. Methodology artifacts add the exact saved AIC recipes and native configuration hashes; no new AIC solve, simulation or hardware job is launched when regenerating this report.

From the repository root, with NumPy, Matplotlib, PyYAML and markdown-it-py installed:

```bash
python kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_agentx_serving_report.py
```

The earlier [agg report](agentx-agg-kv-rr-report.md) and [disagg report](agentx-disagg-kv-rr-report.md) remain dated snapshots. This report preserves all 37 current agg/D88 hardware jobs and all 12 original native agg comparisons. Section 4 additionally preserves eight existing native disagg runs and two historical 72-GPU hardware references; errored profiling runs remain visible as diagnostics, and failed warmups contribute no performance point.
