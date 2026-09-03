#!/usr/bin/env python3
"""Generate the BigQuery INSERT for ml-workload-benchmarks.benchmark_dataset_v2
.inference_run_summary from the final DSR1-FP4 + DSv4-FP4 slims (18 rows).
Writes reports/inference_run_summary_insert.sql. Conventions:
  - per_chip metrics divide by TOTAL chips used; per-decode-GPU values go in comments
  - TP/EP/DP columns describe the DECODE worker; prefill parallelism in slice fields
  - interactivity = 1000 / mean TPOT (matches InferenceX identity)
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, '..')
RS1 = os.path.join(ROOT, 'dsr1-sweep', 'results-summary')
RS4 = os.path.join(ROOT, 'dsv4-sweep', 'results-summary')
REPO = 'https://github.com/alisachen-google/gcp-dynamo-cuj/tree/main/dynamo-disagg-sweep'

# (pid, conc, slim glob, transport, plane, pg, dg, tot, nodes, prefill_workers, gap_note)
# All-RDMA arm (user request 2026-09-03): every DSR1 row is the GPUDirect-RDMA KV run
# (Table 2 of benchmark-report-rdma-kv.md), not the mixed optimal finals.
DSR1 = [
    ('p1', 4,    'llr-results_concurrency_4_*',    'RDMA', 'NATS', 4, 16, 20, 5, 1,  '+8.1% vs InferenceMax'),
    ('p2', 8,    'llr-results_concurrency_8_*',    'RDMA', 'NATS', 4, 16, 20, 5, 1,  '+8.2% vs InferenceMax'),
    ('p3', 32,   'llr-results_concurrency_32_*',   'RDMA', 'NATS', 4, 16, 20, 5, 1,  '+1.5% vs InferenceMax'),
    ('p4', 64,   'llr-results_concurrency_64_*',   'RDMA', 'NATS', 4, 16, 20, 5, 1,  '+3.4% vs InferenceMax'),
    ('p5', 512,  'm4r-results_concurrency_512_*',  'RDMA', 'NATS', 24, 48, 72, 18, 6, '+11.1% vs InferenceMax'),
    ('p6', 2048, 'm4r-results_concurrency_2048_*', 'RDMA', 'NATS', 24, 48, 72, 18, 6, '-1.2% vs InferenceMax'),
    ('p7', 4096, 'm4r-results_concurrency_4096_*', 'RDMA', 'NATS', 24, 48, 72, 18, 6, '+8.5% vs InferenceMax'),
    ('p8', 2048, 'mxr-results_concurrency_2048_*', 'RDMA', 'NATS', 40, 32, 72, 18, 10, '-0.6% vs InferenceMax'),
]
DSV4 = [  # (pid, conc, pg, dg, tot, nodes, prefill_workers, drift, gap_note)
    ('p1', 1,    4, 4,  8,  2, 1, False, '+11.7% vs InferenceMax; TP4+TP4 point'),
    ('p2', 8,    4, 24, 28, 7, 1, False, '+18.6% vs InferenceMax'),
    ('p2', 32,   4, 24, 28, 7, 1, False, '+9.5% vs InferenceMax'),
    ('p2', 64,   4, 24, 28, 7, 1, False, '+8.0% vs InferenceMax'),
    ('p3', 256,  4, 8,  12, 3, 1, False, '+0.3% vs InferenceMax'),
    ('p4', 256,  4, 16, 20, 5, 1, False, '+0.0% vs InferenceMax (exact reproduction)'),
    ('p5', 512,  8, 8,  16, 4, 2, True,  '+1.8% vs InferenceMax'),
    ('p6', 1024, 16, 8, 24, 6, 4, True,  '+0.5% vs InferenceMax'),
    ('p7', 4096, 24, 8, 32, 8, 6, True,  '-0.4% vs InferenceMax'),
    # p8 handled specially: throughput from measured record, latency from parity rerun
]


def load_slim(rsdir, pat):
    files = [f for f in glob.glob(os.path.join(rsdir, pat)) if 'INVALID' not in f]
    if len(files) > 1:  # prefer mult10 finals over superseded base-methodology runs
        files = [f for f in files if '_mult10' in f] or files
    assert len(files) == 1, f'{pat}: {files}'
    return json.load(open(files[0]))


def n(v, f='.2f'):
    return 'NULL' if v is None else format(v, f)


def row(run_id, run_name, model_id, ckpt, tok, precision, conc, num_prompts, tp, ep, dp,
        nodes, tot, pslices, ssize, d, comment, attn):
    dur = d.get('duration')
    outs = d.get('output_throughput'); ins = d.get('total_input_tokens', 0) / dur if dur else None
    tots = d.get('total_token_throughput')
    inter = 1000.0 / d['mean_tpot_ms'] if d.get('mean_tpot_ms') else None
    rps = d.get('completed', num_prompts) / dur if dur else None
    return f"""(
  '{run_id}', 'gcp-dynamo-cuj (DynamoBench)', TRUE, 'server', '{run_name}',
  '{model_id}', 'Dynamo+SGLang', 'a4x_max', 'a4x_max', 'a4x_max',
  'serving-disagg-8k1k', '{ckpt}', '{tok}', 'random (sa-bench)',
  {num_prompts}, {conc}, NULL, 8192, 1024, TRUE, '{precision}',
  {tp}, 1, {ep}, {dp}, TRUE, TRUE, {3*conc},
  {nodes}, 4, {tot}, {pslices}, 1, '{ssize}', 'disaggregated', 'GB300 NVL72',
  TRUE, {n(dur)}, {n(rps, '.4f')},
  {n(outs)}, {n(ins)}, {n(outs/tot if outs else None, '.4f')}, {n(tots)},
  {n(d.get('mean_e2el_ms'))}, {n(d.get('median_e2el_ms'))}, {n(d.get('p99_e2el_ms'))}, {n(d.get('std_e2el_ms'))},
  {n(d.get('mean_ttft_ms'))}, {n(d.get('median_ttft_ms'))}, {n(d.get('p99_ttft_ms'))}, {n(d.get('std_ttft_ms'))},
  {n(d.get('mean_tpot_ms'))}, {n(d.get('median_tpot_ms'))}, {n(d.get('p99_tpot_ms'))}, {n(d.get('std_tpot_ms'))},
  {n(d.get('mean_itl_ms'))}, {n(d.get('median_itl_ms'))}, {n(d.get('p99_itl_ms'))}, {n(d.get('std_itl_ms'))},
  {n(inter, '.2f')}, {('NULL' if not attn else "'" + attn + "'")},
  'alisachen', CURRENT_TIMESTAMP(),
  '{comment}'
)"""


COLS = """run_id, run_source, is_run_externally_visible, run_type, run_name,
  model_id, inference_software_id, hardware_id, prefill_hardware_id, decode_hardware_id,
  workload_type, workload_checkpoint_path, workload_tokenizer_name_or_path, workload_dataset_name_or_path,
  workload_num_prompts, workload_global_batch_size, workload_request_rate_qps,
  workload_max_input_length, workload_max_output_length, workload_quantization_enabled, workload_precision_config,
  workload_tensor_parallel_size, workload_pipeline_parallel_size, workload_expert_parallel_size, data_parallel_size,
  workload_is_disaggregated_compute, workload_has_warmup, workload_num_warmups,
  hardware_num_nodes, hardware_num_chips_per_node_used, hardware_total_chips_used,
  hardware_prefill_num_slices, hardware_decode_num_slices, hardware_slice_size, hardware_serving_type, hardware_runtime_topology,
  result_success, result_duration_seconds, metrics_achieved_request_rate_rps,
  metrics_output_tokens_per_sec, metrics_input_tokens_per_sec, metrics_output_tokens_per_sec_per_chip, metrics_total_tokens_per_sec,
  metrics_e2e_latency_avg_ms, metrics_e2e_latency_p50_ms, metrics_e2e_latency_p99_ms, metrics_e2e_latency_stddev_ms,
  metrics_ttft_avg_ms, metrics_ttft_p50_ms, metrics_ttft_p99_ms, metrics_ttft_stddev_ms,
  metrics_tpot_avg_ms, metrics_tpot_p50_ms, metrics_tpot_p99_ms, metrics_tpot_stddev_ms,
  metrics_itl_avg_ms, metrics_itl_p50_ms, metrics_itl_p99_ms, metrics_itl_stddev_ms,
  metrics_interactivity_tokens_per_sec_per_user, attention_backend,
  update_person_ldap, update_timestamp,
  logs_comments_string"""

rows = []
for pid, conc, pat, kv, plane, pg, dg, tot, nodes, pw, gap in DSR1:
    d = load_slim(RS1, pat)
    cm = (f'DSR1-FP4 8k1k GKE disagg final {pid} (conc {conc}): total tput/GPU {gap}; '
          f'KV={kv}, request plane={plane}; out/decode-GPU={d["output_throughput"]/dg:.1f}, '
          f'in/prefill-GPU={d["total_input_tokens"]/d["duration"]/pg:.0f}; per_chip cols divide by total {tot} chips. '
          f'Closed-loop max-concurrency={conc}, 10x measured. Configs+logs: {REPO}')
    rows.append(row(f'dsr1-fp4-a4xmax-gke-dynamo-sglang-rdmakv-{pid}-c{conc}-01',
                    f'DSR1-FP4 8k1k {pid} ({kv} KV, conc {conc})', 'deepseek-r1-0528',
                    'nvidia/DeepSeek-R1-0528-NVFP4-v2', 'nvidia/DeepSeek-R1-0528-NVFP4-v2',
                    'NVFP4', conc, 10*conc, dg, dg, dg, nodes, tot, pw,
                    f'prefill DEP4 x{pw} / decode DEP{dg} x1', d, cm, 'trtllm_mla'))

for pid, conc, pg, dg, tot, nodes, pw, drift, gap in DSV4:
    pat = f'{"drift/" if drift else ""}{pid}-results_concurrency_{conc}_*'
    d = load_slim(RS4, pat)
    cm = (f'DSv4-FP4 8k1k GKE disagg final {pid} (conc {conc}): total tput/GPU {gap}; '
          f'KV=MNNVL (mooncake), NATS plane, EAGLE MTP; out/decode-GPU={d["output_throughput"]/dg:.1f}; '
          f'per_chip cols divide by total {tot} chips.'
          f'{" Drift-labeled run (MTP template), documented in report." if drift else ""} '
          f'Closed-loop max-concurrency={conc}, 10x measured. Configs+logs: {REPO}')
    rows.append(row(f'dsv4-fp4-a4xmax-gke-dynamo-sglang-{pid}-c{conc}-01',
                    f'DSv4-FP4 8k1k {pid} (conc {conc})', 'deepseek-v4-pro',
                    'deepseek-ai/DeepSeek-V4-Pro', 'deepseek-ai/DeepSeek-V4-Pro',
                    'FP4 (mxfp4)', conc, 10*conc,
                    dg, dg, dg, nodes, tot, pw, f'prefill DEP4 x{pw} / decode {"TP" if pid=="p1" else "DEP"}{dg} x1',
                    d, cm, None))

# DSv4 p8: throughput from the 2026-08-07 measured record; latency from the parity rerun
p8 = json.load(open(os.path.join(RS4, 'p8-rerun-latency-only.extracted.json')))
m = p8['_measured_throughput_record_20260807']
d8 = dict(p8)
d8['output_throughput'] = m['out_per_decode_gpu'] * 8
d8['total_token_throughput'] = m['total_per_gpu'] * 40
d8['total_input_tokens'] = None
d8['duration'] = None
cm = ('DSv4-FP4 8k1k GKE disagg final p8 (conc 8192): total tput/GPU -3.2% vs InferenceMax '
      '(SUSPECT-JIT conservative floor, clean est ~-2%); throughput from 2026-08-07 measured record '
      f'(81919/81920 reqs), latency metrics from 2026-08-01 parity rerun; out/decode-GPU={m["out_per_decode_gpu"]:.1f}, '
      f'in/prefill-GPU={m["in_per_prefill_gpu"]:.0f}; per_chip cols divide by 40 chips. '
      f'Closed-loop max-concurrency=8192, 10x measured. Configs+logs: {REPO}')
rows.append(row('dsv4-fp4-a4xmax-gke-dynamo-sglang-p8-c8192-01',
                'DSv4-FP4 8k1k p8 (conc 8192)', 'deepseek-v4-pro',
                'deepseek-ai/DeepSeek-V4-Pro', 'deepseek-ai/DeepSeek-V4-Pro',
                'FP4 (mxfp4)', 8192, 81920, 8, 8, 8, 10, 40, 8,
                'prefill DEP4 x8 / decode DEP8 x1', d8, cm, None))

sql = (f'-- Final DSR1-FP4 (8 points) + DSv4-FP4 (10 points) GKE A4X Max disagg results\n'
       f'-- Generated {os.popen("date -u +%F").read().strip()} from DynamoBench slims; '
       f'provenance: {REPO}\n'
       f'-- Prereq (already applied): ALTER TABLE `ml-workload-benchmarks.benchmark_dataset_v2.inference_run_summary`\n'
       f'--   ADD COLUMN IF NOT EXISTS metrics_interactivity_tokens_per_sec_per_user FLOAT64;\n\n'
       f'INSERT INTO `ml-workload-benchmarks.benchmark_dataset_v2.inference_run_summary` (\n  {COLS}\n)\nVALUES\n'
       + ',\n'.join(rows) + ';\n')
out = os.path.join(ROOT, 'reports', 'inference_run_summary_insert.sql')
open(out, 'w').write(sql)
print(f'wrote {os.path.relpath(out, ROOT)}: {len(rows)} rows, {len(sql)} bytes')
