#!/usr/bin/env python3
"""Audit observed AgentX source identities; does not dispatch or simulate requests.

Warmup checks include branch identity and prompt length. Profiling overlap is a
diagnostic: closed-loop completion order changes the subset completed in an hour.
Token counts alone do not establish byte-identical request payloads.
"""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import time


def read_records(path):
    rows = []
    with path.open() as stream:
        for line in stream:
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                # Only tolerate an incomplete final line of an active export.
                if line.endswith("\n") or stream.read(1):
                    raise
                break
    return rows


def identity(record):
    m = record["metadata"]
    return (m.get("source_trace_id"), m.get("conversation_id"),
            m.get("source_kind"), m.get("source_outer_idx"), m.get("turn_index"))


def metric(record, name):
    return record.get("metrics", {}).get(name, {}).get("value")


def phase_records(rows, phase):
    return [r for r in rows if r.get("metadata", {}).get("benchmark_phase") == phase]


def successful(record):
    return (not record.get("error")
            and not record["metadata"].get("was_cancelled")
            and not record["metadata"].get("context_overflow_skip")
            and metric(record, "usage_prompt_tokens") is not None
            and metric(record, "usage_completion_tokens") is not None)


def audit(hardware, simulation):
    real_warmup = phase_records(hardware, "warmup")
    sim_warmup = phase_records(simulation, "warmup")
    counters = [Counter(identity(r) + (metric(r, "usage_prompt_tokens"),)
                        for r in rows) for rows in [real_warmup, sim_warmup]]
    expected, actual = counters
    warmup = dict(
        hardware_requests=len(real_warmup), simulation_requests=len(sim_warmup),
        identity_fields=["source_trace_id", "conversation_id", "source_kind",
                         "source_outer_idx", "turn_index", "usage_prompt_tokens"],
        inputs_match=bool(expected) and expected == actual,
        missing_count=sum((expected - actual).values()),
        unexpected_count=sum((actual - expected).values()),
        missing_examples=list((expected - actual).elements())[:10],
        unexpected_examples=list((actual - expected).elements())[:10],
        simulation_all_successful=bool(sim_warmup) and all(map(successful, sim_warmup)),
        simulation_all_one_token=bool(sim_warmup) and all(
            metric(r, "usage_completion_tokens") == 1 for r in sim_warmup),
    )
    warmup["all_checks_pass"] = all(warmup[k] for k in
        ["inputs_match", "simulation_all_successful", "simulation_all_one_token"])

    groups = []
    for rows in [hardware, simulation]:
        grouped = defaultdict(list)
        for r in phase_records(rows, "profiling"):
            if successful(r):
                grouped[identity(r)].append(r)
        groups.append(grouped)
    real, sim = groups
    common = real.keys() & sim.keys()
    single = [key for key in common if len(real[key]) == len(sim[key]) == 1]
    input_deltas, output_deltas = Counter(), Counter()
    examples = []
    for key in sorted(single, key=str):
        a, b = real[key][0], sim[key][0]
        di = metric(b, "usage_prompt_tokens") - metric(a, "usage_prompt_tokens")
        do = metric(b, "usage_completion_tokens") - metric(a, "usage_completion_tokens")
        input_deltas[di] += 1
        output_deltas[do] += 1
        if (di or do) and len(examples) < 15:
            examples.append(dict(identity=key, prompt_token_delta=di, output_token_delta=do))
    profiling = dict(
        scope="Successful completed requests only; input lengths are not payload hashes. Runtime lane/cache-bust markers can change prompt token counts.",
        hardware_successes=sum(map(len, real.values())),
        simulation_successes=sum(map(len, sim.values())),
        hardware_unique_source_identities=len(real),
        simulation_unique_source_identities=len(sim),
        common_unique_source_identities=len(common),
        common_identities_occurring_once_in_each=len(single),
        prompt_token_delta_counts=dict(sorted(input_deltas.items())),
        output_token_delta_counts=dict(sorted(output_deltas.items())),
        differing_count_examples=examples,
    )
    return dict(warmup=warmup, profiling=profiling)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardware-records", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    simulated = args.run_dir / "artifacts/profile_export.jsonl"
    report = audit(read_records(args.hardware_records), read_records(simulated))
    report.update(observed_unix=time.time(), hardware_records=str(args.hardware_records),
                  run_dir=str(args.run_dir),
                  run_state=json.loads((args.run_dir / "status.json").read_text())["state"],
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"run": args.run_dir.name,
                      "warmup_pass": report["warmup"]["all_checks_pass"],
                      "warmup_requests": report["warmup"]["simulation_requests"],
                      "profiling_pairs": report["profiling"]["common_identities_occurring_once_in_each"],
                      "prompt_deltas": report["profiling"]["prompt_token_delta_counts"],
                      "output_deltas": report["profiling"]["output_token_delta_counts"]}))


if __name__ == "__main__":
    main()
