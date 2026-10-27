#!/usr/bin/env python3
"""Summarize SGLang prefill/decode telemetry of the DSR1 NVFP4 mid-curve run per benchmark window.

Usage: analyze-midcurve-telemetry.py --logs <dir with dsr1-m4-*.log> --results <results-summary dir>

Benchmark windows come from the sa-bench result JSONs (end = `date`, start = end - `duration`).
Decode lines are logged every `decode-log-interval` (1000) steps per DP rank; prefill lines once
per prefill batch on TP0 of each worker.
"""

import argparse
import datetime
import glob
import json
import re
import statistics as st
from pathlib import Path

DECODE = re.compile(
    r"\[\d{4}-\d\d-\d\d (\S+) DP(\d+) .*?Decode batch, #running-req: (\d+), #token: \d+, "
    r"token usage: ([\d.]+), pre-allocated usage: ([\d.]+), #prealloc-req: \d+, "
    r"#transfer-req: (\d+), #retracted-req: (\d+), cuda graph: (\w+), "
    r"gen throughput \(token/s\): ([\d.]+)"
)
PREFILL = re.compile(
    r"\[\d{4}-\d\d-\d\d (\S+) TP0\] Prefill batch, #new-seq: \d+, #new-token: (\d+), .*?"
    r"#queue-req: (\d+),"
)
ANSI = re.compile(r"\x1b\[[0-9;]*m")


def seconds(hms):
    h, m, s = map(int, hms.split(":"))
    return h * 3600 + m * 60 + s


def windows(results):
    out = {}
    for path in sorted(Path(results).glob("m4-results_concurrency_*.json")):
        d = json.loads(path.read_text())
        end = datetime.datetime.strptime(d["date"], "%Y%m%d-%H%M%S")
        start = end - datetime.timedelta(seconds=d["duration"])
        tag = path.stem.split("concurrency_")[1].split("_gpus")[0]
        tag += "-tuned" if path.stem.endswith("tuned") else ""
        out[tag] = (seconds(start.strftime("%H:%M:%S")), seconds(end.strftime("%H:%M:%S")))
    return out


def rows(path, pattern):
    with open(path, errors="replace") as f:
        return [m.groups() for m in (pattern.search(ANSI.sub("", line)) for line in f) if m]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--logs", required=True)
    parser.add_argument("--results", required=True)
    args = parser.parse_args()
    wins = windows(args.results)
    for log in sorted(glob.glob(f"{args.logs}/dsr1-m4-decode-*.log")):
        data = rows(log, DECODE)
        for tag, (a, b) in wins.items():
            r = [x for x in data if a <= seconds(x[0]) <= b]
            if not r:
                continue
            run = st.mean(int(x[2]) for x in r)
            tput = st.mean(float(x[8]) for x in r)
            print(
                f"decode  {Path(log).stem:<22} c{tag:<11} running/rank={run:5.1f} "
                f"tok/s/rank={tput:6.0f} step_ms={1000 * run / tput:5.1f} "
                f"transfer/rank={st.mean(int(x[5]) for x in r):5.1f} "
                f"kv_usage={st.mean(float(x[3]) for x in r):.2f} "
                f"retracted={sum(int(x[6]) for x in r)} "
                f"cuda_graph_always={all(x[7] == 'True' for x in r)}"
            )
    for log in sorted(glob.glob(f"{args.logs}/dsr1-m4-prefill-*.log")):
        data = rows(log, PREFILL)
        for tag, (a, b) in wins.items():
            r = [x for x in data if a <= seconds(x[0]) <= b]
            if not r:
                continue
            tokens = sum(int(x[1]) for x in r)
            queued = sum(1 for x in r if int(x[2]) > 0) / len(r)
            print(
                f"prefill {Path(log).stem[-5:]:<22} c{tag:<11} tok/s/worker={tokens / (b - a):7.0f} "
                f"tok/s/GPU={tokens / (b - a) / 4:6.0f} batches={len(r)} "
                f"queue_nonempty={queued:.2f} median_batch_tokens={st.median(int(x[1]) for x in r):.0f}"
            )


if __name__ == "__main__":
    main()
