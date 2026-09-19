"""Silicon knee check: is a completed bench point at/pre-knee, or post-knee?

Reads the per-request export (profile_export.jsonl) of an aiperf artifact dir on
GCS and applies the queue-drain criteria to the measured window:
  - within-run stationarity: TTFT p50 of the first vs last time-quarter
  - saturation level: overall TTFT p50 vs an absolute standing-queue bound
  - health: cancelled/errored request rate
Verdict AT/PRE-KNEE -> exit 0; POST-KNEE -> exit 2 (halts the sweep sequencer).

Usage: knee_check.py <job-name-substring>   # picks latest matching GCS dir
"""
import json, subprocess, sys, tempfile
from pathlib import Path

BUCKET = "gs://alisachen-models/perf/"
GROWTH_RATIO = 1.5      # last-quarter p50 > 1.5x first-quarter p50 ...
GROWTH_ABS_MS = 2000    # ... and by more than 2s absolute => accumulating
SATURATION_P50_MS = 20000  # standing queue this deep is saturation regardless
MAX_BAD_RATE = 0.05

job = sys.argv[1]
ls = subprocess.run(["gcloud", "storage", "ls", BUCKET], capture_output=True, text=True)
dirs = [l for l in ls.stdout.splitlines() if job in l]
if not dirs:
    print(f"KNEE-CHECK ERROR: no artifact dir matching {job}"); sys.exit(3)
root = sorted(dirs)[-1]
sub = subprocess.run(["gcloud", "storage", "ls", root], capture_output=True, text=True)
runs = [l for l in sub.stdout.splitlines()
        if l.rstrip("/").split("/")[-1][:1].isupper() and "cache-warmup" not in l
        and "VARIANT" not in l]
if not runs:
    print(f"KNEE-CHECK ERROR: no measured-run dir under {root}"); sys.exit(3)
run = sorted(runs)[-1]

with tempfile.TemporaryDirectory() as td:
    dst = Path(td) / "profile_export.jsonl"
    cp = subprocess.run(["gcloud", "storage", "cp", run + "profile_export.jsonl", str(dst)],
                        capture_output=True, text=True)
    if not dst.exists():
        print(f"KNEE-CHECK ERROR: fetch failed: {cp.stderr[-200:]}"); sys.exit(3)
    recs, bad = [], 0
    for line in dst.open():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            continue
        md = r.get("metadata", {})
        if md.get("benchmark_phase") != "profiling":
            continue
        if md.get("was_cancelled") or "time_to_first_token" not in r.get("metrics", {}):
            bad += 1
            continue
        recs.append((md["request_start_ns"], r["metrics"]["time_to_first_token"]["value"]))

if len(recs) < 40:
    print(f"KNEE-CHECK ERROR: only {len(recs)} usable records"); sys.exit(3)
recs.sort()
bad_rate = bad / (bad + len(recs))
t0, t1 = recs[0][0], recs[-1][0]
span = (t1 - t0) or 1


def p50(xs):
    s = sorted(xs); return s[len(s) // 2]


q1 = [v for t, v in recs if (t - t0) / span < 0.25]
q4 = [v for t, v in recs if (t - t0) / span >= 0.75]
h1, h4 = p50(q1), p50(q4)
overall = p50([v for _, v in recs])
growing = h4 > GROWTH_RATIO * h1 and (h4 - h1) > GROWTH_ABS_MS
saturated = overall > SATURATION_P50_MS
unhealthy = bad_rate > MAX_BAD_RATE
verdict = "POST-KNEE" if (growing or saturated or unhealthy) else "AT/PRE-KNEE"
print(f"KNEE-CHECK {job}: {verdict} | n={len(recs)} bad={bad_rate:.1%} "
      f"ttft_p50 overall={overall/1000:.2f}s q1={h1/1000:.2f}s q4={h4/1000:.2f}s "
      f"({'growing' if growing else 'stationary'}"
      f"{', saturated' if saturated else ''}{', unhealthy' if unhealthy else ''}) "
      f"[{run.rsplit('/', 2)[-2]}]")
sys.exit(2 if verdict == "POST-KNEE" else 0)
