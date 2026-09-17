#!/usr/bin/env python3
"""Execute independent full aiperf replays from a saved targets file."""

import argparse
import concurrent.futures
import json
import subprocess
import sys
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("aiperf-source", "arrow-dir", "tokenizer", "targets", "run-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--points", help="Comma-separated subset; default all targets")
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--duration", type=float, default=3600)
    parser.add_argument("--cache-capacity-tokens", type=int)
    args = parser.parse_args()
    assert args.jobs > 0
    targets = json.loads(args.targets.read_text())["points"]
    selected = (
        set(args.points.split(",")) if args.points else {p["id"] for p in targets}
    )
    assert selected <= {p["id"] for p in targets}, "Unknown point ID"
    points = [p["id"] for p in targets if p["id"] in selected]
    args.run_root.mkdir(parents=True, exist_ok=True)
    for point in points:
        folder = args.run_root / point
        if folder.exists() and any(folder.iterdir()):
            raise ValueError(f"Artifact directory must be empty: {folder}")

    def run(point):
        command = [
            sys.executable,
            str(Path(__file__).with_name("dynosim_aiperf_replay.py")),
        ]
        for name in ("aiperf_source", "arrow_dir", "tokenizer", "targets"):
            command.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
        command.extend(
            [
                "--point",
                point,
                "--artifact-dir",
                str(args.run_root / point),
                "--duration",
                str(args.duration),
            ]
        )
        if args.cache_capacity_tokens is not None:
            command.extend(["--cache-capacity-tokens", str(args.cache_capacity_tokens)])
        start = time.perf_counter()
        print(f"START {point}", flush=True)
        with (args.run_root / f"{point}.log").open("w") as stream:
            result = subprocess.run(
                command, stdout=stream, stderr=subprocess.STDOUT, check=False
            )
        row = {
            "point": point,
            "exit_code": result.returncode,
            "wall_seconds": time.perf_counter() - start,
            "command": command,
        }
        print(
            f"END {point} exit={result.returncode} wall_s={row['wall_seconds']:.1f}",
            flush=True,
        )
        return row

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(run, point) for point in points]
        for future in concurrent.futures.as_completed(futures):
            results.append(future.result())
            (args.run_root / "sweep-execution.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
    if any(row["exit_code"] for row in results):
        raise SystemExit("One or more replays failed; inspect point logs")


if __name__ == "__main__":
    main()
