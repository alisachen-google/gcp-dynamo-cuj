#!/usr/bin/env python3
"""Plot audited topology curves and record a reproducible sampled-knee summary."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def summarize(points):
    result = []
    for policy, arm in sorted({(p["policy"], p["arm"]) for p in points}):
        curve = sorted(
            (p for p in points if (p["policy"], p["arm"]) == (policy, arm)),
            key=lambda p: p["clients"],
        )
        peak = max(curve, key=lambda p: p["simulation"]["total_tokens_s_gpu"])
        threshold = 0.9 * peak["simulation"]["total_tokens_s_gpu"]
        knee = next(
            p for p in curve if p["simulation"]["total_tokens_s_gpu"] >= threshold
        )
        goodput = max(curve, key=lambda p: p["simulation"]["goodput_requests_s"])
        last_gain = (
            (
                curve[-1]["simulation"]["total_tokens_s_gpu"]
                / curve[-2]["simulation"]["total_tokens_s_gpu"]
                - 1
            )
            if len(curve) > 1
            else None
        )
        result.append(
            {
                "policy": policy,
                "arm": arm,
                "sampled_clients": [p["clients"] for p in curve],
                "observed_peak": {"clients": peak["clients"], **peak["simulation"]},
                "first_sample_at_90_percent_of_observed_peak": {
                    "clients": knee["clients"],
                    **knee["simulation"],
                },
                "highest_sampled_goodput": {
                    "clients": goodput["clients"],
                    **goodput["simulation"],
                },
                "last_interval_throughput_gain_percent": 100 * last_gain
                if last_gain is not None
                else None,
                "still_rising_at_upper_boundary": last_gain is None or last_gain > 0.10,
            }
        )
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="Path without extension"
    )
    args = parser.parse_args()
    points = json.loads(args.audit.read_text())["points"]
    assert points and {p["gpus"] for p in points} == {64}
    arms = sorted({p["arm"] for p in points}, key=lambda s: int(s.split(":")[0]))
    colors = dict(zip(arms, plt.get_cmap("tab10").colors, strict=False))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    figure, axes = plt.subplots(
        3, 2, figsize=(14, 11), layout="constrained", sharey="row", sharex=True
    )
    specs = [
        ("total_tokens_s_gpu", "Input + output tokens/s/GPU", False),
        ("ttft_p95_s", "TTFT p95 (seconds; log scale)", True),
        ("itl_mean_ms", "Mean ITL (ms)", False),
    ]
    for column, policy in enumerate(("kv", "rr")):
        selected = [p for p in points if p["policy"] == policy]
        clients = sorted({p["clients"] for p in points})
        for row, (metric, label, log_y) in enumerate(specs):
            axis = axes[row, column]
            for arm in arms:
                curve = sorted(
                    (p for p in selected if p["arm"] == arm), key=lambda p: p["clients"]
                )
                if not curve:
                    continue
                p, d = arm.split(":")
                axis.plot(
                    [p["clients"] for p in curve],
                    [p["simulation"][metric] for p in curve],
                    "o-",
                    color=colors[arm],
                    linewidth=1.8,
                    markersize=5,
                    label=f"{p}P : {d}D",
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(clients, [str(c) for c in clients], rotation=25)
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if log_y:
                axis.set_yscale("log")
                axis.yaxis.set_major_locator(
                    matplotlib.ticker.LogLocator(base=10, subs=(1, 2, 5))
                )
                axis.yaxis.set_major_formatter(
                    matplotlib.ticker.FuncFormatter(lambda value, pos: f"{value:g}")
                )
                axis.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            else:
                axis.set_ylim(bottom=0)
            if row == 0:
                axis.set_title(f"{policy.upper()} routing", fontsize=13)
                axis.legend(frameon=False, ncols=2, fontsize=9, loc="upper left")
            if row == 2:
                axis.set_xlabel("Concurrent AgentX session trees")
    figure.suptitle(
        "64 GPUs · 16 TP4 workers · Nemotron-3-Ultra AgentX\nSimulation predictions; full 3,600-second AIPerf replay at every point",
        fontsize=15,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".svg"))
    svg = args.output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    figure.savefig(args.output.with_suffix(".png"), dpi=140)
    result = {
        "knee_definition": "First sampled concurrency reaching 90% of that topology/policy curve's highest observed input+output throughput. Not a fitted physical knee or an SLO guarantee. Curves with >10% growth in the last interval require an upper-boundary extension.",
        "curves": summarize(points),
    }
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n")
    print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
