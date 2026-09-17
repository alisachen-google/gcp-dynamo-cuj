#!/usr/bin/env python3
"""Plot audited RR throughput and latency curves for agg and disagg."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--arm", choices=["both", "agg6", "12:6"], default="both")
    parser.add_argument(
        "--output", type=Path, required=True, help="Output path without extension"
    )
    args = parser.parse_args()
    points = json.loads(args.audit.read_text())["points"]
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    arms = ["agg6", "12:6"] if args.arm == "both" else [args.arm]
    figure, axes = plt.subplots(
        3 if len(arms) == 2 else 1,
        2 if len(arms) == 2 else 3,
        figsize=(12, 10) if len(arms) == 2 else (15, 4.5),
        layout="constrained",
        squeeze=False,
    )
    specs = [
        ("total_tokens_s_gpu", "Input + output tokens/s/GPU", False),
        ("ttft_p95_s", "TTFT p95 (seconds; log scale)", True),
        ("itl_mean_ms", "Mean ITL (ms)", False),
    ]
    for column, arm in enumerate(arms):
        selected = sorted(
            (p for p in points if p["arm"] == arm), key=lambda p: p["clients"]
        )
        clients = [p["clients"] for p in selected]
        for row, (metric, label, log_y) in enumerate(specs):
            axis = axes[row, column] if len(arms) == 2 else axes[0, row]
            axis.plot(
                clients,
                [p["simulation"][metric] for p in selected],
                "o-",
                color="#2563eb",
                linewidth=2,
                label="RR simulation",
            )
            real = [p for p in selected if "real" in p]
            if real:
                axis.plot(
                    [p["clients"] for p in real],
                    [p["real"][metric] for p in real],
                    "s--",
                    color="#d97706",
                    linewidth=2,
                    label="RR hardware",
                )
            if arm == "12:6":
                recipe = [p for p in selected if p["recipe_point"]]
                axis.scatter(
                    [p["clients"] for p in recipe],
                    [p["simulation"][metric] for p in recipe],
                    s=100,
                    facecolors="none",
                    edgecolors="#111827",
                    linewidths=1.5,
                    zorder=5,
                    label="GitHub recipe points",
                )
            axis.set_xscale("log", base=2)
            axis.set_xticks(
                clients,
                [str(c) for c in clients],
                rotation=25 if len(clients) > 4 else 0,
            )
            axis.set_ylabel(label)
            axis.grid(alpha=0.2)
            if log_y:
                axis.set_yscale("log")
            else:
                axis.set_ylim(bottom=0)
            if row == 0:
                axis.set_title(
                    "Agg: 6 × TP4, 24 GPUs"
                    if arm == "agg6"
                    else "Disagg: 12P + 6D × TP4, 72 GPUs\nSimulation predictions; no RR hardware reference"
                )
                axis.legend(frameon=False, loc="upper left", fontsize=9)
            if row == 2 or len(arms) == 1:
                axis.set_xlabel("Concurrent AgentX session trees")
    figure.suptitle(
        "Nemotron-3-Ultra AgentX RR sweep · 3,600-second aiperf replay", fontsize=15
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output.with_suffix(".svg"))
    svg = args.output.with_suffix(".svg")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n"
    )
    figure.savefig(args.output.with_suffix(".png"), dpi=150)
    print(args.output.with_suffix(".svg"))


if __name__ == "__main__":
    main()
