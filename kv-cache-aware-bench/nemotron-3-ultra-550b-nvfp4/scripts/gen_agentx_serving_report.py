#!/usr/bin/env python3
"""Generate the current hardware decisions for agg and 8:8 disagg AgentX.

All numerical conclusions are derived from preserved per-request inputs.
"""

from __future__ import annotations

import base64
import csv
import gzip
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import gen_agg_kv_rr_report as agg
import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt
from matplotlib.ticker import FuncFormatter

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
DATA = REPORTS / "agentx-serving-perf-data"
STEM = "agentx-serving-perf-report"
I90 = "e2e_normalized_interactivity_p90_tps"
LABELS = {
    "kv": "Default KV",
    "rr": "RR",
    "kvs3c08": "KV scale 3 / credit 0.8",
    "kvs2c08": "KV scale 2 / credit 0.8",
    "kvt05": "KV temperature 0.5",
    "kvc15": "KV credit 1.5",
    "kvc20": "KV credit 2.0",
    "kvd05": "KV decay 0.5",
    "kvd10": "KV decay 1.0",
    "kvs3c08d05": "KV scale 3 / credit 0.8 / decay 0.5",
    "kvt02": "KV temperature 0.2",
}
COLORS = {
    "kv": "#2463b3",
    "rr": "#d65b29",
    "kvs3c08": "#16836b",
    "kvs2c08": "#7d75b7",
    "kvt05": "#987242",
    "kvc15": "#16836b",
    "kvc20": "#7d75b7",
    "kvd05": "#b58a27",
    "kvd10": "#ab5374",
    "kvs3c08d05": "#4b817d",
    "kvt02": "#b5936f",
}
ORDER = list(LABELS)
FIGURES = ["curves", "agg-flags", "disagg-flags", "operating-points"]


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit(folder, entry):
    with gzip.open(folder / entry["file"], "rt", newline="") as stream:
        records = list(csv.DictReader(stream))
    warm = [r for r in records if r["phase"] == "warmup"]
    profile = [r for r in records if r["phase"] == "profiling"]
    good = [r for r in profile if r["status"] == "success"]
    result = {
        "phase_status_counts": dict(
            Counter(f"{r['phase']}/{r['status']}" for r in records)
        ),
        "warmup_requests": len(warm),
        "warmup_errors": sum(r["status"] != "success" for r in warm),
        "request_error_rate_pct": 100
        * sum(r["status"] != "success" for r in profile)
        / len(profile),
        "warmup_wall_s": None,
        "client_inflight_mean": None,
        "client_inflight_peak": None,
        "ttft_q1_p50_s": None,
        "ttft_q4_p50_s": None,
        "source_trace_count": None,
        "turn_index_p50": None,
    }
    assert result["warmup_errors"] == 0
    if "request_start_ns" not in records[0]:
        return result
    result["warmup_wall_s"] = (
        max(int(r["request_end_ns"]) for r in warm)
        - min(int(r["request_start_ns"]) for r in warm)
    ) / 1e9
    events = sorted(
        pair
        for r in profile
        for pair in [(int(r["request_start_ns"]), 1), (int(r["request_end_ns"]), -1)]
    )
    current = area = peak = 0
    previous = events[0][0]
    for t, delta in events:
        area += current * (t - previous)
        current += delta
        peak = max(peak, current)
        previous = t
    assert current == 0
    result.update(
        client_inflight_mean=area / (events[-1][0] - events[0][0]),
        client_inflight_peak=peak,
    )
    starts = np.array([int(r["request_start_ns"]) for r in good], dtype=np.int64)
    relative = (starts - starts.min()) / max(1, starts.max() - starts.min())
    ttft = np.array([float(r["ttft_ms"]) / 1000 for r in good])
    upper_middle = lambda x: float(sorted(x)[len(x) // 2])
    result.update(
        ttft_q1_p50_s=upper_middle(ttft[relative < 0.25]),
        ttft_q4_p50_s=upper_middle(ttft[relative >= 0.75]),
        source_trace_count=len({r["source_trace_id"] for r in good}),
        turn_index_p50=float(np.median([int(r["turn_index"]) for r in good])),
    )
    return result


def load():
    manifest = read(DATA / "manifest.json")
    verified = {}
    requests = {}
    for root in {r["input_root"] for r in manifest["runs"]}:
        folder = REPORTS / root
        m = read(folder / "manifest.json")
        for path, spec in m["files"].items():
            assert sha(folder / path) == spec["sha256"], path
        verified[root] = {
            "manifest_sha256": sha(folder / "manifest.json"),
            "files": len(m["files"]),
        }
        requests[root] = read(folder / "request-metrics/manifest.json")
    recipe = (DATA / "source/agentx_runner_flags.sh.txt").read_text()
    router_flags = dict(re.findall(r'\[(\w+)\]="(--router-mode [^"]+)"', recipe))
    points = []
    for item in manifest["runs"]:
        folder = REPORTS / item["input_root"]
        summary = read(folder / item["summary"])
        agg.validate_client(summary, item["clients"])
        config = summary["input_config"]
        assert (
            config["datasets"][0]["dataset"]
            == "semianalysis_cc_traces_weka_062126_256k"
        )
        assert config["datasets"][0]["cache_bust"]["target"] == "first_turn_prefix"
        assert (
            config["endpoint"]["use_server_token_count"]
            and config["endpoint"]["streaming"]
            and config["endpoint"]["extra"]["ignore_eos"]
        )
        gpus = 24 if item["architecture"] == "agg" else 64
        p = {
            **item,
            **agg.metrics(summary, gpus),
            "gpus": gpus,
            "fleet": re.search(r"alisachen-(.+)-agentx-", item["artifact"])[1],
            "router_flags": router_flags[item["policy"]],
            "ttft_p50_s": summary["time_to_first_token"]["p50"] / 1000,
            "ttft_p99_s": summary["time_to_first_token"]["p99"] / 1000,
            "submission_valid": summary["metadata"]["submission_valid"],
            "benchmark_id": summary["benchmark_id"],
            "itl_p50_ms": summary["inter_token_latency"]["p50"],
            "output_tokens_p99": summary["output_sequence_length"]["p99"],
        }
        p.update(agg.request_metrics(folder, requests[item["input_root"]], p, summary))
        p.update(audit(folder, requests[item["input_root"]]["runs"][p["id"]]))
        q1, q4 = p["ttft_q1_p50_s"], p["ttft_q4_p50_s"]
        growing = q1 is not None and q4 > 1.5 * q1 and q4 - q1 > 2
        p["post_knee"] = bool(
            growing or p["ttft_p50_s"] > 20 or p["request_error_rate_pct"] > 5
        )
        p["eligible"] = p["combined_slo_pass"] and not p["post_knee"]
        p["clients_per_gpu"] = p["clients"] / gpus
        p["total_tok_s_fleet"] = p["total_tok_s_gpu"] * gpus
        p["output_tok_s_fleet"] = p["output_tok_s_gpu"] * gpus
        if "source_cell" in p:
            source = p["source_cell"]
            assert abs(p["total_tok_s_gpu"] - source["tot"]) < 0.0001
            assert abs(p["ttft_p95_s"] - source["p95"]) < 0.0001
            assert np.isclose(p["warmup_wall_s"], source["wu_wall"], rtol=1e-10)
            assert np.isclose(p["client_inflight_mean"], source["infl"], rtol=1e-10)
            assert p["post_knee"] == source["knee"].startswith("POST")
        points.append(p)
    points.sort(
        key=lambda p: (
            p["architecture"],
            ORDER.index(p["policy"]),
            p["clients"],
            p["artifact"],
        )
    )
    _, _, native = agg.load_inputs(REPORTS / "agentx-agg-kv-rr-data")
    status = (
        read(DATA / manifest["native_c480_status"])
        if manifest.get("native_c480_status")
        else None
    )
    return manifest, points, native, status, verified


def cells(points, arch, policy=None):
    return [
        p
        for p in points
        if p["architecture"] == arch and (policy is None or p["policy"] == policy)
    ]


def one(points, arch, policy, concurrency, campaign=None):
    candidates = [
        p
        for p in cells(points, arch, policy)
        if p["clients"] == concurrency
        and (campaign is None or p["campaign"] == campaign)
    ]
    assert len(candidates) == 1, (arch, policy, concurrency, len(candidates))
    return candidates[0]


def best(points, arch, policy=None, criterion="eligible"):
    candidates = [
        p for p in cells(points, arch, policy) if p[criterion] and not p["post_knee"]
    ]
    return max(candidates, key=lambda p: p["total_tok_s_gpu"], default=None)


def chosen(points):
    return [
        best(points, arch, policy)
        for arch in ["agg", "disagg"]
        for policy in ["rr", "kv", "kvs3c08" if arch == "agg" else "kvc15"]
    ]


def agg_contrasts(points):
    campaign = "agg-np2-20260919"
    rows = []
    for treatment, control in [
        ("kvs3c08", "kv"),
        ("kvd05", "kv"),
        ("kvd10", "kv"),
        ("kvs3c08d05", "kvs3c08"),
    ]:
        a = one(points, "agg", treatment, 192, campaign)
        b = one(points, "agg", control, 192, campaign)
        rows.append(
            {
                "treatment_id": a["id"],
                "control_id": b["id"],
                "treatment": LABELS[treatment],
                "control": LABELS[control],
                "total_change_pct": delta(a, b, "total_tok_s_gpu"),
                "ttft_change_pct": delta(a, b, "ttft_p95_s"),
                "i90_change_pct": delta(a, b, I90),
                "cache_change_pp": a["cache_pct"] - b["cache_pct"],
            }
        )
    return rows


def name(p, full=False):
    return (
        ("Agg " if p["architecture"] == "agg" else "D88 ")
        + LABELS[p["policy"]]
        + (f" C{p['clients']}" if full else "")
    )


def delta(a, b, field):
    return (a[field] / b[field] - 1) * 100


def figure(name_, description):
    return f"![{description}]({STEM}-{name_}.png)\n\n[SVG]({STEM}-{name_}.svg) · [PDF]({STEM}-{name_}.pdf)\n"


def save(fig, tag):
    for ext in ["png", "svg", "pdf"]:
        path = REPORTS / f"{STEM}-{tag}.{ext}"
        fig.savefig(
            path,
            dpi=180,
            facecolor="white",
            bbox_inches="tight",
            metadata={"Creator": "AgentX measured serving report"},
        )
        if ext == "svg":
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def plots(points):
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.labelcolor": "#162b45",
            "text.color": "#162b45",
            "svg.fonttype": "none",
            "svg.hashsalt": STEM,
        }
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 10.2))
    for row, arch in enumerate(["agg", "disagg"]):
        ticks = (
            [48, 96, 192, 384]
            if arch == "agg"
            else [72, 96, 144, 192, 384, 480, 672, 768, 1152]
        )
        for column, metric, title in [
            (0, "total_tok_s_gpu", "Total input + output tok/s/GPU"),
            (1, "ttft_p95_s", "TTFT p95 (seconds; log scale)"),
            (2, I90, "E2E I90 (output tok/s/user)"),
        ]:
            ax = axes[row, column]
            for pol in ["kv", "rr"]:
                series = [
                    p
                    for p in cells(points, arch, pol)
                    if arch != "agg" or p["campaign"] == "agg-20260916"
                ]
                ax.plot(
                    [p["clients"] for p in series],
                    [p[metric] for p in series],
                    color=COLORS[pol],
                    marker={"kv": "o", "rr": "s"}.get(pol, "D"),
                    linewidth=2,
                    markersize=5,
                    label=LABELS[pol],
                )
                if arch == "agg":
                    repeats = [
                        p
                        for p in cells(points, arch, pol)
                        if p["campaign"] == "agg-np2-20260919"
                    ]
                    ax.scatter(
                        [p["clients"] for p in repeats],
                        [p[metric] for p in repeats],
                        marker="x",
                        color=COLORS[pol],
                        s=65,
                        linewidths=1.8,
                        zorder=7,
                    )
                if column == 0:
                    peak = max(series, key=lambda p: p[metric])
                    ax.scatter(
                        peak["clients"],
                        peak[metric],
                        s=110,
                        facecolors="none",
                        edgecolors=COLORS[pol],
                        linewidths=1.5,
                        zorder=5,
                    )
                else:
                    selected = best(points, arch, pol)
                    if selected:
                        ax.scatter(
                            selected["clients"],
                            selected[metric],
                            marker="*",
                            color=COLORS[pol],
                            edgecolor="#172c43",
                            s=150,
                            zorder=6,
                        )
            ax.set_xscale("log", base=2)
            ax.set_xticks(
                ticks, [str(n) for n in ticks], rotation=45 if arch == "disagg" else 0
            )
            ax.set_xlabel("Concurrency · live sessions")
            ax.set_title(title, loc="left", fontsize=11)
            ax.grid(alpha=0.18)
            ax.spines[["top", "right"]].set_visible(False)
            if column == 0:
                ax.set_ylim(0, 16000 if arch == "agg" else 21000)
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
                ax.text(
                    0.03,
                    0.97,
                    "24 GPUs · agg"
                    if arch == "agg"
                    else "64 GPUs · 8 prefill + 8 decode",
                    transform=ax.transAxes,
                    va="top",
                    weight="bold",
                    fontsize=10,
                )
                ax.legend(
                    loc="upper left",
                    bbox_to_anchor=(0, 0.90),
                    fontsize=8,
                    frameon=False,
                )
                ax.axvspan(192, 384, color=COLORS["rr"], alpha=0.06)
                if arch == "disagg":
                    ax.axvspan(672, 768, color=COLORS["kv"], alpha=0.10)
                    ax.annotate(
                        "KV 672 → 768:\n+2.4% throughput, +59% TTFT",
                        xy=(768, 15543),
                        xytext=(0.22, 0.57),
                        textcoords="axes fraction",
                        fontsize=8,
                        arrowprops={"arrowstyle": "->"},
                    )
                else:
                    ax.text(
                        0.52,
                        0.93,
                        "Sampled KV/RR peak: C192\nC384: less throughput,\nlonger TTFT",
                        transform=ax.transAxes,
                        fontsize=8,
                        va="top",
                    )
            else:
                ax.set_yscale("log")
                ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                threshold = 10 if column == 1 else 20
                ax.axhline(threshold, color="#a62b3a", linestyle="--", linewidth=1.3)
                ax.set_ylim((0.8, 850) if column == 1 else (0.35, 140))
                ax.text(
                    0.02,
                    0.98,
                    "SLO ≤10 s" if column == 1 else "SLO ≥20 tok/s",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                    color="#a62b3a",
                )
        if arch == "agg":
            boundary = one(points, "agg", "kv", 192, "agg-20260916")
            axes[row, 2].annotate(
                "Default KV: C96 passes both SLOs\nC192 fails; boundary lies in 96–192",
                xy=(192, boundary[I90]),
                xytext=(0.06, 0.27),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
        else:
            boundary = one(points, "disagg", "kv", 480)
            axes[row, 1].axvspan(480, 672, alpha=0.07, color=COLORS["kv"])
            axes[row, 1].annotate(
                "Default KV: C480 passes both SLOs\nC672 fails; boundary lies in 480–672",
                xy=(480, boundary["ttft_p95_s"]),
                xytext=(0.04, 0.20),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
    fig.suptitle(
        "Measured AgentX curves · default KV versus round-robin",
        x=0.04,
        ha="left",
        fontsize=16,
    )
    fig.text(
        0.04,
        0.015,
        "Rings: sampled throughput peaks. Stars: highest-throughput samples passing both SLOs. Crosses: fresh agg default-KV C192 reference.\nLines show the original default-KV and RR ladders. Most cells have one trial; shaded intervals do not locate an exact knee.",
        fontsize=10,
        color="#52657a",
    )
    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.91, bottom=0.14, hspace=0.37, wspace=0.25
    )
    save(fig, "curves")

    for arch, concurrency, specs in [
        (
            "agg",
            192,
            [(p, "agg-20260916") for p in ["kv", "kvs2c08", "kvs3c08", "kvt05"]]
            + [
                (p, "agg-np2-20260919")
                for p in ["kv", "kvs3c08", "kvd05", "kvd10", "kvs3c08d05"]
            ],
        ),
        (
            "disagg",
            480,
            [(p, None) for p in ["kv", "kvc15", "kvc20", "kvs3c08", "kvd05"]],
        ),
    ]:
        selected = [
            one(points, arch, pol, concurrency, campaign) for pol, campaign in specs
        ]
        labels = [
            (
                "np-2: "
                if p["campaign"] == "agg-np2-20260919"
                else "Sep 16: "
                if arch == "agg"
                else ""
            )
            + LABELS[p["policy"]].replace("KV ", "")
            for p in selected
        ]
        fig, axes = plt.subplots(1, 3, figsize=(16, 8.2 if arch == "agg" else 6.0))
        for ax, key, title, threshold in [
            (axes[0], "total_tok_s_gpu", "Total input + output tok/s/GPU", None),
            (axes[1], "ttft_p95_s", "TTFT p95 (seconds)", 10),
            (axes[2], I90, "E2E I90 (output tok/s/user)", 20),
        ]:
            values = [p[key] for p in selected]
            bars = ax.barh(
                np.arange(len(selected)),
                values,
                color=[COLORS[p["policy"]] for p in selected],
                height=0.62,
            )
            for bar, point in zip(bars, selected):
                if arch == "agg" and point["campaign"] == "agg-20260916":
                    bar.set_hatch("///")
                    bar.set_edgecolor("#52657a")
                    bar.set_alpha(0.55)
            ax.set_yticks(np.arange(len(selected)), labels if ax is axes[0] else [])
            ax.invert_yaxis()
            ax.bar_label(
                bars,
                labels=[
                    f"{v:,.0f}" if key == "total_tok_s_gpu" else f"{v:.2f}"
                    for v in values
                ],
                padding=4,
                fontsize=9,
            )
            ax.set_xlim(0, max(values) * 1.20)
            ax.set_title(title, loc="left", fontsize=11)
            if threshold:
                ax.axvline(threshold, color="#a62b3a", linestyle="--")
                ax.set_xlabel(
                    "Dashed: ≤10 s" if threshold == 10 else "Dashed: ≥20 tok/s",
                    color="#a62b3a",
                    fontsize=9,
                )
            ax.spines[["top", "right", "left"]].set_visible(False)
            ax.grid(axis="x", alpha=0.15)
            ax.set_axisbelow(True)
        fig.suptitle(
            f"{'Agg · 24 GPUs' if arch == 'agg' else 'D88 · 64 GPUs'} · measured router flags at C{concurrency}",
            x=0.04,
            ha="left",
            fontsize=15,
        )
        footer = (
            "Hatched: Sep 16. Solid: current np-2 campaign, including both fresh references. Decay loses alone and on top of scale 3 / credit 0.8.\nNo measured agg C192 variant passes both SLOs. Both scale-3 / credit-0.8 trials pass TTFT and miss I90; each C192 run has 2–3 client errors."
            if arch == "agg"
            else "Credit 1.5 has the lowest measured C480 TTFT and highest E2E I90. Credit 2.0 gives essentially identical throughput.\nAll five KV rows have zero errors. RR480 is omitted from this scale: 387.54 s TTFT, 0.6525 I90, 39 client errors; see the full table."
        )
        fig.text(0.04, 0.015, footer, fontsize=9, color="#52657a")
        fig.subplots_adjust(
            left=0.27 if arch == "agg" else 0.19,
            right=0.98,
            top=0.88,
            bottom=0.20,
            wspace=0.20,
        )
        save(fig, f"{arch}-flags")

    selected = chosen(points)
    labels = [
        name(p, True)
        .replace("KV scale 3 / credit 0.8", "tuned KV")
        .replace("KV credit 1.5", "tuned KV")
        for p in selected
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.8))
    for ax, key, title in [
        (axes[0], "total_tok_s_gpu", "Total tokens/s/GPU"),
        (axes[1], "output_tok_s_gpu", "Output tokens/s/GPU"),
        (axes[2], "clients_per_gpu", "Measured sessions/GPU"),
    ]:
        values = [p[key] for p in selected]
        bars = ax.barh(
            np.arange(len(selected)),
            values,
            color=[COLORS[p["policy"]] for p in selected],
            height=0.62,
        )
        ax.set_yticks(np.arange(len(selected)), labels if ax is axes[0] else [])
        ax.invert_yaxis()
        ax.bar_label(
            bars,
            labels=[
                f"{v:,.0f}" if key == "total_tok_s_gpu" else f"{v:.2f}" for v in values
            ],
            padding=4,
            fontsize=9,
        )
        ax.set_xlim(0, max(values) * 1.23)
        ax.set_title(title, loc="left", fontsize=11)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
    fig.suptitle(
        "Best measured operating points · TTFT p95 ≤10 s and E2E I90 ≥20 tok/s",
        x=0.04,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.04,
        0.015,
        "Tuned agg: scale 3 / credit 0.8, C96. Tuned D88: credit 1.5, C576. All selected rows have zero exported errors.\n24 versus 64 GPUs: these are normalized observations from different fleet sizes, not proof of linear scaling or a cost forecast.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.21, right=0.98, top=0.86, bottom=0.19, wspace=0.20)
    save(fig, "operating-points")


def table(headers, rows):
    return agg.table(headers, rows)


def pf(value):
    return "Pass" if value else "Fail"


def markdown(points, native, manifest, status):
    a = best(points, "agg")
    d = best(points, "disagg")
    dbase = one(points, "disagg", "kv", 480)
    d480 = one(points, "disagg", "kvc15", 480)
    d20 = one(points, "disagg", "kvc20", 480)
    a192 = one(points, "agg", "kvs3c08", 192, "agg-20260916")
    a192_repeat = one(points, "agg", "kvs3c08", 192, "agg-np2-20260919")
    fresh_default = one(points, "agg", "kv", 192, "agg-np2-20260919")
    combined_decay = one(points, "agg", "kvs3c08d05", 192)
    additions = [p for p in points if p["new_since_previous_report"]]
    date = manifest["collected_utc"][:16].replace("T", " ") + " UTC"
    source_link = f"https://github.com/alisachen-google/gcp-dynamo-cuj/blob/{manifest['source_commit']}/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_D88_RESULTS.md"
    link = lambda p: f"[{name(p, True)}]({p['gcs_console']})"
    out = [
        f"""# AgentX serving performance: measured agg and disagg decisions

**Updated {date} · {len(points)} completed hardware runs · 24-GPU agg and 64-GPU 8:8 disagg · Weka 256K**

**The fresh agg controls are complete, and decay has no observed benefit in any tested combination.** At C192, scale 3 / credit 0.8 now repeats at **{a192_repeat["total_tok_s_gpu"]:,.0f} total tokens/s/GPU**, **{a192_repeat["ttft_p95_s"]:.2f} s TTFT p95**, and **{a192_repeat[I90]:.4f} E2E-normalized output tokens/s**. Both C192 trials miss the I90 limit of 20. Adding decay 0.5 reduces throughput by **{-delta(combined_decay, a192_repeat, "total_tok_s_gpu"):.1f}%** and I90 to **{combined_decay[I90]:.4f}**. **Agg's best sampled point meeting both SLOs remains scale 3 / credit 0.8 at C96.**

**Disagg's measured choice remains KV overlap credit 1.5 at C576:** **{d["total_tok_s_gpu"]:,.0f} total tokens/s/GPU**, **{d["ttft_p95_s"]:.2f} s TTFT p95**, and **{d[I90]:.4f} I90**. The temperature follow-up has no completed summary at this snapshot. Default KV C576 remains queued, so the maximum concurrency gain caused by tuning is still unmeasured.

Compared at each fleet's best sampled point meeting both SLOs, tuned D88 delivers **{d["total_tok_s_gpu"] / a["total_tok_s_gpu"]:.2f}× total throughput/GPU** and **{d["output_tok_s_gpu"] / a["output_tok_s_gpu"]:.2f}× output-only throughput/GPU** versus tuned agg. The fleet sizes differ, **64 versus 24 GPUs**; this is an observed operating-point comparison, not a controlled estimate of architecture scaling.

[Standalone HTML]({STEM}.html) · [all hardware data CSV]({STEM}.csv) · [data and comparison JSON]({STEM}.json) · [source manifest](agentx-serving-perf-data/manifest.json) · [validation]({STEM}-validation.json)

## 1. Concrete operating choices

The limits are **TTFT p95 ≤10 seconds** and **E2E-normalized interactivity I90 ≥20 output tokens/s/user**. For each successful profiling request, calculate `r_i = request_latency_seconds / output_tokens_i`; then **`I90 = 1 / P90(r_i)`**, using linear percentile interpolation. This includes TTFT and generation. It is not inverse ITL/TPOT, not a percentile ratio, and not whole-session latency including think time or tools.

Select the highest measured total throughput/GPU that meets both limits and has no post-knee queue evidence. Errors are shown separately; no new availability SLO is assumed. **All six selected rows below have zero exported errors.** The lower-concurrency passing point is retained when a higher point fails either limit; there is no interpolation of passing results.
"""
    ]
    out.append(
        table(
            [
                "Serving / setting",
                "GPUs",
                "Sessions",
                "Total tok/s/GPU",
                "Output tok/s/GPU",
                "TTFT p95 (s)",
                "E2E I90 (tok/s)",
                "Total/RR in same topology",
            ],
            [
                [
                    name(p),
                    p["gpus"],
                    p["clients"],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['output_tok_s_gpu']:.2f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p[I90]:.4f}",
                    f"{p['total_tok_s_gpu'] / best(points, p['architecture'], 'rr')['total_tok_s_gpu']:.2f}×",
                ]
                for p in chosen(points)
            ],
        )
    )
    out.append(
        figure(
            "operating-points",
            "Selected measured operating points under both SLOs, showing total and output throughput per GPU and sessions per GPU",
        )
    )
    out.append(f"""
**Use the following settings as measured operating candidates, then repeat them on the intended deployment:**

| Deployment | Router arguments beyond `--router-mode kv` | Measured concurrency to use | Limit of this choice |
| --- | --- | --- | --- |
| Agg, 6 × TP4/EP4, 24 GPUs | `--router-temperature 0 --router-queue-policy fcfs --router-prefill-load-scale 3 --router-kv-overlap-score-credit 0.8` | **96** under both SLOs | C192 has I90 **{a192[I90]:.4f} / {a192_repeat[I90]:.4f}** in two trials; both fail. Keep decay at default 0. |
| D88, 8 prefill + 8 decode TP4, 64 GPUs | `--router-temperature 0 --router-queue-policy fcfs --router-kv-overlap-score-credit 1.5` | **576** under both SLOs | Highest tested passing point for this setting; its next failing point has not been measured. Load scale and decay retain defaults 1 and 0. |

Tuned D88 C576 has **{10 - d["ttft_p95_s"]:.2f} s TTFT headroom** and **{d[I90] - 20:.2f} tok/s I90 headroom**, based on one trial. RR72's TTFT is **9.9115 s**, only **0.0885 s** below the limit; its capacity ratio needs a repeat before being treated as stable. These are replay-based choices, not production arrival-rate guarantees.

### Fleet totals and what can be compared
""")
    out.append(
        table(
            [
                "Selected tuned fleet",
                "GPUs",
                "Sessions",
                "Sessions/GPU",
                "Total tok/s, fleet",
                "Output tok/s, fleet",
                "Requests/s",
            ],
            [
                [
                    name(p),
                    p["gpus"],
                    p["clients"],
                    f"{p['clients_per_gpu']:.2f}",
                    f"{p['total_tok_s_fleet']:,.0f}",
                    f"{p['output_tok_s_fleet']:,.0f}",
                    f"{p['requests_s']:.2f}",
                ]
                for p in [a, d]
            ],
        )
    )
    out.append(f"""
D88 uses **2.67× as many GPUs**, serves **{d["total_tok_s_fleet"] / a["total_tok_s_fleet"]:.2f}× total fleet tokens/s** and **{d["output_tok_s_fleet"] / a["output_tok_s_fleet"]:.2f}× fleet output tokens/s** at these selected samples. Report these separately from the per-GPU ratios. Total-token throughput counts cached input tokens, so output-only throughput and requests/s are included to expose workload-mix differences. No GPU-hour price, linear extrapolation to a different fleet size, or GPU requirement for 1,000 users is inferred.

## 2. What the new measurements add

The inventory now contains **{len(cells(points, "agg"))} agg and {len(cells(points, "disagg"))} D88 hardware runs**: **{len(additions)} newly completed agg runs** since the September 19 22:34 UTC combined report ({manifest["previous_snapshot"]["hardware_runs"]} runs). The 20 D88 measurements, including credit 2.0 at C480 and credit 1.5 at C576, were already in that snapshot and remain included. This is **{sum(p["is_new"] for p in points)} more runs** than the original separate agg/D88 reports. Inputs come from the [agg source report]({source_link.replace("AGENTX_D88_RESULTS", "AGENTX_AGG_RESULTS")}), [D88 source report]({source_link}), and full GCS summaries and request records.
""")
    out.append(
        table(
            [
                "New measurement / artifacts",
                "Total tok/s/GPU",
                "TTFT p95 (s)",
                "E2E I90 (tok/s)",
                "Both SLOs",
                "Client errors",
                "What it adds",
            ],
            [
                [
                    link(p),
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p[I90]:.4f}",
                    pf(p["combined_slo_pass"]),
                    p["request_errors"],
                    {
                        (
                            "agg",
                            "kv",
                            192,
                        ): "Fresh default control for the decay-alone rows",
                        (
                            "agg",
                            "kvs3c08",
                            192,
                        ): "Tuned result reproduces; I90 misses 20 again",
                        (
                            "agg",
                            "kvs3c08d05",
                            192,
                        ): "Adding decay to tuned KV reduces throughput and interactivity",
                    }.get(
                        (p["architecture"], p["policy"], p["clients"]),
                        "Additional hardware sample",
                    ),
                ]
                for p in points
                if p["new_since_previous_report"]
            ],
        )
    )
    out.append(f"""
**What changed in the decision:** agg's previously pending controls are now measured. The no-decay scale-3/credit-0.8 setting remains the leading C192 candidate, and its small I90 failure has reproduced. The next useful load point is C144; adding more credit-decay variants is lower priority. The detailed contrasts are in section 4.1.

**What stays the same for disagg:** tuned C576 has **{delta(d, dbase, "total_tok_s_gpu"):.1f}% more total throughput/GPU** and **20% more concurrent sessions** than default KV's best sampled combined-SLO cell at C480. **Default KV576 is unmeasured**, so this is not a controlled claim that the flag itself creates all of that capacity increase. The measured same-C480 flag effect remains **{delta(d480, dbase, "ttft_p95_s"):.1f}% TTFT p95**, **{delta(d480, dbase, I90):+.1f}% I90**, and only **{delta(d480, dbase, "total_tok_s_gpu"):+.2f}% throughput**.

## 3. Curves, knees and same-concurrency comparisons

This section compares **default KV and RR** for agg and disagg.

{figure("curves", "Agg and disagg default KV versus RR: throughput, TTFT p95 and E2E interactivity versus concurrency, with knee evidence")}

### Throughput peak, SLO crossing and highest tested pass are distinct

| Series | Measured throughput evidence | SLO evidence | Concrete next point |
| --- | --- | --- | --- |
| Agg default KV | Highest sampled throughput C192; C384 falls 14.5% | C96 passes both; C192 fails both | Sample 144 to narrow 96–192. |
| Agg RR | Highest sampled throughput C192; C384 falls 25.4% | C48 passes both; C96 fails both | Repeat C48, then test 72 if RR capacity matters. |
| D88 default KV | C672→768 adds only 2.4% throughput for 59% more TTFT; C1152 then falls 31.2% | Both SLO boundaries lie in 480–672 | Test C576 to narrow the 480–672 boundary. |
| D88 RR | New C384 is 20.7% below C192; C480 falls further | TTFT crossing 72–96; I90 crossing 96–144 | Repeat C72, then C84; throughput overload bracket is now 192–384. |

Most cells have one trial. Default agg KV192 has two trials across deployment campaigns; the fresh result appears as a cross on the original curves. This repeat is a useful reproducibility check, not a calibrated tail-variance distribution. The figures mark sampled points and unsampled intervals, not exact optimized knees. GPU utilization is not used to pick these knees because no aligned per-engine GPU series is supplied here.

### Default KV versus RR at the same concurrency

GPU count is fixed **within each architecture**. Ratios below compare matching session counts; agg uses the original September 16 ladder so its repeated KV192 is not mixed with the older RR control. Overloaded RR rows describe overload behavior, not sustainable capacity.
""")
    rows = []
    for arch in ["agg", "disagg"]:
        common = sorted(
            {p["clients"] for p in cells(points, arch, "kv")}
            & {p["clients"] for p in cells(points, arch, "rr")}
        )
        for c in common:
            k = one(points, arch, "kv", c, "agg-20260916" if arch == "agg" else None)
            r = one(points, arch, "rr", c)
            rows.append(
                [
                    "Agg24" if arch == "agg" else "D88",
                    c,
                    f"{k['total_tok_s_gpu']:,.0f} / {r['total_tok_s_gpu']:,.0f}",
                    f"{k['total_tok_s_gpu'] / r['total_tok_s_gpu']:.2f}×",
                    f"{k['ttft_p95_s']:.2f} / {r['ttft_p95_s']:.2f}",
                    f"{r['ttft_p95_s'] / k['ttft_p95_s']:.2f}×",
                    f"{k[I90]:.2f} / {r[I90]:.2f}",
                    "RR overloaded"
                    if r["post_knee"]
                    else "Below queue-check cutoff; check SLOs separately",
                ]
            )
    out.append(
        table(
            [
                "Topology",
                "Sessions",
                "KV / RR total tok/s/GPU",
                "KV/RR throughput",
                "KV / RR TTFT p95 (s)",
                "RR/KV TTFT",
                "KV / RR I90",
                "Interpretation",
            ],
            rows,
        )
    )
    out.append("""
The D88 C96 and C144 pairs show the same direction as C192: throughput gains are modest at low load, while KV substantially reduces TTFT. At C384, KV/RR throughput reaches 2.85×, but RR is already overloaded. This pattern supports prefix reuse and queue pressure as contributors; it does not independently isolate prefill computation, admission wait and KV transfer time.

## 4. Router tuning: measured effects and remaining controls

### 4.1 Agg at C192: fresh controls confirm the tuning direction
""")
    out.append(
        figure(
            "agg-flags",
            "Agg C192 flag comparison with fresh default and tuned controls and the scale-3/credit-0.8 plus decay combination",
        )
    )
    flags = [p for p in cells(points, "agg") if p["clients"] == 192]
    out.append(
        table(
            [
                "C192 setting",
                "Campaign",
                "Total tok/s/GPU",
                "TTFT p95 (s)",
                "I90 (tok/s)",
                "Cached input %",
                "Errors",
                "Both pass",
            ],
            [
                [
                    LABELS[p["policy"]],
                    "Sep 19 np-2" if p["campaign"] == "agg-np2-20260919" else "Sep 16",
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p[I90]:.4f}",
                    f"{p['cache_pct']:.1f}",
                    p["request_errors"],
                    pf(p["combined_slo_pass"]),
                ]
                for p in flags
            ],
        )
    )
    control = one(points, "agg", "kv", 192, "agg-20260916")
    decay = [one(points, "agg", pol, 192) for pol in ["kvd05", "kvd10"]]
    out.append("""
**Compare against controls from the same np-2 campaign.** These contrasts hold the replay configuration, concurrency and declared fleet recipe constant. The campaign uses two separate 24-GPU fleets; the treatment and reference were not all run on the same physical workers or in randomized order. Read the deltas as measured comparisons, not confidence intervals.
""")
    out.append(
        table(
            [
                "Treatment",
                "Reference",
                "Δ total tok/s/GPU",
                "Δ TTFT p95",
                "Δ E2E I90",
                "Δ cached input",
            ],
            [
                [
                    r["treatment"],
                    r["control"],
                    f"{r['total_change_pct']:+.2f}%",
                    f"{r['ttft_change_pct']:+.1f}%",
                    f"{r['i90_change_pct']:+.1f}%",
                    f"{r['cache_change_pp']:+.2f} pp",
                ]
                for r in agg_contrasts(points)
            ],
        )
    )
    out.append(f"""
**Decision:** retain **scale 3 / credit 0.8, decay 0** for agg. Against the fresh default, it gives **{delta(a192_repeat, fresh_default, "total_tok_s_gpu"):+.1f}% total throughput**, **{delta(a192_repeat, fresh_default, "ttft_p95_s"):.1f}% TTFT p95**, and **{delta(a192_repeat, fresh_default, I90):+.1f}% I90** at C192. Decay 0.5 and 1.0 alone lose on all three metrics versus the fresh default. Adding decay 0.5 to the tuned setting also loses on all three, despite passing the TTFT limit. No tested decay setting is the preferred candidate for the next concurrency sweep.

**The repeat changes our confidence, not the combined-SLO operating point.** The original tuned C192 I90 is **{a192[I90]:.4f}** and the fresh repeat is **{a192_repeat[I90]:.4f}**; both are below 20. The combined-decay I90 is **{combined_decay[I90]:.4f}**. **No collected agg C192 variant passes both SLOs**, so the best sampled point remains tuned C96. Source tables sometimes call `1000 / ITL p90` interactivity; that decode-only metric can exceed 20 while the request-level E2E metric used here fails. This report always uses `1 / P90(E2E / output_tokens)`.

**Reference reproducibility:**
""")
    out.append(
        table(
            [
                "C192 reference",
                "Total/GPU, Sep 16 → np-2",
                "Δ total",
                "TTFT p95, Sep 16 → np-2 (s)",
                "I90, Sep 16 → np-2",
                "Warmup, Sep 16 → np-2 (s)",
            ],
            [
                [
                    LABELS[b["policy"]],
                    f"{b['total_tok_s_gpu']:,.0f} → {r['total_tok_s_gpu']:,.0f}",
                    f"{delta(r, b, 'total_tok_s_gpu'):+.2f}%",
                    f"{b['ttft_p95_s']:.2f} → {r['ttft_p95_s']:.2f}",
                    f"{b[I90]:.4f} → {r[I90]:.4f}",
                    f"{b['warmup_wall_s']:,.1f} → {r['warmup_wall_s']:,.1f}",
                ]
                for b, r in [(control, fresh_default), (a192, a192_repeat)]
            ],
        )
    )
    out.append(f"""
The new controls reproduce the earlier throughput within **2%** and warm up in **{fresh_default["warmup_wall_s"]:,.1f} / {a192_repeat["warmup_wall_s"]:,.1f} s**. The decay-alone runs had **{decay[0]["warmup_wall_s"]:,.1f} / {decay[1]["warmup_wall_s"]:,.1f} s** warmups immediately after deployment. The later controls weaken the hypothesis that np-2 is uniformly slower; the longer first warmup is consistent with a startup transient. Cache state and run ordering remain possible contributors. Two reference pairs do not establish a full noise floor for the other flags, especially for the small throughput changes.

### 4.2 Disagg at C480: credit 1.5 is the strongest observed latency candidate

{figure("disagg-flags", "Disagg C480 default and tuned KV router comparison, including overlap credit 2.0")}
""")
    out.append(
        table(
            [
                "C480 setting",
                "Total tok/s/GPU",
                "Δ total vs default",
                "TTFT p95 (s)",
                "Δ TTFT",
                "E2E I90",
                "Cached input %",
                "Both pass",
            ],
            [
                [
                    LABELS[p["policy"]],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{delta(p, dbase, 'total_tok_s_gpu'):+.2f}%",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{delta(p, dbase, 'ttft_p95_s'):+.1f}%",
                    f"{p[I90]:.4f}",
                    f"{p['cache_pct']:.1f}",
                    pf(p["combined_slo_pass"]),
                ]
                for p in cells(points, "disagg")
                if p["clients"] == 480 and p["policy"] != "rr"
            ],
        )
    )
    out.append(f"""
Credit **2.0** has effectively the same throughput as credit 1.5 (**{delta(d20, d480, "total_tok_s_gpu"):+.4f}%**), but TTFT p95 is **{d20["ttft_p95_s"]:.2f} versus {d480["ttft_p95_s"]:.2f} s** and I90 is **{d20[I90]:.4f} versus {d480[I90]:.4f}**. More reported cached input does not translate into a better latency result at this point. One trial cannot establish the precision of this ordering, but there is no observed reason to prefer 2.0 over 1.5.

At **C192**, credit 1.5 and default KV deliver **5,138 versus 5,142 total tok/s/GPU**: there is no material throughput gain at light load. The useful change appears in the heavier-load latency measurements. **C576 is a new operating point, not a same-concurrency A/B test.**

Scale 3 / credit 0.8 misses the D88 TTFT limit at C480; decay 0.5 misses both limits. The scale-3 setting changes two variables together, so it cannot identify which variable caused the regression. The measured agg and D88 tuning choices differ; keep separate profiles for the two deployments.

### Exact measured flag coverage
""")
    coverage = []
    for arch in ["agg", "disagg"]:
        for pol in ORDER:
            group = cells(points, arch, pol)
            if group:
                coverage.append(
                    [
                        "Agg24" if arch == "agg" else "D88",
                        LABELS[pol],
                        ", ".join(
                            str(c)
                            + (
                                f" ({n} runs)"
                                if (n := sum(p["clients"] == c for p in group)) > 1
                                else ""
                            )
                            for c in sorted({p["clients"] for p in group})
                        ),
                        f"`{group[0]['router_flags']}`",
                    ]
                )
    out.append(
        table(
            ["Topology", "Setting", "Completed session counts", "Router command"],
            coverage,
        )
    )
    out.append("""
## 5. Evidence quality and simulation status

All included hardware summaries pass `inferencex-agentx-mvp` validation. The shared replay uses the pinned-name Weka 256K corpus with 393 sessions, seed 42, 3,600-second profiling, 60-second grace, trajectory start ratios 0.25–0.75, streaming, server token counts, `ignore_eos` and per-play first-turn prefix cache busting. Time-bounded closed-loop progression produces different completed request cohorts; a fixed seed does not imply identical completed turns.

The fleet recipes declare SGLang 0.5.16 and Dynamo 1.4.2. These summaries do not attest an immutable tokenizer revision or server image digest. Cached-input percentages in this report consistently use AIPerf's `overall_usage_prompt_cache_read_pct` export; the source D88 report instead quotes a frontend histogram ratio, which can differ slightly. Neither is presented as a direct per-engine KV occupancy measurement.

**Errors and cache state:** D88 RR384 has **68 client errors** and RR480 has **39**; both are overloaded. The source's separate 353 server-side timeout count for RR480 is not a client request-error count. The source notes that KV144's guard saw drain-tail timeouts from the preceding RR run before KV144 sent traffic; KV144 itself has zero exported request errors. The runner now drains after overload. Caches are not explicitly flushed between hardware cells, and cache busting does not prove identical physical cache occupancy. Warmup is a useful comparison check, not a complete health certificate.

### Native simulations: retain the evidence boundary

The existing [agg Native DynoSim V10 report](agentx-agg-kv-rr-report.md#31-native-dynosim-v10-current-completed-calibration-samples) contains **12 matching native results for the original 12 agg hardware cells**. Their stored inputs and E2E calculations are revalidated here. No new native run was added in this update, and the agg decay settings have no matching completed V10 result in this inventory. The error table retains the original hardware pairings; the fresh hardware references are not additional simulation trials. Calibration points and held-out samples must be read separately.
""")
    out.append(
        table(
            [
                "Native agg V10 subset",
                "Points",
                "Mean absolute total-throughput error",
                "Mean absolute TTFT p95 error",
                "Mean absolute I90 error",
            ],
            [
                [
                    role,
                    len(group),
                    *[
                        f"{100 * np.mean([abs(p['hardware_comparison'][metric]['relative_error']) for p in group]):.1f}%"
                        for metric in ["total_tok_s_gpu", "ttft_p95_s", I90]
                    ],
                ]
                for role in sorted({p["role"] for p in native})
                if (group := [p for p in native if p["role"] == role])
            ],
        )
    )
    if status:
        failures = [r for r in status["runs"] if r["state"] == "failed"]
        evidence = sum(bool(r["handoff_limit_evidence"]) for r in failures)
        out.append(f"""
**The C480 disagg native flag sweep did not produce usable performance results:** **{len(failures)}/{len(status["runs"])} runs failed**, with no completed measurement exports. All **{evidence}** failed runs contain **`mocker handoff session limit reached`** in prefill logs. The queue's `completed_with_failures` status means scheduling finished, not that simulations succeeded. This is a native mocker admission/handoff failure, not a measured hardware capacity limit. [Failure evidence and log hashes](agentx-serving-perf-data/source/native-c480-status.json) are preserved. The build was a V11 disaggregation extension with frozen V10 timing coefficients, not the unchanged V10 binary.

Resolve and verify the native admission/backpressure behavior before spending another full sweep on these configurations; keep the hardware batch limit intact rather than enlarging it merely to avoid errors. Then validate the corrected build at one matched hardware point and record the new build identity. No simulated flag ranking or simulated D88 knee is claimed from these failed runs.
""")
    out.append("""
## 6. Next measurements that would change the decision

| Priority | Exact measurement | Decision it resolves |
| --- | --- | --- |
| 1 | Measure **agg scale 3 / credit 0.8, decay 0 at C144**; if it passes, test **C168** and repeat the chosen point | Finds a higher passing concurrency than C96. C192's small I90 miss has now reproduced; its controls are complete. |
| 2 | Collect the already queued **D88 default KV C576** and repeat **credit 1.5 C576** | Determines whether tuning increases capacity at equal concurrency and verifies the selected operating point. |
| 3 | Measure **D88 credit 1.5 C624**; move to **C672** if both limits pass | Brackets the tuned latency boundary without calling C576 an exact knee. |
| 4 | Complete the existing **D88 temperature 0.5 / 0.2 C480** follow-up | Measures randomness in routing on disagg; the agg temperature result does not establish its effect here. |
| 5 | For further agg tuning, test **scale 4 / credit 0.8** against scale 3 / credit 0.8; separately test **scale 3 / credit 0** at C192 | Varies one flag at a time to distinguish the load-scale and overlap-credit effects. Decay alone and on the tuned setting already lost in the observed trials. |
| 6 | Repeat **D88 RR72**, then **RR84** if needed | Stabilizes the RR denominator, currently only 0.089 s below the TTFT limit. |
| 7 | After the native handoff fix, rerun one C480 baseline before the remaining grid | Establishes that a full warmup and profiling window can complete and that predictions can be compared to hardware. |

The [D88 follow-up](agentx-serving-perf-data/source/run_agentx_88_followup.sh.txt) runs **temperature 0.5 and 0.2 at C480**, then **default KV at C576**. The temperature-0.5 run has started, but its GCS folder has no completed AIPerf summary at collection time. Temperature 0.2 and default KV576 are still queued in that sequence. These cells contribute no measurements yet; do not submit duplicates. The agg default/tuned references and combined-decay test are complete. Capture request-level E2E, actual output length, errors, trace identity and warmup timestamps on every point. To explain the architecture difference, add per-engine prefill/queue/running counts, KV occupancy, cache read counters and transfer wait/timing aligned to profiling.

## 7. Complete hardware inventory and reproduction

The table below contains every completed hardware sample used in the report. **Neither failed native runs nor an artifact directory without a completed AIPerf summary contributes a plotted point.** The unfinished D88 temperature run and historical failed smoke directories are listed in the source manifest. The report is an explicit collection-time snapshot; later temperature and default-KV576 results can change the disagg choice.
""")
    out.append(
        table(
            [
                "Run / artifacts",
                "Campaign",
                "Fleet",
                "Total tok/s/GPU",
                "Output tok/s/GPU",
                "TTFT p95 (s)",
                "E2E I90",
                "Errors",
                "Both pass",
            ],
            [
                [
                    link(p),
                    p["campaign"],
                    p["fleet"],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['output_tok_s_gpu']:.2f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p[I90]:.4f}",
                    p["request_errors"],
                    pf(p["combined_slo_pass"]),
                ]
                for p in points
            ],
        )
    )
    out.append(f"""
The [manifest](agentx-serving-perf-data/manifest.json) records the source commit, GCS inventory, every new summary and request projection, and links to the already preserved agg/D88 inputs. Per-request projection manifests include hashes of the original full JSONL exports. The generator validates those hashes, request/error counts, TTFT p95, E2E p95 and AIPerf's P10 output/E2E against the raw summaries, then computes canonical I90 independently. Source D88 warmup, in-flight counts and queue verdicts are reconciled where supplied. The HTML embeds all plots and the plotted CSV/JSON; supporting input files remain linked in GitHub to keep the standalone report compact.

Regenerate offline from the repository root, using Python with NumPy, Matplotlib and markdown-it-py:

```bash
python kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_agentx_serving_report.py
```

The read-only collector is `scripts/collect_agentx_serving_results.py`; it imports completed GCS artifacts and preserves a new inventory snapshot. Neither script submits benchmark traffic. Previous [agg](agentx-agg-kv-rr-report.md) and [D88](agentx-disagg-kv-rr-report.md) reports remain dated snapshots of their original inventories; this report carries the latest combined decisions as of **{date}**.
""")
    return "\n\n".join(out)


def html(markdown_text):
    body = MarkdownIt("commonmark").enable("table").render(markdown_text)
    for tag in FIGURES:
        data = base64.b64encode((REPORTS / f"{STEM}-{tag}.svg").read_bytes()).decode()
        body = body.replace(
            f'src="{STEM}-{tag}.png"', f'src="data:image/svg+xml;base64,{data}"'
        )

    def heading(match):
        level, text = match.groups()
        anchor = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]*>", "", text).lower()).strip(
            "-"
        )
        return f'<h{level} id="{anchor}">{text}</h{level}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", heading, body)

    def portable(match):
        url = match[1]
        if ":" in url or url.startswith("#"):
            return match[0]
        if url == f"{STEM}.html":
            return 'href="#"'
        filename, _, anchor = url.partition("#")
        path = (REPORTS / filename).resolve()
        if filename.startswith(STEM) and path.suffix in {
            ".csv",
            ".json",
            ".svg",
            ".pdf",
        }:
            mime = {
                ".csv": "text/csv",
                ".json": "application/json",
                ".svg": "image/svg+xml",
                ".pdf": "application/pdf",
            }[path.suffix]
            data = base64.b64encode(path.read_bytes()).decode()
            return f'download="{path.name}" href="data:{mime};base64,{data}"'
        if path.is_relative_to(ROOT.parents[1]):
            target = (
                "https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/"
                + path.relative_to(ROOT.parents[1]).as_posix()
            )
            return f'href="{target}{"#" + anchor if anchor else ""}"'
        return match[0]

    body = re.sub(r'href="([^"]+)"', portable, body)
    body = re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-scroll">\1</div>',
        body,
        flags=re.DOTALL,
    )
    document = (
        """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AgentX serving performance · measured agg and disagg decisions</title><style>
:root{color-scheme:light;--ink:#162b45;--muted:#52657a;--line:#dce4ed}*{box-sizing:border-box}
body{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1360px;margin:28px auto 64px;padding:38px 54px 64px;background:#fff;border:1px solid var(--line);border-radius:14px;min-width:0}
h1{font-size:36px;line-height:1.2;margin:0 0 18px;letter-spacing:-.7px}h2{font-size:27px;line-height:1.3;margin-top:46px;padding-top:20px;border-top:2px solid var(--line)}h3{font-size:20px;margin-top:30px}
p,li{overflow-wrap:anywhere}a{color:#205db1;text-underline-offset:3px}.table-scroll{overflow-x:auto;margin:24px 0}
table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}th{background:#edf3f9}tr:nth-child(even) td{background:#f8fafc}
img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}code{font:13px ui-monospace,SFMono-Regular,monospace;background:#edf2f7;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}
pre{padding:18px;background:#edf2f7;border-radius:8px;overflow:auto}pre code{padding:0;overflow-wrap:normal}nav{display:flex;gap:20px;flex-wrap:wrap;font-size:14px;margin:0 0 26px}.eyebrow{font-size:12px;letter-spacing:1.4px;text-transform:uppercase;color:var(--muted);margin-bottom:14px}
@media(max-width:800px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:29px}h2{font-size:23px}table{font-size:12px}th,td{padding:8px}}
@media print{body{background:#fff}main{border:0;max-width:none;margin:0;padding:0}nav{display:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}.table-scroll{overflow:visible}table{font-size:9px}}
</style></head><body><main><div class="eyebrow">Nemotron-3-Ultra 550B · AgentX · measured hardware</div>
<nav><a href="#1-concrete-operating-choices">Operating choices</a><a href="#2-what-the-new-measurements-add">New results</a><a href="#3-curves-knees-and-same-concurrency-comparisons">Curves and knees</a><a href="#4-router-tuning-measured-effects-and-remaining-controls">Router tuning</a><a href="#5-evidence-quality-and-simulation-status">Evidence</a><a href="#6-next-measurements-that-would-change-the-decision">Next measurements</a><a href="#7-complete-hardware-inventory-and-reproduction">All results</a></nav>
"""
        + body
        + "</main></body></html>\n"
    )
    (REPORTS / f"{STEM}.html").write_text(document)


def main():
    manifest, points, native, status, verified = load()
    validation = {
        "passed": True,
        "hardware_runs": len(points),
        "new_hardware_runs": sum(p["is_new"] for p in points),
        "new_since_previous_report": sum(
            p["new_since_previous_report"] for p in points
        ),
        "by_architecture": dict(Counter(p["architecture"] for p in points)),
        "native_v10_agg_runs_revalidated": len(native),
        "input_manifests": verified,
        "profiling_successes": sum(p["successful_requests"] for p in points),
        "profiling_errors": sum(p["request_errors"] for p in points),
        "excluded_successes_from_I90": sum(
            p["interactivity_excluded_successful_requests"] for p in points
        ),
        "checks": [
            "source hashes",
            "shared replay configuration",
            "canonical inverse-P90 E2E calculation",
            "TTFT/E2E percentiles reconciled to summaries",
            "request and error counts",
            "D88 rounded source metrics, warmup, inflight, queue verdicts",
            "existing native V10 inputs and hardware matches",
        ],
    }
    (REPORTS / f"{STEM}-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    result = {
        "collected_utc": manifest["collected_utc"],
        "source_commit": manifest["source_commit"],
        "previous_snapshot": manifest["previous_snapshot"],
        "slo_ttft_p95_s": 10,
        "slo_e2e_normalized_interactivity_p90_tps": 20,
        "hardware": points,
        "selected_operating_points": chosen(points),
        "agg_c192_current_campaign_contrasts": agg_contrasts(points),
        "native_v10_agg": native,
        "native_disagg_status": status,
        "unavailable": manifest["unavailable"],
    }
    (REPORTS / f"{STEM}.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [k for k in points[0] if not isinstance(points[0][k], (dict, list))]
    with (REPORTS / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(points)
    plots(points)
    document = markdown(points, native, manifest, status)
    (REPORTS / f"{STEM}.md").write_text(document)
    html(document)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
