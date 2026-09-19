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
        tuned = "kvs3c08" if arch == "agg" else "kvc15"
        ticks = (
            [48, 96, 192, 384]
            if arch == "agg"
            else [72, 96, 144, 192, 384, 480, 576, 768, 1152]
        )
        for column, metric, title in [
            (0, "total_tok_s_gpu", "Total input + output tok/s/GPU"),
            (1, "ttft_p95_s", "TTFT p95 (seconds; log scale)"),
            (2, I90, "E2E I90 (output tok/s/user)"),
        ]:
            ax = axes[row, column]
            for pol in ["kv", "rr", tuned]:
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
            axes[row, 2].annotate(
                "Tuned C192: 19.7795 <20\nC96 remains the combined-SLO choice",
                xy=(192, 19.7795),
                xytext=(0.06, 0.27),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
        else:
            axes[row, 1].axvspan(480, 672, alpha=0.07, color=COLORS["kv"])
            axes[row, 1].annotate(
                "Tuned C576: 8.75 s, passes\nIts next failing point is unmeasured",
                xy=(576, 8.75054),
                xytext=(0.04, 0.20),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
    fig.suptitle(
        "Measured AgentX curves · default KV, RR and each architecture's selected tuning",
        x=0.04,
        ha="left",
        fontsize=16,
    )
    fig.text(
        0.04,
        0.015,
        "Rings: highest sampled throughput in each series. Stars: highest-throughput samples passing both SLOs. Connecting lines are guides.\nSingle-point flag variants are shown in the following charts. One trial per cell; shaded intervals do not locate an exact knee.",
        fontsize=10,
        color="#52657a",
    )
    fig.subplots_adjust(
        left=0.065, right=0.99, top=0.91, bottom=0.14, hspace=0.37, wspace=0.25
    )
    save(fig, "curves")

    for arch, concurrency, policies in [
        ("agg", 192, ["kv", "kvs2c08", "kvs3c08", "kvt05", "kvd05", "kvd10", "rr"]),
        ("disagg", 480, ["kv", "kvc15", "kvc20", "kvs3c08", "kvd05"]),
    ]:
        selected = [
            one(
                points,
                arch,
                pol,
                concurrency,
                "agg-20260916"
                if arch == "agg" and pol in {"kv", "rr", "kvs3c08"}
                else None,
            )
            for pol in policies
        ]
        labels = [
            LABELS[p["policy"]].replace("KV ", "")
            + (" *" if p["campaign"] == "agg-np2-20260919" else "")
            for p in selected
        ]
        fig, axes = plt.subplots(1, 3, figsize=(15, 6.0))
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
                if point["campaign"] == "agg-np2-20260919":
                    bar.set_hatch("///")
                    bar.set_edgecolor("#755224")
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
            "* Hatched decay rows are the new np-2 campaign: warmup is slower and matched references are not yet available.\nBoth new decay runs fail both SLOs. All seven C192 rows show 3 exported request errors; differences across campaigns are not causal estimates."
            if arch == "agg"
            else "Credit 1.5 has the lowest measured C480 TTFT and highest E2E I90. Credit 2.0 gives essentially identical throughput.\nAll five KV rows have zero errors. RR480 is omitted from this scale: 387.54 s TTFT, 0.6525 I90, 39 client errors; see the full table."
        )
        fig.text(0.04, 0.015, footer, fontsize=9, color="#52657a")
        fig.subplots_adjust(left=0.19, right=0.98, top=0.88, bottom=0.20, wspace=0.20)
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
    a192 = one(points, "agg", "kvs3c08", 192)
    date = manifest["collected_utc"][:16].replace("T", " ") + " UTC"
    source_link = f"https://github.com/alisachen-google/gcp-dynamo-cuj/blob/{manifest['source_commit']}/kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/AGENTX_D88_RESULTS.md"
    link = lambda p: f"[{name(p, True)}]({p['gcs_console']})"
    out = [
        f"""# AgentX serving performance: measured agg and disagg decisions

**Updated {date} · {len(points)} completed hardware runs · 24-GPU agg and 64-GPU 8:8 disagg · Weka 256K**

**The new measured disagg choice is KV overlap credit 1.5 at C576:** **{d["total_tok_s_gpu"]:,.0f} total tokens/s/GPU**, **{d["ttft_p95_s"]:.2f} s TTFT p95**, and **{d[I90]:.4f} E2E-normalized output tokens/s**. It passes both chosen SLOs. **Agg remains scale 3 / credit 0.8 at C96** under the combined SLO: its C192 result passes TTFT but narrowly misses E2E interactivity. The two new agg decay settings at C192 miss both limits and do not establish an improvement.

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
| Agg, 6 × TP4/EP4, 24 GPUs | `--router-temperature 0 --router-queue-policy fcfs --router-prefill-load-scale 3 --router-kv-overlap-score-credit 0.8` | **96** under both SLOs | 192 has I90 **{a192[I90]:.4f}**, below 20; it is not an eligible combined-SLO operating point. |
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

The inventory now contains **{len(cells(points, "agg"))} agg and {len(cells(points, "disagg"))} D88 hardware runs**, up from 12 and 14 in the prior reports: **{sum(p["is_new"] for p in points)} additional completed cells**. D88's inventory comes from the [updated source report]({source_link}); the new agg decay jobs were discovered directly in GCS and validated from their summaries and request records.
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
                        ("disagg", "kv", 96): "Matched low-load KV/RR comparison",
                        ("disagg", "kv", 144): "Matched low-load KV/RR comparison",
                        (
                            "disagg",
                            "rr",
                            384,
                        ): "Tightens RR overload bracket to 192–384",
                        (
                            "disagg",
                            "kvc20",
                            480,
                        ): "No observed advantage over credit 1.5 on TTFT/I90",
                        (
                            "disagg",
                            "kvc15",
                            192,
                        ): "No material throughput gain at light load",
                        ("disagg", "kvc15", 576): "New highest tested tuned pass",
                        (
                            "agg",
                            "kvd05",
                            192,
                        ): "Both fail; matched np-2 reference pending",
                        (
                            "agg",
                            "kvd10",
                            192,
                        ): "Both fail; matched np-2 reference pending",
                    }.get(
                        (p["architecture"], p["policy"], p["clients"]),
                        "Additional hardware sample",
                    ),
                ]
                for p in points
                if p["is_new"]
            ],
        )
    )
    out.append(f"""
**What changed in the decision:** tuned D88 moves from the previously tested C480 to the now-tested **C576**. Relative to default KV's best sampled combined-SLO cell at C480, that is **{delta(d, dbase, "total_tok_s_gpu"):.1f}% more total throughput/GPU** and **20% more concurrent sessions**. **Default KV576 is unmeasured**, so this is not a controlled claim that the flag itself creates all of that capacity increase. The measured same-C480 flag effect remains **{delta(d480, dbase, "ttft_p95_s"):.1f}% TTFT p95**, **{delta(d480, dbase, I90):+.1f}% I90**, and only **{delta(d480, dbase, "total_tok_s_gpu"):+.2f}% throughput**.

## 3. Curves, knees and same-concurrency comparisons

{figure("curves", "Agg and disagg throughput, TTFT p95 and E2E interactivity versus concurrency, including measured tuned curves and knee evidence")}

### Throughput peak, SLO crossing and highest tested pass are distinct

| Series | Measured throughput evidence | SLO evidence | Concrete next point |
| --- | --- | --- | --- |
| Agg default KV | Highest sampled throughput C192; C384 falls 14.5% | C96 passes both; C192 fails both | Sample 144 to narrow 96–192. |
| Agg RR | Highest sampled throughput C192; C384 falls 25.4% | C48 passes both; C96 fails both | Repeat C48, then test 72 if RR capacity matters. |
| Agg scale 3 / credit 0.8 | Two points, C96 and C192; throughput still rises | C192 passes TTFT but misses I90 by **1.10%** | Repeat C192 and test C144; no tuned throughput knee is known. |
| D88 default KV | C672→768 adds only 2.4% throughput for 59% more TTFT; C1152 then falls 31.2% | Both SLO boundaries lie in 480–672 | Default C576 is the missing direct control for the tuned result. |
| D88 RR | New C384 is 20.7% below C192; C480 falls further | TTFT crossing 72–96; I90 crossing 96–144 | Repeat C72, then C84; throughput overload bracket is now 192–384. |
| D88 credit 1.5 | C192, C480 and C576 measured; throughput still rises | All three pass both; C576 is the highest tested pass | Repeat C576, then C624 or C672 to bracket this setting's boundary. |

Each cell has one trial. The figures mark sampled points and unsampled intervals, not exact optimized knees. A one-point flag variant has no measurable concurrency knee. GPU utilization is not used to pick these knees because no aligned per-engine GPU series is supplied here.

### Default KV versus RR at the same concurrency

GPU count is fixed **within each architecture**. Ratios below compare matching session counts; overloaded RR rows describe overload behavior, not sustainable capacity.
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
The new D88 C96 and C144 pairs show the same direction as C192: throughput gains are modest at low load, while KV substantially reduces TTFT. At C384, KV/RR throughput reaches 2.85×, but RR is already overloaded. This pattern supports prefix reuse and queue pressure as contributors; it does not independently isolate prefill computation, admission wait and KV transfer time.

## 4. Router tuning: measured effects and remaining controls

### 4.1 Agg at C192: include the new decay data, preserve the campaign distinction
""")
    out.append(
        figure(
            "agg-flags",
            "Agg C192 flag comparison including the new credit-decay runs, marked as a different hardware campaign",
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
    out.append(f"""
**Observed result:** decay 0.5 gives **{decay[0]["total_tok_s_gpu"]:,.0f} total tok/s/GPU, {decay[0]["ttft_p95_s"]:.2f} s TTFT and {decay[0][I90]:.4f} I90**; decay 1.0 gives **{decay[1]["total_tok_s_gpu"]:,.0f}, {decay[1]["ttft_p95_s"]:.2f} s and {decay[1][I90]:.4f} I90**. Neither passes either SLO. Both use default scale/credit and change only decay, but they run on separate np-2 fleets from the September 16 references.

**Why the default-reference comparison is provisional:** the original default-KV192 warmup is **{control["warmup_wall_s"]:,.1f} s** and the original scale-3/credit-0.8 warmup is **{a192["warmup_wall_s"]:,.1f} s**. New decay warmups are **{decay[0]["warmup_wall_s"]:,.1f} / {decay[1]["warmup_wall_s"]:,.1f} s**. The source orchestration therefore remeasures both references on np-2. Until those summaries arrive, lower throughput versus the old runs cannot be attributed solely to decay. The new settings' failure to meet the fixed SLOs is still directly observed.

**What to use now:** the previously validated scale-3/credit-0.8 C96 point remains the best sampled combined-SLO agg result. At C192 the same setting gives the best observed TTFT among the collected agg variants, but **I90 {a192[I90]:.4f} must not be rounded up to a pass**. An improvement in TTFT alone does not settle the joint latency criterion.

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
                            str(c) for c in sorted({p["clients"] for p in group})
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

The existing [agg Native DynoSim V10 report](agentx-agg-kv-rr-report.md#31-native-dynosim-v10-current-completed-calibration-samples) contains **12 matching native results for the original 12 agg hardware cells**. Their stored inputs and E2E calculations are revalidated here; the new agg decay settings have no matching completed V10 result in this inventory. Calibration points and held-out samples must be read separately.
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
| 1 | Finish the in-flight np-2 agg **default KV192** and **scale 3 / credit 0.8 C192** references | Separates the new decay results from the approximately 6% warmup drift between fleets/campaigns. |
| 2 | Repeat **D88 credit 1.5 C576**, and collect the already queued **default KV C576** control | Verifies the selected operating point and isolates the tuning benefit at equal concurrency. |
| 3 | Repeat **agg scale 3 / credit 0.8 C192**, then measure **C144** under the same campaign | Determines whether the 1.10% I90 miss is repeatable and finds a larger passing point than C96. |
| 4 | Measure **D88 credit 1.5 C624**; move to **C672** if both limits pass | Brackets the tuned latency boundary without calling C576 an exact knee. |
| 5 | Repeat **D88 RR72**, then **RR84** if needed | Stabilizes the RR denominator, currently only 0.089 s below the TTFT limit. |
| 6 | After the native handoff fix, rerun one C480 baseline before the remaining grid | Establishes that a full warmup and profiling window can complete and that predictions can be compared to hardware. |

The [queued D88 follow-up](agentx-serving-perf-data/source/run_agentx_88_followup.sh.txt) runs **temperature 0.5 and 0.2 at C480**, then **default KV at C576**, after the agg program releases np-2. These are planned cells, not measurements in the current inventory. There is no need to submit duplicates of those queued controls. Capture request-level E2E, actual output length, errors, trace identity and warmup timestamps on every point. To explain the architecture difference, add per-engine prefill/queue/running counts, KV occupancy, cache read counters and transfer wait/timing aligned to profiling.

## 7. Complete hardware inventory and reproduction

The table below contains every completed hardware sample used in the report. **Neither failed native runs nor an artifact directory without a completed AIPerf summary contributes a plotted point.** The pending np-2 reference and historical failed smoke directories are listed in the source manifest. New reference results can change the campaign comparison when they finish; this report is an explicit collection-time snapshot.
""")
    out.append(
        table(
            [
                "Run / artifacts",
                "Campaign",
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
        "slo_ttft_p95_s": 10,
        "slo_e2e_normalized_interactivity_p90_tps": 20,
        "hardware": points,
        "selected_operating_points": chosen(points),
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
