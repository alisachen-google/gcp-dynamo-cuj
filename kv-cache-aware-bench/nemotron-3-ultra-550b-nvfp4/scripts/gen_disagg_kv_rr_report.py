#!/usr/bin/env python3
"""Build the D88 hardware report in the format used for the agg KV/RR report.

Uses preserved inputs only. No benchmarks, network requests or simulations run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import re
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from gen_agg_kv_rr_report import metrics, request_metrics, table, validate_client
from markdown_it import MarkdownIt
from matplotlib.ticker import FuncFormatter

STUDY = Path(__file__).resolve().parents[1]
STEM = "agentx-disagg-kv-rr-report"
NAMES = {
    "kv": "Default KV",
    "rr": "RR",
    "kvc15": "KV: credit 1.5",
    "kvs3c08": "KV: scale 3, credit 0.8",
    "kvd05": "KV: decay 0.5",
}
COLORS = {
    "kv": "#2463b3",
    "rr": "#d65b29",
    "kvc15": "#16836b",
    "kvs3c08": "#7d75b7",
    "kvd05": "#bf923e",
}
MARKERS = {"kv": "o", "rr": "s", "kvc15": "D", "kvs3c08": "^", "kvd05": "X"}
FLAGS = {
    "kv": (1, 1, 0),
    "kvc15": (1, 1.5, 0),
    "kvs3c08": (3, 0.8, 0),
    "kvd05": (1, 1, 0.5),
}
ORDER = ["kv", "rr", "kvc15", "kvs3c08", "kvd05"]
FIGURES = ["hardware", "interactivity", "flag-sweep", "cache-load"]


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_records(folder, entry, point):
    with gzip.open(folder / entry["file"], "rt", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert all(r["status"] in {"success", "error"} for r in rows)
    profiling = [r for r in rows if r["phase"] == "profiling"]
    good = [r for r in profiling if r["status"] == "success"]
    warm = [r for r in rows if r["phase"] == "warmup"]
    events = sorted(
        pair
        for r in profiling
        if r["request_start_ns"] and r["request_end_ns"]
        for pair in [(int(r["request_start_ns"]), 1), (int(r["request_end_ns"]), -1)]
    )
    current = area = peak = 0
    previous = events[0][0]
    for time, delta in events:
        area += current * (time - previous)
        current += delta
        peak = max(peak, current)
        previous = time
    assert current == 0
    inflight = area / (events[-1][0] - events[0][0])
    starts = np.array([int(r["request_start_ns"]) for r in good], dtype=np.int64)
    times = (starts - starts.min()) / max(1, starts.max() - starts.min())
    ttft = np.array([float(r["ttft_ms"]) / 1000 for r in good])
    # Reproduce knee_check.py's upper-middle sample convention exactly.
    median = lambda values: float(sorted(values)[len(values) // 2])
    q1, q4 = median(ttft[times < 0.25]), median(ttft[times >= 0.75])
    growing = q4 > 1.5 * q1 and q4 - q1 > 2
    standing_queue = median(ttft) > 20
    error_rate = sum(r["status"] != "success" for r in profiling) / len(profiling)
    post_knee = growing or standing_queue or error_rate > 0.05
    source = point["source_cell"]
    assert post_knee == source["knee"].startswith("POST"), point["id"]
    assert np.isclose(inflight, source["infl"], rtol=1e-10), point["id"]
    assert peak == source["peak"] and len(profiling) == source["n"]
    warm_wall = (
        max(int(r["request_end_ns"]) for r in warm)
        - min(int(r["request_start_ns"]) for r in warm)
    ) / 1e9
    assert np.isclose(warm_wall, source["wu_wall"], rtol=1e-10)
    traces = sorted({r["source_trace_id"] for r in good if r["source_trace_id"]})
    sessions = {r["root_correlation_id"] for r in good if r["root_correlation_id"]}
    return {
        "profiling_records": len(profiling),
        "request_error_rate": error_rate,
        "client_inflight_mean": inflight,
        "client_inflight_peak": peak,
        "profile_envelope_s": (events[-1][0] - events[0][0]) / 1e9,
        "warmup_requests": len(warm),
        "warmup_wall_s": warm_wall,
        "warmup_latency_p50_s": float(
            np.median([float(r["request_latency_ms"]) / 1000 for r in warm])
        ),
        "warmup_errors": sum(r["status"] != "success" for r in warm),
        "ttft_quarter1_p50_s": q1,
        "ttft_quarter4_p50_s": q4,
        "queue_growing": growing,
        "standing_queue": standing_queue,
        "post_knee": post_knee,
        "knee_check": "POST-KNEE" if post_knee else "AT/PRE-KNEE",
        "source_trace_count": len(traces),
        "source_trace_ids": traces,
        "profiled_sessions_per_lane": len(sessions) / point["clients"],
        "turn_index_p50": float(np.median([int(r["turn_index"]) for r in good])),
        "subagent_request_share": sum(int(r["agent_depth"]) > 0 for r in good)
        / len(good),
    }


def load_inputs(folder):
    manifest = read(folder / "manifest.json")
    for path, meta in manifest["files"].items():
        assert digest(folder / path) == meta["sha256"], path
    provenance = read(folder / manifest["request_metrics_manifest"])
    points = []
    for source in manifest["hardware"]:
        summary = read(folder / source["summary"])
        validate_client(summary, source["clients"])
        cfg = summary["input_config"]
        assert (
            cfg["datasets"][0]["dataset"] == "semianalysis_cc_traces_weka_062126_256k"
        )
        assert (
            cfg["endpoint"]["use_server_token_count"] and cfg["endpoint"]["streaming"]
        )
        assert cfg["endpoint"]["extra"]["ignore_eos"]
        assert cfg["datasets"][0]["cache_bust"]["target"] == "first_turn_prefix"
        assert cfg["runtime"]["workers"] == 200
        point = {**source, "kind": "hardware", "gpus": 64, **metrics(summary, 64)}
        point.update(request_metrics(folder, provenance, point, summary))
        point.update(audit_records(folder, provenance["runs"][point["id"]], point))
        point.update(
            ttft_p50_s=summary["time_to_first_token"]["p50"] / 1000,
            ttft_p99_s=summary["time_to_first_token"]["p99"] / 1000,
            itl_p50_ms=summary["inter_token_latency"]["p50"],
            output_tokens_p90=summary["output_sequence_length"]["p90"],
            output_tokens_p99=summary["output_sequence_length"]["p99"],
            submission_valid=summary["metadata"]["submission_valid"],
            decode_only_inverse_itl_p90_tps=1000
            / summary["inter_token_latency"]["p90"],
        )
        with (folder / source["server_metrics"]).open() as stream:
            rows = [r for r in csv.reader(stream) if len(r) > 6]
        sums = {
            name: float(next(r for r in rows if r[2] == name)[6])
            for name in [
                "dynamo_frontend_cached_tokens",
                "dynamo_frontend_input_sequence_tokens",
            ]
        }
        ratio = (
            sums["dynamo_frontend_cached_tokens"]
            / sums["dynamo_frontend_input_sequence_tokens"]
        )
        assert np.isclose(ratio, source["source_cell"]["hit"], rtol=1e-10)
        point["frontend_cached_input_pct"] = 100 * ratio
        point["eligible_combined_slo"] = (
            point["combined_slo_pass"]
            and not point["post_knee"]
            and point["request_errors"] == 0
        )
        point["flags"] = (
            dict(
                zip(
                    ["load_scale", "overlap_credit", "credit_decay"],
                    FLAGS[point["policy"]],
                )
            )
            if point["policy"] != "rr"
            else {}
        )
        for key, value in [("total_tok_s_gpu", "tot"), ("ttft_p95_s", "p95")]:
            # The source MD generator consumes a rounded CSV, not this JSON.
            assert abs(point[key] - source["source_cell"][value]) < 0.0001
        points.append(point)
    points.sort(key=lambda p: (ORDER.index(p["policy"]), p["clients"]))
    return manifest, points


def pick(points, policy, criterion="eligible_combined_slo"):
    candidates = [
        p
        for p in points
        if p["policy"] == policy
        and p[criterion]
        and not p["post_knee"]
        and p["request_errors"] == 0
    ]
    return max(candidates, key=lambda p: p["total_tok_s_gpu"], default=None)


def save(fig, output, name):
    for ext in ["png", "svg", "pdf"]:
        fig.savefig(
            output / f"{STEM}-{name}.{ext}",
            dpi=180,
            bbox_inches="tight",
            facecolor="white",
            metadata={"Creator": "N3U AgentX report generator"},
        )
        if ext == "svg":
            path = output / f"{STEM}-{name}.{ext}"
            path.write_text(
                "\n".join(line.rstrip() for line in path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def axis_style(ax, ylabel, log=False):
    ax.set_xscale("log", base=2)
    ticks = [72, 96, 144, 192, 384, 480, 672, 768, 1152]
    ax.set_xticks(ticks, [str(v) for v in ticks], rotation=40)
    ax.set_xlim(64, 1300)
    ax.set_xlabel("Concurrency · live AgentX sessions")
    ax.set_ylabel(ylabel)
    if log:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(FuncFormatter(lambda y, _: f"{y:g}"))
    ax.grid(True, alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)


def curves(ax, points, metric, tuned=False):
    for policy in ORDER if tuned else ["kv", "rr"]:
        series = [p for p in points if p["policy"] == policy]
        ax.plot(
            [p["clients"] for p in series],
            [p[metric] for p in series],
            color=COLORS[policy],
            marker=MARKERS[policy],
            markersize=7,
            linewidth=2,
            linestyle="-" if len(series) > 1 else "None",
            label=NAMES[policy],
        )


def plots(points, output):
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
    by = {(p["policy"], p["clients"]): p for p in points}
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.8))
    left, right = axes
    curves(left, points, "total_tok_s_gpu")
    curves(right, points, "ttft_p95_s")
    axis_style(left, "Total input + output tokens/s/GPU")
    axis_style(right, "TTFT p95 (seconds; log scale)", log=True)
    left.set_ylim(0, 19300)
    left.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    right.set_ylim(1.4, 1100)
    left.set_title(
        "Throughput: diminishing returns, then collapse", loc="left", fontsize=12
    )
    right.set_title("Latency: the 10-second SLO binds earlier", loc="left", fontsize=12)
    for policy in ["kv", "rr"]:
        peak = max(
            (p for p in points if p["policy"] == policy),
            key=lambda p: p["total_tok_s_gpu"],
        )
        left.scatter(
            peak["clients"],
            peak["total_tok_s_gpu"],
            s=240,
            marker="*",
            edgecolor="#10243b",
            color=COLORS[policy],
            zorder=6,
        )
    left.axvspan(672, 768, color=COLORS["kv"], alpha=0.10)
    left.annotate(
        "KV 672 → 768: throughput +2.4%\nTTFT +59%; peak sampled at 768",
        xy=(768, by["kv", 768]["total_tok_s_gpu"]),
        xytext=(0.28, 0.97),
        textcoords="axes fraction",
        va="top",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLORS["kv"]},
    )
    left.annotate(
        "1,152: throughput −31%\nTTFT 164 s; queue grows",
        xy=(1152, by["kv", 1152]["total_tok_s_gpu"]),
        xytext=(0.51, 0.39),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->"},
    )
    left.annotate(
        "RR peak sampled at 192\n480: −27% throughput; 39 client errors",
        xy=(192, by["rr", 192]["total_tok_s_gpu"]),
        xytext=(0.03, 0.43),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->", "color": COLORS["rr"]},
    )
    left.legend(loc="lower right", frameon=False)
    right.axhline(10, color="#9b2432", linestyle="--", linewidth=1.5)
    right.text(140, 8.3, "TTFT p95 SLO ≤10 s", color="#9b2432", fontsize=9)
    for policy, low, high, position in [
        ("kv", 480, 672, (0.44, 0.05)),
        ("rr", 72, 96, (0.03, 0.59)),
    ]:
        point = by[policy, low]
        right.axvspan(low, high, color=COLORS[policy], alpha=0.09)
        right.scatter(
            low,
            point["ttft_p95_s"],
            marker="*",
            s=230,
            color=COLORS[policy],
            edgecolor="#10243b",
            zorder=6,
        )
        right.annotate(
            f"{NAMES[policy]} SLO crossing: {low}–{high}\n{point['ttft_p95_s']:.2f} → {by[policy, high]['ttft_p95_s']:.2f} s",
            xy=(low, point["ttft_p95_s"]),
            xytext=position,
            textcoords="axes fraction",
            fontsize=9,
            arrowprops={"arrowstyle": "->", "color": COLORS[policy]},
        )
    right.annotate(
        "RR480: 387.5 s\n0.47% client request errors",
        xy=(480, by["rr", 480]["ttft_p95_s"]),
        xytext=(0.50, 0.88),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->"},
    )
    fig.suptitle(
        "Measured D88 · 64 GB300 GPUs · 8 prefill + 8 decode TP4 workers",
        fontsize=15,
        x=0.05,
        ha="left",
    )
    fig.text(
        0.05,
        0.01,
        "Stars: sampled throughput peaks (left) or last sampled SLO passes (right). Shading marks unsampled transition intervals.\nSingle trial per point; connecting lines guide the eye. The exact capacity or latency knee is not localized.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.075, right=0.98, top=0.87, bottom=0.24, wspace=0.25)
    save(fig, output, "hardware")

    fig, ax = plt.subplots(figsize=(12, 6.5))
    curves(ax, points, "e2e_normalized_interactivity_p90_tps", tuned=True)
    axis_style(ax, "I90 · E2E-normalized output tokens/s/user", log=True)
    ax.set_ylim(0.15, 180)
    ax.axhline(20, color="#9b2432", linestyle="--")
    ax.text(
        66, 21.5, "Chosen I90 SLO ≥20 output tok/s/user", color="#9b2432", fontsize=10
    )
    ax.axvspan(480, 672, color=COLORS["kv"], alpha=0.08)
    ax.axvspan(96, 144, color=COLORS["rr"], alpha=0.08)
    ax.annotate(
        "Default KV: I90 crosses between 480–672\nBoth SLOs select KV480; RR selects 72",
        xy=(480, by["kv", 480]["e2e_normalized_interactivity_p90_tps"]),
        xytext=(0.37, 0.97),
        textcoords="axes fraction",
        va="top",
        fontsize=10,
        arrowprops={"arrowstyle": "->"},
    )
    ax.legend(loc="lower left", ncol=2, frameon=False, fontsize=9)
    ax.set_title(
        "E2E interactivity uses each request's full latency, including TTFT",
        loc="left",
        pad=17,
    )
    fig.text(
        0.08,
        0.015,
        "I90 = 1 / P90(E2E seconds ÷ output tokens), linear percentile interpolation. Successful profiling requests only.\nErrors remain visible in the tables; RR480 is overloaded. Flag variants are single measured points, not fitted curves.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(bottom=0.25, top=0.89, left=0.09, right=0.98)
    save(fig, output, "interactivity")

    selected = [by[p, 480] for p in ["kv", "kvc15", "kvs3c08", "kvd05", "rr"]]
    labels = [
        "Default KV",
        "KV credit 1.5",
        "KV scale 3 / credit 0.8",
        "KV decay 0.5",
        "RR (overloaded)",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(14, 8.3))
    for ax, metric, title, fmt, threshold in [
        (
            axes[0, 0],
            "total_tok_s_gpu",
            "Total input + output tokens/s/GPU",
            ",.0f",
            None,
        ),
        (axes[0, 1], "ttft_p95_s", "TTFT p95 (seconds; lower is better)", ".2f", 10),
        (
            axes[1, 0],
            "e2e_normalized_interactivity_p90_tps",
            "E2E I90 (output tok/s/user; higher is better)",
            ".2f",
            20,
        ),
        (
            axes[1, 1],
            "frontend_cached_input_pct",
            "Frontend-reported cached input (%)",
            ".1f",
            None,
        ),
    ]:
        values = [p[metric] for p in selected]
        bars = ax.barh(
            labels, values, color=[COLORS[p["policy"]] for p in selected], height=0.6
        )
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontsize=11)
        ax.bar_label(
            bars, labels=[format(v, fmt) for v in values], padding=5, fontsize=9
        )
        if metric == "ttft_p95_s":
            ax.set_xscale("log")
            ax.set_xlim(1, 1000)
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        else:
            ax.set_xlim(0, max(values) * 1.22)
        if threshold:
            ax.axvline(threshold, color="#9b2432", linestyle="--", linewidth=1.2)
            ax.set_xlabel(
                "Dashed line: TTFT p95 ≤10 s"
                if threshold == 10
                else "Dashed line: E2E I90 ≥20 output tok/s/user",
                color="#9b2432",
                fontsize=9,
            )
        ax.grid(axis="x", alpha=0.15)
        ax.set_axisbelow(True)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle(
        "Measured KV-router flag sweep · concurrency 480 · same 64-GPU fleet",
        fontsize=15,
        x=0.05,
        ha="left",
    )
    fig.text(
        0.05,
        0.015,
        "Credit 1.5 vs default: TTFT p95 −15.4%, I90 +13.1%, throughput +0.3%; repeats are needed.\nAll four KV runs have zero exported errors. RR has 39 client errors / 8,317 profiling records and is an overload reference.",
        fontsize=10,
        color="#52657a",
    )
    fig.subplots_adjust(
        left=0.18, right=0.96, top=0.88, bottom=0.13, hspace=0.42, wspace=0.70
    )
    save(fig, output, "flag-sweep")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.7))
    curves(axes[0], points, "frontend_cached_input_pct")
    curves(axes[1], points, "client_inflight_mean")
    axis_style(axes[0], "Frontend cached-input token ratio (%)")
    axis_style(axes[1], "Mean requests in flight · client records")
    axes[0].set_ylim(0, 100)
    axes[0].set_title(
        "Prefix reuse declines with offered population", loc="left", fontsize=12
    )
    axes[1].set_title(
        "Live sessions and active requests are different", loc="left", fontsize=12
    )
    axes[0].legend(frameon=False)
    axes[1].annotate(
        "KV1152: 700 mean in flight\nGrowing TTFT and lower throughput",
        xy=(1152, by["kv", 1152]["client_inflight_mean"]),
        xytext=(0.10, 0.82),
        textcoords="axes fraction",
        fontsize=9,
        arrowprops={"arrowstyle": "->"},
    )
    fig.suptitle(
        "Evidence behind the latency rise · measured hardware",
        x=0.06,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.06,
        0.01,
        "Cache ratio comes from frontend histogram sums, not per-engine cache scrapes. In-flight includes failed requests and is\ntime-weighted over each profiling request envelope. These signals support a prefill/queue hypothesis, not a complete bottleneck proof.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.86, bottom=0.25, wspace=0.26)
    save(fig, output, "cache-load")


def figure(name, alt):
    return f"![{alt}]({STEM}-{name}.png)\n\n[SVG]({STEM}-{name}.svg) · [PDF]({STEM}-{name}.pdf)\n"


def pass_label(value):
    return "Pass" if value else "Fail"


def report(points, manifest, data_name):
    by = {(p["policy"], p["clients"]): p for p in points}
    kv, rr, tuned = (pick(points, policy) for policy in ["kv", "rr", "kvc15"])
    base = by["kv", 480]
    source = manifest["source_report"]
    source_commit = manifest["source_commit"]
    link = lambda p: f"[{NAMES[p['policy']]} C{p['clients']}]({p['gcs_console']})"
    gain = lambda a, b, field: (a[field] / b[field] - 1) * 100
    o = [
        f"""# N3U disaggregated serving: KV-aware routing versus round-robin

Hardware collected **2026-09-17–19** · **64 GB300 GPUs, 8 prefill + 8 decode TP4 workers** · **14 measured runs: 6 default KV, 5 RR, 3 tuned KV**

With **TTFT p95 ≤10 seconds** and **E2E-normalized interactivity at P90 ≥20 output tokens/s/user**, the best measured default-KV point is **{kv["clients"]} sessions**, versus **{rr["clients"]} for RR**: **{kv["total_tok_s_gpu"] / rr["total_tok_s_gpu"]:.2f}× total tokens/s/GPU**. Overlap credit **1.5 at 480** also passes both limits, reducing TTFT p95 **7.12 → 6.02 s (−15.4%)** and increasing E2E interactivity **{base["e2e_normalized_interactivity_p90_tps"]:.2f} → {tuned["e2e_normalized_interactivity_p90_tps"]:.2f} tok/s**. Its **+0.3% throughput change needs repeat testing**. These are best sampled points, not established capacity limits.

[Standalone HTML]({STEM}.html) · [all plotted data (CSV)]({STEM}.csv) · [data and provenance (JSON)]({STEM}.json) · [preserved inputs (ZIP)]({STEM}-inputs.zip) · [agg companion report](agentx-agg-kv-rr-report.md)

This report follows the agg report's sweep, knee, SLO and flag-comparison structure. Its hardware inventory is pinned to [AGENTX_D88_RESULTS.md at `{source_commit[:7]}`]({source}); the original AIPerf summaries and request exports provide full-precision metrics. Later measurements are outside this snapshot. No new hardware jobs were launched to prepare it.

## 1. Real data sweep and knee

### Setup and metric definitions

The fleet has **8 prefill and 8 decode TP4 workers** across 16 × a4x-maxgpu-4g nodes, with one MNNVL ComputeDomain and Mooncake/MNNVL KV transfer. The source reports SGLang 0.5.16, Dynamo 1.4.2 and FlashInfer 0.6.18. The exported client configuration verifies AIPerf's `inferencex-agentx-mvp` scenario, Weka 256K corpus (**393 sessions**), seed **42**, **3,600 s** profiling, **60 s** grace, **200 client workers**, streaming, server token counts, `ignore_eos`, end-to-start agentic replay and per-play `first_turn_prefix` cache busting. Trajectory starts are sampled in **25–75%** of the trace before warmup; do not compare to a different start range without labeling that workload change.

| Metric | Definition used here |
| --- | --- |
| Concurrency | Live AgentX session trees, including spawned subagents and recorded think time; not requests on the wire. |
| Total throughput/GPU | AIPerf input + output token throughput divided by **all 64 GPUs**. Cached inputs count; this is not newly computed token throughput. |
| Output throughput/GPU | Output-only throughput divided by all 64 GPUs, reported beside the total. |
| TTFT p95 | Successful profiling requests' time to first token; chosen limit **≤10 s**. |
| E2E I90 | **`1 / P90(E2E_seconds / output_tokens)`**, calculated per successful profiling request with linear percentile interpolation; chosen limit **≥20 output tok/s/user**. |
| Cached input | `dynamo_frontend_cached_tokens` histogram sum / `dynamo_frontend_input_sequence_tokens` histogram sum from the exported frontend scrape. **Not per-engine cache hit telemetry.** |
| In-flight | Time-weighted number of open client requests, including errors, over the first-start to last-end envelope of profiling-tagged records. Not an engine running-batch gauge. |
| Error / health | Errors are counted separately; latency percentiles exclude them. The selected SLO points all have zero errors and pass the source's queue check. |

The source's column named “P90 interactivity” is **`1000 / ITL_p90_ms`**, which describes decode speed. This report replaces that comparison with the same **E2E I90** used in the agg report. Both values remain in the downloadable data to make the distinction explicit. The chosen 20 tok/s threshold is our engineering criterion, not a dataset requirement. AIPerf's embedded request-goodput settings (TTFT 5 s / ITL 10 ms) are separate and are not used to select these points.

{figure("hardware", "Measured disagg concurrency versus throughput and TTFT, with throughput peaks and SLO crossing brackets")}

### Measured hardware points
"""
    ]
    o.append(
        table(
            [
                "Policy",
                "Sessions",
                "Total tok/s/GPU",
                "Output tok/s/GPU",
                "Req/s",
                "TTFT p95 (s)",
                "E2E I90 (tok/s)",
                "Cached input",
                "Errors",
                "Queue check",
            ],
            [
                [
                    NAMES[p["policy"]],
                    p["clients"],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['output_tok_s_gpu']:.2f}",
                    f"{p['requests_s']:.2f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                    f"{p['frontend_cached_input_pct']:.1f}%",
                    p["request_errors"],
                    p["knee_check"],
                ]
                for p in points
            ],
        )
    )
    o.append("""
### What is a knee in these plots?

**Throughput and SLO limits answer different questions.** Default KV has its highest sampled throughput at **768**, but **480** is the highest sampled point satisfying both latency limits. RR's highest sampled throughput is **192**, while its last sampled TTFT pass is **72**. The source queue check labels a run AT/PRE-KNEE or POST-KNEE using within-run queue behavior; that label alone does not locate a cross-concurrency optimum.

| Evidence | Measured change | Interpretation |
| --- | --- | --- |
| KV 672 → 768 | Total throughput +2.4%; TTFT p95 17.04 → 27.14 s (+59%) | Diminishing returns; both points already fail the 10 s SLO. |
| KV 768 → 1,152 | Throughput −31.2%; TTFT p95 27.14 → 164.32 s; TTFT p50 first/last quarter 40.40 → 137.82 s at 1,152 | Measured collapse and a growing queue. Refine 672–1,152 to localize the throughput transition. |
| RR 192 → 480 | Throughput −27.2%; TTFT p95 31.45 → 387.54 s; 39 client errors | A coarse overload bracket. 192 is a sampled peak, not proof that the exact knee is 192. |
| Default KV 480 → 672 | TTFT p95 7.12 → 17.04 s; E2E I90 crosses below 20 | Both chosen latency boundaries lie within this unsampled interval. |
| RR 72 → 96 | TTFT p95 9.91 → 13.03 s | TTFT boundary lies in 72–96. RR72 has only **0.089 s (0.89%)** headroom; repeat it. |

There is one trial per cell, so no repeat-derived confidence interval or noise floor supports an exact knee. Connecting lines do not add measured points. The flag variants have one concurrency each; their knees cannot be inferred from that single point.

### Source jobs
""")
    o.append(
        table(
            [
                "Run / original artifacts",
                "Sessions",
                "Completed UTC",
                "Summary and per-request provenance",
            ],
            [
                [
                    link(p),
                    p["clients"],
                    p["source_cell"]["done"],
                    f"[summary]({data_name}/{p['summary']}) · [request manifest]({data_name}/request-metrics/manifest.json)",
                ]
                for p in points
            ],
        )
    )
    o.append("""
## 2. KV versus RR at equal configuration and equal SLO

### 2.1 Same configuration and client count

Both policies use the same 64-GPU 8:8 fleet shape, corpus, seed and profiling settings. Runs are sequential; the frontend is restarted with different router flags. Per-run cache-bust IDs and the turns completed in a time-bounded closed loop differ. Only **192 and 480** were measured for both default policies.
""")
    o.append(
        table(
            [
                "Sessions",
                "KV total/GPU",
                "RR total/GPU",
                "KV/RR throughput",
                "KV TTFT p95",
                "RR TTFT p95",
                "RR/KV TTFT",
                "Qualification",
            ],
            [
                [
                    c,
                    f"{by['kv', c]['total_tok_s_gpu']:,.0f}",
                    f"{by['rr', c]['total_tok_s_gpu']:,.0f}",
                    f"{by['kv', c]['total_tok_s_gpu'] / by['rr', c]['total_tok_s_gpu']:.2f}×",
                    f"{by['kv', c]['ttft_p95_s']:.2f} s",
                    f"{by['rr', c]['ttft_p95_s']:.2f} s",
                    f"{by['rr', c]['ttft_p95_s'] / by['kv', c]['ttft_p95_s']:.2f}×",
                    "Both zero errors; RR misses SLO"
                    if c == 192
                    else "RR overloaded; 39 client errors",
                ]
                for c in [192, 480]
            ],
        )
    )
    o.append("""
The **192-session** comparison is the cleaner equal-population result: **16.4% more total throughput** for KV and **13.12× shorter p95 TTFT**. At 480, the **3.79× throughput ratio** describes operation against an overloaded RR reference and must not be presented as sustainable capacity at equal load.

### 2.2 TTFT-only comparison: p95 ≤10 seconds

Select the highest-throughput measured point passing the same TTFT threshold, with a passing queue check and zero exported errors. Concurrency may differ; GPU count and topology remain fixed.
""")
    o.append(
        table(
            [
                "Policy",
                "Selected sessions",
                "Total tok/s/GPU",
                "Output tok/s/GPU",
                "TTFT p95 (s)",
                "Total throughput/RR",
            ],
            [
                [
                    NAMES[p["policy"]],
                    p["clients"],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['output_tok_s_gpu']:.2f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}×",
                ]
                for p in [
                    pick(points, pol, "ttft_slo_pass") for pol in ["kv", "rr", "kvc15"]
                ]
            ],
        )
    )
    o.append(f"""
Default **KV480 versus RR72 gives {kv["total_tok_s_gpu"] / rr["total_tok_s_gpu"]:.2f}× total throughput/GPU**, **{kv["output_tok_s_gpu"] / rr["output_tok_s_gpu"]:.2f}× output-only throughput/GPU**, and **{kv["requests_s"] / rr["requests_s"]:.2f}× requests/s**. These ratios differ because the completed request mix differs. Credit 1.5 at 480 gives **{tuned["total_tok_s_gpu"] / rr["total_tok_s_gpu"]:.2f}×** RR72's total throughput. Neither scale 3 / credit 0.8 nor decay 0.5 passes TTFT at its only measured point, 480; their lower-concurrency capacity is unmeasured.

### 2.3 E2E-normalized interactivity at P90 ≥20 output tokens/s

E2E covers one inference request's TTFT and generation, excluding human think time, tools and later turns. For each successful profiling request, divide its `request_latency` by its server-counted `output_sequence_length`, take P90, then invert. Warmup and failed/cancelled requests are excluded and counted separately. All successful exported requests in this snapshot have valid positive E2E and output length. Full-precision TTFT, E2E p95 and AIPerf's P10 E2E output rate are independently reconciled to each summary; P10 output/E2E is not substituted for the canonical inverse-P90 calculation.

{figure("interactivity", "Measured E2E-normalized interactivity versus concurrency with the 20 output tokens per second threshold")}
""")
    o.append(
        table(
            [
                "Hardware setting",
                "Sessions",
                "TTFT p95 (s)",
                "E2E p95 (s)",
                "I90 (tok/s/user)",
                "TTFT ≤10",
                "I90 ≥20",
                "Both pass",
                "Errors",
            ],
            [
                [
                    NAMES[p["policy"]],
                    p["clients"],
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p['e2e_p95_s']:.2f}",
                    f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                    pass_label(p["ttft_slo_pass"]),
                    pass_label(p["interactivity_slo_pass"]),
                    pass_label(p["combined_slo_pass"]),
                    p["request_errors"],
                ]
                for p in points
            ],
        )
    )
    o.append("\n**Highest-throughput eligible samples under both run-level limits:**\n")
    o.append(
        table(
            [
                "Policy",
                "Sessions",
                "Total tok/s/GPU",
                "Output tok/s/GPU",
                "TTFT p95 (s)",
                "E2E I90 (tok/s)",
                "Total/RR",
            ],
            [
                [
                    NAMES[p["policy"]],
                    p["clients"],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['output_tok_s_gpu']:.2f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                    f"{p['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}×",
                ]
                for p in [kv, rr, tuned]
            ],
        )
    )
    o.append("""
The combined selections match the TTFT-only selections in this snapshot. The two limits still reject different flag variants: **scale 3 / credit 0.8 passes I90 but fails TTFT**, while **decay 0.5 fails both**. For the interactivity-only criterion, RR can reach 96 sessions, while default KV remains at 480. Two separate run-level percentile limits do not establish the fraction of requests meeting both individual limits. These are closed-loop replay operating points; a production arrival-rate guarantee needs separate validation.

### 2.4 Real KV-router flag sweep at 480 sessions

Temperature **0** and **FCFS** are held fixed for all measured KV rows. Numeric defaults below follow the source recipe's stated defaults; unset flags inherit Dynamo defaults. The exact arguments are preserved with the runner. There are **four KV configurations including baseline**, plus an overloaded RR reference.
""")
    selected = [by[pol, 480] for pol in ["kv", "kvc15", "kvs3c08", "kvd05", "rr"]]
    o.append(
        table(
            [
                "Setting",
                "Prefill-load scale",
                "Overlap credit",
                "Credit decay",
                "Changed arguments",
            ],
            [
                [
                    NAMES[pol],
                    *FLAGS[pol],
                    {
                        "kv": "None; KV defaults",
                        "kvc15": "`--router-kv-overlap-score-credit 1.5`",
                        "kvs3c08": "`--router-prefill-load-scale 3 --router-kv-overlap-score-credit 0.8`",
                        "kvd05": "`--router-kv-overlap-score-credit-decay 0.5`",
                    }[pol],
                ]
                for pol in ["kv", "kvc15", "kvs3c08", "kvd05"]
            ],
        )
    )
    o.append(
        figure(
            "flag-sweep",
            "Measured disagg router flags at 480 sessions: throughput, TTFT, E2E interactivity and cached-input ratio",
        )
    )
    o.append(
        table(
            [
                "480-session setting",
                "Total tok/s/GPU",
                "Δ throughput",
                "TTFT p95 (s)",
                "Δ TTFT",
                "E2E I90",
                "Cached input",
                "Both pass",
            ],
            [
                [
                    NAMES[p["policy"]],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{gain(p, base, 'total_tok_s_gpu'):+.2f}%",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{gain(p, base, 'ttft_p95_s'):+.1f}%",
                    f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                    f"{p['frontend_cached_input_pct']:.2f}%",
                    pass_label(p["combined_slo_pass"]),
                ]
                for p in selected
            ],
        )
    )
    o.append(f"""
**Credit 1.5 is the strongest measured candidate at 480.** Compared with default KV, TTFT p95 decreases **{abs(gain(tuned, base, "ttft_p95_s")):.1f}%**, E2E I90 increases **{gain(tuned, base, "e2e_normalized_interactivity_p90_tps"):.1f}%**, and the frontend cached-input ratio increases **{tuned["frontend_cached_input_pct"] - base["frontend_cached_input_pct"]:.2f} percentage points**. Throughput changes only **{gain(tuned, base, "total_tok_s_gpu"):+.2f}%**. The tail improvement is observed in one trial; repeat both baseline and candidate before calling it reproducible.

**Scale 3 / credit 0.8** reduces total throughput **1.5%** and raises TTFT p95 to **10.48 s**. **Decay 0.5** reduces throughput **3.0%**, raises TTFT p95 to **13.13 s**, and drops I90 below 20. The first variant changes two flags together, so this experiment cannot attribute its regression separately to load scale and credit. Decay 0.5 and credit 1.5 each change one flag from default.

The observed direction differs from the [agg flag sweep](agentx-agg-kv-rr-report.md#24-real-kv-router-flag-sweep), where scale 3 / credit 0.8 helped at 192 sessions. An agg setting should not be copied into disagg without measuring it: the prefill and decode queues, GPU allocation and active request mix differ. The present data support testing stronger prefix preference on D88; they do not prove that any one router formula or bottleneck explains the entire difference.

The source prose still describes a **>2% throughput** winner gate. Its pinned `pick_agentx_flag_winner.py` also accepts **>10% TTFT reduction** with throughput no more than **1% below default** and a passing queue check. Credit 1.5 meets that updated operational rule. Neither rule replaces repeat-based uncertainty analysis.

Candidate router arguments, for a controlled repeat:

```text
--router-mode kv
--router-temperature 0.0
--router-queue-policy fcfs
--router-kv-overlap-score-credit 1.5
```

Load scale 1 and decay 0 remain the source's defaults. Credit 2.0 and a 576-session follow-up are listed in the source orchestration but have no completed row in this 14-run snapshot. They are not plotted as results.

### 2.5 Why KV helps, and what the measurements establish

{figure("cache-load", "Frontend cached-input ratio and client requests in flight versus concurrency")}

At 192 sessions, KV reports **92.6% cached input** versus **61.1% for RR**, alongside TTFT p95 **2.40 versus 31.45 s**. At 480, RR falls to **29.2% cached input**, carries about **336 mean requests in flight**, and has **39 exported client request failures**; KV reports **89.1% cached input** with about **119 in flight** and no request errors. This is consistent with extra prefill work and queue pressure when prefix reuse is lost.

ITL remains relatively low even as TTFT explodes; decode-only speed therefore hides the request-level slowdown. This supports investigating the prefill/admission path first. It does **not** isolate prefill compute, scheduler wait and KV handoff latency: the export scrapes the frontend, not every prefill/decode engine, and this report has no phase-aligned per-engine GPU, KV occupancy or transfer-latency series. Those measurements are needed for a complete bottleneck attribution.

### 2.6 Replay health and comparability

All 14 summaries report `submission_valid: true`; none of the preserved request records is a context-overflow skip or cancellation. **RR480 has 39 client request errors / 8,317 profiling records (0.469%)**, leaving **8,278 successes** for latency percentiles. The summary identifies all 39 as responses without actual content. Separately, the source report cites **353 server-side 300-second disaggregation queue-wait timeouts** and a MNNVL guard verdict of **overload with healthy transport**. Those 353 events have not been reconciled to the client records and must not be used as the AIPerf request-error count. Matching server events to request IDs is an outstanding audit item. A valid replay stamp is not a serving-health certificate; no per-engine transport logs were newly audited here.

The source queue checker marks a run POST-KNEE if TTFT p50 grows by both >1.5× and >2 s from first to last time-quarter, overall TTFT p50 exceeds 20 s, or bad-request fraction exceeds 5%. Recalculation from the preserved records reproduces its verdicts. **RR480 is post-knee because of its standing queue**, despite a lower last-quarter median; **KV1152 has both a standing and growing queue**. AT/PRE-KNEE does not mean that the run passes our TTFT or E2E SLO.
""")
    o.append(
        table(
            [
                "Run",
                "Warmup requests",
                "Warmup wall (s)",
                "Client in-flight mean / peak",
                "TTFT p50 Q1 → Q4 (s)",
                "Queue check",
            ],
            [
                [
                    f"{NAMES[p['policy']]} C{p['clients']}",
                    p["warmup_requests"],
                    f"{p['warmup_wall_s']:,.1f}",
                    f"{p['client_inflight_mean']:.1f} / {p['client_inflight_peak']}",
                    f"{p['ttft_quarter1_p50_s']:.2f} → {p['ttft_quarter4_p50_s']:.2f}",
                    p["knee_check"],
                ]
                for p in points
            ],
        )
    )
    walls = [p["warmup_wall_s"] for p in points if p["clients"] == 480]
    o.append(f"""
All five C480 runs have **445 successful warmup requests**. Their warmup wall times span **{min(walls):,.1f}–{max(walls):,.1f} s**, a **{(max(walls) - min(walls)) / base["warmup_wall_s"] * 100:.2f}%** range relative to default KV. This corrects the source prose's “within 0.5%” statement for the full C480 group. Similar warmup duration is a useful check, but does not establish an undegraded fleet or identical physical cache state. Caches were not flushed between runs; per-play cache busting prevents recycled trace content from simply reusing a previous play's marked prefix, while old blocks can still occupy cache capacity.

The separate plain-replay warmup in the job template exits on a duplicate flag according to the source report; the successful AgentX trajectory warmup is the one used here. Tokenizer name is recorded, but an immutable tokenizer revision and server image digest are not attested by these summaries; this limits future byte-for-byte reproduction. GPU utilization is unmeasured here, not zero.

Closed-loop progression changes the request cohort even at fixed concurrency. The following descriptors make that visible; no assumption of identical completed turns is made.
""")
    o.append(
        table(
            [
                "Run",
                "Successful requests",
                "ISL p50",
                "OSL p90 / p99",
                "Turn index p50",
                "Distinct source traces",
                "Profiled sessions/lane",
            ],
            [
                [
                    f"{NAMES[p['policy']]} C{p['clients']}",
                    f"{p['successful_requests']:,}",
                    f"{p['input_tokens_p50']:,.0f}",
                    f"{p['output_tokens_p90']:,.0f} / {p['output_tokens_p99']:,.0f}",
                    f"{p['turn_index_p50']:g}",
                    p["source_trace_count"],
                    f"{p['profiled_sessions_per_lane']:.2f}",
                ]
                for p in points
            ],
        )
    )
    o.append("""
## 3. Simulation method, calibration gaps and next measurements

### 3.1 Native simulation status

**All 14 hardware samples above are measured, not simulation.** There is no completed, matched Native DynoSim result set for this entire D88 concurrency/flag grid in the pinned source. The [C480 native flag simulation report](agentx-disagg-c480-kv-flags.md) tracks that work separately. As checked at **2026-09-19 03:25 UTC**, it has **0 completed runs; baseline and credit 0.8 failed before exporting results**, and two scale-control runs are active. It uses a **V11 disaggregation extension with frozen V10 AIC timing coefficients**, and should not be labeled the unchanged Native V10 build. Consult that live report for later completion status. Failed or active jobs do not supply a comparison point.

This report does not substitute the source's older custom Python **v5** model for Native V10, or fill missing points with a fitted curve. Matching native-vs-hardware comparisons require the same 8:8 topology, concurrency, router settings and replay configuration, with a recorded native binary hash. Completed agg V10 calibration does not by itself validate disaggregated scheduling or KV handoff.

### 3.2 Historical simulation gap in the source report

The source used the older v5 stream-level model to choose its initial ladder. The table below is a **historical error audit only**; those forecasts are not part of the current Native V10/V11 comparison. Measured latency is much higher than forecast, especially at large populations. This makes the model unsuitable for choosing the D88 SLO boundary without further calibration.
""")
    historical = []
    with (
        STUDY / "reports" / data_name / "source/dynosim_n3u_agentx_d64_v5.csv.txt"
    ).open() as stream:
        for row in csv.DictReader(stream):
            key = (row["policy"], int(row["clients"]))
            if row["pd"] == "8:8" and key in by:
                p = by[key]
                historical.append(
                    [
                        NAMES[key[0]],
                        key[1],
                        f"{float(row['total_tok_s_gpu']):,.0f}",
                        f"{p['total_tok_s_gpu']:,.0f}",
                        f"{float(row['ttft_p95_s']):.2f}",
                        f"{p['ttft_p95_s']:.2f}",
                        f"{p['ttft_p95_s'] / float(row['ttft_p95_s']):.1f}×",
                    ]
                )
    o.append(
        table(
            [
                "Policy",
                "Sessions",
                "Old v5 total/GPU",
                "Measured total/GPU",
                "Old v5 TTFT p95 (s)",
                "Measured TTFT p95 (s)",
                "Measured/old TTFT",
            ],
            historical,
        )
    )
    o.append("""
The measurements establish a prediction gap, not its complete cause. A calibration investigation should separate **prefix residency and frontend overlap estimates**, **prefill admission/queueing**, **prefill–decode handoff and timeout behavior**, and **decode timing under actual batches**. Router tuning is an implementation experiment; changing AIC timing coefficients to fit a queueing error would mix mechanisms. Fit on declared calibration points, then retain a concurrency and flag variant as held-out validation before trusting a simulated knee.

### 3.3 Recommended additional measurements

| Priority | Additional points | Purpose |
| --- | --- | --- |
| 1 | Repeat default KV480 and credit-1.5 KV480; repeat RR72 | Establish noise at the candidate improvement and at RR's very narrow TTFT margin. Alternate run order with identical cache protocol. |
| 2 | KV default and credit 1.5 at 576; refine toward 528 or 624 based on the result | Bisect the measured 480–672 latency boundary under **both** limits. No presumed pass at an unmeasured point. |
| 3 | RR84, then 78 or 90 as appropriate | Refine the 72–96 TTFT boundary after confirming RR72. |
| 4 | Credit 1.25 / 1.5 / 2.0 at C480, with load scale 1 and decay 0 | Isolate credit response; credit 1.5 is currently one trial. Credit 2.0 is planned in the source, not measured in this snapshot. |
| 5 | Scale 1 / 2 / 3 at fixed credit; decay 0 / 0.25 / 0.5 at fixed scale and credit | Separate the two changes in the scale-3/credit-0.8 regression and identify whether mild decay helps before expanding the grid. |
| 6 | Matched native C480 default, credit 1.5, scale 3 / credit 0.8, and decay 0.5 | Test simulation ranking against these measured flags. The initial native sweep needs a credit-1.5 case to cover the new hardware candidate. |

For every added cell, record per-engine cache and KV occupancy, queue/running/prefill counts, phase-aligned GPU telemetry and transfer/wait time, alongside request-level E2E, output length, errors and source-trace IDs. Preserve corpus, tokenizer revision, scenario settings, duration and native/server build identities. Geometric expansion serves capacity exploration; bisection and repeats serve the chosen latency boundary.

## 4. Reproduction and audit trail
""")
    o.append(f"""
The [input manifest]({data_name}/manifest.json) pins the original source commit, every GCS summary, frontend metrics export and SHA-256. The [request-metrics manifest]({data_name}/request-metrics/manifest.json) records hashes of the original full JSONL exports and of compact CSV projections; all phases and statuses are retained without prompt/response text. The input ZIP includes these files and the two report scripts. Summary/token/latency checks, source rounding agreement, queue verdicts, warmup and in-flight reconciliation run every time the report is generated.

The source fleet manifest, benchmark template, router runner and queue-check logic are preserved under [`{data_name}/source/`]({data_name}/manifest.json). [Validation results]({STEM}-validation.json) list the checked inventory and exclusions.

Regenerate offline from the repository root (Python with NumPy, Matplotlib and markdown-it-py):

```bash
python kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_disagg_kv_rr_report.py
```

The initial read-only GCS import used:

```bash
python kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/import_agentx_d88.py \\
  --scratch /tmp/agentx-d88-import
```

Re-importing follows the current source inventory; it intentionally requires network access. Offline regeneration uses the pinned preserved snapshot instead. No simulation or GPU benchmark is launched by either report-generation command.
""")
    return "\n".join(o)


def html_report(markdown, output):
    body = MarkdownIt("commonmark").enable("table").render(markdown)
    for name in FIGURES:
        encoded = base64.b64encode(
            (output / f"{STEM}-{name}.svg").read_bytes()
        ).decode()
        body = body.replace(
            f'src="{STEM}-{name}.png"', f'src="data:image/svg+xml;base64,{encoded}"'
        )

    def heading(match):
        level, text = match.groups()
        ident = re.sub(r"[^a-z0-9]+", "-", re.sub(r"<[^>]*>", "", text).lower()).strip(
            "-"
        )
        return f'<h{level} id="{ident}">{text}</h{level}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", heading, body)
    types = {
        ".svg": "image/svg+xml",
        ".pdf": "application/pdf",
        ".csv": "text/csv",
        ".json": "application/json",
        ".zip": "application/zip",
    }

    def portable_link(match):
        link = match[1]
        if ":" in link or link.startswith("#"):
            return match[0]
        if link == f"{STEM}.html":
            return 'href="#"'
        filename, _, anchor = link.partition("#")
        path = (output / filename).resolve()
        if (
            path.is_relative_to(output.resolve())
            and path.is_file()
            and path.suffix in types
        ):
            encoded = base64.b64encode(path.read_bytes()).decode()
            return f'download="{path.name}" href="data:{types[path.suffix]};base64,{encoded}"'
        if path.is_relative_to(STUDY.parents[1]):
            url = (
                "https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/"
                + path.relative_to(STUDY.parents[1]).as_posix()
            )
            return f'href="{url}{"#" + anchor if anchor else ""}"'
        return match[0]

    body = re.sub(r'href="([^"]+)"', portable_link, body)
    body = re.sub(
        r"(<table>.*?</table>)",
        r'<div class="table-scroll">\1</div>',
        body,
        flags=re.DOTALL,
    )
    html = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>N3U disagg KV vs RR · 64-GPU measured performance</title>
<style>
:root{color-scheme:light;--ink:#162b45;--muted:#52657a;--line:#dce4ed}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1260px;margin:28px auto 64px;padding:38px 54px 64px;background:#fff;border:1px solid var(--line);border-radius:14px;min-width:0}
h1{font-size:36px;line-height:1.2;letter-spacing:-.8px;margin:0 0 18px}h2{font-size:27px;line-height:1.3;margin-top:52px;padding-top:20px;border-top:2px solid var(--line)}
h3{font-size:20px;margin-top:30px}p,li{max-width:1120px;overflow-wrap:anywhere}a{color:#205db1;text-decoration-thickness:1px;text-underline-offset:3px}
.table-scroll{overflow-x:auto;margin:24px 0}table{border-collapse:collapse;width:100%;font-size:13px;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#edf3f9;font-weight:650}tr:nth-child(even) td{background:#f8fafc}img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}
code{font:13px ui-monospace,SFMono-Regular,monospace;background:#edf2f7;padding:2px 4px;border-radius:3px;overflow-wrap:anywhere}pre{padding:18px 20px;background:#edf2f7;border-radius:8px;overflow:auto}pre code{padding:0;background:none;overflow-wrap:normal}
li{margin:9px 0}.eyebrow{font-size:12px;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted);margin-bottom:14px}nav{display:flex;gap:20px;flex-wrap:wrap;font-size:14px;margin:0 0 26px}
@media(max-width:800px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:29px}table{font-size:12px}th,td{padding:8px}h2{font-size:23px}}
@media print{body{background:#fff}main{border:0;max-width:none;margin:0;padding:0}nav{display:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}.table-scroll{overflow:visible}table{font-size:9px}}
</style></head><body><main><div class="eyebrow">Nemotron-3-Ultra 550B · AgentX · 64 GB300 GPUs · D88</div>
<nav><a href="#1-real-data-sweep-and-knee">1. Real sweep</a><a href="#2-kv-versus-rr-at-equal-configuration-and-equal-slo">2. SLO comparisons</a><a href="#2-4-real-kv-router-flag-sweep-at-480-sessions">KV flags at 480</a><a href="#3-simulation-method-calibration-gaps-and-next-measurements">3. Simulation gaps</a><a href="#3-3-recommended-additional-measurements">Next measurements</a><a href="#4-reproduction-and-audit-trail">4. Sources</a></nav>
"""
        + body
        + "</main></body></html>\n"
    )
    (output / f"{STEM}.html").write_text(html)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=STUDY / "reports/agentx-disagg-kv-rr-data"
    )
    parser.add_argument("--output", type=Path, default=STUDY / "reports")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    manifest, points = load_inputs(args.data_dir)
    output = args.output
    # Derived results retain full precision; rounding happens only for display.
    result = {
        "schema_version": 1,
        "kind": "measured_hardware",
        "gpus": 64,
        "topology": "8 prefill + 8 decode TP4",
        "source_commit": manifest["source_commit"],
        "source_report": manifest["source_report"],
        "ttft_p95_slo_s": 10,
        "interactivity_p90_slo_tps": 20,
        "interactivity_formula": "1 / percentile(E2E_seconds / output_tokens, 90, method='linear')",
        "hardware": points,
        "combined_slo_selections": {
            p: pick(points, p)["id"] if pick(points, p) else None for p in ORDER
        },
        "simulation": {
            "matched_native_grid_complete": False,
            "historical_v5_is_native": False,
        },
    }
    (output / f"{STEM}.json").write_text(json.dumps(result, indent=2) + "\n")
    fields = [
        key for key, value in points[0].items() if not isinstance(value, (dict, list))
    ]
    with (output / f"{STEM}.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(points)
    validation = {
        "passed": True,
        "hardware_runs": len(points),
        "counts": dict(Counter(p["policy"] for p in points)),
        "source_commit": manifest["source_commit"],
        "verified_input_hashes": len(manifest["files"]),
        "checks": [
            "source hashes",
            "scenario validity and shared replay configuration",
            "request counts/errors",
            "canonical E2E I90 from per-request records",
            "TTFT p95, E2E p95 and P10 output/E2E versus original summaries",
            "rounded source summary agreement",
            "frontend cache ratio",
            "client in-flight mean and peak",
            "warmup durations",
            "source queue-check verdicts",
        ],
        "profiling_successes": sum(p["successful_requests"] for p in points),
        "profiling_errors": sum(p["request_errors"] for p in points),
        "excluded_successes_from_interactivity": sum(
            p["interactivity_excluded_successful_requests"] for p in points
        ),
        "single_trials_no_confidence_intervals": True,
    }
    (output / f"{STEM}-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    plots(points, output)
    markdown = report(points, manifest, args.data_dir.name)
    (output / f"{STEM}.md").write_text(markdown)
    with zipfile.ZipFile(
        output / f"{STEM}-inputs.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(args.data_dir.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(output).as_posix())
        for name in [
            "gen_disagg_kv_rr_report.py",
            "import_agentx_d88.py",
            "gen_agg_kv_rr_report.py",
        ]:
            archive.write(STUDY / "scripts" / name, f"scripts/{name}")
    html_report(markdown, output)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
