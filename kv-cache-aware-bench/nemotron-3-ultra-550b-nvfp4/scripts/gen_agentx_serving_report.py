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
import yaml
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
    "kvs3c10": "KV scale 3 / default credit 1.0",
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
    "kvs3c10": "#0d705c",
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
FIGURES = [
    "agg-curves",
    "disagg-curves",
    "agg-flags",
    "agg-flags-c160",
    "disagg-flags",
    "operating-points",
    "simulation-agg",
    "simulation-disagg-d88",
    "simulation-disagg-p12d6",
]


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
        p["ttft_slo_pass"] = p["ttft_p95_s"] < 10
        p["combined_slo_pass"] = p["ttft_slo_pass"] and p["interactivity_slo_pass"]
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


def load_native_disagg():
    folder = REPORTS / "agentx-native-disagg-data"
    manifest = read(folder / "manifest.json")
    for filename, spec in manifest["files"].items():
        assert sha(folder / filename) == spec["sha256"], filename
    requests = read(folder / manifest["request_metrics_manifest"])
    points = []
    for entry in manifest["native_runs"] + manifest["legacy_hardware"]:
        summary = read(folder / entry["summary"])
        agg.validate_client(summary, entry["clients"])
        cfg = summary["input_config"]
        assert (
            cfg["datasets"][0]["dataset"] == "semianalysis_cc_traces_weka_062126_256k"
        )
        assert cfg["datasets"][0]["cache_bust"]["target"] == "first_turn_prefix"
        assert cfg["endpoint"]["timeout"] == 1200
        assert (
            cfg["endpoint"]["use_server_token_count"] and cfg["endpoint"]["streaming"]
        )
        assert cfg["endpoint"]["extra"]["ignore_eos"]
        point = entry | agg.metrics(summary, entry["gpus"])
        point.update(agg.request_metrics(folder, requests, point, summary))
        point.update(audit(folder, requests["runs"][point["id"]]))
        point["ttft_slo_pass"] = point["ttft_p95_s"] < 10
        point["combined_slo_pass"] = (
            point["ttft_slo_pass"] and point["interactivity_slo_pass"]
        )
        point["benchmark_id"] = summary["benchmark_id"]
        point["summary_link"] = f"agentx-native-disagg-data/{entry['summary']}"
        point["diagnostic_only"] = point["request_error_rate_pct"] > 5
        if entry["kind"] == "simulation":
            root = folder / "native" / entry["id"]
            assert (
                read(root / "server-environment.json")["core_sha256"]
                == manifest["native_core_sha256"]
            )
            provenance = read(root / "provenance.json")
            assert provenance["clock"] == "wall" and provenance["speedup"] == 1
            topology = read(root / "topology.json")
            assert sum(p["workers"] * 4 for p in topology["pools"]) == entry["gpus"]
            for role in ["prefill", "decode"]:
                engine = read(root / role / "engine.json")
                shared = read(
                    ROOT
                    / "sim-results/agentx_disagg_c480_native_20260919/configs"
                    / f"{role}-engine.json"
                )
                assert engine == shared, (entry["id"], role)
            assert read(root / "warmup-inputs.json")["all_successful_one_token"]
        points.append(point)
    by_id = {p["id"]: p for p in points}
    pairs = []
    for point in points:
        if not point.get("hardware_id"):
            continue
        real = by_id[point["hardware_id"]]
        assert point["benchmark_id"] == real["benchmark_id"]
        assert point["gpus"] == real["gpus"] == 72
        assert (
            point["clients"] == real["clients"]
            and point["policy"] == real["policy"] == "kv"
        )

        def warmup_identity(p):
            with gzip.open(folder / p["request_metrics_file"], "rt") as stream:
                rows = [r for r in csv.DictReader(stream) if r["phase"] == "warmup"]
            assert all(
                r["status"] == "success" and float(r["output_sequence_length"]) == 1
                for r in rows
            )
            return Counter(
                tuple(
                    r[k]
                    for k in [
                        "source_trace_id",
                        "conversation_id",
                        "source_kind",
                        "source_outer_idx",
                        "turn_index",
                        "input_sequence_length",
                    ]
                )
                for r in rows
            )

        assert warmup_identity(point) == warmup_identity(real)
        errors = {
            m: delta(point, real, m) for m in ["total_tok_s_gpu", "ttft_p95_s", I90]
        }
        pairs.append(
            {
                "simulation_id": point["id"],
                "hardware_id": real["id"],
                "relative_error_pct": errors,
                "warmup_inputs_match": True,
            }
        )
    assert len(points) == 10 and len(pairs) == 2
    return {"manifest": manifest, "points": points, "pairs": pairs}


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
        point
        for arch in ["agg", "disagg"]
        for point in [
            best(points, arch, "rr"),
            best(points, arch, "kv"),
            best_tuned(points, arch),
        ]
    ]


def agg_contrasts(points, concurrency=192):
    campaign = "agg-np2-20260919" if concurrency == 192 else "agg-slo10-20260920"
    rows = []
    comparisons = (
        [
            ("kvs3c08", "kv"),
            ("kvd05", "kv"),
            ("kvd10", "kv"),
            ("kvs3c08d05", "kvs3c08"),
        ]
        if concurrency == 192
        else [(p, "kv") for p in ["kvs3c10", "kvs3c08", "kvs2c08", "kvd05", "kvt05"]]
        + [("kvs3c10", "kvs3c08")]
    )
    for treatment, control in comparisons:
        a = one(points, "agg", treatment, concurrency, campaign)
        b = one(points, "agg", control, concurrency, campaign)
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
    for arch in ["agg", "disagg"]:
        fig, axes = plt.subplots(1, 3, figsize=(16, 5.6))
        ticks = (
            [48, 64, 96, 160, 192, 384]
            if arch == "agg"
            else [72, 96, 144, 192, 384, 480, 672, 768, 1152]
        )
        for column, metric, title in [
            (0, "total_tok_s_gpu", "Total input + output tok/s/GPU"),
            (1, "ttft_p95_s", "TTFT p95 (seconds; log scale)"),
            (2, I90, "E2E I90 (output tok/s/user)"),
        ]:
            ax = axes[column]
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
                    additions = [
                        p
                        for p in cells(points, arch, pol)
                        if p["campaign"] == "agg-slo10-20260920"
                    ]
                    ax.scatter(
                        [p["clients"] for p in additions],
                        [p[metric] for p in additions],
                        marker="D",
                        color=COLORS[pol],
                        s=50,
                        zorder=7,
                    )
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
                    selected = best(
                        points,
                        arch,
                        pol,
                        "ttft_slo_pass" if column == 1 else "eligible",
                    )
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
            ax.set_xticks(ticks, [str(n) for n in ticks], rotation=45)
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
                    ax.annotate(
                        "RR peak: C192\nC384: −20.7%; TTFT 289 s",
                        xy=(192, 4419),
                        xytext=(0.48, 0.23),
                        textcoords="axes fraction",
                        fontsize=8,
                        arrowprops={"arrowstyle": "->"},
                    )
                else:
                    ax.text(
                        0.43,
                        0.93,
                        "Sampled KV/RR peak: C192\nC384: KV −14.5%; RR −25.4%\nTTFT rises to 119 / 495 s",
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
                    "SLO <10 s" if column == 1 else "SLO ≥20 tok/s",
                    transform=ax.transAxes,
                    va="top",
                    fontsize=9,
                    color="#a62b3a",
                )
        if arch == "agg":
            boundary = one(points, "agg", "kv", 160)
            axes[1].axvspan(160, 192, alpha=0.08, color=COLORS["kv"])
            axes[1].axvspan(64, 96, alpha=0.06, color=COLORS["rr"])
            axes[1].annotate(
                "TTFT boundary: KV C160–192\nRR C64–96",
                xy=(160, boundary["ttft_p95_s"]),
                xytext=(0.08, 0.08),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
            axes[2].annotate(
                "Both SLOs: KV C96, RR C64\nKV C160 misses I90; refine C96–160",
                xy=(160, boundary[I90]),
                xytext=(0.06, 0.27),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
        else:
            boundary = one(points, "disagg", "kv", 480)
            axes[1].axvspan(480, 672, alpha=0.07, color=COLORS["kv"])
            axes[1].annotate(
                "Default KV: C480 passes both SLOs\nC672 fails; boundary lies in 480–672",
                xy=(480, boundary["ttft_p95_s"]),
                xytext=(0.04, 0.20),
                textcoords="axes fraction",
                fontsize=8,
                arrowprops={"arrowstyle": "->"},
            )
        fig.suptitle(
            f"{'Agg · 24 GPUs' if arch == 'agg' else 'Disagg · 64 GPUs'} · default KV versus round-robin",
            x=0.04,
            ha="left",
            fontsize=16,
        )
        fig.text(
            0.04,
            0.025,
            "Rings: sampled throughput peaks. Stars: TTFT-only choice in the TTFT panel; both-SLO choice in the I90 panel. Shading brackets transitions.\n"
            + (
                "Diamonds: Sep 20 C64/C160 additions; cross: fresh KV192. Lines: Sep 16 ladder. "
                if arch == "agg"
                else "Lines are guides. "
            )
            + "Knees need intermediate points and repeats.",
            fontsize=9,
            color="#52657a",
        )
        fig.subplots_adjust(left=0.065, right=0.99, top=0.83, bottom=0.22, wspace=0.25)
        save(fig, f"{arch}-curves")

    for arch, concurrency, tag, specs in [
        (
            "agg",
            192,
            "agg-flags",
            [(p, "agg-20260916") for p in ["kv", "kvs2c08", "kvs3c08", "kvt05"]]
            + [
                (p, "agg-np2-20260919")
                for p in ["kv", "kvs3c08", "kvd05", "kvd10", "kvs3c08d05"]
            ],
        ),
        (
            "agg",
            160,
            "agg-flags-c160",
            [
                (p, "agg-slo10-20260920")
                for p in ["kv", "kvs3c10", "kvs3c08", "kvs2c08", "kvd05", "kvt05"]
            ],
        ),
        (
            "disagg",
            480,
            "disagg-flags",
            [(p, None) for p in ["kv", "kvc15", "kvc20", "kvs3c08", "kvd05"]],
        ),
    ]:
        selected = [
            one(points, arch, pol, concurrency, campaign) for pol, campaign in specs
        ]
        labels = [
            (
                ""
                if concurrency == 160
                else "np-2: "
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
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.5},
            )
            ax.set_xlim(0, max(values) * 1.20)
            ax.set_title(title, loc="left", fontsize=11)
            if threshold:
                ax.axvline(threshold, color="#a62b3a", linestyle="--")
                ax.set_xlabel(
                    "Dashed: <10 s" if threshold == 10 else "Dashed: ≥20 tok/s",
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
            "Scale 3 / default credit 1.0: +15.0% throughput, −50.4% TTFT versus default KV; both SLOs pass, zero client errors.\nCredit 0.8 also passes both at C160. Their 1.8% throughput gap needs repeats. Other rows have 1–3 client errors; temperature 0.5 fails TTFT."
            if concurrency == 160
            else "Hatched: Sep 16. Solid: later np-2 campaign, including both fresh references. Decay loses alone and on top of scale 3 / credit 0.8.\nNo measured agg C192 variant passes both SLOs. Both scale-3 / credit-0.8 trials pass TTFT and miss I90; each C192 run has 2–3 client errors."
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
        save(fig, tag)

    selected = chosen(points)
    labels = [
        name(p, True)
        .replace("KV scale 3 / default credit 1.0", "tuned KV")
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
        "Best measured operating points · TTFT p95 <10 s and E2E I90 ≥20 tok/s",
        x=0.04,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.04,
        0.015,
        "Tuned agg: scale 3 / default credit 1.0, C160. Tuned D88: credit 1.5, C576. All selected rows have zero exported errors.\n24 versus 64 GPUs: these are normalized observations from different fleet sizes, not proof of linear scaling or a cost forecast.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.21, right=0.98, top=0.86, bottom=0.19, wspace=0.20)
    save(fig, "operating-points")


def table(headers, rows):
    return agg.table(headers, rows)


def pf(value):
    return "Pass" if value else "Fail"


def load_methodology():
    sources = {}

    def record(path):
        key = path.relative_to(ROOT.parents[1]).as_posix()
        sources[key] = {"sha256": sha(path), "bytes": path.stat().st_size}
        return "../" + path.relative_to(ROOT).as_posix()

    candidates = []
    root = ROOT / "sim-results/aic-nvfp4"
    for bracket in ["warm24", "cold24", "warm72", "cold72"]:
        for mode in ["agg", "disagg"]:
            configs = list(
                (root / bracket).glob(f"*/{mode}/top1/generator_config.yaml")
            )
            assert len(configs) == 1
            path = configs[0]
            config = yaml.safe_load(path.read_text())
            experiment = path.parents[1] / "exp_config.yaml"
            exp = yaml.safe_load(experiment.read_text())
            ranking = path.parents[1] / "best_config_topn.csv"
            with ranking.open() as stream:
                result = next(csv.DictReader(stream))
            assert exp["isl"] == 96000 and exp["osl"] == 900
            prefixes = [""] if mode == "agg" else ["prefill_", "decode_"]
            assert all(
                exp[f"{prefix}backend_version"] == "0.5.14" for prefix in prefixes
            )
            assert exp["prefix"] == (92000 if bracket.startswith("warm") else 0)
            worker = config["WorkerConfig"]
            used = sum(
                worker.get(f"{stage}_workers", 0)
                * (worker.get(f"{stage}_gpus_per_worker") or 0)
                for stage in ["agg", "prefill", "decode"]
            )
            candidates.append(
                {
                    "bracket": bracket,
                    "mode": mode,
                    "gpu_budget": exp["total_gpus"],
                    "gpus_in_generated_recipe": used,
                    "workers": worker,
                    "params": config["params"],
                    "predicted": result,
                    "experiment": exp,
                    "config_link": record(path),
                    "ranking_link": record(ranking),
                    "experiment_link": record(experiment),
                }
            )
    engine_paths = {
        "agg": REPORTS
        / "agentx-agg-kv-rr-data/native-v10/agg-kv-c192-calibrated-v10/engine.json",
        "prefill": ROOT
        / "sim-results/agentx_disagg_c480_native_20260919/configs/prefill-engine.json",
        "decode": ROOT
        / "sim-results/agentx_disagg_c480_native_20260919/configs/decode-engine.json",
    }
    engines = {
        role: {"config": read(path), "source": record(path)}
        for role, path in engine_paths.items()
    }
    for role in engines:
        assert (
            engines[role]["config"]["aic_calibration"]
            == engines["agg"]["config"]["aic_calibration"]
        )
    return {
        "aic_version": "0.11.0",
        "aic_recommendations": candidates,
        "native_engines": engines,
        "sources": sources,
    }


def best_tuned(points, arch, criterion="eligible"):
    candidates = [
        p
        for p in cells(points, arch)
        if p["policy"] not in {"kv", "rr"} and p[criterion] and not p["post_knee"]
    ]
    return max(candidates, key=lambda p: p["total_tok_s_gpu"])


def comparison_data(points):
    result = {}
    for arch in ["agg", "disagg"]:
        campaign = "agg-20260916" if arch == "agg" else None
        tuned = "kvs3c08" if arch == "agg" else "kvc15"
        same = [one(points, arch, p, 192, campaign) for p in ["rr", "kv", tuned]]
        slo = [best(points, arch, p, "ttft_slo_pass") for p in ["rr", "kv"]] + [
            best_tuned(points, arch, "ttft_slo_pass")
        ]
        joint = [best(points, arch, p) for p in ["rr", "kv"]] + [
            best_tuned(points, arch)
        ]
        result[arch] = {
            "same_concurrency_reference": 192,
            "same_concurrency_ids": [p["id"] for p in same],
            "ttft_only_selected_ids": [p["id"] for p in slo],
            "joint_slo_selected_ids": [p["id"] for p in joint],
        }
    return result


def simulation_plot(points, native):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for ax, metric, title in zip(
        axes,
        ["total_tok_s_gpu", "ttft_p95_s", I90],
        [
            "Total tokens/s/GPU",
            "TTFT p95 (seconds; log)",
            "E2E I90 (output tok/s/user; log)",
        ],
    ):
        for pol in ["kv", "rr"]:
            hardware = [
                p for p in cells(points, "agg", pol) if p["campaign"] == "agg-20260916"
            ]
            simulated = sorted(
                [p for p in native if p["policy"] == pol], key=lambda p: p["clients"]
            )
            ax.plot(
                [p["clients"] for p in hardware],
                [p[metric] for p in hardware],
                color=COLORS[pol],
                marker="o",
                label=f"{LABELS[pol]} hardware",
            )
            ax.plot(
                [p["clients"] for p in simulated],
                [p[metric] for p in simulated],
                color=COLORS[pol],
                marker="s",
                linestyle="--",
                label=f"{LABELS[pol]} Native V10",
            )
        if metric != "total_tok_s_gpu":
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            ax.axhline(
                10 if metric == "ttft_p95_s" else 20, linestyle=":", color="#a62b3a"
            )
            ax.text(
                0.02,
                10.8 if metric == "ttft_p95_s" else 21.6,
                "TTFT <10 s" if metric == "ttft_p95_s" else "I90 ≥20 tok/s",
                transform=ax.get_yaxis_transform(),
                color="#a62b3a",
                fontsize=8,
            )
        else:
            ax.set_ylim(bottom=0)
            ax.legend(fontsize=8, frameon=False)
        ax.set_xscale("log", base=2)
        ax.set_xticks([48, 96, 192, 384], ["48", "96", "192", "384"])
        ax.set_xlabel("Concurrency · live sessions")
        ax.set_title(title, fontsize=11, loc="left")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle(
        "Agg · hardware versus Native DynoSim V10 · default KV and RR",
        x=0.04,
        ha="left",
        fontsize=16,
    )
    fig.text(
        0.04,
        0.02,
        "C192/C384 were calibration points; C48/C96 are holdouts. The table includes all 12 pairs, including tuned settings.\nThroughput agreement does not guarantee correct latency tails or SLO classifications.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.06, right=0.99, top=0.82, bottom=0.21, wspace=0.25)
    save(fig, "simulation-agg")


def disagg_simulation_plots(points, data):
    native = [
        p
        for p in data["points"]
        if p["kind"] == "simulation" and p["topology"] == "p8d8"
    ]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.8))
    for ax, metric, title in zip(
        axes,
        ["total_tok_s_gpu", "ttft_p95_s", I90],
        [
            "Total input + output tok/s/GPU",
            "TTFT p95 (seconds; log)",
            "E2E I90 (output tok/s/user; log)",
        ],
    ):
        for policy in ["kv", "rr"]:
            real = cells(points, "disagg", policy)
            clean = [
                p for p in native if p["policy"] == policy and p["request_errors"] == 0
            ]
            errored = [
                p for p in native if p["policy"] == policy and p["request_errors"] > 0
            ]
            for group, style, marker, label in [
                (real, "-", "o", "hardware"),
                (clean, "--", "s", "native, zero errors"),
            ]:
                ax.plot(
                    [p["clients"] for p in group],
                    [p[metric] for p in group],
                    color=COLORS[policy],
                    linestyle=style,
                    marker=marker,
                    markersize=4,
                    label=f"{LABELS[policy]} {label}",
                )
            ax.scatter(
                [p["clients"] for p in errored],
                [p[metric] for p in errored],
                color=COLORS[policy],
                marker="x",
                s=85,
                linewidths=2,
                zorder=5,
            )
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_xlabel("Concurrency · live sessions")
        ax.set_xscale("log", base=2)
        ticks = [16, 64, 96, 192, 256, 480, 768, 1152]
        ax.set_xticks(ticks, [str(t) for t in ticks], rotation=45)
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "total_tok_s_gpu":
            ax.set_ylim(0, 18500)
            ax.legend(fontsize=7.5, frameon=False, loc="upper left")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
        else:
            ax.set_yscale("log")
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
            threshold = 10 if metric == "ttft_p95_s" else 20
            ax.axhline(threshold, color="#a62b3a", linestyle=":")
            ax.text(
                0.02,
                threshold * 1.12,
                "TTFT <10 s" if threshold == 10 else "I90 ≥20 tok/s",
                transform=ax.get_yaxis_transform(),
                color="#a62b3a",
                fontsize=8,
            )
    fig.suptitle(
        "Disagg · 8P+8D / 64 GPUs · collected native forecasts alongside hardware",
        x=0.04,
        ha="left",
        fontsize=15,
    )
    fig.text(
        0.04,
        0.02,
        "No sampled concurrency values coincide: these curves do not establish matched-point accuracy. Native V11 uses frozen V10 timing.\n"
        "× = native C256 with errors: KV 11/17,300 (0.064%); RR 3,710/14,871 (24.95%, diagnostic only). No native result at C480 or above.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.83, bottom=0.25, wspace=0.25)
    save(fig, "simulation-disagg-d88")

    matched = [p for p in data["points"] if p["topology"] == "p12d6"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.3))
    for ax, metric, title in zip(
        axes,
        ["total_tok_s_gpu", "ttft_p95_s", I90],
        [
            "Total input + output tok/s/GPU",
            "TTFT p95 (seconds)",
            "E2E I90 (output tok/s/user)",
        ],
    ):
        for kind, color, style, marker in [
            ("hardware", "#2463b3", "-", "o"),
            ("simulation", "#16836b", "--", "s"),
        ]:
            group = sorted(
                [p for p in matched if p["kind"] == kind], key=lambda p: p["clients"]
            )
            ax.plot(
                [p["clients"] for p in group],
                [p[metric] for p in group],
                color=color,
                linestyle=style,
                marker=marker,
                label="KV hardware" if kind == "hardware" else "KV native V11",
            )
            for p in group:
                offset = (0, -17 if kind == "hardware" else 9)
                ax.annotate(
                    f"{p[metric]:,.0f}"
                    if metric == "total_tok_s_gpu"
                    else f"{p[metric]:.2f}",
                    xy=(p["clients"], p[metric]),
                    xytext=offset,
                    textcoords="offset points",
                    ha="center",
                    color=color,
                    fontsize=8,
                )
        ax.set_title(title, fontsize=11, loc="left")
        ax.set_xticks([192, 384], ["192", "384"])
        ax.set_xlim(165, 411)
        ax.set_ylim(bottom=0, top=max(p[metric] for p in matched) * 1.25)
        ax.set_xlabel("Concurrency · live sessions")
        ax.grid(alpha=0.18)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "total_tok_s_gpu":
            ax.legend(frameon=False, fontsize=9)
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    fig.suptitle(
        "Matched disagg checks · 12P+6D / 72 GPUs · KV routing only",
        x=0.04,
        ha="left",
        fontsize=16,
    )
    fig.text(
        0.04,
        0.02,
        "Same topology and concurrency, matched warmup inputs, one-hour profiles; native V11 with frozen V10 timing.\n"
        "Two historical reference points, not the current 64-GPU D88 fleet or an RR validation. Lines connect samples; they do not locate a knee.",
        fontsize=9,
        color="#52657a",
    )
    fig.subplots_adjust(left=0.065, right=0.99, top=0.82, bottom=0.23, wspace=0.25)
    save(fig, "simulation-disagg-p12d6")


def disagg_simulation_markdown(data, status):
    native = [
        p
        for p in data["points"]
        if p["kind"] == "simulation" and p["topology"] == "p8d8"
    ]
    out = [
        """### 4.2 D88: native forecasts alongside the measured KV/RR curves

Six completed native runs cover **8 prefill + 8 decode TP4 workers / 64 GPUs**, with default KV and RR at **C16, C64 and C256**. They use the **V11 disaggregation extension with frozen V10 timing**, the same Weka corpus and replay settings, and the engine configurations in section 5.3. These are existing September 17–18 runs, separate from the failed C480 flag sweep.

**No sampled concurrency values coincide between hardware and native runs.** The graph shows their collected trends; it does not compute accuracy by interpolating an unmeasured hardware or simulation point. No tuned KV points enter these curves.
""",
        figure(
            "simulation-disagg-d88",
            "D88 default KV and RR: hardware and native results at different concurrency grids, with errored native results marked",
        ),
    ]
    out.append(
        table(
            [
                "Native run / summary",
                "C",
                "Total tok/s/GPU",
                "TTFT p95 (s)",
                "E2E I90",
                "Successes / errors",
                "Error rate",
                "Use",
            ],
            [
                [
                    f"[{LABELS[p['policy']]}]({p['summary_link']})",
                    p["clients"],
                    f"{p['total_tok_s_gpu']:,.0f}",
                    f"{p['ttft_p95_s']:.2f}",
                    f"{p[I90]:.4f}",
                    f"{p['successful_requests']:,} / {p['request_errors']:,}",
                    f"{p['request_error_rate_pct']:.3f}%",
                    "Diagnostic only"
                    if p["diagnostic_only"]
                    else "Forecast with errors"
                    if p["request_errors"]
                    else "Completed forecast",
                ]
                for p in native
            ],
        )
    )
    out.append("""**RR256 is an admission-failure diagnostic, not a usable capacity prediction:** 3,710 of 14,871 profiling requests fail (24.95%). Its TTFT and I90 describe successful requests only, so dropping those errors would make the curve misleading. KV256 also has 11 errors (0.064%). Both are crosses outside the zero-error prediction lines. Worker logs contain handoff-session-limit failures; counts and log hashes are preserved with each run. The scenario-valid stamp alone does not validate the serving model.

The lower-concurrency native points show the direction of KV's latency advantage. They do not validate the measured D88 knee, credit-1.5 tuning, or C480/C576 SLO choices. Those require successful native runs at the same hardware concurrency and settings.

### 4.3 Matched disagg comparison: 12P+6D, 72 GPUs, KV only

Two completed native checks **do** have matching real hardware jobs: the earlier **12P+6D TP4 / 72-GPU KV** recipe at **C192 and C384**. These two historical hardware references are additional to the 37 agg/D88 jobs in sections 2–3; they are not substituted for 64-GPU D88 or for RR. Structured pool counts and launch commands establish 72 GPUs; the original topology JSON retains a stale 64-GPU sentence, documented in the source manifest.
""")
    out.append(
        figure(
            "simulation-disagg-p12d6",
            "Matched 72-GPU 12P+6D KV hardware and native throughput, TTFT and E2E interactivity at C192 and C384",
        )
    )
    by_id = {p["id"]: p for p in data["points"]}
    rows = []
    for pair in data["pairs"]:
        s, h = by_id[pair["simulation_id"]], by_id[pair["hardware_id"]]
        rows.append(
            [
                f"[C{h['clients']} hardware]({h['gcs_console']}) / [native]({s['summary_link']})",
                f"{h['total_tok_s_gpu']:,.0f} / {s['total_tok_s_gpu']:,.0f}",
                f"{pair['relative_error_pct']['total_tok_s_gpu']:+.2f}%",
                f"{h['ttft_p95_s']:.3f} / {s['ttft_p95_s']:.3f}",
                f"{pair['relative_error_pct']['ttft_p95_s']:+.2f}%",
                f"{h[I90]:.4f} / {s[I90]:.4f}",
                f"{h['request_errors']} / {s['request_errors']}",
            ]
        )
    out.append(
        table(
            [
                "Matched job",
                "Total/GPU real / native",
                "Throughput error",
                "TTFT p95 real / native (s)",
                "TTFT error",
                "I90 real / native",
                "Errors real / native",
            ],
            rows,
        )
    )
    out.append("""The throughput errors are **−1.10% at C192** and **+0.65% at C384**; TTFT p95 errors are **+0.54%** and **+5.74%**. Both native runs complete without profiling errors. Their 185/349 warmup requests match the respective hardware source/turn/input-token identities; the report independently recomputes request-level TTFT and I90. The timing coefficients remain those derived for agg, and transfer bandwidth/prefill cache sizing remain assumptions. Two checks on this earlier topology do not establish general disagg accuracy or a knee.

### 4.4 The remaining C480 tuning gap

All **12 C480 native flag attempts failed during warmup** with `mocker handoff session limit reached`; no full profiling result exists at that load. The grid tested credits 0.6/0.8/1.0 and did not produce a forecast for the hardware-selected credit-1.5 configuration. Failed warmups are not plotted as zero throughput or as hardware limits.

Fix native handoff admission/backpressure while preserving the intended batch limits, then collect matched D88 default-KV/RR points at C96/C192 and C480. Only after those complete should the tuning grid and C576 SLO choice be evaluated. Validate transfer contention and the prefill cache allocation alongside that work.

[Native disagg input manifest](agentx-native-disagg-data/manifest.json) · [numeric request provenance](agentx-native-disagg-data/request-metrics/manifest.json) · [C480 failure evidence](agentx-serving-perf-data/source/native-c480-status.json) · [C480 sweep manifest](../sim-results/agentx_disagg_c480_native_20260919/plan.json)
""")
    if status:
        assert len(status["runs"]) == 12 and all(
            r["state"] == "failed" and r["handoff_limit_evidence"]
            for r in status["runs"]
        )
    return "\n\n".join(out)


def markdown(points, native, manifest, status, methodology, disagg):
    date = manifest["collected_utc"][:16].replace("T", " ") + " UTC"
    decisions = comparison_data(points)
    by_id = {p["id"]: p for p in points}
    link = lambda p: f"[{LABELS[p['policy']]}]({p['gcs_console']})"
    records = lambda ids: [by_id[i] for i in ids]
    routing_gains = {
        arch: best(points, arch, "kv")["total_tok_s_gpu"]
        / best(points, arch, "rr")["total_tok_s_gpu"]
        for arch in ["agg", "disagg"]
    }
    agg_rr_joint, agg_kv_joint, _ = records(
        decisions["agg"]["joint_slo_selected_ids"]
    )
    disagg_rr_joint, disagg_kv_joint, _ = records(
        decisions["disagg"]["joint_slo_selected_ids"]
    )
    previous_date = (
        manifest["previous_snapshot"]["collected_utc"][:16].replace("T", " ") + " UTC"
    )
    out = [
        f"""# NVIDIA Dynamo on Google Cloud: KV-Aware vs. Round-Robin Routing

*Technical evaluation of prefix-cache locality, throughput, TTFT and end-to-end latency under AgentX replay on GKE.*

[NVIDIA Dynamo](https://github.com/ai-dynamo/dynamo) is an open-source distributed inference framework that coordinates request placement and execution across GPU workers. It integrates with inference engines including SGLang, vLLM and TensorRT-LLM. This Google Cloud **customer use journey (CUJ)** evaluates Dynamo's **KV-cache-aware routing** for agentic inference, using SGLang as the execution backend.

During **prefill**, a model processes an input sequence and materializes **key-value (KV) tensors** for its attention layers. A subsequent request with a matching token prefix can reuse compatible cached state on the selected worker, reducing repeated prefill computation. The reusable prefix depends on cache residency and the model state required by the serving backend.

Dynamo's **KV-aware router** scores eligible workers using prefix overlap and projected active load. Greater overlap reduces estimated prefill work, while assigned prefill and decode work increases the estimated load. **Round-robin (RR)** distributes requests without including prefix overlap in worker selection. Both configurations retain engine prefix caching; the experimental variable is the routing policy and, in the tuning runs, its explicit parameters. [NVIDIA's routing documentation](https://docs.nvidia.com/dynamo/dev/knowledge-base/concepts/system-architecture/kv-aware-routing) describes the placement model.

Agentic coding generates temporally correlated requests. Successive turns repeat instructions, repository context, conversation history and tool outputs; subagent branches can share prefixes with their parent session. Inter-turn delays and competing sessions affect whether that state remains resident. Routing therefore influences both repeated prefill computation and load distribution. Its net effect must be measured through request latency and throughput, because increased cache reuse can coincide with higher queueing delay on a busy worker.

The CUJ covers deployment, workload replay, routing-policy comparison and parameter tuning on **Google Kubernetes Engine (GKE)**. The serving workload is **Nemotron-3-Ultra on NVIDIA GB300 GPUs**, deployed as **24-GPU aggregated serving** and **64-GPU disaggregated serving**. Aggregated workers execute prefill and decode; the disaggregated deployment uses separate prefill and decode pools with a state-transfer stage. Each KV/RR comparison uses the same topology and GPU count within its architecture.

**AIPerf replays AgentX sessions from the Weka 256K corpus** as a closed-loop workload. Requests are reconstructed from trace metadata with the model tokenizer, retaining shared-prefix structure, recorded inter-turn delays and subagent dependencies. Concurrency **C** denotes live session trees. The active request count varies with response completion, think time and subagent execution. Cache-hit rate is an observed result of replay, placement and cache residency.

The evaluation compares routing policies at **fixed session concurrency** and selects operating points under **common latency constraints**. The TTFT criterion is **p95 <10 seconds**. The additional E2E criterion is **I90 ≥20 output tokens/s**, where `I90 = 1 / P90(E2E_seconds / output_tokens)` over successful profiling requests. This normalization includes the interval from request submission to response completion. Default KV/RR curves establish the baseline; separate tables report parameter sweeps and SLO-constrained selections.

At the highest-throughput sampled points satisfying **both** latency criteria, the default-KV/RR ratio is **{routing_gains["agg"]:.2f}× total served tokens/s/GPU for agg** and **{routing_gains["disagg"]:.2f}× for disagg**. The selected concurrency pairs are C{agg_kv_joint["clients"]}/C{agg_rr_joint["clients"]} for agg and C{disagg_kv_joint["clients"]}/C{disagg_rr_joint["clients"]} for disagg (KV/RR). Total served throughput counts input tokens, including cache hits, plus output tokens. Output-only throughput and client errors are reported separately. These ratios characterize the sampled operating points under the specified replay and fleet configurations.

**Evidence snapshot: {date} · {len(points)} completed hardware jobs ({len(cells(points, "agg"))} agg / {len(cells(points, "disagg"))} disagg)**

Each architecture has its own measured curves, full data table, comparison at a sampled knee, and SLO table. The baseline curves show **default KV and RR only**; tuned settings remain in the data and tuning tables.

The snapshot includes **eight additional agg jobs** from [AGENTX_AGG_RESULTS.md, section vi](../AGENTX_AGG_RESULTS.md): RR64, default KV160, five KV variants at C160, and scale-3/credit-0.8 at C256. The D88 measurement cohort is retained from **{previous_date}**; later D88 follow-ups are outside this snapshot.

[Standalone HTML]({STEM}.html) · [hardware CSV]({STEM}.csv) · [comparison JSON]({STEM}.json) · [configuration provenance]({STEM}-methodology.json) · [validation]({STEM}-validation.json)

## 1. Setup, agentic workload and benchmarking methodology

### 1.1 Hardware and serving recipes

Both fleets serve **NVIDIA Nemotron-3-Ultra-550B-A55B-NVFP4** on **GB300**, with modelopt FP4 quantization and TP4/EP4 per worker. The recipes install Dynamo **1.4.2** with its SGLang dependency **0.5.16**, and FlashInfer **0.6.18**. The container base tag alone is not the final installed-version record; immutable image/tokenizer revisions and complete live package inventories were not captured.

| Setting | Aggregated serving | Disaggregated serving (D88) |
| --- | --- | --- |
| Fleet | 6 workers × 4 GPUs = **24 GPUs** | 8 prefill + 8 decode workers × 4 GPUs = **64 GPUs** |
| Work placement | Each worker does both prefill and decode | Separate prefill and decode pools |
| Parallelism per worker | TP4, EP4 | TP4, EP4 on both stages |
| Context / page size | 262,144 tokens / 64 tokens | 262,144 tokens / 64 tokens |
| Explicit admission limit | 16 running requests per worker | Prefill 8; decode 64 per worker |
| Prefill chunk / static memory fraction | 16,384 tokens / 0.85 | Prefill 16,384 tokens / 0.85 on both stages |
| Request and transfer path | NATS request plane; no P→D handoff | NATS request plane; Mooncake handoff, MNNVL/IMEX domain |
| Speculative decoding | Not enabled in the saved recipe | Not enabled in the saved recipe |
| Sources | [agg manifest](agentx-serving-perf-data/source/n3u-agg-newstack-np2.yaml.txt) | [disagg manifest](agentx-serving-perf-data/source/n3u-mnnvl-88.yaml.txt) |

The original agg concurrency sweep was collected on September 16. The subsequent np-2 campaign contains two reference repeats and three decay variants. The September 20 SLO campaign adds RR64, default KV160, five C160 flag variants and tuned KV256. The agg recipes use two separate fleets (`n3u-agg-ns` / `n3u-agg-ns2`). The fixed-concurrency C192 comparison uses the original matched campaign; subsequent references and C160 tuning runs are reported separately. Comparisons between agg and disagg normalize by GPU count but retain differences in fleet size and request mix, which limit architectural scaling inferences.

### 1.2 Agentic workload and replay

AIPerf **0.12.0** runs `inferencex-agentx-mvp` on **`semianalysisai/cc-traces-weka-062126-256k`**, a corpus of 393 recorded coding-session roots with their subagents. The loader reconstructs prompt content from block identifiers and trace lengths using the model tokenizer. Replay preserves prefix-sharing structure and subagent spawn/join dependencies. Within each stream, the next turn follows response completion and the recorded end-to-start delay. Each run samples from the corpus; completion of all 393 roots is not required by the duration limit. The [AgentX replay documentation](https://github.com/ai-dynamo/aiperf/blob/main/docs/tutorials/agentx-mvp.md) describes the scenario; the controls below specify this experiment.

| Replay control | Value used in the measured jobs |
| --- | --- |
| Concurrency C | Live root sessions including their descendants; active request count and decode batch size vary |
| Seed / dataset | 42 / 393 roots from the named Weka 256K corpus |
| Initial trajectory | Initial trace time sampled within 0.25–0.75 of recorded duration; trajectory warmup precedes profiling |
| Timing | Recorded end-to-start delays; whole-system idle-gap cap 10 seconds |
| Measurement | 3,600-second profiling window; 60-second grace; 1,200-second request timeout |
| Prompt/output handling | Model tokenizer; server token counts; streaming; `ignore_eos` preserves recorded output lengths |
| Recycled traces | A fresh first-turn-prefix cache-bust marker per play |
| Validation | Included runs pass the scenario stamp; success and error counts are retained separately |

The [benchmark template](agentx-serving-perf-data/source/sgl-d72-agentx.yaml.txt), [runner](agentx-serving-perf-data/source/agentx_runner_flags.sh.txt) and [agg SLO sweep](agentx-serving-perf-data/source/run_agentx_agg_slo10_np2_v2.sh.txt) preserve the replay configuration and router arguments. Completed exports determine the reported run inventory. A fixed seed controls sampling, while closed-loop execution allows a lower-latency configuration to complete more turns within the same profiling window. Input sequence length (ISL), output sequence length (OSL), conversation depth and source-trace distributions therefore remain comparison variables. Runs with think time disabled are excluded from this workload.

### 1.3 Fair KV/RR comparison and metric definitions

**Routing is the configured treatment variable.** Within each architecture, the comparison fixes the serving recipe and replay controls. RR uses `--router-mode round-robin`; default KV uses `--router-mode kv --router-temperature 0 --router-queue-policy fcfs`. Both retain prefix caching in the agg/prefill engines. Tuned KV adds the arguments listed in each architecture's configuration table. Deployment and sampling limitations are reported below.

| Metric or comparison | Definition |
| --- | --- |
| Total throughput/GPU | AIPerf input + output tokens/s divided by the total fleet GPU count, including both disagg stages. Cached input counts toward served token volume; this metric does not measure newly computed tokens alone. |
| Output throughput/GPU | Output tokens/s divided by the same GPU count. Reported separately to expose workload-mix differences. |
| TTFT | p95 time to first token over successful profiling requests, in seconds. The selection criterion is **strict TTFT p95 <10 s**. No sampled point is exactly 10 s. |
| E2E interactivity I90 | For each successful profiling request with valid latency and positive output length, compute `r_i = E2E_seconds / output_tokens`; then `I90 = 1 / P90(r_i)` using linear interpolation. The additional criterion is **I90 ≥20 tok/s/user**, equivalent to `P90(r_i) ≤0.05 s/token`. |
| Same configuration | Same topology, GPU count, workload and session concurrency; only router settings differ. |
| Same SLO | For each policy, select the highest-throughput sampled point passing TTFT <10 s and the queue check. Selected concurrency may differ between policies. Repeats remain separate; the selected value is a sample maximum rather than a mean or confidence bound. |
| Combined SLO | Apply TTFT <10 s **and** I90 ≥20. This is shown separately from the TTFT-only table. |
| Errors | Failed profiling requests are excluded from latency percentiles and reported separately. Operating-point selection does not apply an additional availability SLO. |

**Knee identification:** the report examines throughput slope or decline, TTFT tails, errors and within-run latency progression. Throughput knees are reported at the resolution of the sampled concurrency sweep; an SLO crossing defines a separate boundary. The queue check uses sustained median TTFT, first-to-last-quarter growth in median TTFT, or an error rate above 5% as overload indicators. These are client-observed indicators rather than direct measurements of queue occupancy. GPU utilization alone does not identify either boundary.

Most configuration/concurrency combinations have one trial. The agg C192 reference repeats provide limited repeatability evidence; confidence intervals are not estimated. Caches were not explicitly flushed between every hardware run, so cache busting and successful warmup do not establish identical initial cache state. `overall_usage_prompt_cache_read_pct` measures the client-reported cached-input fraction; per-engine KV occupancy is a separate quantity. Scenario validity checks replay rules. The latency results characterize this closed-loop session population and do not establish latency at an independently controlled production arrival rate.

**E2E accounting:** the source results log's “P90 interactivity” uses inverse P90 inter-token latency. It excludes TTFT and is not the E2E-normalized I90 above. This report recomputes I90 from each successful profiling request's full latency and output length; consequently, default KV160 and scale-2/credit-0.8 C160 pass TTFT but fail I90 ≥20.
"""
    ]

    for number, arch in [(2, "agg"), (3, "disagg")]:
        title = (
            "Aggregated serving: measured KV and RR"
            if arch == "agg"
            else "Disaggregated serving: measured KV and RR"
        )
        subset = cells(points, arch)
        out.append(
            f"## {number}. {title}\n\n### {number}.1 Default KV versus RR: curves and knee evidence\n\n"
            + figure(
                f"{arch}-curves",
                f"{title}: default KV and RR only, with throughput peaks and latency boundaries",
            )
        )
        if arch == "agg":
            out.append("""| Policy | Sampled throughput / knee evidence | TTFT <10 s boundary | Additional E2E boundary |
| --- | --- | --- | --- |
| Default KV | Peak at **C192**; C384 loses **14.5%** throughput and TTFT p95 rises **11.66→119.44 s**. Saturation transition lies in 192–384. | **C160 passes at 9.57 s**; both C192 references fail. Refine **160–192**. | C96 passes; **C160 fails at I90 16.1267**. Refine **96–160**. |
| RR | C96→192 adds only **10.8%** throughput; C384 loses **25.4%** versus C192. C192 is the sampled peak; diminishing returns begin over 96–192. | **C64 passes at 9.21 s**; C96 fails. Refine **64–96**. | **C64 passes at I90 35.3472**; C96 fails. Refine **64–96**. |

The throughput-knee comparison below uses **C192 for both arms**. It does not claim that C192 meets the latency SLO. Diamonds mark the new RR64/default-KV160 samples and a cross marks the fresh default-KV192 reference; the original ladder remains the line so campaigns are not silently combined. Stars in the TTFT panel select the best TTFT-only points; stars in the I90 panel select the best points meeting both limits. These SLO brackets combine dated campaigns and need matched repeats before claiming an exact crossing.""")
        else:
            out.append("""| Policy | Sampled throughput / knee evidence | TTFT <10 s boundary | Additional E2E boundary |
| --- | --- | --- | --- |
| Default KV | C672→768 adds only **2.4%** throughput while TTFT rises **59%**. C768 is the sampled peak; C1152 then loses **31.2%** and develops growing queues. Plateau begins around **672–768**. | C480 passes; C672 fails. Refine **480–672**. | I90 also crosses between 480 and 672. |
| RR | C192 is the sampled peak; C384 loses **20.7%**, has 68 errors and TTFT p95 **289.39 s**. Overload transition lies in **192–384**. | C72 passes by only **0.0885 s**; C96 fails. Refine **72–96**. | C96 passes I90; C144 fails. |

There is **no shared throughput knee** for KV and RR. The same-concurrency table uses **C192, RR's sampled peak**, where tuned credit 1.5 also has a real measurement. No tuned/RR pair exists at the default-KV plateau of 672–768; that comparison cannot be filled by extrapolation.""")
        out.append(
            f"### {number}.2 All collected data points, including tuned KV\n\nEvery completed {arch} run in the scoped inventory is shown, including repeated references. The flags identify the recipe; each linked name opens its hardware artifacts."
        )
        out.append(
            table(
                [
                    "Setting / artifacts",
                    "C",
                    "Campaign",
                    "Total tok/s/GPU",
                    "Output tok/s/GPU",
                    "TTFT p95 (s)",
                    "E2E I90",
                    "TTFT <10",
                    "Both SLOs",
                    "Errors",
                ],
                [
                    [
                        link(p),
                        p["clients"],
                        "Sep 16"
                        if p["campaign"] == "agg-20260916"
                        else "Sep 20 SLO"
                        if p["campaign"] == "agg-slo10-20260920"
                        else "np-2"
                        if arch == "agg"
                        else "D88",
                        f"{p['total_tok_s_gpu']:,.0f}",
                        f"{p['output_tok_s_gpu']:.2f}",
                        f"{p['ttft_p95_s']:.2f}",
                        f"{p[I90]:.4f}",
                        pf(p["ttft_slo_pass"]),
                        pf(p["combined_slo_pass"]),
                        p["request_errors"],
                    ]
                    for p in subset
                ],
            )
        )
        coverage = []
        for pol in ORDER:
            group = cells(points, arch, pol)
            if group:
                counts = Counter(p["clients"] for p in group)
                coverage.append(
                    [
                        LABELS[pol],
                        ", ".join(
                            str(c) + (f" ({n} runs)" if n > 1 else "")
                            for c, n in sorted(counts.items())
                        ),
                        f"`{group[0]['router_flags']}`",
                    ]
                )
        out.append(
            table(
                ["Recipe", "Collected concurrency", "Exact router arguments"], coverage
            )
        )
        out.append(
            f"### {number}.3 Tuned KV versus RR at the same configuration, near the sampled knee"
        )
        same = records(decisions[arch]["same_concurrency_ids"])
        rr = same[0]
        out.append(
            table(
                [
                    "C192 setting",
                    "Total tok/s/GPU",
                    "Throughput / RR",
                    "TTFT p95 (s)",
                    "TTFT ratio (RR / policy)",
                    "E2E I90",
                    "Errors",
                ],
                [
                    [
                        link(p),
                        f"{p['total_tok_s_gpu']:,.0f}",
                        f"{p['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}×",
                        f"{p['ttft_p95_s']:.2f}",
                        f"{rr['ttft_p95_s'] / p['ttft_p95_s']:.2f}×",
                        f"{p[I90]:.4f}",
                        p["request_errors"],
                    ]
                    for p in same
                ],
            )
        )
        if arch == "agg":
            out.append(
                "At C192, tuned KV has **1.62× RR total served throughput/GPU**. TTFT p95 is **6.33 s for tuned KV** and **60.08 s for RR**, giving a **9.49× RR/KV latency ratio**. These measurements come from the September 16 campaign. The subsequent tuned repeat is reported separately; that campaign did not include a matched RR192 repeat."
            )
        else:
            out.append(
                "At C192, KV with overlap credit 1.5 has **1.16× RR total served throughput/GPU**. TTFT p95 is **2.30 s for tuned KV** and **31.45 s for RR**, giving a **13.68× RR/KV latency ratio**. Default and tuned KV have similar throughput at this concurrency. At C480, the tuned-KV/RR throughput ratio is **3.80×**; the RR run exhibits overload. The C480 ratio therefore compares different saturation states and does not establish a capacity ratio under a common SLO."
            )
        out.append(
            f"### {number}.4 Tuned KV versus RR under the same SLO: TTFT p95 <10 seconds\n\nFor each policy, select the highest measured total served throughput passing the **TTFT-only** criterion and queue check. I90 ≥20 is evaluated separately in the last column."
        )
        chosen_ttft = records(decisions[arch]["ttft_only_selected_ids"])
        rr = chosen_ttft[0]
        out.append(
            table(
                [
                    "Policy / selected run",
                    "C",
                    "Total tok/s/GPU",
                    "Output tok/s/GPU",
                    "TTFT p95 (s)",
                    "E2E I90",
                    "Throughput / RR",
                    "I90 ≥20",
                ],
                [
                    [
                        link(p),
                        p["clients"],
                        f"{p['total_tok_s_gpu']:,.0f}",
                        f"{p['output_tok_s_gpu']:.2f}",
                        f"{p['ttft_p95_s']:.2f}",
                        f"{p[I90]:.4f}",
                        f"{p['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}×",
                        pf(p["interactivity_slo_pass"]),
                    ]
                    for p in chosen_ttft
                ],
            )
        )
        if arch == "agg":
            fresh = one(points, "agg", "kvs3c08", 192, "agg-np2-20260919")
            default = chosen_ttft[1]
            tuned = chosen_ttft[2]
            out.append(
                f"**TTFT-only selection:** default KV160/RR64 has a **{default['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}× total served throughput/GPU ratio**. Tuned KV192/RR64 has a **{tuned['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}×** ratio using the highest-throughput original sample. The np-2 C192 repeat measures **{fresh['total_tok_s_gpu']:,.0f} total tok/s/GPU, {fresh['ttft_p95_s']:.2f} s TTFT p95 and {fresh['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}× RR64 throughput**, corresponding to the comparison in the source log. The original and repeated C192 trials have three and two client errors, respectively. Default KV160 has three errors; RR64 has zero.\n\nTuned C256 measures **12,328 total tok/s/GPU** with **18.87 s TTFT p95** and fails the TTFT criterion. For scale 3/credit 0.8, the sampled TTFT crossing is bounded by **C192–256**. Both tuned C192 trials and default KV160 fail the additional E2E criterion."
            )
        else:
            out.append(
                "**TTFT-only selection:** KV576 with credit 1.5/RR72 has a **7.95× total served throughput/GPU ratio**. Both runs also satisfy I90 ≥20 and have zero exported client errors. RR72's TTFT p95 is close to the 10-second threshold; repeated measurements are required to establish its margin. Default KV576 is outside the retained D88 snapshot, so the table does not quantify the tuning effect at fixed concurrency."
            )
        out.append("**Combined TTFT and E2E selection:** apply both latency criteria:")
        joint = records(decisions[arch]["joint_slo_selected_ids"])
        rr = joint[0]
        out.append(
            table(
                [
                    "Policy",
                    "Selected C",
                    "Total tok/s/GPU",
                    "TTFT p95 (s)",
                    "E2E I90",
                    "Throughput / RR",
                ],
                [
                    [
                        LABELS[p["policy"]],
                        p["clients"],
                        f"{p['total_tok_s_gpu']:,.0f}",
                        f"{p['ttft_p95_s']:.2f}",
                        f"{p[I90]:.4f}",
                        f"{p['total_tok_s_gpu'] / rr['total_tok_s_gpu']:.2f}×",
                    ]
                    for p in joint
                ],
            )
        )
        if arch == "agg":
            out.append(
                "The default-KV/RR throughput ratio under the combined SLO is **1.75×**, compared with 2.11× in the previous snapshot. This change results from selecting RR64 in place of RR48; the default-KV96 measurement is unchanged."
            )
        out.append(f"### {number}.5 What the flag sweep establishes")
        if arch == "agg":
            baseline = one(points, "agg", "kv", 160)
            out.append(
                "**C160 parameter sweep:** load scale 3 with default overlap credit 1.0 has the highest measured agg throughput among the sampled configurations satisfying both SLOs.\n\n"
                + figure(
                    "agg-flags-c160",
                    "New agg C160 flag sweep: measured throughput, TTFT and E2E interactivity",
                )
            )
            out.append(
                table(
                    [
                        "C160 KV setting",
                        "Total tok/s/GPU",
                        "Δ throughput",
                        "TTFT p95 (s)",
                        "E2E I90",
                        "Cached input",
                        "Both SLOs",
                        "Errors",
                    ],
                    [
                        [
                            link(p),
                            f"{p['total_tok_s_gpu']:,.0f}",
                            f"{delta(p, baseline, 'total_tok_s_gpu'):+.2f}%",
                            f"{p['ttft_p95_s']:.2f}",
                            f"{p[I90]:.4f}",
                            f"{p['cache_pct']:.1f}%",
                            pf(p["combined_slo_pass"]),
                            p["request_errors"],
                        ]
                        for p in [
                            one(points, "agg", pol, 160)
                            for pol in [
                                "kv",
                                "kvs3c10",
                                "kvs3c08",
                                "kvs2c08",
                                "kvd05",
                                "kvt05",
                            ]
                        ]
                    ],
                )
            )
            scale3 = one(points, "agg", "kvs3c10", 160)
            credit08 = one(points, "agg", "kvs3c08", 160)
            out.append(
                f"**Selected agg candidate:** prefill load scale 3, temperature 0, default overlap credit 1.0 and default decay 0. The additional argument is `--router-prefill-load-scale 3.0`. Relative to default KV at C160, the measured differences are **{delta(scale3, baseline, 'total_tok_s_gpu'):+.1f}% total throughput** and **{delta(scale3, baseline, 'ttft_p95_s'):.1f}% TTFT p95**. The candidate has **I90 {scale3[I90]:.4f}** and zero exported client errors. The cached-input fraction is **{baseline['cache_pct']:.1f}% for default KV and {scale3['cache_pct']:.1f}% for this candidate**.\n\nScale 3 with credit 0.8 also satisfies both criteria at C160, with one client error. Default credit has **{delta(scale3, credit08, 'total_tok_s_gpu'):+.1f}%** observed throughput relative to credit 0.8. Repeats are required to distinguish this difference from run-to-run variation. The default-credit configuration changes only load scale relative to default KV; the use of separate fleets limits causal attribution from these individual runs.\n\nScale 2/credit 0.8 satisfies TTFT but fails I90 at **19.6589**. Decay 0.5 has **−0.7%** throughput relative to default KV at C160 and fails I90. Temperature 0.5 has **15.3% lower throughput** and fails TTFT at **19.17 s**. C160 tuning-run warmups span approximately **1,389–1,397 s**, compared with **1,492 s** for default KV. Warmup duration is a diagnostic; repeatability requires repeated profiling measurements.\n\n**C192 parameter sweep:** the retained sweep evaluates a higher session concurrency. No sampled C192 configuration satisfies both SLOs."
            )
            out.append(
                figure(
                    "agg-flags",
                    "Retained agg C192 flag sweep, with TTFT and E2E thresholds",
                )
            )
            out.append(
                table(
                    [
                        "C192 np-2 treatment",
                        "C192 np-2 reference",
                        "Δ total throughput",
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
            out.append(
                "The C192 reference repeats differ from the original throughput measurements by less than 2%. The tuned repeat has I90 **19.7385**, compared with **19.7795** originally; both fail the E2E criterion. Decay 0.5/1.0 alone and decay 0.5 added to scale 3/credit 0.8 have lower measured throughput and higher TTFT than their corresponding references. Small throughput differences require additional repeats. Initial decay-only warmups lasted approximately 1,733 s, compared with 1,632–1,634 s for subsequent controls. A startup transient is a possible explanation; these observations do not establish a systematic np-2 performance difference.\n\n**Additional measurements:** repeat default KV160, RR64 and both scale-3 C160 variants; collect **scale 3/default credit 1.0 at C192**, which is absent from this snapshot. For the measured credit-0.8 recipe, refine **C160–192** for the combined SLO and **C192–256** for TTFT alone. Additional default-KV160–192 and RR64–96 points would refine their TTFT crossings. The C160 table uses default KV as its reference because no RR160 measurement is available."
            )
        else:
            out.append(
                figure(
                    "disagg-flags",
                    "Measured disagg KV routing flags, with TTFT and E2E thresholds",
                )
            )
            baseline = one(points, "disagg", "kv", 480)
            out.append(
                table(
                    [
                        "C480 KV setting",
                        "Total tok/s/GPU",
                        "Δ throughput",
                        "TTFT p95 (s)",
                        "Δ TTFT",
                        "E2E I90",
                        "Both SLOs",
                    ],
                    [
                        [
                            LABELS[p["policy"]],
                            f"{p['total_tok_s_gpu']:,.0f}",
                            f"{delta(p, baseline, 'total_tok_s_gpu'):+.2f}%",
                            f"{p['ttft_p95_s']:.2f}",
                            f"{delta(p, baseline, 'ttft_p95_s'):+.1f}%",
                            f"{p[I90]:.4f}",
                            pf(p["combined_slo_pass"]),
                        ]
                        for p in subset
                        if p["clients"] == 480 and p["policy"] != "rr"
                    ],
                )
            )
            out.append(
                "**Overlap credit 1.5 has the lowest measured TTFT p95 among the sampled C480 configurations:** TTFT p95 −15.4%, I90 +13.1%, total throughput +0.29% relative to default KV. Credit 2.0 has similar throughput, higher TTFT and lower I90 than credit 1.5. Scale 3/credit 0.8 fails TTFT; decay 0.5 fails both criteria. The agg-selected configuration therefore requires independent validation on the disaggregated topology.\n\n**D88 measurement scope:** temperature 0.5/0.2 at C480 and default KV576 are outside the retained cohort and are excluded from these selections. RR384 and RR480 have 68 and 39 client errors, respectively. The 353 server timeout events reported separately for RR480 represent a different counting scope from exported client errors."
            )

    a, d = best_tuned(points, "agg"), best_tuned(points, "disagg")
    out.append(f"""### 3.6 Agg and disagg under both SLOs

At their best sampled points satisfying **both** limits, tuned agg uses **C{a["clients"]} with load scale 3/default credit 1.0** and tuned D88 uses **C{d["clients"]} with credit 1.5**. D88 delivers **{d["total_tok_s_gpu"] / a["total_tok_s_gpu"]:.2f}× total throughput/GPU** and **{d["output_tok_s_gpu"] / a["output_tok_s_gpu"]:.2f}× output throughput/GPU**, on 64 versus 24 GPUs. The fleet totals are **{d["total_tok_s_fleet"]:,.0f} versus {a["total_tok_s_fleet"]:,.0f} total tok/s**. Different fleet sizes and completed request mixes prevent interpreting this as a controlled scaling or cost result.

{figure("operating-points", "Best measured agg and disagg operating points passing both SLOs")}

## 4. Simulation versus real hardware jobs

### 4.1 Agg: twelve paired Native DynoSim V10 results

These are **12 actual native simulations paired with the original 12 agg hardware jobs**: four calibration points (default KV/RR at C192/C384) and eight holdouts (default KV/RR at C48/C96 and the four original tuned jobs). The simulator build, engine configuration and timing coefficients are shared. **{len(cells(points, "agg")) - len(native)} later agg hardware jobs** have no new native execution paired to their run IDs: two repeats, three C192 decay variants, and the eight new C64/C160/C256 jobs. In particular, the new load-scale-3/default-credit C160 selection is supported by hardware measurements, not a new simulation.

{figure("simulation-agg", "Default KV and RR: original hardware versus Native DynoSim V10 throughput, TTFT and E2E interactivity")}
""")
    rows = []
    for s in sorted(native, key=lambda p: (ORDER.index(p["policy"]), p["clients"])):
        h = by_id[s["hardware_id"]]
        rows.append(
            [
                LABELS[s["policy"]],
                s["clients"],
                s["role"],
                f"{h['total_tok_s_gpu']:,.0f} / {s['total_tok_s_gpu']:,.0f}",
                f"{delta(s, h, 'total_tok_s_gpu'):+.1f}%",
                f"{h['ttft_p95_s']:.2f} / {s['ttft_p95_s']:.2f}",
                f"{h[I90]:.4f} / {s[I90]:.4f}",
                f"{pf(h['combined_slo_pass'])} / {pf(s['combined_slo_pass'])}",
                f"{h['request_errors']} / {s['request_errors']}",
            ]
        )
    out.append(
        table(
            [
                "Setting",
                "C",
                "Role",
                "Total/GPU real / sim",
                "Throughput error",
                "TTFT p95 real / sim (s)",
                "I90 real / sim",
                "Both SLOs real / sim",
                "Errors real / sim",
            ],
            rows,
        )
    )
    out.append(
        table(
            [
                "Subset",
                "Points",
                "Mean absolute throughput error",
                "Mean absolute TTFT p95 error",
                "Mean absolute I90 error",
            ],
            [
                [
                    role,
                    len(group),
                    *[
                        f"{100 * np.mean([abs(p['hardware_comparison'][m]['relative_error']) for p in group]):.1f}%"
                        for m in ["total_tok_s_gpu", "ttft_p95_s", I90]
                    ],
                ]
                for role in ["calibration", "holdout"]
                if (group := [p for p in native if p["role"] == role])
            ],
        )
    )
    out.append("""The four-point acceptance gate covered **±20% total throughput**, not TTFT or I90. All eight holdouts also fall within ±20% throughput. The original model reproduces the sampled default-policy throughput decline from C192 to C384 and the direction of the scale-3/credit-0.8 and temperature-0.5 effects, but it underestimates the RR384 TTFT tail by **38.4%**.

**SLO classification error:** simulated default KV192 passes TTFT (8.71 s) while hardware fails (11.66 s). Simulated tuned KV192 gives I90 **22.5354** while hardware gives **19.7795**; it incorrectly passes the combined SLO and selects C192 where hardware selects C96 **within the original paired cohort**. The expanded hardware inventory now selects tuned C160, which has no native counterpart. There is **one combined-SLO classification disagreement among 12 pairs**. Throughput calibration therefore supports candidate screening; SLO selection still requires hardware validation. [Original paired inputs and calibration/holdout provenance](agentx-agg-kv-rr-report.md#31-native-dynosim-v10-current-completed-calibration-samples).
""")
    out.append(disagg_simulation_markdown(disagg, status))

    out.append("""## 5. How we simulate performance: AIC, DynoSim and recipe selection

### 5.1 What each component contributes

| Component | Input and role | Output used here |
| --- | --- | --- |
| AIPerf | Recorded sessions, tokenizer, seed, lane count and replay rules | Actual request timing, branches, warmup, recycling, streaming metrics and request exports |
| AIConfigurator (AIC) | Model, accelerator, backend, precision and parallelism; pass shape or search constraints | Estimated prefill/decode forward-pass duration; candidate worker shapes, batch limits, P:D ratios, memory estimates and generated deployment/benchmark files |
| Native Dynamo / DynoSim mocker | Requests, router flags, worker topology, scheduler rules and cache capacity | Placement, prefix reuse, admission, batch/chunk composition, cache state and time spent waiting around model work |
| Hardware validation | AIPerf against the real serving fleet | Determines whether a candidate's throughput, latency tails, errors and SLO result actually reproduce |

AIC estimates forward-pass execution time. The native scheduler determines batch composition and execution timing; the saved calibration coefficients adjust the AIC estimates for this experiment. This division is described in [NVIDIA's DynoSim explanation](https://developer.nvidia.com/blog/dynosim-simulating-the-pareto-frontier/) and [AIC 0.11.0](https://github.com/ai-dynamo/aiconfigurator/tree/v0.11.0). The configuration tables below describe **our saved experiment**, not every capability of the current upstream projects.

Our native route is:

```text
AIPerf AgentX (normal wall clock, streaming HTTP)
    → Dynamo 1.4.2 frontend and actual KV / RR routing
    → native SGLang mocker scheduler and hybrid cache state
    → AIC 0.11.0 forward-pass timing + shared calibration
    → streaming response timing → AIPerf TTFT / E2E / throughput
```

The native V10 build is a local patched candidate, not an NVIDIA release named V10. It uses speedup **1.0** for prefill and decode. AIPerf still owns the AgentX replay; no chosen cache-hit rate or per-concurrency throughput multiplier substitutes for that replay.

### 5.2 The detailed configurations AIC generated

The repository preserves **SILICON-mode AIC 0.11.0 solves** for 24- and 72-GPU budgets using SGLang **0.5.14**, fixed **ISL 96,000 / OSL 900**, context 262,144 and chunked prefill. The **warm** bracket sets a synthetic 92,000-token prefix (95.83% of ISL) and a 5,000 ms TTFT solver target; the **cold** bracket sets prefix 0 and a 30,000 ms target. Both use a **10 ms TPOT** target. These are fixed-shape solver inputs and predicted point estimates, not AgentX session concurrency, measured cache hit, or p95 SLO results.

The table shows each saved **rank-1 generated recipe**. MoE parallelism is written as tensor-parallel size / expert-parallel size. The predicted throughput column counts **output tokens only** and uses the GPUs active in that candidate; it must not be compared directly with our total-token AgentX throughput.
""")
    rows = []
    for c in methodology["aic_recommendations"]:
        w, params, predicted = c["workers"], c["params"], c["predicted"]
        if c["mode"] == "agg":
            p = params["agg"]
            layout = f"{w['agg_workers']} × TP{p['tensor_parallel_size']}"
            moe = f"{p['moe_tensor_parallel_size']} / {p['moe_expert_parallel_size']}"
            batch = str(p["max_batch_size"])
        else:
            p, d = params["prefill"], params["decode"]
            layout = f"{w['prefill_workers']}P × TP{p['tensor_parallel_size']} + {w['decode_workers']}D × TP{d['tensor_parallel_size']}"
            moe = f"P {p['moe_tensor_parallel_size']}/{p['moe_expert_parallel_size']}; D {d['moe_tensor_parallel_size']}/{d['moe_expert_parallel_size']}"
            batch = f"P {p['max_batch_size']}; D {d['max_batch_size']}"
        rows.append(
            [
                f"[{c['bracket']} {c['mode']}]({c['config_link']})",
                f"{c['gpu_budget']} / {c['gpus_in_generated_recipe']}",
                layout,
                moe,
                batch,
                f"{float(predicted['tokens/s/gpu']):.2f}",
                f"{float(predicted['ttft']) / 1000:.3f} / {float(predicted['tpot']):.3f}",
            ]
        )
    out.append(
        table(
            [
                "AIC case / generated config",
                "GPU budget / used",
                "Worker recipe",
                "MoE TP / EP",
                "Batch per worker",
                "Predicted output tok/s/active GPU",
                "Predicted TTFT s / TPOT ms",
            ],
            rows,
        )
    )
    out.append("""All listed candidates use PP1/DP1, FP8 static GEMM, NVFP4 MoE, FP8 KV/FMHA and half-precision communication in the saved AIC search. The generator emits `generator_config.yaml`, worker launch scripts, Kubernetes deployment/benchmark YAML and benchmark scripts; the ranking CSV and experiment YAML retain the objective and search inputs. Exact configurations, hashes and links are in the [methodology data](agentx-serving-perf-report-methodology.json).

**The proposed and deployed recipes differ.** Warm24 agg's top generated proposal is **3×TP8**, whereas our real agg fleet is **6×TP4**. Warm72 disagg's top proposal is **1P+7D at TP8**, using **64 of the 72 budgeted GPUs**; our tested 64-GPU D88 fleet is **8P+8D at TP4**. Cold and warm searches select very different P:D ratios. AIC therefore supplies starting shapes and timing data; these saved solves do not certify the measured TP4 recipes, router flags, or AgentX knees. The generated templates also contain old runtime defaults and local model paths, so they require reconciliation with the serving manifests rather than direct deployment unchanged.

### 5.3 Exact native agg and disagg simulation inputs

The timing identity below is common: **AIC 0.11.0, GB300, SGLang 0.5.14 tables, TP4, MoE TP1/EP4, attention DP1, FP8 GEMM, NVFP4 MoE, FP8 KV and BF16 FMHA**. The BF16 FMHA choice differs from the historical AIC search's FP8 FMHA. Hardware recipes specify SGLang 0.5.16; database/framework mismatch is part of the remaining modeling gap.
""")
    engines = methodology["native_engines"]
    field_rows = []
    descriptors = [
        ("Workers / total GPUs", ["6 agg / 24", "8 prefill / 32", "8 decode / 32"]),
        (
            "Role",
            [engines[r]["config"]["worker_type"] for r in ["agg", "prefill", "decode"]],
        ),
        (
            "Maximum sequences",
            [
                engines[r]["config"]["max_num_seqs"]
                for r in ["agg", "prefill", "decode"]
            ],
        ),
        (
            "Attention blocks × block size",
            [
                f"{engines[r]['config']['num_gpu_blocks']:,} × {engines[r]['config']['block_size']}"
                for r in ["agg", "prefill", "decode"]
            ],
        ),
        (
            "Attention token capacity per worker",
            [
                f"{engines[r]['config']['num_gpu_blocks'] * engines[r]['config']['block_size']:,}"
                for r in ["agg", "prefill", "decode"]
            ],
        ),
        (
            "Token budget / chunk size",
            [
                f"{engines[r]['config']['max_num_batched_tokens']:,} / {engines[r]['config']['sglang']['chunked_prefill_size']:,}"
                for r in ["agg", "prefill", "decode"]
            ],
        ),
        (
            "Prefix caching",
            [
                str(engines[r]["config"]["enable_prefix_caching"])
                for r in ["agg", "prefill", "decode"]
            ],
        ),
        (
            "Mamba state slots",
            [
                engines[r]["config"]["sglang"]["mamba_state_capacity"]
                for r in ["agg", "prefill", "decode"]
            ],
        ),
        ("State cache chunk / tracking interval", ["64 / 256 tokens"] * 3),
        (
            "Transfer bandwidth per rank",
            ["No handoff", "64 GB/s assumed", "64 GB/s assumed"],
        ),
        (
            "Transfer payload per rank",
            [
                "No handoff",
                "3,072 bytes/input token + 101,990,400 state bytes",
                "Same payload",
            ],
        ),
        ("Decode tokens reserved at handoff", ["Not applicable", "0", "512"]),
        (
            "Config source",
            [
                f"[{r} engine JSON]({engines[r]['source']})"
                for r in ["agg", "prefill", "decode"]
            ],
        ),
    ]
    field_rows = [[label, *values] for label, values in descriptors]
    out.append(
        table(
            [
                "Native configuration",
                "Agg V10",
                "Disagg prefill V11 extension",
                "Disagg decode V11 extension",
            ],
            field_rows,
        )
    )
    out.append("""**Cache size provenance:** the agg attention allocation (443,697 pages ×64 =28,396,608 tokens/worker) comes from observed serving metadata. The 769 Mamba slots and checkpoint settings remain assumptions because the live state pool was not captured. Disagg prefill inherits that allocation as an assumption; decode uses 809,406 observed attention pages and 64 state/request slots in the saved model. AIC did not measure these fleet cache allocations. Cache-hit rate emerges from the replay, placement and finite cache state.

**Transfer provenance:** the 64 GB/s value is a per-rank modeling assumption, not measured Mooncake bandwidth. The native handoff moves the full prompt's modeled KV plus recurrent state; independent delays omit shared-link contention. These assumptions need direct transfer/cache measurements and a matched D88 hardware check. The saved input files are linked above; the failed sweep is not evidence that these values are accurate.

### 5.4 Shared timing calibration and what DynoSim tells us

The same coefficients are used for every policy and concurrency:

```text
prefill_ms = 1.7050018888 × AIC_prefill_ms
           + 165.5695513 × batch × new_tokens × (prefix_tokens + new_tokens / 2) / 1e9
decode_ms  = AIC_decode_ms + 0.288 + 0.03145728 × ready_decode_requests
```

The prefill fit uses **93 isolated one-token RR192 hardware warmup requests**. Decode additions model missing recurrent-state work from geometry/bandwidth and launch-overhead assumptions; they are not directly measured Nemotron kernel times. See the [prefill fit](agentx-agg-kv-rr-data/native-v10/prefill_calibration_v2.json), [decode derivation](agentx-agg-kv-rr-data/native-v10/decode_mamba_timing_candidate_v10.json), and [native build reconstruction](agentx-agg-kv-rr-data/native-v10/v10_patch_reconstruction.json).

| Recipe question | What the completed native evidence tells us | What still needs hardware or a simulator fix |
| --- | --- | --- |
| Agg 6×TP4, default KV versus RR | Reproduces the sampled C192 throughput peak and C384 decline, and the direction of the KV advantage. Eight holdouts average 1.7% absolute throughput error. | TTFT tails remain biased; the simulator falsely passes default KV192 under TTFT-only. |
| Agg router tuning | Frozen V10 reproduces the original scale-3/credit-0.8 throughput benefit and temperature-0.5 loss. | Tuned KV192 falsely passes combined SLO in simulation. Decay variants and the new C160/C256 samples lack native counterparts. Hardware now selects scale 3/default credit 1.0 at C160 under both limits; repeat it and measure C192 with those flags. |
| Disagg 8P+8D TP4 | Six completed native runs at C16/C64/C256 show the sampled routing trends, with separate admission/cache budgets and transfer timing. | No native point matches the hardware concurrency grid; RR256 has 24.95% errors. All C480 tuning attempts fail warmup, so no native tuning ranking or high-load SLO capacity is established. |
| Earlier disagg 12P+6D TP4 | Two matched 72-GPU KV checks have throughput errors of −1.10%/+0.65% and TTFT errors of +0.54%/+5.74%. | These are two historical KV points, not validation of D88, RR, or the full latency boundary. Transfer/cache assumptions still need measurement. |
| Choosing TP or the P:D ratio | AIC supplies candidate shapes; a working replay model can compare them under the workload. | The current evidence does not establish that 6×TP4 or 8P+8D is globally optimal. Compare candidates at fixed total GPUs and validate on hardware. |

Fix native disagg admission/backpressure while preserving the intended batch limits, complete one full matched baseline, and only then repeat the flag grid. Validate transfer timing/contended bandwidth, prefill cache capacity and SGLang-version effects. For agg, use the frozen model to prioritize measurements; the real-job SLO remains the decision source.

### 5.5 Artifacts, validation and regeneration

The [hardware manifest](agentx-serving-perf-data/manifest.json) records the collection timestamp, source commit, summaries, source GCS paths and hashes. Numeric per-request projections retain original full-export hashes without prompt/response text. The generator checks the shared replay controls, request/error counts, raw TTFT and E2E percentiles, I90 normalization and the native build/engine identity. Methodology artifacts add the exact saved AIC recipes and native configuration hashes; no new AIC solve, simulation or hardware job is launched when regenerating this report.

From the repository root, with NumPy, Matplotlib, PyYAML and markdown-it-py installed:

```bash
python kv-cache-aware-bench/nemotron-3-ultra-550b-nvfp4/scripts/gen_agentx_serving_report.py
```

The earlier [agg report](agentx-agg-kv-rr-report.md) and [disagg report](agentx-disagg-kv-rr-report.md) remain dated snapshots. This report preserves 45 hardware jobs in the scoped inventory (25 agg and the retained 20 D88 jobs) and all 12 original native agg comparisons. Section 4 additionally preserves eight existing native disagg runs and two historical 72-GPU hardware references; errored profiling runs remain visible as diagnostics, and failed warmups contribute no performance point.
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
<title>NVIDIA Dynamo on Google Cloud: KV-Aware vs. Round-Robin Routing</title>
<meta name="description" content="Google Cloud CUJ technical evaluation of NVIDIA Dynamo KV-aware and round-robin routing on GKE: AgentX replay, prefix-cache reuse, total served throughput, TTFT and E2E-normalized interactivity."><style>
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
</style></head><body><main><div class="eyebrow">Google Cloud CUJ · NVIDIA Dynamo · AgentX</div>
<nav><a href="#1-setup-agentic-workload-and-benchmarking-methodology">1. Setup and methodology</a><a href="#2-aggregated-serving-measured-kv-and-rr">2. Agg hardware</a><a href="#3-disaggregated-serving-measured-kv-and-rr">3. Disagg hardware</a><a href="#4-simulation-versus-real-hardware-jobs">4. Simulation vs hardware</a><a href="#5-how-we-simulate-performance-aic-dynosim-and-recipe-selection">5. AIC and DynoSim</a></nav>
"""
        + body
        + "</main></body></html>\n"
    )
    (REPORTS / f"{STEM}.html").write_text(document)


def main():
    manifest, points, native, status, verified = load()
    disagg = load_native_disagg()
    methodology = load_methodology()
    methodology["source_commit"] = manifest["source_commit"]
    (REPORTS / f"{STEM}-methodology.json").write_text(
        json.dumps(methodology, indent=2) + "\n"
    )
    validation = {
        "passed": True,
        "hardware_runs": len(points),
        "new_hardware_runs": sum(p["is_new"] for p in points),
        "new_since_previous_report": sum(
            p["new_since_previous_report"] for p in points
        ),
        "by_architecture": dict(Counter(p["architecture"] for p in points)),
        "native_v10_agg_runs_revalidated": len(native),
        "native_disagg_runs_revalidated": len(disagg["manifest"]["native_runs"]),
        "additional_legacy_disagg_hardware_runs": len(disagg["pairs"]),
        "native_disagg_source_files_hashed": len(disagg["manifest"]["files"]),
        "methodology_sources_hashed": len(methodology["sources"]),
        "aic_saved_recipes": len(methodology["aic_recommendations"]),
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
            "native disagg core, engine, replay, request counts and error rates",
            "two 72-GPU KV hardware pairs and exact warmup input matches",
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
        "slo_ttft_operator": "<",
        "comparisons": comparison_data(points),
        "methodology_file": f"{STEM}-methodology.json",
        "slo_e2e_normalized_interactivity_p90_tps": 20,
        "hardware": points,
        "selected_operating_points": chosen(points),
        "agg_c192_current_campaign_contrasts": agg_contrasts(points),
        "agg_c160_slo_campaign_contrasts": agg_contrasts(points, 160),
        "native_v10_agg": native,
        "native_disagg": {
            "input_manifest": "agentx-native-disagg-data/manifest.json",
            "input_manifest_sha256": sha(
                REPORTS / "agentx-native-disagg-data/manifest.json"
            ),
            "points": disagg["points"],
            "matched_pairs": disagg["pairs"],
        },
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
    simulation_plot(points, native)
    disagg_simulation_plots(points, disagg)
    document = markdown(points, native, manifest, status, methodology, disagg)
    (REPORTS / f"{STEM}.md").write_text(document)
    html(document)
    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
