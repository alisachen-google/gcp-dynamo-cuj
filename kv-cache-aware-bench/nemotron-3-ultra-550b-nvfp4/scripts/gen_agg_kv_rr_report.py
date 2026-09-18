#!/usr/bin/env python3
"""Build the agg KV/RR report from preserved hardware and simulation artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import gzip
import hashlib
import json
import re
import shlex
import zipfile
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from markdown_it import MarkdownIt
from matplotlib.ticker import FuncFormatter

STUDY = Path(__file__).resolve().parents[1]
STEM = "agentx-agg-kv-rr-report"
COLORS = {"kv": "#2463b3", "rr": "#d65b29"}
MARKERS = {"kv": "o", "rr": "s"}
TUNED_STYLES = {
    "kvs3c08": ("#16836b", "D", "KV scale 3 / credit 0.8"),
    "kvs2c08": ("#7d75b7", "^", "KV scale 2 / credit 0.8"),
    "kvt05": ("#bf923e", "X", "KV temperature 0.5"),
}
CLIENTS = [48, 96, 192, 384]
TTFT_SLO_S = 10
INTERACTIVITY_SLO_TPS = 20
FLAG_ORDER = ["kv", "kvs2c08", "kvs3c08", "kvt05", "rr"]
FLAG_LABELS = {
    "kv": "Default KV",
    "kvs2c08": "KV: scale 2, credit 0.8",
    "kvs3c08": "KV: scale 3, credit 0.8",
    "kvt05": "KV: temperature 0.5",
    "rr": "RR reference",
}


def read(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(summary, gpus=24):
    return {
        "total_tok_s_gpu": summary["total_token_throughput"]["avg"] / gpus,
        "output_tok_s_gpu": summary["output_token_throughput"]["avg"] / gpus,
        "requests_s": summary["request_throughput"]["avg"],
        "input_tokens_p50": summary["input_sequence_length"]["p50"],
        "output_tokens_p50": summary["output_sequence_length"]["p50"],
        "ttft_p95_s": summary["time_to_first_token"]["p95"] / 1000,
        "itl_mean_ms": summary["inter_token_latency"]["avg"],
        "itl_p90_ms": summary["inter_token_latency"]["p90"],
        "cache_pct": summary.get("overall_usage_prompt_cache_read_pct", {}).get("avg"),
        "successful_requests": int(summary["request_count"]["avg"]),
        "request_errors": int(summary.get("error_request_count", {}).get("avg", 0)),
    }


def normalized_interactivity(e2e_ms, output_tokens):
    """Invert the slow-tail percentile after normalizing each request's E2E."""
    ratios_s = np.asarray(e2e_ms) / 1000 / np.asarray(output_tokens)
    assert np.all(np.isfinite(ratios_s)) and np.all(ratios_s > 0)
    return float(1 / np.percentile(ratios_s, 90, method="linear"))


def request_metrics(folder, provenance, point, summary=None):
    entry = provenance["runs"][point["id"]]
    with gzip.open(folder / entry["file"], "rt", newline="") as f:
        records = list(csv.DictReader(f))
    counts = Counter(f"{r['phase']}/{r['status']}" for r in records)
    assert dict(counts) == entry["rows_by_phase_and_status"]
    successful = [
        r for r in records if r["phase"] == "profiling" and r["status"] == "success"
    ]
    assert len(successful) == point["successful_requests"], point["id"]
    assert counts["profiling/error"] == point["request_errors"], point["id"]
    values = np.array(
        [
            [
                float(r.get(k) or "nan")
                for k in ["request_latency_ms", "output_sequence_length", "ttft_ms"]
            ]
            for r in successful
        ]
    )
    valid = np.all(np.isfinite(values[:, :2]) & (values[:, :2] > 0), axis=1)
    e2e, osl, _ttft = values[valid].T
    assert len(e2e), point["id"]
    assert np.isclose(
        np.percentile(values[:, 2], 95) / 1000, point["ttft_p95_s"], rtol=1e-10
    ), point["id"]
    if summary is not None:
        assert np.isclose(
            np.percentile(values[:, 0], 95),
            summary["request_latency"]["p95"],
            rtol=1e-10,
        )
        assert np.isclose(
            np.percentile(osl / (e2e / 1000), 10),
            summary["e2e_output_token_throughput"]["p10"],
            rtol=1e-10,
        )
    interactivity = normalized_interactivity(e2e, osl)
    ttft_pass = point["ttft_p95_s"] <= TTFT_SLO_S
    e2e_pass = interactivity >= INTERACTIVITY_SLO_TPS
    return {
        "e2e_normalized_interactivity_p90_tps": interactivity,
        "e2e_normalized_latency_p90_ms_per_token": 1000 / interactivity,
        "e2e_p95_s": float(np.percentile(e2e, 95) / 1000),
        "interactivity_valid_requests": len(e2e),
        "interactivity_excluded_successful_requests": int(np.count_nonzero(~valid)),
        "ttft_slo_pass": ttft_pass,
        "interactivity_slo_pass": e2e_pass,
        "combined_slo_pass": ttft_pass and e2e_pass,
        "request_metrics_file": entry["file"],
        "request_metrics_source_sha256": entry["source_sha256"],
    }


def best_point(points, policy, criterion="combined_slo_pass"):
    eligible = [p for p in points if p["policy"] == policy and p[criterion]]
    return max(eligible, key=lambda p: p["total_tok_s_gpu"], default=None)


def proposed_flag_sweep():
    anchor = {
        "prefill_load_scale": 3.0,
        "overlap_score_credit": 0.8,
        "temperature": 0.0,
        "queue_policy": "fcfs",
    }
    variants = []
    for knob, values in [
        ("prefill_load_scale", [4.0, 5.0]),
        ("overlap_score_credit", [0.6, 1.0]),
        ("temperature", [0.1, 0.25]),
    ]:
        for value in values:
            settings = anchor | {knob: value}
            variants.append(
                {
                    "changed_flag": knob,
                    "settings": settings,
                    "concurrency": 192,
                    "router_args": [
                        "--router-mode",
                        "kv",
                        "--router-temperature",
                        str(settings["temperature"]),
                        "--router-queue-policy",
                        settings["queue_policy"],
                        "--router-prefill-load-scale",
                        str(settings["prefill_load_scale"]),
                        "--router-kv-overlap-score-credit",
                        str(settings["overlap_score_credit"]),
                    ],
                }
            )
    return {
        "status": "proposed_not_launched",
        "serving_shape": "24 GB300 GPUs; six TP4/EP4 aggregated workers",
        "anchor": anchor,
        "anchor_concurrency": 192,
        "additional_anchor_repeats": 2,
        "profiling_duration_s": 3600,
        "initial_additional_profiles": 8,
        "initial_profiling_gpu_hours_excluding_warmup_and_drain": 192,
        "slo_ttft_p95_s": TTFT_SLO_S,
        "slo_e2e_normalized_interactivity_p90_tps": INTERACTIVITY_SLO_TPS,
        "variants": variants,
        "followup": "Repeat finalists; test 128/160 and 96 sessions; then cross the strongest scale/credit values. Match every collected hardware cell with frozen V10, retaining mismatches.",
    }


def validate_client(summary, clients):
    assert summary["metadata"]["scenario"] == "inferencex-agentx-mvp"
    assert summary["metadata"]["submission_valid"]
    phase = summary["input_config"]["phases"][0]
    assert phase["concurrency"] == clients
    assert phase["duration"] == 3600 and phase["grace_period"] == 60
    assert phase["timing_mode"] == "agentic_replay"
    assert (
        phase["trajectory_start_min_ratio"],
        phase["trajectory_start_max_ratio"],
    ) == (0.25, 0.75)
    dataset = summary["input_config"]["datasets"][0]
    assert dataset["entries"] == 393 and dataset["random_seed"] == 42


def load_inputs(folder):
    manifest = read(folder / "manifest.json")
    for name, metadata in manifest["files"].items():
        assert digest(folder / name) == metadata["sha256"], name
    assert manifest["slo_ttft_p95_s"] == TTFT_SLO_S
    assert manifest["slo_e2e_normalized_interactivity_p90_tps"] == INTERACTIVITY_SLO_TPS
    requests = read(folder / manifest["request_metrics_manifest"])
    recipe = (folder / "hardware/agentx_runner.sh.txt").read_text()
    router_flags = dict(re.findall(r'\[(\w+)\]="(--router-mode [^"]+)"', recipe))
    assert set(router_flags) == set(FLAG_ORDER) | {"kvwspt"}
    hardware = []
    for entry in manifest["hardware"]:
        summary = read(folder / entry["summary"])
        validate_client(summary, entry["clients"])
        hardware.append(
            entry
            | metrics(summary)
            | {
                "kind": "hardware",
                "benchmark_id": summary["benchmark_id"],
                "router_flags": router_flags[entry["policy"]],
            }
        )
        hardware[-1].update(request_metrics(folder, requests, hardware[-1], summary))
    baseline = [p for p in hardware if p["policy"] in COLORS]
    assert {(p["policy"], p["clients"]) for p in baseline} == {
        (p, c) for p in COLORS for c in CLIENTS
    }
    reference = {(p["policy"], p["clients"]): p for p in hardware}
    assert {
        (p["policy"], p["clients"]) for p in hardware if p["policy"] not in COLORS
    } == {("kvs2c08", 192), ("kvs3c08", 192), ("kvt05", 192), ("kvs3c08", 96)}

    matrix = read(folder / "native-v10/matrix.json")
    assert (
        matrix["all_runs_complete"]
        and matrix["one_shared_build"]
        and matrix["one_shared_engine"]
    )
    assert not matrix["issues"] and matrix["calibration_targets"] == [
        "total_tok_s_per_gpu"
    ]
    holdouts = read(folder / "native-v10/holdouts.json")
    assert holdouts["all_runs_complete"] and holdouts["replay_valid"]
    assert holdouts["calibration_frozen"]
    assert len(matrix["points"]) == 4 and len(holdouts["points"]) == 8
    native = []
    engine_hashes, core_hashes = set(), set()
    native_points = [p | {"role": "calibration"} for p in matrix["points"]] + [
        p | {"role": "holdout"} for p in holdouts["points"]
    ]
    for point in native_points:
        run = folder / "native-v10" / point["run"]
        summary = read(run / "simulation_summary.json")
        validate_client(summary, point["clients"])
        values = metrics(summary)
        policy = point.get("policy", point["router"])
        hardware_point = reference[(policy, point["clients"])]
        assert (
            abs(
                values["total_tok_s_gpu"]
                - point["metrics"]["total_tok_s_per_gpu"]["simulation"]
            )
            < 1e-7
        )
        assert (
            abs(
                values["ttft_p95_s"] * 1000
                - point["metrics"]["p95_ttft_ms"]["simulation"]
            )
            < 1e-6
        )
        assert (
            abs(
                hardware_point["total_tok_s_gpu"]
                - point["metrics"]["total_tok_s_per_gpu"]["hardware"]
            )
            < 1e-7
        )
        assert summary["benchmark_id"] == hardware_point["benchmark_id"]
        assert (
            point["replay_valid"]
            and point["hardware_settings_match"]
            and point["hardware_serving_shape_matches"]
        )
        engine = read(run / "engine.json")
        assert engine["engine_type"] == "sglang" and engine["max_num_seqs"] == 16
        assert engine["speedup_ratio"] == engine["decode_speedup_ratio"] == 1
        assert engine["num_gpu_blocks"] * engine["block_size"] == 28396608
        environment = read(run / "server-environment.json")
        assert environment["core_sha256"] == point["native_core_sha256"]
        engine_hashes.add(digest(run / "engine.json"))
        core_hashes.add(environment["core_sha256"])
        native.append(
            values
            | {
                "kind": "native_v10",
                "id": point["run"],
                "policy": policy,
                "clients": point["clients"],
                "role": point["role"],
                "hardware_id": hardware_point["id"],
                "router_flags": hardware_point["router_flags"],
            }
        )
        native[-1].update(request_metrics(folder, requests, native[-1], summary))
        native[-1]["hardware_comparison"] = {
            metric: {
                "hardware": hardware_point[metric],
                "native_v10": native[-1][metric],
                "relative_error": native[-1][metric] / hardware_point[metric] - 1,
            }
            for metric in [
                "total_tok_s_gpu",
                "output_tok_s_gpu",
                "ttft_p95_s",
                "e2e_normalized_interactivity_p90_tps",
            ]
        }
    assert len(engine_hashes) == len(core_hashes) == 1
    assert {(p["policy"], p["clients"]) for p in native} == set(reference)
    return manifest, hardware, native


def curve(ax, points, policy, metric, label=None, faint=False):
    selected = sorted(
        (p for p in points if p["policy"] == policy), key=lambda p: p["clients"]
    )
    ax.plot(
        [p["clients"] for p in selected],
        [p[metric] for p in selected],
        color=COLORS[policy],
        marker=MARKERS[policy],
        markersize=5 if faint else 7,
        linewidth=1.5 if faint else 2.6,
        linestyle="--" if faint else "-",
        alpha=0.45 if faint else 1,
        label=label or policy.upper(),
        zorder=2 if faint else 4,
    )
    return selected


def knee_evidence(points):
    lookup = {(p["policy"], p["clients"]): p for p in points}
    evidence = []
    for policy, start, end in [("kv", 192, 384), ("rr", 96, 192)]:
        before, after = lookup[(policy, start)], lookup[(policy, end)]
        throughput = 100 * (after["total_tok_s_gpu"] / before["total_tok_s_gpu"] - 1)
        tail = after["ttft_p95_s"] / before["ttft_p95_s"]
        evidence.append(
            f"{policy.upper()} {start}→{end}: throughput {throughput:+.1f}%, TTFT p95 ×{tail:.1f}"
        )
    return ".  ".join(evidence) + "."


def plot_pair(
    output, name, points, title, subtitle, caption, mode, hardware=(), tuned=()
):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#334155",
            "text.color": "#172b45",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.29, top=0.70, wspace=0.26)
    fig.text(0.075, 0.94, title, fontsize=18, weight="bold")
    fig.text(0.075, 0.885, subtitle, fontsize=10.5, color="#53667a")
    for ax, metric, panel in zip(
        axes,
        ["total_tok_s_gpu", "ttft_p95_s"],
        ["Throughput", "P95 time to first token"],
    ):
        ax.set_title(panel, loc="left", fontsize=12, weight="bold", pad=12)
        ax.set_xscale("log", base=2)
        ax.set_xticks(CLIENTS, [str(c) for c in CLIENTS])
        ax.set_xlim(43, 432)
        ax.set_xlabel("Concurrent AgentX session trees", labelpad=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")
        ax.tick_params(length=0, pad=7)
        ax.grid(axis="y", color="#dde4ec", linewidth=0.7)
        ax.set_axisbelow(True)
        if metric == "total_tok_s_gpu":
            ax.set_ylabel("Input + output tokens/s/GPU")
            ax.set_ylim(0, max(13000, max(p[metric] for p in [*points, *tuned]) * 1.18))
            ax.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        else:
            ax.set_ylabel("Seconds · log scale")
            ax.set_yscale("log")
            ax.set_ylim(2, 900)
            ax.set_yticks(
                [2, 5, 10, 20, 50, 100, 200, 500],
                ["2", "5", "10", "20", "50", "100", "200", "500"],
            )
            ax.axhline(
                TTFT_SLO_S,
                color="#687b8f",
                linestyle=(0, (4, 3)),
                linewidth=1.2,
                zorder=1,
            )
            ax.text(
                0.02,
                TTFT_SLO_S,
                " 10 s TTFT SLO",
                transform=ax.get_yaxis_transform(),
                va="bottom",
                color="#53667a",
                fontsize=9,
            )
        if mode == "hardware":
            ax.axvspan(192, 384, color="#eef2f6", zorder=0)
            ax.axvline(192, color="#8190a3", linewidth=1.1, linestyle=":")
        if mode == "native":
            for policy in COLORS:
                curve(
                    ax,
                    hardware,
                    policy,
                    metric,
                    policy.upper() + " hardware",
                    faint=True,
                )
        for policy, color in COLORS.items():
            selected = curve(
                ax,
                points,
                policy,
                metric,
                policy.upper() + (" V10" if mode == "native" else ""),
            )
            if mode in {"hardware", "native"}:
                peak = max(selected, key=lambda p: p["total_tok_s_gpu"])
                ax.scatter(
                    [peak["clients"]],
                    [peak[metric]],
                    marker="*",
                    s=220,
                    color=color,
                    edgecolor="white",
                    linewidth=0.8,
                    zorder=6,
                )
                post = next(p for p in selected if p["clients"] == 384)
                ax.scatter(
                    [384],
                    [post[metric]],
                    marker=MARKERS[policy],
                    s=60,
                    facecolor="white",
                    edgecolor=color,
                    linewidth=1.8,
                    zorder=6,
                )
            for point in selected:
                if point["clients"] not in (192, 384):
                    continue
                value = point[metric]
                text = (
                    f"{value:,.0f}" if metric == "total_tok_s_gpu" else f"{value:.1f} s"
                )
                dy = 12 if policy == "kv" else -17
                if metric == "ttft_p95_s":
                    dy = -16 if policy == "kv" else 11
                    if mode == "native" and policy == "rr" and point["clients"] == 384:
                        dy = -17  # Keep the native label away from the hardware marker.
                ax.annotate(
                    text,
                    (point["clients"], value),
                    xytext=(-4, dy),
                    textcoords="offset points",
                    ha="right",
                    fontsize=9,
                    color=color,
                    weight="bold",
                    zorder=7,
                )
        for policy, (color, marker, label) in TUNED_STYLES.items():
            selected = sorted(
                (p for p in tuned if p["policy"] == policy), key=lambda p: p["clients"]
            )
            if selected:
                ax.plot(
                    [p["clients"] for p in selected],
                    [p[metric] for p in selected],
                    marker=marker,
                    color=color,
                    linewidth=2.3,
                    label=label,
                    zorder=5,
                )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.069, 0.855),
        ncol=4 if mode == "native" else 3,
        frameon=False,
        fontsize=9,
    )
    fig.text(0.075, 0.045, caption, fontsize=9.2, color="#53667a", linespacing=1.6)
    save_figure(fig, output, name)


def save_figure(fig, output, name):
    for ext in ("png", "svg", "pdf"):
        fig.savefig(
            output / f"{STEM}-{name}.{ext}",
            dpi=180,
            facecolor="white",
            metadata={"Creator": "N3U AgentX report generator"},
        )
        if ext == "svg":
            svg_path = output / f"{STEM}-{name}.svg"
            svg_path.write_text(
                "\n".join(line.rstrip() for line in svg_path.read_text().splitlines())
                + "\n"
            )
    plt.close(fig)


def flag_sweep_points(hardware):
    get = {(p["policy"], p["clients"]): p for p in hardware}
    result = []
    for policy, clients in [(p, 192) for p in FLAG_ORDER] + [
        ("kv", 96),
        ("kvs3c08", 96),
    ]:
        point = get[(policy, clients)]
        default = get[("kv", clients)]
        tokens = shlex.split(point["router_flags"])
        flags = dict(zip(tokens[::2], tokens[1::2], strict=True))
        result.append(
            point
            | {
                "label": FLAG_LABELS[policy],
                "prefill_load_scale": flags.get(
                    "--router-prefill-load-scale", "not explicitly set"
                ),
                "overlap_score_credit": flags.get(
                    "--router-kv-overlap-score-credit", "not explicitly set"
                ),
                "temperature": flags.get("--router-temperature", "not applicable"),
                "queue_policy": flags.get("--router-queue-policy", "not applicable"),
                "throughput_change_vs_default_kv_pct": 100
                * (point["total_tok_s_gpu"] / default["total_tok_s_gpu"] - 1),
                "ttft_change_vs_default_kv_pct": 100
                * (point["ttft_p95_s"] / default["ttft_p95_s"] - 1),
            }
        )
    return result


def plot_flag_sweep(output, points):
    selected = [p for p in points if p["clients"] == 192]
    colors = [
        COLORS[p["policy"]]
        if p["policy"] in COLORS
        else TUNED_STYLES[p["policy"]][0]
        for p in selected
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), sharey=True)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.27, top=0.75, wspace=0.20)
    fig.text(
        0.045,
        0.94,
        "Real KV-router flag sweep · 192 AgentX clients",
        fontsize=18,
        weight="bold",
    )
    fig.text(
        0.045,
        0.885,
        "24 GB300 GPUs · 6 × TP4/EP4 · three tested variants, with default KV and RR references",
        fontsize=10.5,
        color="#53667a",
    )
    for ax, metric, title, limit in zip(
        axes,
        ["total_tok_s_gpu", "ttft_p95_s"],
        ["Throughput · higher is better", "P95 TTFT · lower is better"],
        [13000, 72],
    ):
        values = [p[metric] for p in selected]
        ax.barh(range(len(selected)), values, color=colors, height=0.58, zorder=3)
        ax.set_yticks(
            range(len(selected)), [FLAG_LABELS[p["policy"]] for p in selected]
        )
        ax.set_ylim(len(selected) - 0.45, -0.65)
        ax.set_xlim(0, limit)
        ax.set_title(title, loc="left", fontsize=12, weight="bold", pad=13)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.spines["bottom"].set_color("#cbd5e1")
        ax.tick_params(length=0, pad=9)
        ax.grid(axis="x", color="#dde4ec", linewidth=0.7)
        ax.set_axisbelow(True)
        for index, value in enumerate(values):
            label = f"{value:,.0f}" if metric == "total_tok_s_gpu" else f"{value:.2f} s"
            ax.text(
                value + limit * 0.02,
                index,
                label,
                va="center",
                fontsize=10,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                weight="bold" if selected[index]["policy"] == "kvs3c08" else "normal",
            )
        if metric == "total_tok_s_gpu":
            ax.set_xlabel("Input + output tokens/s/GPU", labelpad=10)
            ax.set_xticks([0, 4000, 8000, 12000])
            ax.xaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:,.0f}"))
        else:
            ax.set_xlabel("Seconds", labelpad=10)
            ax.set_xticks([0, 20, 40, 60])
            ax.axvline(
                TTFT_SLO_S, color="#687b8f", linestyle=(0, (4, 3)), linewidth=1.2
            )
            ax.text(11.1, -0.48, "10 s TTFT SLO", fontsize=9, color="#53667a")
    fig.text(
        0.045,
        0.045,
        "Best measured: scale 3, credit 0.8, temperature 0, FCFS. Versus default KV: +14.1% throughput, −45.7% p95 TTFT.\n"
        "All five settings at 192 fail I90 ≥20 output tok/s/user; the best is 19.78. Passing TTFT alone is insufficient.\n"
        "One run per setting. Scale 3/credit 0.8 was also measured at 96; its throughput knee remains unresolved.",
        fontsize=9.2,
        color="#53667a",
        linespacing=1.6,
    )
    save_figure(fig, output, "flag-sweep")


def plot_interactivity(output, hardware, native):
    metric = "e2e_normalized_interactivity_p90_tps"
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 6.2), sharey=True)
    fig.subplots_adjust(left=0.065, right=0.98, bottom=0.26, top=0.70, wspace=0.13)
    fig.text(
        0.065, 0.94, "E2E-normalized interactivity · P90", fontsize=19, weight="bold"
    )
    fig.text(
        0.065,
        0.885,
        "1 / P90(request E2E seconds / output tokens) · includes TTFT and generation · higher is better",
        fontsize=10.5,
        color="#53667a",
    )
    for ax, points, title in zip(
        axes,
        [hardware, native],
        ["Measured hardware", f"Native V10 · {len(native)} matching hardware settings"],
    ):
        ax.set_title(title, fontsize=11, weight="bold", loc="left", pad=13)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xticks(CLIENTS, [str(c) for c in CLIENTS])
        ax.set_xlim(43, 432)
        ax.set_ylim(0.4, 100)
        ax.set_yticks(
            [0.5, 1, 2, 5, 10, 20, 50, 100],
            ["0.5", "1", "2", "5", "10", "20", "50", "100"],
        )
        ax.set_xlabel("Concurrent AgentX session trees", labelpad=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color("#cbd5e1")
        ax.tick_params(length=0, pad=7)
        ax.grid(axis="y", color="#dde4ec", linewidth=0.7)
        ax.set_axisbelow(True)
        ax.axhspan(INTERACTIVITY_SLO_TPS, 100, color="#edf7f2", zorder=0)
        ax.axhline(
            INTERACTIVITY_SLO_TPS, color="#526b61", linestyle="--", linewidth=1.2
        )
        for policy in COLORS:
            if points is native:
                curve(ax, hardware, policy, metric, faint=True)
            curve(ax, points, policy, metric)
        for policy, (color, marker, label) in TUNED_STYLES.items():
            selected = sorted(
                (p for p in points if p["policy"] == policy), key=lambda p: p["clients"]
            )
            ax.plot(
                [p["clients"] for p in selected],
                [p[metric] for p in selected],
                marker=marker,
                color=color,
                linewidth=2.3,
                label=label,
                zorder=5,
            )
    axes[0].set_ylabel("Output tokens/s/user · log scale")
    tuned = sorted(
        (p for p in hardware if p["policy"] == "kvs3c08"), key=lambda p: p["clients"]
    )
    for policy, x_offset, y_offset in [("kv", 10, -28), ("rr", 9, 12)]:
        p = best_point(hardware, policy)
        axes[0].scatter(
            p["clients"],
            p[metric],
            marker="*",
            s=220,
            color=COLORS[policy],
            edgecolor="white",
            zorder=6,
        )
        axes[0].annotate(
            f"{policy.upper()} {p['clients']}: {p[metric]:.2f}",
            (p["clients"], p[metric]),
            xytext=(x_offset, y_offset),
            textcoords="offset points",
            fontsize=9,
            color=COLORS[policy],
        )
    axes[0].annotate(
        f"Tuned 192: {tuned[-1][metric]:.2f} < 20",
        (192, tuned[-1][metric]),
        xytext=(12, 22),
        textcoords="offset points",
        fontsize=9,
        color="#126653",
        arrowprops={"arrowstyle": "-", "color": "#16836b"},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    native_tuned192 = next(
        p for p in native if (p["policy"], p["clients"]) == ("kvs3c08", 192)
    )
    axes[1].annotate(
        f"Tuned 192: {native_tuned192[metric]:.2f} "
        + ("≥ 20" if native_tuned192["interactivity_slo_pass"] else "< 20"),
        (192, native_tuned192[metric]),
        xytext=(-50, 27),
        textcoords="offset points",
        fontsize=9,
        color="#126653",
        arrowprops={"arrowstyle": "-", "color": "#16836b"},
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.85, "pad": 2},
    )
    axes[1].text(
        0.98,
        21.7,
        "20 tok/s threshold",
        ha="right",
        fontsize=9,
        color="#526b61",
        transform=axes[1].get_yaxis_transform(),
    )
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.065, 0.835),
        frameon=False,
        ncol=3,
        fontsize=9.5,
    )
    fig.text(
        0.065,
        0.055,
        "Stars: highest-throughput measured default-policy cells passing BOTH I90 ≥20 tok/s and TTFT p95 ≤10 s.\n"
        f"Native V10: four original calibration points plus {len(native) - 4} new holdouts; faint dashed curves show default-policy hardware.\n"
        "Lines connect samples; threshold crossings are not measured. Hardware has one trial per cell, with no uncertainty bands.",
        fontsize=9.3,
        color="#53667a",
        linespacing=1.55,
    )
    save_figure(fig, output, "interactivity")


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def report(hardware, native, data_name):
    get = {(p["policy"], p["clients"]): p for p in hardware}
    rows = []
    for clients in CLIENTS:
        k, r = get[("kv", clients)], get[("rr", clients)]
        rows.append(
            [
                clients,
                f"{k['total_tok_s_gpu']:,.0f}",
                f"{r['total_tok_s_gpu']:,.0f}",
                f"+{100 * (k['total_tok_s_gpu'] / r['total_tok_s_gpu'] - 1):.1f}%",
                f"{k['ttft_p95_s']:.2f}",
                f"{r['ttft_p95_s']:.2f}",
                f"{r['ttft_p95_s'] / k['ttft_p95_s']:.2f}×",
            ]
        )
    same_config = table(
        [
            "Clients",
            "KV total tok/s/GPU",
            "RR total tok/s/GPU",
            "KV throughput gain",
            "KV TTFT p95 (s)",
            "RR TTFT p95 (s)",
            "RR/KV TTFT p95",
        ],
        rows,
    )

    k, r, tuned = [best_point(hardware, policy) for policy in ["kv", "rr", "kvs3c08"]]
    kt, rt, tuned_ttft = [
        best_point(hardware, policy, "ttft_slo_pass")
        for policy in ["kv", "rr", "kvs3c08"]
    ]
    ttft_slo = table(
        [
            "Policy",
            "Selected clients",
            "Total tok/s/GPU",
            "Output tok/s/GPU",
            "TTFT p95 (s)",
            "Throughput/RR",
        ],
        [
            [
                name,
                p["clients"],
                f"{p['total_tok_s_gpu']:,.0f}",
                f"{p['output_tok_s_gpu']:.2f}",
                f"{p['ttft_p95_s']:.2f}",
                f"{p['total_tok_s_gpu'] / rt['total_tok_s_gpu']:.2f}×",
            ]
            for name, p in [
                ("Default KV", kt),
                ("RR", rt),
                ("Tuned KV: scale 3, credit 0.8", tuned_ttft),
            ]
        ],
    )
    interactivity_table = table(
        [
            "Hardware setting",
            "Sessions",
            "TTFT p95 (s)",
            "I90 (output tok/s/user)",
            "I90 ≥20",
            "Both limits pass",
            "Errors",
        ],
        [
            [
                FLAG_LABELS[p["policy"]],
                p["clients"],
                f"{p['ttft_p95_s']:.2f}",
                f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                "Pass" if p["interactivity_slo_pass"] else "Fail",
                "Pass" if p["combined_slo_pass"] else "Fail",
                p["request_errors"],
            ]
            for p in sorted(
                hardware, key=lambda p: (FLAG_ORDER.index(p["policy"]), p["clients"])
            )
        ],
    )
    joint_slo = table(
        [
            "Hardware policy",
            "Selected sessions",
            "Total tok/s/GPU",
            "Output tok/s/GPU",
            "TTFT p95 (s)",
            "I90 (output tok/s/user)",
            "Total throughput/RR",
        ],
        [
            [
                name,
                p["clients"],
                f"{p['total_tok_s_gpu']:,.0f}",
                f"{p['output_tok_s_gpu']:.2f}",
                f"{p['ttft_p95_s']:.2f}",
                f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                f"{p['total_tok_s_gpu'] / r['total_tok_s_gpu']:.2f}×",
            ]
            for name, p in [
                ("Default KV", k),
                ("RR", r),
                ("Tuned KV: scale 3, credit 0.8", tuned),
            ]
        ],
    )
    flags = flag_sweep_points(hardware)
    flags192 = [p for p in flags if p["clients"] == 192]
    settings_table = table(
        [
            "Recipe variant",
            "Prefill-load scale",
            "Overlap credit",
            "Temperature",
            "Queue policy",
            "Measured clients",
        ],
        [
            [
                f"`{p['policy']}`",
                p["prefill_load_scale"],
                p["overlap_score_credit"],
                p["temperature"],
                p["queue_policy"].upper(),
                ", ".join(
                    str(c)
                    for c in sorted(
                        q["clients"] for q in hardware if q["policy"] == p["policy"]
                    )
                ),
            ]
            for p in flags192
            if p["policy"] != "rr"
        ],
    )
    flags_table = table(
        [
            "192-client setting / artifacts",
            "Total tok/s/GPU",
            "Change vs default KV",
            "TTFT p95 (s)",
            "TTFT change vs default KV",
            "Cached input",
            "I90 (tok/s/user)",
            "Both limits pass",
        ],
        [
            [
                f"[{p['label']}]({p['gcs_console']})",
                f"{p['total_tok_s_gpu']:,.0f}",
                f"{p['throughput_change_vs_default_kv_pct']:+.1f}%",
                f"{p['ttft_p95_s']:.2f}",
                f"{p['ttft_change_vs_default_kv_pct']:+.1f}%",
                f"{p['cache_pct']:.1f}%",
                f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                "Pass" if p["combined_slo_pass"] else "Fail",
            ]
            for p in flags192
        ],
    )
    flags96 = {p["policy"]: p for p in flags if p["clients"] == 96}
    tuned96, default96 = flags96["kvs3c08"], flags96["kv"]
    tuned_sweep_table = table(
        [
            "Collected tuned-KV job / artifacts",
            "Sessions",
            "Total tok/s/GPU",
            "Output tok/s/GPU",
            "TTFT p95 (s)",
            "I90 (tok/s/user)",
            "Both limits pass",
            "Errors",
        ],
        [
            [
                f"[{FLAG_LABELS[p['policy']]}]({p['gcs_console']})",
                p["clients"],
                f"{p['total_tok_s_gpu']:,.0f}",
                f"{p['output_tok_s_gpu']:.2f}",
                f"{p['ttft_p95_s']:.2f}",
                f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                "Pass" if p["combined_slo_pass"] else "Fail",
                p["request_errors"],
            ]
            for p in sorted(
                (p for p in hardware if p["policy"] not in COLORS),
                key=lambda p: (p["clients"], FLAG_ORDER.index(p["policy"])),
            )
        ],
    )
    comparison = []
    for p in sorted(native, key=lambda p: (p["policy"], p["clients"])):
        h = get[(p["policy"], p["clients"])]
        comparison.append(
            [
                FLAG_LABELS[p["policy"]],
                p["clients"],
                p["role"],
                f"{h['total_tok_s_gpu']:,.0f}",
                f"{p['total_tok_s_gpu']:,.0f}",
                f"{100 * (p['total_tok_s_gpu'] / h['total_tok_s_gpu'] - 1):+.1f}%",
                f"{p['ttft_p95_s']:.2f} / {h['ttft_p95_s']:.2f}",
                f"{100 * (p['ttft_p95_s'] / h['ttft_p95_s'] - 1):+.1f}%",
            ]
        )
    native_table = table(
        [
            "Policy",
            "Clients",
            "V10 sample role",
            "Real total tok/s/GPU",
            "V10 total tok/s/GPU",
            "Throughput error",
            "TTFT p95 sim / real (s)",
            "TTFT error",
        ],
        comparison,
    )
    native_interactivity = table(
        [
            "Policy",
            "Sessions",
            "Hardware I90",
            "V10 I90",
            "I90 error",
            "V10 passes both limits",
            "Hardware / V10 request errors",
        ],
        [
            [
                FLAG_LABELS[p["policy"]],
                p["clients"],
                f"{get[(p['policy'], p['clients'])]['e2e_normalized_interactivity_p90_tps']:.4f}",
                f"{p['e2e_normalized_interactivity_p90_tps']:.4f}",
                f"{100 * (p['e2e_normalized_interactivity_p90_tps'] / get[(p['policy'], p['clients'])]['e2e_normalized_interactivity_p90_tps'] - 1):+.1f}%",
                "Pass" if p["combined_slo_pass"] else "Fail",
                f"{get[(p['policy'], p['clients'])]['request_errors']} / {p['request_errors']}",
            ]
            for p in sorted(native, key=lambda p: (p["policy"], p["clients"]))
        ],
    )
    native_tuned_table = table(
        [
            "Tuned setting",
            "Sessions",
            "Hardware / V10 total tok/s/GPU",
            "Hardware / V10 output tok/s/GPU",
            "Hardware / V10 TTFT p95 (s)",
            "Hardware / V10 I90 (tok/s/user)",
            "Both limits: hardware / V10",
        ],
        [
            [
                FLAG_LABELS[p["policy"]],
                p["clients"],
                f"{get[(p['policy'], p['clients'])]['total_tok_s_gpu']:,.0f} / {p['total_tok_s_gpu']:,.0f}",
                f"{get[(p['policy'], p['clients'])]['output_tok_s_gpu']:.2f} / {p['output_tok_s_gpu']:.2f}",
                f"{get[(p['policy'], p['clients'])]['ttft_p95_s']:.2f} / {p['ttft_p95_s']:.2f}",
                f"{get[(p['policy'], p['clients'])]['e2e_normalized_interactivity_p90_tps']:.4f} / {p['e2e_normalized_interactivity_p90_tps']:.4f}",
                (
                    "Pass"
                    if get[(p["policy"], p["clients"])]["combined_slo_pass"]
                    else "Fail"
                )
                + " / "
                + ("Pass" if p["combined_slo_pass"] else "Fail"),
            ]
            for p in sorted(
                (p for p in native if p["policy"] not in COLORS),
                key=lambda p: (p["clients"], FLAG_ORDER.index(p["policy"])),
            )
        ],
    )
    native_slo_rows = []
    for policy in ["kv", "rr", "kvs3c08"]:
        h, n = best_point(hardware, policy), best_point(native, policy)
        native_slo_rows.append(
            [
                FLAG_LABELS[policy],
                h["clients"],
                n["clients"] if n else "None",
                f"{h['total_tok_s_gpu']:,.0f}",
                f"{n['total_tok_s_gpu']:,.0f}" if n else "—",
                "Same sampled concurrency"
                if n and h["clients"] == n["clients"]
                else "Different selection",
            ]
        )
    native_slo_table = table(
        [
            "Policy",
            "Hardware selected sessions",
            "V10 selected sessions",
            "Hardware total tok/s/GPU",
            "V10 total tok/s/GPU",
            "Combined-SLO decision",
        ],
        native_slo_rows,
    )
    holdout_points = [p for p in native if p["role"] == "holdout"]
    throughput_holdouts_pass = sum(
        abs(
            p["total_tok_s_gpu"] / get[(p["policy"], p["clients"])]["total_tok_s_gpu"]
            - 1
        )
        <= 0.2
        for p in holdout_points
    )
    slo_mismatches = [
        p
        for p in native
        if p["combined_slo_pass"]
        != get[(p["policy"], p["clients"])]["combined_slo_pass"]
    ]
    native_mismatches_text = (
        "; ".join(
            f"{FLAG_LABELS[p['policy']]} at {p['clients']}" for p in slo_mismatches
        )
        or "none"
    )
    hardware_tuned192 = get[("kvs3c08", 192)]
    native_tuned192 = next(
        p for p in native if (p["policy"], p["clients"]) == ("kvs3c08", 192)
    )
    raw_table = table(
        ["Clients", "KV hardware artifacts", "RR hardware artifacts"],
        [
            [
                c,
                f"[KV c{c}]({get[('kv', c)]['gcs_console']})",
                f"[RR c{c}]({get[('rr', c)]['gcs_console']})",
            ]
            for c in CLIENTS
        ],
    )
    errors = "; ".join(
        f"{c} clients: KV {get[('kv', c)]['request_errors']}, RR {get[('rr', c)]['request_errors']}"
        for c in CLIENTS
    )
    return f"""# N3U aggregated serving: KV-aware routing versus round-robin

Hardware collected **2026-09-16** · V10 simulations **2026-09-17–18** · SLO comparison updated **2026-09-18** · **24 GB300 GPUs, 6 × TP4/EP4 workers**

With **TTFT p95 ≤10 s** and **E2E-normalized interactivity at P90 ≥20 output tokens/s/user**, the best measured default-KV point is **{k["clients"]} sessions**, versus **{r["clients"]} for RR**: **{k["total_tok_s_gpu"] / r["total_tok_s_gpu"]:.2f}× total tokens/s/GPU**. Tuned KV at 192 passes TTFT but narrowly fails interactivity (**{tuned_ttft["e2e_normalized_interactivity_p90_tps"]:.4f} <20**); its highest-throughput measured point passing both limits is also **{tuned["clients"]}**. These are single-run comparisons. The throughput peak remains 192 for both default policies, while the SLO boundary needs more points and repeats.

[Standalone HTML]({STEM}.html) · [all plotted data (CSV)]({STEM}.csv) · [data and provenance (JSON)]({STEM}.json) · [preserved inputs (ZIP)]({STEM}-inputs.zip)

## 1. Real data sweep and knee

### Setup and metric definitions

These are completed **hardware jobs**, using AIPerf 0.12.0's `inferencex-agentx-mvp` scenario and the public Weka 256K trace. Both default policies were measured at **48, 96, 192 and 384 clients**, with a **3,600-second profiling window**, 60-second grace and 1,200-second request timeout. AgentX concurrency counts live session trees, including their subagents; it does not equal requests simultaneously decoding.

Throughput charts use AIPerf **input + output tokens/s divided by all 24 GPUs**, including cached input tokens. This is served token volume, not GPU compute throughput. TTFT is AIPerf's **p95 over successful profiling requests**, converted from milliseconds to seconds. Output-only throughput is shown separately in the SLO tables. Interactivity charts use per-request E2E latency and output length; that per-user rate is not divided by GPU count.

The configured serving shape is six TP4/EP4 replicas, context 262,144, max running 16 per worker, 16,384-token prefill chunks, and 443,697 attention-KV pages at 64 tokens/page. The recipe installs Dynamo 1.4.2, whose SGLang dependency is 0.5.16. The initial container tag alone does not prove the installed package version; the benchmark's actual worker package inventory was not captured.

KV and RR used separate deployments (`n3u-agg-ns` and `n3u-agg-ns2`) with the same configured serving shape. Each baseline cell has one collected run; these are not repeated-trial estimates, and no confidence intervals are claimed. The busy-stream measurements in [AGG24_RESULTS.md](../AGG24_RESULTS.md) use a different concurrency definition and are excluded here.

![Real agg KV/RR: concurrency versus throughput and p95 TTFT]({STEM}-hardware.png)

[Hardware plot SVG]({STEM}-hardware.svg) · [PDF]({STEM}-hardware.pdf)

Stars mark **192, the highest-throughput sampled default-policy point**. Hollow markers at 384 identify the measured saturation region. The shaded 192–384 interval highlights the unsampled default-KV saturation transition; it is not a confidence band or an assertion that the two policies have identical knees. Green diamonds add the **measured tuned-KV scale-3/credit-0.8 points at 96 and 192**. That two-point curve does not establish a tuned throughput knee.

### What the sweep establishes

- **Default KV:** 192 remains stable in the reported within-run check. At 384 throughput falls **14.5%**, and TTFT p95 rises from **11.66 to 119.44 s**. The original request-record analysis reports TTFT p50 rising from **37.9 to 99.6 s** between the first and last quarters at 384. Thus **192 is the last stable sampled point**, and the saturation transition is bracketed between **192 and 384**.
- **RR:** doubling clients from 96 to 192 buys only **10.8%** more throughput while TTFT p95 rises **12.56 → 60.08 s**. This indicates diminishing returns in **96–192**. At 384 throughput falls another **25.4%** and TTFT p95 reaches **494.98 s**. The highest sampled throughput is at **192**, but even 96 fails the 10-second TTFT limit and the 20-token/s interactivity limit.
- **Resolution:** there are no collected default-policy AgentX points between 192 and 384. The original ladders stopped at 384; 768/1,536 were not measured. A continuous optimum or an exact knee at 192 is not established.
- **Errors:** exported request-error counts are {errors}. Latency percentiles exclude failed requests; their counts remain visible rather than being treated as zero.

The quarter-by-quarter queue evidence comes from the existing [hardware report, section iii](../AGENTX_AGG_RESULTS.md#iii-real-runs-performance-curve-and-the-kv-vs-rr-points). The plots and numerical comparisons here are regenerated from the original AIPerf summaries, not rounded plot-point JSON.

### Source jobs

{raw_table}

Each GCS directory contains the AIPerf summary (`profile_export_aiperf.json`), per-request metrics (`profile_export.jsonl`), raw request/response exports and server metrics. All 12 known agg AgentX summaries, including tuning runs, were verified accessible and preserved in the [input manifest]({data_name}/manifest.json). The [run index](../RUN_INDEX.md#bench-jobs-chronological) contains the complete job list.

## 2. KV versus RR at equal configuration and equal SLO

### 2.1 Same configuration and client count

GPU count, worker shape, dataset, scenario and profiling duration are fixed. The policies run on separate equivalent fleets; benchmark IDs, cache-bust namespaces and completed closed-loop request cohorts are not identical. Throughput gain is `100 × (KV / RR − 1)`; the final column is `RR TTFT p95 / KV TTFT p95`.

{same_config}

The main stable comparison is **192 clients**: KV has **1.42× total throughput** and **5.15× shorter p95 TTFT**. The 384 comparison describes overloaded operation and should not be presented as a sustainable capacity advantage.

### 2.2 TTFT-only comparison: p95 ≤10 seconds

For each policy, select the **highest-throughput measured cell that passes the same p95 threshold**. Concurrency is allowed to differ; topology and GPU count remain fixed. This is a run-level TTFT percentile constraint, not AIPerf's separate request-goodput metric and not a combined TTFT/ITL or error-rate SLO.

{ttft_slo}

The default-policy selections are **KV96 versus RR48**, giving **{kt["total_tok_s_gpu"] / rt["total_tok_s_gpu"]:.2f}× total throughput/GPU**. The measured TTFT transition is bracketed by **48–96 for RR** and **96–192 for default KV**. No interpolation supplies an unmeasured passing point. This replaces the report's former 20-second TTFT comparison with the chosen **10-second** limit.

The tuned-KV row uses prefill-load scale 3, overlap credit 0.8 and temperature 0. Its 192-session run passes this **TTFT-only** criterion, giving **{tuned_ttft["total_tok_s_gpu"] / rt["total_tok_s_gpu"]:.2f}×** RR48's total throughput. It **does not pass the added interactivity criterion**, as shown next.

### 2.3 Added comparison: E2E-normalized interactivity at P90 ≥20 output tokens/s

For each valid successful profiling request, compute `r_i = E2E_i_seconds / output_tokens_i`, then **`I90 = 1 / P90(r_i)`**. The selected threshold is **`I90 ≥20 output tokens/s/user`**, equivalently **`P90(r_i) ≤0.050 s/output token`**. E2E covers one inference request including TTFT and generation; human think time, tools and the rest of the session are outside this latency. This follows the [public InferenceX AgentX metric definition](https://github.com/SemiAnalysisAI/InferenceX/blob/main/MODELS.md#agentx-guidelines). The **20-token/s cutoff is our chosen engineering criterion**, not a Weka-mandated threshold.

The calculation uses individual AIPerf `request_latency` and `output_sequence_length` values, with linear percentile interpolation. It does **not** divide aggregate percentiles or use decode-only `1/TPOT`. AIPerf's P10 of per-request output/E2E rate is close but can differ under interpolation; it is checked against the export for provenance and is not substituted for the formula above. The exported data records the valid and excluded request counts for every run. Warmup, errors and cancelled records are excluded from the percentile, with counts preserved separately.

![P90 E2E-normalized interactivity versus concurrency: hardware and matching Native DynoSim V10 settings]({STEM}-interactivity.png)

[Interactivity SVG]({STEM}-interactivity.svg) · [PDF]({STEM}-interactivity.pdf) · [per-request metric provenance]({data_name}/request-metrics/manifest.json)

{interactivity_table}

**Tuned KV192 is a measured fail:** **{tuned_ttft["e2e_normalized_interactivity_p90_tps"]:.4f} tok/s**, or **{tuned_ttft["e2e_normalized_latency_p90_ms_per_token"]:.4f} ms/output token** against the 50 ms budget. It misses by **{100 * (1 - tuned_ttft["e2e_normalized_interactivity_p90_tps"] / INTERACTIVITY_SLO_TPS):.2f}%** in interactivity. Rounding to 20 would hide the failure. With one trial, its distance from the threshold cannot be distinguished from run-to-run variation; it is a priority repeat.

Select maximum measured throughput under **both run-level limits**: TTFT p95 ≤10 seconds and I90 ≥20 output tokens/s/user.

{joint_slo}

For the existing hardware samples, the interactivity-only and combined selections coincide. Default KV96 gives **{100 * (k["total_tok_s_gpu"] / r["total_tok_s_gpu"] - 1):.1f}% more total throughput/GPU than RR48**. Tuned KV96 gives **{tuned["total_tok_s_gpu"] / r["total_tok_s_gpu"]:.2f}× RR48** and **{100 * (tuned["total_tok_s_gpu"] / k["total_tok_s_gpu"] - 1):.1f}% more than default KV96**; the latter small difference needs repeats. These are the best **sampled eligible** cells, not proven capacity maxima.

Two run-level percentiles do not establish a joint per-request success fraction. Keep request errors and raw E2E distributions visible. AgentX is closed-loop, and policies can complete different request mixes within the same duration; these are criteria for this replay population. A production arrival-rate guarantee requires a separate load-controlled validation.

### 2.4 Real KV-router flag sweep

**Yes: four additional real AgentX jobs were collected**—three KV variants at **192 clients**, plus the best measured variant at **96 clients**. The default-KV and RR comparisons reuse the baseline jobs above. All use the same 24-GPU serving shape and 3,600-second AgentX profiling configuration. These are measured hardware results; no new hardware jobs were launched to generate this report.

**Complete collected tuned-KV agg sweep:**

{tuned_sweep_table}

Scale 3/credit 0.8 is the only tuned setting measured at two concurrencies. Scale 2/credit 0.8 and temperature 0.5 each have one 192-session sample; no curves or additional points are inferred for them. The upper agg plot overlays the two measured scale-3 throughput/TTFT points, and the interactivity plot shows its I90 curve. The 192-session bar chart below compares every tested setting at the same concurrency.

The settings below come from the preserved [runner recipe]({data_name}/hardware/agentx_runner.sh.txt), which installs Dynamo 1.4.2 and changes the frontend router arguments for each named variant. “Not explicitly set” means the recipe inherits the frontend's defaults; the table does not infer a numeric value from a variant name.

{settings_table}

The explicit flag names are `--router-prefill-load-scale`, `--router-kv-overlap-score-credit`, `--router-temperature` and `--router-queue-policy`. All four KV recipes use `--router-mode kv`. The RR reference uses `--router-mode round-robin`.

![Measured agg KV-router flags at 192 AgentX clients]({STEM}-flag-sweep.png)

[Flag-sweep SVG]({STEM}-flag-sweep.svg) · [PDF]({STEM}-flag-sweep.pdf) · [settings, metrics and deltas (CSV)]({STEM}-flag-sweep.csv)

{flags_table}

At **192 clients**, `kvs3c08` is the **best measured setting**: **11,012 total tok/s/GPU** and **6.33 s p95 TTFT**, versus default KV's **9,655** and **11.66 s**. That is **14.1% more throughput** and **45.7% lower p95 TTFT**. Output-only throughput also rises from **96.81 to 108.87 tok/s/GPU**. Relative to RR at the same 192 clients, this setting gives **1.62× total throughput** and **9.49× shorter p95 TTFT**.

The lower scale of 2 with the same 0.8 credit improves throughput by **6.1%** and p95 TTFT by **30.4%** versus default KV. Temperature 0.5 reduces throughput by **19.0%** and increases p95 TTFT by **90.0%**, to **22.15 s**, which misses the 10-second TTFT SLO. **All five 192-client settings fail I90 ≥20**, including the two tuned settings that pass TTFT. All five rows have **three exported request errors**; latency percentiles cover successful requests.

At **96 clients**, [the same scale-3/credit-0.8 variant]({tuned96["gcs_console"]}) delivers **{tuned96["total_tok_s_gpu"]:,.0f} total tok/s/GPU** versus [{default96["total_tok_s_gpu"]:,.0f} for default KV]({default96["gcs_console"]}), a **{tuned96["throughput_change_vs_default_kv_pct"]:.1f}%** increase. P95 TTFT falls **{default96["ttft_p95_s"]:.2f} → {tuned96["ttft_p95_s"]:.2f} s** (**{-tuned96["ttft_change_vs_default_kv_pct"]:.1f}% lower**). Both jobs have zero exported errors. There are no repeat trials to determine the statistical significance of the small throughput difference.

The exact best measured router arguments are:

```text
--router-mode kv
--router-temperature 0.0
--router-queue-policy fcfs
--router-prefill-load-scale 3.0
--router-kv-overlap-score-credit 0.8
```

**Scope of the conclusion:** this is a small flag sweep, not a full factorial search. Scale and overlap credit change together versus default KV, so their individual contributions cannot be separated; scale 2 versus scale 3 does hold credit at 0.8. A `kvwspt` queue-policy recipe exists, but no completed agg AgentX measurement for it is present in the collected run set. There are no measured scale-4/5 variants, independent overlap-credit sweep, or tuned runs above 192 clients. Each cell has one trial on one of the two equivalent deployments. Thus this selects the best **observed** setting without establishing a global optimum or tuned knee.

The higher cached-input share and lower latency are consistent with a better balance between cache reuse and queued work, but the aggregate summaries do not isolate that mechanism. Section 3 now replays every collected flag variant through Native DynoSim V10 and compares its metrics with hardware. Those new points use the original calibration unchanged, so they test whether the simulator reproduces the measured tuning effect. The [original tuning analysis](../AGENTX_AGG_RESULTS.md#v-kv-router-flag-sweep-at-the-192-client-comparison-point-measured) is retained as background; the tables here are regenerated from the full AIPerf summaries and keep the tuned points separate from the default-KV concurrency curve.

### 2.5 Why temperature 0.5 and overlap credit 0.8; what to sweep next

The preserved recipe and experiment notes show a **small exploratory set**, not a parameter search: temperature **0 versus 0.5** with inherited default scale/credit, and prefill scales **2 versus 3** with credit fixed at **0.8** and temperature **0**. The notes compare the tuned recipe with an earlier simulator prediction, but do **not record an optimization or quantitative rule that selected exactly 0.5 or 0.8**. We cannot claim either value was optimal or that an independent credit sweep was completed.

Temperature here is **router worker-selection randomness**, not model generation temperature. Overlap credit **0.8 is a routing cost multiplier**, not an 80% cache-hit target or cache-size fraction. These meanings follow [NVIDIA's router configuration documentation](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/router/configuration-and-tuning). Record the installed version and resolved flags for every new run; the baseline recipe did not explicitly set scale/credit, so this report does not infer their actual runtime values from the historical label.

The existing temperature-0.5 run lost **19.0% throughput** against default KV at 192 and produced **7.0315 I90**, but it does not rule out smaller temperatures or the same temperature on a tuned scale/credit configuration. Similarly, scale 3 outperformed scale 2 **at fixed credit 0.8**; that does not establish a monotone trend from default KV because scale and credit both change against the baseline.

Use the measured `(scale=3, credit=0.8, temperature=0, queue=fcfs)` setting as the anchor. Sweep one parameter at a time first, then check interactions among the strongest candidates:

| Knob | Proposed values | Hold fixed | Existing evidence / next question |
| --- | --- | --- | --- |
| Prefill-load scale | **1, 2, 3, 4, 5** | Credit 0.8, temperature 0, FCFS | 2 and 3 exist at 192. Add 1, 4 and 5 to isolate scale and find whether the gain continues. |
| Device-local overlap credit | **0.4, 0.6, 0.8, 1.0** | Scale 3, temperature 0, FCFS | Only 0.8 is measured here. Determine how much preference for cached prefixes helps near the SLO boundary. |
| Router temperature | **0, 0.1, 0.25, 0.5** | Scale 3, credit 0.8, FCFS | Only 0 exists on this anchor. The old 0.5 job used inherited defaults, so it is not the 0.5 cell of this new sweep. |
| Queue policy, second priority | **FCFS, WSPT** | Same selected scale, credit, temperature and enabled queue threshold | A WSPT recipe exists but has no completed agg AgentX run. Confirm that requests actually queue; otherwise the policy may not affect dispatch. Compare tails and interactivity as well as throughput. |

**First small batch:** repeat the anchor at 192, then test **scale 4/5**, **credit 0.6/1.0**, and **temperature 0.1/0.25** at 192 with the other anchor values fixed. These are six new settings, each isolated against the same anchor. At this load the current best I90 is only 1.10% below target, so a reproducible improvement could turn 192 into an eligible point. Initial runs screen candidates; repeat finalists and validate them at **128/160** and **96**, then compare the best measured throughput satisfying **both** TTFT p95 ≤10 s and I90 ≥20. Add a small crossed scale/credit grid around any winner; isolated sweeps alone can miss interactions.

The [proposed six-setting sweep (JSON)]({STEM}-next-sweep.json) lists the exact router arguments. Six candidate runs plus two additional anchor repeats cost **eight one-hour profiles, or 192 GPU-hours on a 24-GPU fleet**, before warmup, drain and setup. This is a staged proposal; it does not schedule jobs. Finalist repeats and new concurrency points add to that budget.

After router tuning, engine admission (`max-running-requests`) and prefill chunk size are separate useful sweeps, with memory pressure, errors and queue statistics recorded. They change the serving configuration and should form a separate comparison group. Keep the Weka workload, think time, duration and tokenizer fixed across tuning arms. These proposed jobs have **not been launched**.

### 2.6 Which KV flags also apply to disagg?

Disagg has additional choices, but the agg winner is not a validated disagg recipe. Our [historical disagg sweep](../D72_RESULTS.md#kv-router-flag-configuration-and-sweep-for-the-disagg-selected-points) tested scale 2, credit 0.8 and temperature 0.5 separately on **6 prefill + 12 decode workers, host-staged transport, request concurrency 12**. Relative output-throughput changes were **+1.2%, +2.9% and −16%**; TTFT p95 was **6.3, 6.3 and 12.5 seconds**, versus **6.5 seconds** for default KV. These single-cell results use the earlier busy-stream workload, not this AgentX session-concurrency replay. The small gains do not establish an optimum. The historical report also contains an older simulation grid; it is not Native V10 validation and is excluded from this report's comparisons.

Dynamo routes prefill and decode separately: the documented prefill stage disables active-block tracking, and ordinary disagg decode routing disables overlap scoring and prefill-token tracking. Consequently, cache-credit tuning targets prefill placement; decode needs its own load-balance evidence. See [Dynamo's disagg routing behavior](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/router/disaggregated-serving).

For a new disagg AgentX study, first record the effective configuration and per-worker placement on both stages. Start with temperature **0** and FCFS. Prioritize `--router-kv-overlap-score-credit` at **0.6, 0.8 and 1.0**, then an independent `--router-kv-overlap-score-credit-decay` sweep if hot-cache workers accumulate prefill backlog. Test queue threshold and FCFS/WSPT together with evidence that router queueing is active. Treat `--router-prefill-load-scale` **1/2/3** as conditional: if every candidate's cost is only the same positive scale times adjusted prefill work, the scale cancels from deterministic ranking. That is an inference from the [documented cost equation](https://docs.nvidia.com/dynamo/knowledge-base/modular-components/router/routing-concepts), not a measured improvement; confirm a competing cost term or changed placements before spending a full sweep on it.

Hold the P:D split, total GPUs, transport, cache tiers and workload fixed. Select throughput under **TTFT p95 ≤10 seconds and I90 ≥20 output tokens/s**, using actual AgentX measurements. This disagg plan is separate from the completed agg settings and has not been launched here.

## 3. Simulation method, curves and knee selection

### 3.1 Native DynoSim V10: current completed calibration samples

The workspace contains a newer, separate **native DynoSim experiment**. Its path is:

```text
AIPerf 0.12.0 AgentX client, normal wall clock and streaming HTTP
  → Dynamo 1.4.2 frontend and KV / round-robin router
  → native SGLang Mocker/DynoSim worker scheduler
  → AIConfigurator 0.11.0 forward-pass timing
```

**V10 is a local candidate built on Dynamo 1.4.2 with eleven patches; it is not an unmodified NVIDIA release or a Dynamo release named v10.** The patches cover CPU execution, visible first-token handling, prefix/chunk admission, hybrid-state cache behavior and timing corrections. All **12 plotted native runs** use one native core build and one shared engine configuration. The original **four calibration points** are default KV/RR at 192/384; **eight new holdouts** cover default KV/RR at 48/96 and all four tuned-KV hardware settings. The preserved [calibration matrix]({data_name}/native-v10/matrix.json) and [holdout matrix]({data_name}/native-v10/holdouts.json) record these checks; the [patch reconstruction record]({data_name}/native-v10/v10_patch_reconstruction.json) identifies the modified source.

The actual AgentX loader, branches/joins, seed 42, 393 roots, 25–75% sampled starts, one-token trajectory warmup, whole-system idle cap, recycling and metrics export remain in AIPerf. Original benchmark IDs reproduce each hardware point's initial cache-bust namespace. Both simulator speedup ratios are **1.0**. A separate 900-second hardware prewarm failed and is not invented for simulation. Latency still changes closed-loop completion order and the completed request cohort, so this is not a claim of byte-identical full-run request sequences.

The shared native configuration uses six TP4/EP4 workers, max running 16, 16,384-token chunks and the measured **28,396,608 attention-KV tokens per worker**. Its hybrid-state settings are **769 state slots**, 64-token cache chunks and a 256-token tracking interval. Those hybrid settings remain model assumptions; the actual hardware Mamba-state pool was not captured. AIC uses its SGLang **0.5.14** tables, while the hardware recipe's Dynamo dependency specifies SGLang **0.5.16**.

Prefill calibration was fitted to **93 isolated one-token RR192 hardware warmups**, using shared coefficients for every policy and concurrency:

```text
prefill_ms = 1.7050018888 × AIC_prefill_ms
           + 165.5695513 × batch × new_tokens × (prefix_tokens + new_tokens / 2) / 1e9
decode_ms  = AIC_decode_ms + 0.288 + 0.03145728 × ready_decode_requests
```

The decode additions are a candidate correction for missing recurrent-state work in AIC's analytical fallback, derived from model geometry and bandwidth/kernel-overhead assumptions. They are not directly measured Nemotron kernel times. No per-concurrency throughput multiplier is applied. See the preserved [prefill fit]({data_name}/native-v10/prefill_calibration_v2.json) and [decode derivation]({data_name}/native-v10/decode_mamba_timing_candidate_v10.json).

![Native V10 agg KV/RR simulation versus hardware]({STEM}-native-simulation.png)

[Native simulation SVG]({STEM}-native-simulation.svg) · [PDF]({STEM}-native-simulation.pdf)

Solid KV/RR curves now contain **48, 96, 192 and 384** for each default policy. The tuned scale-3/credit-0.8 curve contains **96 and 192**; scale 2/credit 0.8 and temperature 0.5 each have one 192-session point. Faint dashed curves show default-policy hardware. **Every hardware point has a matching Native DynoSim V10 run; all current simulation plots and comparisons use that same V10 build.**

{native_table}

The recorded calibration gate is **±20% total throughput error at each of the four selected points**; all four pass. **TTFT was not an acceptance target**, and three points miss ±20% TTFT. The RR384 tail is underestimated by **38.4%**. These four workloads were used during model development; passing them is calibration evidence, not independent validation of arbitrary loads, policies or latency SLOs. The latest candidate narrows the earlier throughput gap, but does not establish TTFT fidelity.

Among the **eight new holdouts**, **{throughput_holdouts_pass}/8** fall within ±20% total throughput error. This is a check against the frozen model, not a newly fitted calibration gate. Latency and SLO classification are assessed separately below.

The interactivity comparison uses each native run's own request records. Units below are output tokens/s/user; it was **not a calibration acceptance target**.

{native_interactivity}

Across all 12 paired settings, the combined-SLO pass/fail decision differs at **{len(slo_mismatches)}** points: **{native_mismatches_text}**. Native KV192 passes TTFT alone while hardware KV192 fails it; interactivity rejects both. Matching throughput at the four calibration points does not by itself validate latency or SLO decisions.

{native_slo_table}

Each model selects its highest-throughput measured cell passing both limits. A matching selected concurrency is evidence about this sampled ladder, not proof of an exact boundary or independent confirmation of the hardware capacity.

### 3.2 Native V10 comparison for the complete tuned-KV agg sweep

All four collected tuned-KV hardware jobs now have a corresponding **Native DynoSim V10 holdout** with the same hardware workload configuration, exact router arguments and unchanged engine calibration. Temperature, scale and credit are applied to the native Dynamo frontend; they are not replaced with fitted throughput multipliers.

{native_tuned_table}

For the scale-3/credit-0.8 **192-session** pair, median input lengths are **{hardware_tuned192["input_tokens_p50"]:,.0f} / {native_tuned192["input_tokens_p50"]:,.0f} tokens** (hardware/V10), and median output lengths are **{hardware_tuned192["output_tokens_p50"]:,.0f} / {native_tuned192["output_tokens_p50"]:,.0f}**. The interactivity difference therefore accompanies very similar median sequence lengths. TTFT p95 is **{hardware_tuned192["ttft_p95_s"]:.2f} / {native_tuned192["ttft_p95_s"]:.2f} s**, and ITL p90 is **{hardware_tuned192["itl_p90_ms"]:.2f} / {native_tuned192["itl_p90_ms"]:.2f} ms**. Both latency components remain relevant; these aggregate percentiles cannot apportion the I90 error causally, and the completed request cohorts differ. Use the hardware result for the operating-point decision until repeats and intermediate points resolve the boundary.

These are direct comparisons of the same named setting. The two scale-3 points form the measured concurrency curve; the other settings have one point each. Errors and missed SLO classifications remain in the comparison. The earlier custom Python model is archived in the preserved inputs for historical provenance and is **excluded from every current simulation chart, selection and comparison**.

### 3.3 How a knee is selected

1. **Hold the workload and serving configuration fixed.** For hardware or one simulator build, sweep concurrency from low load upward. Compare the same AIPerf throughput definition and GPU denominator.
2. **Locate diminishing returns and saturation.** Examine throughput gains between adjacent samples, p95 TTFT, request errors, and within-run queue/TTFT progression. The highest-throughput sample is a candidate operating point, not proof of an exact continuous knee.
3. **Bracket the transition.** Hardware default KV is stable at 192 and saturated at 384; RR is already flattening over 96–192 and collapses by 384. Native V10 also declines from 192 to 384 for the default policies; its added 48/96 points complete the same geometric ladder. Intermediate concurrencies and repeats are still needed to localize a knee, and the two-point tuned curve does not establish one.
4. **Refine and validate.** Prioritize the SLO boundaries below. Further throughput-peak refinement at KV240/288/336 or RR120/144/168 can wait: those intervals are already beyond the observed default-policy SLO boundaries. Queue stationarity and failure behavior must accompany throughput before promoting a simulator-selected point to a hardware recommendation.
5. **Apply the SLO separately.** Among measured eligible cells, maximize throughput subject to both thresholds. A throughput knee and the best point under a latency budget are different selections: here RR192 is the sampled throughput peak, while RR48 is the combined-SLO choice. The pass/fail boundary is not itself proof of a sharp knee; tail changes need repeats to establish that they exceed run-to-run variation.

### 3.4 Recommended additional measurements

**Yes—collect targeted boundary points and repeats.** The geometric sweep already shows overload; more points above 384 are low priority for this SLO. The following work is proposed, not collected:

| Priority | Measurement | What it resolves |
| --- | --- | --- |
| 1 | Two additional independent repeats of default KV96, RR48, tuned KV96 and tuned KV192, keeping the same seed and 3,600-second duration | Establish three trials per anchor; tuned KV192 is only 1.10% below the interactivity cutoff. Report each trial and the spread before accepting a borderline point. |
| 2 | Default RR64, then RR80 if needed to narrow the boundary | RR48 passes both limits; RR96 fails both. Start at 64 and refine according to the measured pass/fail bracket. |
| 2 | Default and tuned KV128, then KV160 if needed, keeping the two KV settings paired | Both pass at 96 and fail interactivity at 192. The intermediate points may add throughput while meeting both limits. |
| 3 | Native V10 at any newly collected 64/80/128/160 hardware boundary points, with the build and calibration coefficients frozen | The original 12 hardware settings now have matching native runs. New intermediate hardware points would provide another holdout check of the SLO boundary. |

For direct KV/RR A/B claims at a new concurrency, collect the matching policy at that **same concurrency**; for capacity under the shared SLO, each policy may select a different concurrency. Repeat the final new boundary cells before promoting them. Keep corpus, tokenizer/revision, seed, duration, serving configuration and cache-reset procedure fixed; run close in time or alternate order. Check trace/context/output-length distributions, request failures, cache hit, actual in-flight requests and per-worker queue/KV pressure. After estimating same-seed run noise, use an additional matched seed to check trace-sample sensitivity. Do not fit a new simulator correction to these points and also call them independent validation.

### Reproduce this report

The [manifest]({data_name}/manifest.json) preserves exact source hashes and original locations. The input ZIP contains hardware summaries and the router-flag recipe, native configurations/build identity, the native matrix, calibration evidence, the separate custom-model audit/source, and **numeric per-request metrics for all 24 plotted runs**. The [original native experiment notes]({data_name}/native-v10/original-calibration-report.md) preserve the detailed method and source locations as a readable snapshot; [the original text]({data_name}/native-v10/original-calibration-report.txt) is retained. Full hardware request records remain in the linked GCS artifacts. Temporary source paths are recorded for provenance; regeneration uses the preserved report inputs. The command below rebuilds the report; it does not rerun hardware jobs or native simulations.

From the study directory, with Python 3.12, Matplotlib 3.11.2, NumPy 2.5.3 and markdown-it-py 4.2.0 available:

```bash
python scripts/gen_agg_kv_rr_report.py \\
  --data-dir reports/{data_name} \\
  --output-dir reports
```

The generator checks input hashes, all expected policy/concurrency pairs, the 3,600-second AgentX phase settings, GPU normalization, agreement between native summaries and the comparison matrix, and the shared native engine/build. It also reconciles per-request success/error counts and TTFT with every summary, and raw E2E p95 and AIPerf's P10 rate with the hardware/native exports. It computes the canonical I90 and all selections from unrounded per-request values and writes the [validation record]({STEM}-validation.json). The [numeric extraction script](../scripts/import_agentx_request_metrics.py) preserves phase, status, E2E, OSL and TTFT without prompt/response content; source hashes and row counts are recorded in its manifest. The [frozen native launch adapters](../scripts/native_v10_agg/README.md) describe how the added runs apply each hardware setting without changing the model.
"""


def html_report(markdown, output):
    parser = MarkdownIt("commonmark").enable("table")
    body = parser.render(markdown)
    for name in [
        "hardware",
        "flag-sweep",
        "interactivity",
        "native-simulation",
    ]:
        svg = (output / f"{STEM}-{name}.svg").read_bytes()
        body = body.replace(
            f'src="{STEM}-{name}.png"',
            'src="data:image/svg+xml;base64,' + base64.b64encode(svg).decode() + '"',
        )
    for suffix, mime in [
        ("csv", "text/csv"),
        ("json", "application/json"),
        ("inputs.zip", "application/zip"),
    ]:
        filename = (
            f"{STEM}.{suffix}" if suffix != "inputs.zip" else f"{STEM}-inputs.zip"
        )
        encoded = base64.b64encode((output / filename).read_bytes()).decode()
        body = body.replace(
            f'href="{filename}"',
            f'download="{filename}" href="data:{mime};base64,{encoded}"',
        )

    def heading(match):
        level, content = match.groups()
        ident = re.sub(
            r"[^a-z0-9]+", "-", re.sub(r"<[^>]*>", "", content).lower()
        ).strip("-")
        return f'<h{level} id="{ident}">{content}</h{level}>'

    body = re.sub(r"<h([23])>(.*?)</h\1>", heading, body)

    def portable_link(match):
        link = match[1]
        if ":" in link or link.startswith("#"):
            return match[0]
        if link == f"{STEM}.html":
            return 'href="#"'
        filename, _, anchor = link.partition("#")
        target = (output / filename).resolve()
        types = {
            ".svg": "image/svg+xml",
            ".pdf": "application/pdf",
            ".csv": "text/csv",
            ".json": "application/json",
            ".txt": "text/plain",
        }
        if (
            target.is_relative_to(output.resolve())
            and target.is_file()
            and target.suffix in types
        ):
            encoded = base64.b64encode(target.read_bytes()).decode()
            download_name = "--".join(target.relative_to(output.resolve()).parts)
            return f'download="{download_name}" href="data:{types[target.suffix]};base64,{encoded}"'
        if target.is_relative_to(STUDY.parent.parent):
            remote = (
                "https://github.com/alisachen-google/gcp-dynamo-cuj/blob/main/"
                + str(target.relative_to(STUDY.parent.parent))
            )
            return f'href="{remote}{"#" + anchor if anchor else ""}"'
        return match[0]

    body = re.sub(r'href="([^"]+)"', portable_link, body)
    document = (
        """<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>N3U agg KV vs RR · hardware and simulation</title>
<style>
:root{color-scheme:light;--ink:#162b45;--muted:#52657a;--line:#dce4ed}
*{box-sizing:border-box}body{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.65 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1260px;margin:28px auto 64px;padding:38px 54px 64px;background:#fff;border:1px solid var(--line);border-radius:14px}
h1{font-size:36px;line-height:1.2;letter-spacing:-.8px;margin:0 0 18px}h2{font-size:27px;line-height:1.3;margin-top:52px;padding-top:20px;border-top:2px solid var(--line)}
h3{font-size:20px;margin-top:30px}p,li{max-width:1120px}a{color:#205db1;text-decoration-thickness:1px;text-underline-offset:3px}
table{border-collapse:collapse;width:100%;font-size:13px;margin:24px 0;font-variant-numeric:tabular-nums}th,td{padding:10px 12px;text-align:left;border-bottom:1px solid var(--line)}
th{background:#edf3f9;font-weight:650}tr:nth-child(even) td{background:#f8fafc}img{width:100%;height:auto;border:1px solid var(--line);border-radius:8px;margin:12px 0}
code{font:13px ui-monospace,SFMono-Regular,monospace;background:#edf2f7;padding:2px 4px;border-radius:3px}pre{padding:18px 20px;background:#edf2f7;border-radius:8px;overflow:auto}pre code{padding:0;background:none}
li{margin:9px 0}.eyebrow{font-size:12px;letter-spacing:1.6px;text-transform:uppercase;color:var(--muted);margin-bottom:14px}nav{display:flex;gap:24px;flex-wrap:wrap;font-size:14px;margin:0 0 26px}
@media(max-width:800px){main{margin:0;padding:24px 18px;border-radius:0}h1{font-size:29px}table{display:block;overflow-x:auto;font-size:12px}th,td{padding:8px}h2{font-size:23px}}
@media print{body{background:#fff}main{border:0;max-width:none;margin:0;padding:0}nav{display:none}h2,h3{break-after:avoid}img,table{break-inside:avoid}a{color:inherit}}
</style></head><body><main><div class="eyebrow">Nemotron-3-Ultra 550B · AgentX · 24 GB300 GPUs</div>
<nav><a href="#1-real-data-sweep-and-knee">1. Real sweep</a><a href="#2-kv-versus-rr-at-equal-configuration-and-equal-slo">2. SLO comparisons</a><a href="#2-4-real-kv-router-flag-sweep">Tuned KV agg sweep</a><a href="#3-simulation-method-curves-and-knee-selection">3. Simulation</a><a href="#3-4-recommended-additional-measurements">Next measurements</a></nav>
"""
        + body
        + "</main></body></html>\n"
    )
    (output / f"{STEM}.html").write_text(document)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir", type=Path, default=STUDY / "reports/agentx-agg-kv-rr-data"
    )
    parser.add_argument("--output-dir", type=Path, default=STUDY / "reports")
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    manifest, hardware, native = load_inputs(args.data_dir)
    baseline = [p for p in hardware if p["policy"] in COLORS]
    plot_pair(
        output,
        "hardware",
        baseline,
        "Measured hardware · default KV, tuned KV and RR",
        "N3U · 24 GB300 GPUs · 6 × TP4/EP4 · AgentX · one-hour profiles",
        knee_evidence(baseline)
        + "\n★: sampled default-policy throughput peaks; not the SLO choices. Shading: KV saturation bracket, 192–384."
        + "\nTuned scale 3: only 96/192 measured; other variants: one point each. Their throughput knees remain unresolved.",
        "hardware",
        tuned=[p for p in hardware if p["policy"] not in COLORS],
    )
    plot_pair(
        output,
        "native-simulation",
        native,
        "Native DynoSim V10 · throughput and latency",
        "Dynamo 1.4.2 + local V10 patches · AIC 0.11.0 · 24-GPU model · normal AIPerf HTTP replay",
        knee_evidence(native)
        + "\nSolid: V10; faint dashed: hardware. ★: sampled default-policy throughput peaks; not SLO choices."
        + "\nFour calibration points + eight holdouts, one frozen build. Tuned knees remain unresolved; latency requires validation.",
        "native",
        baseline,
        tuned=[p for p in native if p["policy"] not in COLORS],
    )
    flag_points = flag_sweep_points(hardware)
    plot_flag_sweep(output, flag_points)
    plot_interactivity(output, hardware, native)
    all_points = hardware + native
    fields = [
        "kind",
        "id",
        "role",
        "hardware_id",
        "policy",
        "clients",
        "total_tok_s_gpu",
        "output_tok_s_gpu",
        "requests_s",
        "input_tokens_p50",
        "output_tokens_p50",
        "ttft_p95_s",
        "e2e_p95_s",
        "e2e_normalized_interactivity_p90_tps",
        "e2e_normalized_latency_p90_ms_per_token",
        "interactivity_valid_requests",
        "interactivity_excluded_successful_requests",
        "ttft_slo_pass",
        "interactivity_slo_pass",
        "combined_slo_pass",
        "itl_mean_ms",
        "itl_p90_ms",
        "cache_pct",
        "successful_requests",
        "request_errors",
        "router_flags",
    ]
    with (output / f"{STEM}.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(all_points)
    flag_fields = [
        "id",
        "policy",
        "clients",
        "label",
        "prefill_load_scale",
        "overlap_score_credit",
        "temperature",
        "queue_policy",
        "total_tok_s_gpu",
        "output_tok_s_gpu",
        "throughput_change_vs_default_kv_pct",
        "ttft_p95_s",
        "e2e_normalized_interactivity_p90_tps",
        "ttft_slo_pass",
        "interactivity_slo_pass",
        "combined_slo_pass",
        "ttft_change_vs_default_kv_pct",
        "cache_pct",
        "request_errors",
        "router_flags",
        "gcs_summary",
        "gcs_console",
    ]
    with (output / f"{STEM}-flag-sweep.csv").open("w") as f:
        writer = csv.DictWriter(
            f, fieldnames=flag_fields, extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(flag_points)
    exported = {
        "metric_basis": "AIPerf total input+output tokens/s / 24; p95 TTFT in seconds",
        "slo_ttft_p95_s": TTFT_SLO_S,
        "slo_e2e_normalized_interactivity_p90_tps": INTERACTIVITY_SLO_TPS,
        "interactivity_definition": {
            "formula": "1 / percentile_90(request_latency_ms / 1000 / output_sequence_length)",
            "unit": "output tokens/s/user",
            "percentile_method": "linear",
            "population": "Successful profiling requests with finite positive E2E and OSL; excludes warmup, errors and cancellations.",
            "source": "https://github.com/SemiAnalysisAI/InferenceX/blob/main/MODELS.md#agentx-guidelines",
            "threshold_origin": "User-selected engineering criterion; not a dataset requirement.",
        },
        "hardware_slo_selections": {
            criterion: {
                policy: best_point(hardware, policy, criterion)["id"]
                for policy in ["kv", "rr", "kvs3c08"]
            }
            for criterion in [
                "ttft_slo_pass",
                "interactivity_slo_pass",
                "combined_slo_pass",
            ]
        },
        "native_v10_slo_selections": {
            policy: best_point(native, policy)["id"]
            if best_point(native, policy)
            else None
            for policy in ["kv", "rr", "kvs3c08"]
        },
        "simulation_scope": "Native DynoSim V10 only; all 12 hardware policy/concurrency settings matched with one frozen build and engine.",
        "source_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "points": all_points,
        "flag_sweep": flag_points,
    }
    (output / f"{STEM}.json").write_text(json.dumps(exported, indent=2) + "\n")
    (output / f"{STEM}-next-sweep.json").write_text(
        json.dumps(proposed_flag_sweep(), indent=2) + "\n"
    )
    with zipfile.ZipFile(
        output / f"{STEM}-inputs.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name in ["manifest.json", *sorted(manifest["files"])]:
            archive.write(args.data_dir / name, args.data_dir.name + "/" + name)
    markdown = report(hardware, native, args.data_dir.name)
    (output / f"{STEM}.md").write_text(markdown)
    best_flag = max(
        (p for p in flag_points if p["clients"] == 192),
        key=lambda p: p["total_tok_s_gpu"],
    )
    assert best_flag["policy"] == "kvs3c08"
    validation = {
        "all_checks_passed": True,
        "hardware_summaries": len(hardware),
        "baseline_hardware_points": len(baseline),
        "additional_flag_sweep_jobs": len(hardware) - len(baseline),
        "flag_csv_rows_including_references": len(flag_points),
        "router_arguments_parsed_from_preserved_recipe": True,
        "best_measured_flag_at_192": best_flag["policy"],
        "best_flag_throughput_change_vs_default_kv_pct": best_flag[
            "throughput_change_vs_default_kv_pct"
        ],
        "best_flag_ttft_change_vs_default_kv_pct": best_flag[
            "ttft_change_vs_default_kv_pct"
        ],
        "native_v10_points": len(native),
        "custom_python_points": 0,
        "native_points_match_all_hardware_settings": True,
        "original_native_calibration_points": 4,
        "new_native_holdout_points": 8,
        "native_calibration_frozen": True,
        "verified_input_files": len(manifest["files"]),
        "shared_native_core_and_engine": True,
        "native_matrix_matches_summaries": True,
        "native_ttft_calibrated": False,
        "interactivity_recomputed_from_requests": len(all_points),
        "interactivity_valid_requests": sum(
            p["interactivity_valid_requests"] for p in all_points
        ),
        "interactivity_excluded_successful_requests": sum(
            p["interactivity_excluded_successful_requests"] for p in all_points
        ),
        "request_counts_and_ttft_match_summaries": True,
        "hardware_and_native_e2e_p95_and_rate_p10_match_summaries": True,
        "percentile_method": "linear",
        "slo_ttft_p95_s": TTFT_SLO_S,
        "slo_e2e_normalized_interactivity_p90_tps": INTERACTIVITY_SLO_TPS,
        "slo_selections": {
            criterion: {
                policy: best_point(hardware, policy, criterion)["clients"]
                for policy in ["kv", "rr", "kvs3c08"]
            }
            for criterion in [
                "ttft_slo_pass",
                "interactivity_slo_pass",
                "combined_slo_pass",
            ]
        },
    }
    (output / f"{STEM}-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    # The standalone HTML embeds downloads, including this validation record.
    # Write every artifact first so a prior run's record cannot be embedded.
    html_report(markdown, output)
    print(json.dumps(validation))
    print(output / f"{STEM}.html")


if __name__ == "__main__":
    main()
