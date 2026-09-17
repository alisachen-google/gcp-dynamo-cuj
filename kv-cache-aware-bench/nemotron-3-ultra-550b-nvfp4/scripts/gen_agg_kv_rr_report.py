#!/usr/bin/env python3
"""Build the agg KV/RR report from preserved hardware and simulation artifacts."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import json
import re
import shlex
import zipfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from markdown_it import MarkdownIt
from matplotlib.ticker import FuncFormatter

STUDY = Path(__file__).resolve().parents[1]
STEM = "agentx-agg-kv-rr-report"
COLORS = {"kv": "#2463b3", "rr": "#d65b29"}
MARKERS = {"kv": "o", "rr": "s"}
CLIENTS = [48, 96, 192, 384]
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
        "ttft_p95_s": summary["time_to_first_token"]["p95"] / 1000,
        "itl_mean_ms": summary["inter_token_latency"]["avg"],
        "cache_pct": summary.get("overall_usage_prompt_cache_read_pct", {}).get("avg"),
        "successful_requests": int(summary["request_count"]["avg"]),
        "request_errors": int(summary.get("error_request_count", {}).get("avg", 0)),
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
    baseline = [p for p in hardware if p["policy"] in COLORS]
    assert {(p["policy"], p["clients"]) for p in baseline} == {
        (p, c) for p in COLORS for c in CLIENTS
    }
    reference = {(p["policy"], p["clients"]): p for p in baseline}
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
    native = []
    engine_hashes, core_hashes = set(), set()
    for point in matrix["points"]:
        run = folder / "native-v10" / point["run"]
        summary = read(run / "simulation_summary.json")
        validate_client(summary, point["clients"])
        values = metrics(summary)
        hardware_point = reference[(point["router"], point["clients"])]
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
                "policy": point["router"],
                "clients": point["clients"],
            }
        )
    assert len(engine_hashes) == len(core_hashes) == 1
    assert {(p["policy"], p["clients"]) for p in native} == {
        (p, c) for p in COLORS for c in [192, 384]
    }

    custom = []
    for point in read(folder / "custom-python/audit.json")["points"]:
        assert point["checks"]["profiling_send_window_s"] == 3600
        assert point["provenance"]["cache_capacity_tokens"] == 28396608
        assert point["provenance"].get("kv_decode_block_cost_ablation", False) == (
            point["policy"] == "kv"
        )
        assert point["provenance"].get("scheduled_serving_config") is None
        s = point["simulation"]
        custom.append(
            {
                "kind": "custom_python",
                "id": point["id"],
                "policy": point["policy"],
                "clients": point["clients"],
                "total_tok_s_gpu": s["total_tokens_s_gpu"],
                "output_tok_s_gpu": s["output_tokens_s_gpu"],
                "ttft_p95_s": s["ttft_p95_s"],
                "itl_mean_ms": s["itl_mean_ms"],
                "cache_pct": s["cached_input_percent"],
                "successful_requests": s["successful_requests"],
                "request_errors": s["request_errors"],
            }
        )
    assert {(p["policy"], p["clients"]) for p in custom} == {
        (p, c) for p in COLORS for c in CLIENTS
    }
    return manifest, hardware, native, custom


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


def plot_pair(output, name, points, title, subtitle, caption, mode, hardware=()):
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.labelcolor": "#334155",
            "text.color": "#172b45",
        }
    )
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5))
    fig.subplots_adjust(left=0.075, right=0.98, bottom=0.24, top=0.73, wspace=0.26)
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
            ax.set_ylim(0, 14500 if mode == "custom" else 13000)
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
                20, color="#687b8f", linestyle=(0, (4, 3)), linewidth=1.2, zorder=1
            )
            ax.text(
                0.02,
                20,
                " 20 s SLO",
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
            if mode == "hardware":
                peak = next(p for p in selected if p["clients"] == 192)
                ax.scatter(
                    [192],
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
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper left",
        bbox_to_anchor=(0.069, 0.855),
        ncol=4,
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
    colors = [COLORS["kv"], "#5f8fbd", "#16836b", "#bf923e", COLORS["rr"]]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.5), sharey=True)
    fig.subplots_adjust(left=0.20, right=0.97, bottom=0.23, top=0.75, wspace=0.20)
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
            ax.axvline(20, color="#687b8f", linestyle=(0, (4, 3)), linewidth=1.2)
            ax.text(21.1, -0.48, "20 s SLO", fontsize=9, color="#53667a")
    fig.text(
        0.045,
        0.045,
        "Best measured: scale 3, credit 0.8, temperature 0, FCFS. Versus default KV: +14.1% throughput, −45.7% p95 TTFT.\nOne run per setting. The same variant was also tested at 96 clients; its knee has not been measured.",
        fontsize=9.2,
        color="#53667a",
        linespacing=1.6,
    )
    save_figure(fig, output, "flag-sweep")


def table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        + ["| " + " | ".join(str(v) for v in row) + " |" for row in rows]
    )


def report(hardware, native, custom, data_name):
    baseline = [p for p in hardware if p["policy"] in COLORS]
    get = {(p["policy"], p["clients"]): p for p in baseline}
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

    def best(policy):
        return max(
            (p for p in hardware if p["policy"] == policy and p["ttft_p95_s"] <= 20),
            key=lambda p: p["total_tok_s_gpu"],
        )

    k, r, tuned = best("kv"), best("rr"), best("kvs3c08")
    slo = table(
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
        ],
        [
            [
                f"[{p['label']}]({p['gcs_console']})",
                f"{p['total_tok_s_gpu']:,.0f}",
                f"{p['throughput_change_vs_default_kv_pct']:+.1f}%",
                f"{p['ttft_p95_s']:.2f}",
                f"{p['ttft_change_vs_default_kv_pct']:+.1f}%",
                f"{p['cache_pct']:.1f}%",
            ]
            for p in flags192
        ],
    )
    flags96 = {p["policy"]: p for p in flags if p["clients"] == 96}
    tuned96, default96 = flags96["kvs3c08"], flags96["kv"]
    comparison = []
    for p in sorted(native, key=lambda p: (p["policy"], p["clients"])):
        h = get[(p["policy"], p["clients"])]
        comparison.append(
            [
                p["policy"].upper(),
                p["clients"],
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
            "Real total tok/s/GPU",
            "V10 total tok/s/GPU",
            "Throughput error",
            "TTFT p95 sim / real (s)",
            "TTFT error",
        ],
        comparison,
    )
    old = {(p["policy"], p["clients"]): p for p in custom}
    old_table = table(
        [
            "Clients",
            "Custom KV total tok/s/GPU",
            "Custom RR total tok/s/GPU",
            "Custom KV / RR TTFT p95 (s)",
        ],
        [
            [
                c,
                f"{old[('kv', c)]['total_tok_s_gpu']:,.0f}",
                f"{old[('rr', c)]['total_tok_s_gpu']:,.0f}",
                f"{old[('kv', c)]['ttft_p95_s']:.2f} / {old[('rr', c)]['ttft_p95_s']:.2f}",
            ]
            for c in CLIENTS
        ],
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

Hardware collected **2026-09-16** · simulation artifacts **2026-09-17** · **24 GB300 GPUs, 6 × TP4/EP4 workers**

Default KV delivers **41.9% more total tokens/s/GPU than RR at 192 AgentX clients**, with TTFT p95 **11.66 versus 60.08 seconds**. Under the same **TTFT p95 ≤20 s** budget, the best measured default-KV point is **192 clients**, versus **96 clients** for RR: **1.57× throughput**. The highest-throughput sampled hardware point is 192 for both policies; the precise knees need finer sampling.

[Standalone HTML]({STEM}.html) · [all plotted data (CSV)]({STEM}.csv) · [data and provenance (JSON)]({STEM}.json) · [preserved inputs (ZIP)]({STEM}-inputs.zip)

## 1. Real data sweep and knee

### Setup and metric definitions

These are completed **hardware jobs**, using AIPerf 0.12.0's `inferencex-agentx-mvp` scenario and the public Weka 256K trace. Both default policies were measured at **48, 96, 192 and 384 clients**, with a **3,600-second profiling window**, 60-second grace and 1,200-second request timeout. AgentX concurrency counts live session trees, including their subagents; it does not equal requests simultaneously decoding.

All charts use AIPerf **input + output tokens/s divided by all 24 GPUs**, including cached input tokens. This is served token volume, not GPU compute throughput. TTFT is AIPerf's **p95 over successful profiling requests**, converted from milliseconds to seconds. Output-only throughput is shown separately in the SLO table.

The configured serving shape is six TP4/EP4 replicas, context 262,144, max running 16 per worker, 16,384-token prefill chunks, and 443,697 attention-KV pages at 64 tokens/page. The recipe installs Dynamo 1.4.2, whose SGLang dependency is 0.5.16. The initial container tag alone does not prove the installed package version; the benchmark's actual worker package inventory was not captured.

KV and RR used separate deployments (`n3u-agg-ns` and `n3u-agg-ns2`) with the same configured serving shape. Each baseline cell has one collected run; these are not repeated-trial estimates, and no confidence intervals are claimed. The busy-stream measurements in [AGG24_RESULTS.md](../AGG24_RESULTS.md) use a different concurrency definition and are excluded here.

![Real agg KV/RR: concurrency versus throughput and p95 TTFT]({STEM}-hardware.png)

[Hardware plot SVG]({STEM}-hardware.svg) · [PDF]({STEM}-hardware.pdf)

Stars mark **192, the highest-throughput sampled point**. Hollow markers at 384 identify the measured saturation region. The shaded 192–384 interval highlights the unsampled KV saturation transition; it is not a confidence band or an assertion that the two policies have identical knees.

### What the sweep establishes

- **Default KV:** 192 remains stable in the reported within-run check. At 384 throughput falls **14.5%**, and TTFT p95 rises from **11.66 to 119.44 s**. The original request-record analysis reports TTFT p50 rising from **37.9 to 99.6 s** between the first and last quarters at 384. Thus **192 is the last stable sampled point**, and the saturation transition is bracketed between **192 and 384**.
- **RR:** doubling clients from 96 to 192 buys only **10.8%** more throughput while TTFT p95 rises **12.56 → 60.08 s**. This indicates diminishing returns in **96–192**. At 384 throughput falls another **25.4%** and TTFT p95 reaches **494.98 s**. The highest sampled throughput is at **192**, but this already fails the 20-second latency budget.
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

### 2.2 Same SLO: TTFT p95 ≤20 seconds

For each policy, select the **highest-throughput measured cell that passes the same p95 threshold**. Concurrency is allowed to differ; topology and GPU count remain fixed. This is a run-level TTFT percentile constraint, not AIPerf's separate request-goodput metric and not a combined TTFT/ITL or error-rate SLO.

{slo}

Default KV 384 fails the threshold; RR 192 and 384 fail it. Therefore **default KV192 versus RR96 gives 1.57×**, or **57.3% more total throughput/GPU**, under this SLO. No interpolation supplies an unmeasured passing point. The RR latency-budget crossing lies somewhere between 96 and 192; the KV crossing lies between 192 and 384.

The tuned-KV row is a separate measured variant, using prefill-load scale 3, overlap credit 0.8 and temperature 0. It gives **1.79× RR throughput** under the same SLO, but has only been measured at 96 and 192 clients; its knee is unestablished. The flag sweep below provides the measurements behind this row.

### 2.3 Real KV-router flag sweep

**Yes: four additional real AgentX jobs were collected**—three KV variants at **192 clients**, plus the best measured variant at **96 clients**. The default-KV and RR comparisons reuse the baseline jobs above. All use the same 24-GPU serving shape and 3,600-second AgentX profiling configuration. These are measured hardware results; no new hardware jobs were launched to generate this report.

The settings below come from the preserved [runner recipe]({data_name}/hardware/agentx_runner.sh.txt), which installs Dynamo 1.4.2 and changes the frontend router arguments for each named variant. “Not explicitly set” means the recipe inherits the frontend's defaults; the table does not infer a numeric value from a variant name.

{settings_table}

The explicit flag names are `--router-prefill-load-scale`, `--router-kv-overlap-score-credit`, `--router-temperature` and `--router-queue-policy`. All four KV recipes use `--router-mode kv`. The RR reference uses `--router-mode round-robin`.

![Measured agg KV-router flags at 192 AgentX clients]({STEM}-flag-sweep.png)

[Flag-sweep SVG]({STEM}-flag-sweep.svg) · [PDF]({STEM}-flag-sweep.pdf) · [settings, metrics and deltas (CSV)]({STEM}-flag-sweep.csv)

{flags_table}

At **192 clients**, `kvs3c08` is the **best measured setting**: **11,012 total tok/s/GPU** and **6.33 s p95 TTFT**, versus default KV's **9,655** and **11.66 s**. That is **14.1% more throughput** and **45.7% lower p95 TTFT**. Output-only throughput also rises from **96.81 to 108.87 tok/s/GPU**. Relative to RR at the same 192 clients, this setting gives **1.62× total throughput** and **9.49× shorter p95 TTFT**.

The lower scale of 2 with the same 0.8 credit improves throughput by **6.1%** and p95 TTFT by **30.4%** versus default KV. Temperature 0.5 reduces throughput by **19.0%** and increases p95 TTFT by **90.0%**, to **22.15 s**, which misses the 20-second SLO. All five 192-client rows have **three exported request errors**; TTFT percentiles cover successful requests.

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

The higher cached-input share and lower latency are consistent with a better balance between cache reuse and queued work, but the aggregate summaries do not isolate that mechanism. The earlier fixed-rate custom model predicted the wrong direction for this tuning; neither simulation curve in section 3 includes a validated replay of these flag variants. The measured flag improvement must not be treated as proof that the current simulator models these knobs correctly. The [original tuning analysis](../AGENTX_AGG_RESULTS.md#v-kv-router-flag-sweep-at-the-192-client-comparison-point-measured) is retained as background; the tables here are regenerated from the full AIPerf summaries and keep the tuned points separate from the default-KV concurrency curve.

## 3. Simulation method, curves and knee selection

### 3.1 Native DynoSim V10: current completed calibration samples

The workspace contains a newer, separate **native DynoSim experiment**. Its path is:

```text
AIPerf 0.12.0 AgentX client, normal wall clock and streaming HTTP
  → Dynamo 1.4.2 frontend and KV / round-robin router
  → native SGLang Mocker/DynoSim worker scheduler
  → AIConfigurator 0.11.0 forward-pass timing
```

**V10 is a local candidate built on Dynamo 1.4.2 with eleven patches; it is not an unmodified NVIDIA release or a Dynamo release named v10.** The patches cover CPU execution, visible first-token handling, prefix/chunk admission, hybrid-state cache behavior and timing corrections. All four plotted native runs use one native core build and one shared engine configuration. The preserved [matrix]({data_name}/native-v10/matrix.json) records these checks; the [patch reconstruction record]({data_name}/native-v10/v10_patch_reconstruction.json) identifies the modified source.

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

Solid curves contain **only the completed native points at 192 and 384**. Faint dashed curves show hardware context. There are no native 48/96 points in this calibration set; the plot does not fill those positions with another model.

{native_table}

The recorded calibration gate is **±20% total throughput error at each of the four selected points**; all four pass. **TTFT was not an acceptance target**, and three points miss ±20% TTFT. The RR384 tail is underestimated by **38.4%**. These four workloads were used during model development; passing them is calibration evidence, not independent validation of arbitrary loads, policies or latency SLOs. The latest candidate narrows the earlier throughput gap, but does not establish TTFT fidelity.

### 3.2 Earlier custom Python model: complete four-point diagnostic sweep

The earlier eight-point comparison also uses actual AIPerf AgentX workload replay, but routes it through an accelerated in-process transport into the study's **custom Python serving model**, historically named `dynosim_agentx.py`. It does **not** invoke NVIDIA DynoSim. Its plotted cells all use a 28.4M-token attention-cache capacity; KV additionally includes projected active-prompt-block routing cost. Within each concurrency, the KV/RR pair shares the RR hardware client's configuration and benchmark ID.

That serving model uses fixed-rate serial prefill (**19,700 uncached tokens/s per worker**), immediate cache insertion and a static decode interval (**8.9 + 1.73 × outstanding requests ms**). Outstanding requests include those waiting for prefill. It omits faithful shared-GPU scheduling, running-limit admission and hybrid-state cache eligibility. It therefore cannot be used to select a hardware knee merely because its replay audit passes. Its [source snapshot and audit]({data_name}/custom-python/audit.json) are preserved separately from V10.

![Earlier custom Python agg KV/RR simulation]({STEM}-custom-simulation.png)

[Custom-model SVG]({STEM}-custom-simulation.svg) · [PDF]({STEM}-custom-simulation.pdf)

{old_table}

Both custom-model throughput curves still rise at 384: **the knee is not reached in that sampled range**. Labeling 384 as its knee would confuse the largest tested concurrency with a saturation point. At 384 the custom model overpredicts KV throughput by **47.4%** and RR by **111.5%**, and even incorrectly puts KV inside the 20-second SLO. These diagnostic results explain why the hardware and native-calibration sections must remain separate.

### 3.3 How a knee is selected

1. **Hold the workload and serving configuration fixed.** For hardware or one simulator build, sweep concurrency from low load upward. Compare the same AIPerf throughput definition and GPU denominator.
2. **Locate diminishing returns and saturation.** Examine throughput gains between adjacent samples, p95 TTFT, request errors, and within-run queue/TTFT progression. The highest-throughput sample is a candidate operating point, not proof of an exact continuous knee.
3. **Bracket the transition.** Hardware default KV is stable at 192 and saturated at 384; RR is already flattening over 96–192 and collapses by 384. Native V10 also declines from 192 to 384, but with no lower-concurrency native samples its exact knee is unresolved. The earlier custom model has no observed downturn through 384.
4. **Refine and validate.** Candidate follow-up concurrencies are **240/288/336 for default KV** and **120/144/168 for RR**, plus repeats of adjacent points. These are proposed refinement points, not completed jobs. Queue stationarity and failure behavior must accompany throughput before promoting a simulator-selected point to a hardware recommendation.
5. **Apply the SLO separately.** Among measured eligible cells, maximize throughput subject to the chosen threshold. A throughput knee and the best point under a latency budget are different selections: here RR192 is the sampled throughput peak, while RR96 is the 20-second-SLO choice.

### Reproduce this report

The [manifest]({data_name}/manifest.json) preserves exact source hashes and original locations. The input ZIP contains hardware summaries and the router-flag recipe, native configurations/build identity, the native matrix, calibration evidence, and the separate custom-model audit/source. The [original native experiment notes]({data_name}/native-v10/original-calibration-report.txt) preserve the detailed method and source locations as a text snapshot. Full hardware request records remain in the linked GCS artifacts. Temporary source paths are recorded for provenance; regeneration uses the preserved report inputs. The command below rebuilds the report; it does not rerun hardware jobs or native simulations.

From the study directory, with Python 3.12, Matplotlib 3.11.2 and markdown-it-py 4.2.0 available:

```bash
python scripts/gen_agg_kv_rr_report.py \\
  --data-dir reports/{data_name} \\
  --output-dir reports
```

The generator checks input hashes, all expected policy/concurrency pairs, the 3,600-second AgentX phase settings, GPU normalization, agreement between native summaries and the comparison matrix, and the shared native engine/build. It computes tables and plots from unrounded metrics and writes the [validation record]({STEM}-validation.json).
"""


def html_report(markdown, output):
    parser = MarkdownIt("commonmark").enable("table")
    body = parser.render(markdown)
    for name in ["hardware", "flag-sweep", "native-simulation", "custom-simulation"]:
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
            return f'download="{target.name}" href="data:{types[target.suffix]};base64,{encoded}"'
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
<nav><a href="#1-real-data-sweep-and-knee">1. Real sweep</a><a href="#2-kv-versus-rr-at-equal-configuration-and-equal-slo">2. Comparisons</a><a href="#2-3-real-kv-router-flag-sweep">KV flag sweep</a><a href="#3-simulation-method-curves-and-knee-selection">3. Simulation</a></nav>
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
    manifest, hardware, native, custom = load_inputs(args.data_dir)
    baseline = [p for p in hardware if p["policy"] in COLORS]
    plot_pair(
        output,
        "hardware",
        baseline,
        "Measured hardware · KV versus RR",
        "N3U · 24 GB300 GPUs · 6 × TP4/EP4 · AgentX · one-hour profiles",
        "★ 192 = highest-throughput sampled point for both policies; RR already flattens over 96–192.\nShading: KV transition over 192–384. Hollow markers: saturated runs. Intermediate points were not measured.",
        "hardware",
    )
    plot_pair(
        output,
        "native-simulation",
        native,
        "Native DynoSim V10 · throughput and latency",
        "Dynamo 1.4.2 + local V10 patches · AIC 0.11.0 · 24-GPU model · normal AIPerf HTTP replay",
        "Solid: V10 at 192 and 384 only. Faint dashed: hardware. Native 48/96 points are unavailable.\nThroughput passes the recorded ±20% gate; TTFT remains uncalibrated. The exact knee needs more samples.",
        "native",
        baseline,
    )
    plot_pair(
        output,
        "custom-simulation",
        custom,
        "Earlier custom Python model · full four-point sweep",
        "Actual AIPerf workload replay · custom fixed-rate serving · 28.4M-token cache per worker",
        "Both modeled throughput curves are still rising at 384: no knee is established within this range.\nThis custom model is not NVIDIA DynoSim; its missing saturation and optimistic latency limit hardware/SLO decisions.",
        "custom",
    )
    flag_points = flag_sweep_points(hardware)
    plot_flag_sweep(output, flag_points)
    all_points = hardware + native + custom
    fields = [
        "kind",
        "id",
        "policy",
        "clients",
        "total_tok_s_gpu",
        "output_tok_s_gpu",
        "ttft_p95_s",
        "itl_mean_ms",
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
        "slo_ttft_p95_s": 20,
        "source_manifest_sha256": digest(args.data_dir / "manifest.json"),
        "points": all_points,
        "flag_sweep": flag_points,
    }
    (output / f"{STEM}.json").write_text(json.dumps(exported, indent=2) + "\n")
    with zipfile.ZipFile(
        output / f"{STEM}-inputs.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for path in sorted(args.data_dir.rglob("*")):
            if path.is_file():
                archive.write(
                    path,
                    args.data_dir.name + "/" + str(path.relative_to(args.data_dir)),
                )
    markdown = report(hardware, native, custom, args.data_dir.name)
    (output / f"{STEM}.md").write_text(markdown)
    html_report(markdown, output)
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
        "custom_python_points": len(custom),
        "verified_input_files": len(manifest["files"]),
        "shared_native_core_and_engine": True,
        "native_matrix_matches_summaries": True,
        "native_ttft_calibrated": False,
        "slo_selection": {
            "default_kv_clients": 192,
            "rr_clients": 96,
            "tuned_kv_clients": 192,
        },
    }
    (output / f"{STEM}-validation.json").write_text(
        json.dumps(validation, indent=2) + "\n"
    )
    print(json.dumps(validation))
    print(output / f"{STEM}.html")


if __name__ == "__main__":
    main()
