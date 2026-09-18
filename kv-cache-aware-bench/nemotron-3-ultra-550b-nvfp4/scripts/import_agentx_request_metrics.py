#!/usr/bin/env python3
"""Preserve numeric AgentX request metrics from locally downloaded AIPerf exports.

The sources JSON is a list of {id, records, source}: records is a local JSONL
path; source is its original location. No requests are sent by this importer.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import io
import json
from collections import Counter
from pathlib import Path

METRICS = {
    "request_latency_ms": ("request_latency", "ms"),
    "output_sequence_length": ("output_sequence_length", "tokens"),
    "ttft_ms": ("time_to_first_token", "ms"),
}
FIELDS = ["source_line", "phase", "status", *METRICS]


def extract(entry, folder):
    source = Path(entry["records"])
    target = folder / f"{entry['id']}.csv.gz"
    counts = Counter()
    checksum = hashlib.sha256()
    # Fixed gzip timestamp and no filename keep the preserved input reproducible.
    with (
        source.open("rb") as incoming,
        target.open("wb") as outgoing,
        gzip.GzipFile(fileobj=outgoing, mode="wb", filename="", mtime=0) as zipped,
        io.TextIOWrapper(zipped, encoding="utf-8", newline="") as text,
    ):
        writer = csv.DictWriter(text, fieldnames=FIELDS, lineterminator="\n")
        writer.writeheader()
        for line_number, line in enumerate(incoming, 1):
            checksum.update(line)
            record = json.loads(line)
            metadata = record["metadata"]
            phase = metadata["benchmark_phase"]
            status = (
                "error"
                if record.get("error")
                else "cancelled"
                if metadata.get("was_cancelled")
                else "context_overflow_skip"
                if metadata.get("context_overflow_skip")
                else "success"
            )
            row = {"source_line": line_number, "phase": phase, "status": status}
            for column, (metric, unit) in METRICS.items():
                value = record.get("metrics", {}).get(metric)
                if value is not None:
                    assert value["unit"] == unit, (source, line_number, metric)
                    row[column] = value["value"]
            writer.writerow(row)
            counts[f"{phase}/{status}"] += 1
    return {
        "file": f"request-metrics/{target.name}",
        "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        "source": entry["source"],
        "source_sha256": checksum.hexdigest(),
        "source_bytes": source.stat().st_size,
        "rows_by_phase_and_status": dict(sorted(counts.items())),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    args = parser.parse_args()
    folder = args.data_dir / "request-metrics"
    folder.mkdir(parents=True, exist_ok=True)
    sources = json.loads(args.sources.read_text())
    assert len({entry["id"] for entry in sources}) == len(sources)
    provenance = {
        "schema_version": 1,
        "extraction": "Numeric columns only; all exported phases and statuses retained. source_line is 1-based in the original JSONL.",
        "runs": {entry["id"]: extract(entry, folder) for entry in sources},
    }
    path = folder / "manifest.json"
    path.write_text(json.dumps(provenance, indent=2) + "\n")
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["request_metrics_manifest"] = "request-metrics/manifest.json"
    for item in provenance["runs"].values():
        manifest["files"][item["file"]] = {
            "sha256": item["sha256"],
            "source": item["source"],
        }
    manifest["files"][manifest["request_metrics_manifest"]] = {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source": "Numeric projection of the original per-request exports; see this file for source hashes.",
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Preserved {len(sources)} per-request metric exports in {folder}")


if __name__ == "__main__":
    main()
