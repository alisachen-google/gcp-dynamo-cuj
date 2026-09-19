#!/usr/bin/env python3
"""Losslessly archive completed raw exports after full round-trip verification."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import zstandard as zstd


def archive(run):
    status = json.loads((run / "status.json").read_text())
    if status["state"] != "completed" or time.time() - status["updated_unix"] < 30:
        raise ValueError(f"Run is not complete and settled: {run}")
    manifest = run / "lossless-archives.json"
    records = json.loads(manifest.read_text()) if manifest.exists() else []
    params = zstd.ZstdCompressionParameters.from_level(
        6, window_log=27, enable_ldm=True, threads=2)
    for name in ["profile_export_raw.jsonl", "server_metrics_export.json"]:
        source = run / "artifacts" / name
        if not source.exists():
            continue
        destination = source.with_name(source.name + ".zst")
        partial = destination.with_name(destination.name + ".partial")
        if destination.exists() or partial.exists():
            raise FileExistsError(destination)
        before = source.stat()
        expected = hashlib.sha256()
        with source.open("rb") as src, partial.open("xb") as out:
            with zstd.ZstdCompressor(compression_params=params).stream_writer(out) as dst:
                while chunk := src.read(4 * 1024 * 1024):
                    expected.update(chunk)
                    dst.write(chunk)
        actual = hashlib.sha256()
        recovered_size = 0
        with partial.open("rb") as src, zstd.ZstdDecompressor().stream_reader(src) as stream:
            while chunk := stream.read(4 * 1024 * 1024):
                actual.update(chunk)
                recovered_size += len(chunk)
        after = source.stat()
        if (actual.digest() != expected.digest() or recovered_size != before.st_size
                or (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns)):
            raise RuntimeError(f"Archive verification failed: {source}")
        partial.rename(destination)
        records.append(dict(original_path=str(source), archive_path=str(destination),
                            original_bytes=before.st_size, compressed_bytes=destination.stat().st_size,
                            uncompressed_sha256=actual.hexdigest(), bytes_verified=True,
                            compression="zstd level6, long-distance window 128MiB, 2 threads",
                            completed_unix=time.time()))
        temporary = manifest.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(records, indent=2) + "\n")
        temporary.replace(manifest)
        source.unlink()
    return records


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    print(json.dumps(archive(args.run_dir)), flush=True)
