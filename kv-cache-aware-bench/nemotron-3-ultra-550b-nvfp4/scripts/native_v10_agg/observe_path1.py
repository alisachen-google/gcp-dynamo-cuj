#!/usr/bin/env python3
"""Record native forward-pass events and worker metrics without sending requests."""

import argparse
import asyncio
import gzip
import json
import os
from pathlib import Path
import threading
import time
import urllib.request


async def observe(run_dir):
    provenance = json.loads((run_dir / "provenance.json").read_text())
    os.environ.update(provenance["env_overrides"])
    from dynamo.llm import FpmEventSubscriber
    from dynamo.runtime import DistributedRuntime
    import msgspec

    runtime = DistributedRuntime(asyncio.get_running_loop(), "etcd", "nats", event_plane="zmq")
    endpoint = runtime.endpoint(provenance.get(
        "fpm_endpoint", os.environ["DYN_NAMESPACE"] + ".backend.generate"))
    subscriber = FpmEventSubscriber(endpoint)
    stopping = threading.Event()

    def events():
        with gzip.open(run_dir / "forward-passes.jsonl.gz", "wt", compresslevel=1) as stream:
            count = 0
            while not stopping.is_set():
                payload = subscriber.recv()
                if payload is None:
                    break
                value = msgspec.msgpack.decode(bytes(payload))
                stream.write(json.dumps(dict(received_unix=time.time(), fpm=value)) + "\n")
                count += 1
                if count % 100 == 0:
                    stream.flush()
            print(f"Recorded {count} forward-pass events", flush=True)

    thread = threading.Thread(target=events, daemon=True)
    thread.start()
    ports = [provenance["worker_metrics_port"] + i
             for i, name in enumerate(n for n in provenance["commands"] if n.startswith("worker-"))]

    def metrics(port):
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as response:
                return dict(port=port, text=response.read().decode())
        except OSError as error:
            return dict(port=port, error=str(error))

    try:
        with gzip.open(run_dir / "worker-metrics.jsonl.gz", "wt", compresslevel=1) as stream:
            while True:
                status = json.loads((run_dir / "status.json").read_text())
                if status["state"] in ("completed", "failed"):
                    break
                observations = await asyncio.gather(*(asyncio.to_thread(metrics, p) for p in ports))
                stream.write(json.dumps(dict(time=time.time(), workers=observations)) + "\n")
                stream.flush()
                await asyncio.sleep(5)
    finally:
        stopping.set()
        subscriber.shutdown()
        thread.join(timeout=5)
        runtime.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    asyncio.run(observe(parser.parse_args().run_dir))
