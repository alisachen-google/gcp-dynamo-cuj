#!/usr/bin/env python3
"""Normal AIPerf CLI with the hardware benchmark ID restored for cache-bust parity.

Run with the AIPerf 0.12.0 interpreter. Dataset loading, multiprocessing, clocks,
AgentX scheduling, HTTP transport, token accounting and metric export are upstream.
No serving model is implemented here.
"""

import os


def main():
    import aiperf.cli_runner as runner
    from aiperf.cli import app

    saved_id = os.environ["PATH1_BENCHMARK_ID"]
    original = runner._make_benchmark_run

    def with_saved_id(config, **kwargs):
        if kwargs.get("benchmark_id") not in (None, saved_id):
            raise ValueError("Conflicting benchmark ID")
        kwargs["benchmark_id"] = saved_id
        return original(config, **kwargs)

    runner._make_benchmark_run = with_saved_id
    app()


if __name__ == "__main__":
    main()
