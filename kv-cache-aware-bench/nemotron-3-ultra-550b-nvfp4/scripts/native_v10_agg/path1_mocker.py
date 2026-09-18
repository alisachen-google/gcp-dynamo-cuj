#!/usr/bin/env python3
"""Launch NVIDIA live Mocker, preserving SGLang's advertised context limit.

Dynamo 1.4.2's CLI accepts max_model_len only for vLLM and otherwise advertises
zero. This startup-only adapter sets ModelRuntimeConfig.context_length, which is
the same metadata field the real SGLang worker publishes. It changes no scheduler,
cache, latency prediction, request handler or wall clock.
"""

import os


def main():
    from dynamo.mocker import main as upstream

    original = upstream.build_runtime_config

    def with_context_limit(engine_args):
        block_size, runtime_config = original(engine_args)
        runtime_config.context_length = int(os.environ["PATH1_CONTEXT_LENGTH"])
        return block_size, runtime_config

    upstream.build_runtime_config = with_context_limit
    upstream.main()


if __name__ == "__main__":
    main()
