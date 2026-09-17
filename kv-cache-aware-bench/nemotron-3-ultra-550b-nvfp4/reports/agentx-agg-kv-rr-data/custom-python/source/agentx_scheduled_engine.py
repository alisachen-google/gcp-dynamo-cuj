"""Experimental shared-GPU serving model for actual AIPerf AgentX replay.

Workers admit a bounded number of requests. Chunked prefills and decode batches
share one GPU timeline. Decode advances at batch changes, arrivals and completions
instead of emitting one event per token. This is a fluid batch approximation,
not the SGLang scheduler or kernels. Cache entries appear after computation.
"""

from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from dataclasses import dataclass, field


class PrefixCache:
    def __init__(self, capacity_tokens, checkpoint_capacity=None, checkpoint_stride=64):
        self.capacity = capacity_tokens // 64
        self.cache = OrderedDict()
        self.checkpoints = OrderedDict()
        self.checkpoint_capacity = checkpoint_capacity
        self.checkpoint_stride = checkpoint_stride // 64
        self.peak_blocks = 0
        self.evicted_checkpoints = 0

    def match(self, hashes):
        count = 0
        for block in hashes:
            if block not in self.cache:
                break
            count += 1
        if self.checkpoint_capacity is not None:
            while count and hashes[count - 1] not in self.checkpoints:
                count -= 1
            if count:
                self.checkpoints.move_to_end(hashes[count - 1])
        for block in hashes[:count]:
            self.cache.move_to_end(block)
        return count * 64

    def insert(self, hashes, computed_tokens, reserved_states=0):
        blocks = min(len(hashes), computed_tokens // 64)
        for block in hashes[:blocks]:
            self.cache[block] = None
            self.cache.move_to_end(block)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        self.peak_blocks = max(self.peak_blocks, len(self.cache))
        if self.checkpoint_capacity is not None:
            end = blocks // self.checkpoint_stride * self.checkpoint_stride
            if end:
                self.checkpoints[hashes[end - 1]] = end
                self.checkpoints.move_to_end(hashes[end - 1])
            self.reserve(reserved_states)

    def reserve(self, reserved_states):
        if self.checkpoint_capacity is not None:
            available = max(0, self.checkpoint_capacity - reserved_states)
            while len(self.checkpoints) > available:
                self.checkpoints.popitem(last=False)
                self.evicted_checkpoints += 1


@dataclass(eq=False)
class Request:
    hashes: list[int]
    input_tokens: int
    output_tokens: int
    start_s: float
    worker: int
    first: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    done: asyncio.Future = field(
        default_factory=lambda: asyncio.get_running_loop().create_future()
    )
    admitted_s: float | None = None
    first_s: float | None = None
    end_s: float | None = None
    cached_tokens: int = 0
    computed_tokens: int = 0
    prefill_service_s: float = 0.0
    remaining_output: float = 0.0
    cancelled: bool = False
    active_at_admission: int = 0


class Worker:
    def __init__(self, index, config):
        self.index = index
        self.config = config
        self.cache = PrefixCache(
            config["cache_capacity_tokens"],
            config.get("checkpoint_capacity"),
            config.get("checkpoint_stride_tokens", 64),
        )
        self.pending = deque()
        self.running = []
        self.prefilling = None
        self.wake = asyncio.Event()
        self.task = None
        self.max_running = 0
        self.max_pending = 0
        self.prefill_gpu_s = 0.0
        self.decode_gpu_s = 0.0

    def submit(self, request):
        self.pending.append(request)
        self.max_pending = max(self.max_pending, len(self.pending))
        self.wake.set()
        if self.task is None or self.task.done():
            self.task = asyncio.create_task(self.run())

    def reserved_states(self):
        return self.config.get("state_slots_per_running_request", 0) * len(self.running)

    def complete(self, request, now):
        request.end_s = now
        self.running.remove(request)
        if not request.done.done():
            request.done.set_result(None)

    def cleanup(self):
        self.pending = deque(r for r in self.pending if not r.cancelled)
        self.running = [r for r in self.running if not r.cancelled]
        if self.prefilling is not None and self.prefilling.cancelled:
            self.prefilling = None

    async def run(self):
        loop = asyncio.get_running_loop()
        while True:
            self.cleanup()
            if (
                self.prefilling is None
                and self.pending
                and len(self.running) < self.config["max_running_requests"]
            ):
                request = self.pending.popleft()
                request.admitted_s = loop.time()
                request.cached_tokens = self.cache.match(request.hashes)
                # Even a completely cached prompt needs its last token evaluated.
                request.cached_tokens = min(
                    request.cached_tokens, (request.input_tokens - 1) // 64 * 64
                )
                request.computed_tokens = request.cached_tokens
                self.running.append(request)
                request.active_at_admission = len(self.running)
                self.max_running = max(self.max_running, len(self.running))
                self.cache.reserve(self.reserved_states())
                self.prefilling = request
            if self.prefilling is not None:
                request = self.prefilling
                count = min(
                    self.config["prefill_chunk_tokens"],
                    request.input_tokens - request.computed_tokens,
                )
                cost = max(
                    self.config.get("prefill_min_seconds", 0.005),
                    count / self.config["prefill_tokens_per_second"],
                )
                await asyncio.sleep(cost)
                self.prefill_gpu_s += cost
                request.prefill_service_s += cost
                if request.cancelled:
                    continue
                request.computed_tokens += count
                self.cache.insert(
                    request.hashes, request.computed_tokens, self.reserved_states()
                )
                if request.computed_tokens == request.input_tokens:
                    request.first_s = loop.time()
                    if not request.first.done():
                        request.first.set_result(None)
                    self.prefilling = None
                    request.remaining_output = max(0, request.output_tokens - 1)
                    if not request.remaining_output:
                        self.complete(request, loop.time())
                continue
            if not self.running:
                return
            batch = list(self.running)
            step = (
                self.config["decode_base_ms"]
                + self.config["decode_slope_ms"] * len(batch)
            ) / 1000
            duration = min(r.remaining_output for r in batch) * step
            start = loop.time()
            self.wake.clear()
            try:
                await asyncio.wait_for(self.wake.wait(), timeout=max(duration, 1e-8))
            except TimeoutError:
                pass
            elapsed = loop.time() - start
            self.decode_gpu_s += elapsed
            for request in batch:
                request.remaining_output = max(
                    0, request.remaining_output - elapsed / step
                )
            for request in list(self.running):
                if not request.cancelled and request.remaining_output < 1e-5:
                    self.complete(request, loop.time())


class ScheduledAggEngine:
    def __init__(self, workers, policy, config):
        if policy != "rr":
            raise ValueError(
                "The experimental scheduled model currently validates agg RR only"
            )
        required = [
            "cache_capacity_tokens",
            "max_running_requests",
            "prefill_chunk_tokens",
            "prefill_tokens_per_second",
            "decode_base_ms",
            "decode_slope_ms",
        ]
        for key in required:
            if config[key] <= 0:
                raise ValueError(f"{key} must be positive")
        if config.get("checkpoint_stride_tokens", 64) % 64:
            raise ValueError("Checkpoint stride must be a multiple of 64")
        self.config = config
        self.workers = [Worker(i, config) for i in range(workers)]
        self.rr = 0

    def submit(self, hashes, input_tokens, output_tokens, now):
        index = self.rr % len(self.workers)
        self.rr += 1
        request = Request(hashes, input_tokens, output_tokens, now, index)
        self.workers[index].submit(request)
        return request

    def cancel(self, request):
        request.cancelled = True
        request.first.cancel()
        request.done.cancel()
        self.workers[request.worker].wake.set()

    def statistics(self):
        return [
            {
                "worker": w.index,
                "max_running": w.max_running,
                "max_pending": w.max_pending,
                "prefill_gpu_seconds": w.prefill_gpu_s,
                "decode_gpu_seconds": w.decode_gpu_s,
                "max_cache_blocks": w.cache.peak_blocks,
                "checkpoint_evictions": w.cache.evicted_checkpoints,
                "remaining_pending": len(w.pending),
                "remaining_running": len(w.running),
            }
            for w in self.workers
        ]
