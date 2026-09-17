"""Routing regressions that affect disagg AgentX topology predictions."""

import unittest

from dynosim_agentx import Engine


class DecodeRoutingTest(unittest.TestCase):
    def test_unequal_prompts_balance_memory_instead_of_request_count(self):
        engine = Engine(1, 2, "kv", decode_policy="active_blocks")
        self.assertEqual(engine.serve(list(range(100)), 100, 0)[3], 0)
        self.assertEqual(engine.serve([100], 100, 0)[3], 1)
        # Each worker has one request, but worker 0 has 100x the footprint.
        self.assertEqual(engine.serve([101], 100, 0)[3], 1)

    def test_shared_active_prefix_survives_one_request_finishing(self):
        engine = Engine(1, 2, "kv", decode_policy="active_blocks")
        prompt = list(range(100))
        first = engine.serve(prompt, 100, 0)[3]
        engine.serve([100], 100, 0)
        second = engine.serve(prompt, 100, 0)[3]
        self.assertEqual((first, second), (0, 0))
        engine.release(first, prompt)
        self.assertEqual(len(engine.active_prompt_blocks[0]), 100)
        engine.release(second, prompt)
        self.assertFalse(engine.active_prompt_blocks[0])
        self.assertEqual(engine.D[0], 0)

    def test_rr_cycles_independently_on_unequal_tier_sizes(self):
        engine = Engine(3, 2, "rr", decode_policy="round_robin")
        assignments = []
        for index in range(6):
            decode = engine.serve([index], 1, 0)[3]
            assignments.append((engine.last_prefill_worker, decode))
            engine.release(decode, [index])
        self.assertEqual(assignments, [(0, 0), (1, 1), (2, 0), (0, 1), (1, 0), (2, 1)])
        self.assertEqual(engine.D, [0, 0])


if __name__ == "__main__":
    unittest.main()
