"""Guard the user-facing E2E SLO definition and profiling exclusions."""

import unittest

from analyze import percentile, valid_request


def record(latency_ms=10000, output_tokens=100):
    return {
        "metadata": {"benchmark_phase": "profiling"},
        "metrics": {
            "usage_completion_tokens": {"value": output_tokens, "unit": "tokens"},
            "request_latency": {"value": latency_ms, "unit": "ms"},
            "time_to_first_token": {"value": 2000, "unit": "ms"},
        },
    }


class E2EMetrics(unittest.TestCase):
    def test_normalize_each_request_before_percentile_and_inversion(self):
        rows = [
            valid_request(record(10000, 1000))[0],
            valid_request(record(100000, 100))[0],
        ]
        normalized = [row["normalized_e2e_s_per_token"] for row in rows]
        self.assertAlmostEqual(percentile(normalized, 0.9), 0.901)
        self.assertAlmostEqual(1 / percentile(normalized, 0.9), 1 / 0.901)

    def test_exclude_warmup_failure_cancelled_and_zero_output(self):
        warmup = record()
        warmup["metadata"]["benchmark_phase"] = "warmup"
        failure = record()
        failure["error"] = {"message": "overloaded"}
        cancelled = record()
        cancelled["metadata"]["was_cancelled"] = True
        for value in [warmup, failure, cancelled, record(output_tokens=0)]:
            self.assertIsNone(valid_request(value)[0])

    def test_no_silent_unit_conversion(self):
        value = record()
        value["metrics"]["request_latency"]["unit"] = "s"
        with self.assertRaises(ValueError):
            valid_request(value)


if __name__ == "__main__":
    unittest.main()
