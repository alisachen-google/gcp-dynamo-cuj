"""Regression checks for the canonical AgentX interactivity definition and cohort."""

import csv
import gzip
import tempfile
import unittest
from pathlib import Path

from collect_agg_v10_holdouts import engine_sha
from gen_agg_kv_rr_report import normalized_interactivity, request_metrics


class InteractivityTests(unittest.TestCase):
    def test_engine_identity_uses_configuration_not_json_formatting(self):
        with tempfile.TemporaryDirectory() as directory:
            a, b = Path(directory) / "a.json", Path(directory) / "b.json"
            a.write_text('{"capacity": 100, "scale": 3}')
            b.write_text('{\n  "scale": 3,\n  "capacity": 100\n}\n')
            self.assertEqual(engine_sha(a), engine_sha(b))
            b.write_text('{"scale": 3, "capacity": 101}')
            self.assertNotEqual(engine_sha(a), engine_sha(b))

    def test_invert_after_percentile(self):
        # Times per token are 0.1 and 0.3 seconds. Their linear P90 is 0.28;
        # the P10 of the reciprocal rates would instead be 4 tokens/s.
        self.assertAlmostEqual(
            normalized_interactivity([1000, 3000], [10, 10]), 1 / 0.28
        )
        self.assertNotAlmostEqual(normalized_interactivity([1000, 3000], [10, 10]), 4)

    def test_twenty_token_boundary_includes_e2e_and_output_length(self):
        self.assertEqual(normalized_interactivity([50, 100], [1, 2]), 20)
        self.assertLess(normalized_interactivity([50.1, 100.2], [1, 2]), 20)

    def test_only_valid_successful_profiling_requests_enter_interactivity(self):
        fields = [
            "source_line",
            "phase",
            "status",
            "request_latency_ms",
            "output_sequence_length",
            "ttft_ms",
        ]
        rows = [
            [1, "warmup", "success", 100000, 1, 100000],
            [2, "profiling", "success", 1000, 10, 1000],
            [3, "profiling", "success", 3000, 10, 3000],
            [4, "profiling", "success", 4000, 0, 4000],
            [5, "profiling", "error", "", "", ""],
            [6, "profiling", "cancelled", "", "", ""],
        ]
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            with gzip.open(folder / "records.csv.gz", "wt", newline="") as stream:
                writer = csv.writer(stream)
                writer.writerow(fields)
                writer.writerows(rows)
            provenance = {
                "runs": {
                    "example": {
                        "file": "records.csv.gz",
                        "source_sha256": "test-fixture",
                        "rows_by_phase_and_status": {
                            "warmup/success": 1,
                            "profiling/success": 3,
                            "profiling/error": 1,
                            "profiling/cancelled": 1,
                        },
                    }
                }
            }
            point = {
                "id": "example",
                "successful_requests": 3,
                "request_errors": 1,
                "ttft_p95_s": 3.9,
            }
            result = request_metrics(folder, provenance, point)
        self.assertEqual(result["interactivity_valid_requests"], 2)
        self.assertEqual(result["interactivity_excluded_successful_requests"], 1)
        self.assertAlmostEqual(result["e2e_normalized_interactivity_p90_tps"], 1 / 0.28)
        self.assertTrue(result["ttft_slo_pass"])
        self.assertFalse(result["interactivity_slo_pass"])
        self.assertFalse(result["combined_slo_pass"])


if __name__ == "__main__":
    unittest.main()
