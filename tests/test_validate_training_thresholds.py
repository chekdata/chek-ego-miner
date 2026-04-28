from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from validate_training_thresholds import summarize_time_sync  # noqa: E402


class TimeSyncSummaryTests(unittest.TestCase):
    def test_gate_uses_accepted_for_mapping_samples_and_keeps_raw_diagnostics(self) -> None:
        threshold = {
            "minimum_source_kinds": {"pro": 1},
            "min_accepted_for_mapping_ratio": 0.8,
            "max_rtt_ms": 20,
            "per_source_rtt_ok_ms": {"stereo_pair": 20},
            "max_clock_offset_span_ms": 25,
        }
        rows = [
            {
                "source_kind": "stereo_pair",
                "accepted": True,
                "accepted_for_mapping": True,
                "rtt_ns": 6_000_000,
                "clock_offset_ns": 100_000_000,
            },
            {
                "source_kind": "stereo_pair",
                "accepted": True,
                "accepted_for_mapping": True,
                "rtt_ns": 8_000_000,
                "clock_offset_ns": 103_000_000,
            },
            {
                "source_kind": "stereo_pair",
                "accepted": False,
                "accepted_for_mapping": False,
                "mapping_rejected_reason": "rtt_budget_exceeded",
                "rtt_ns": 60_000_000,
                "clock_offset_ns": 160_000_000,
            },
        ]

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            sync_dir = root / "sync"
            sync_dir.mkdir()
            (sync_dir / "time_sync_samples.jsonl").write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )

            summary = summarize_time_sync(root, threshold, "pro")

        checks = {item["id"]: item for item in summary["checks"]}
        self.assertTrue(checks["time_sync_rtt_p95_within_budget"]["ok"])
        self.assertTrue(checks["time_sync_offset_span_within_budget"]["ok"])
        self.assertEqual(summary["accepted_for_mapping_count"], 2)

        stereo = summary["by_source_kind"]["stereo_pair"]
        self.assertEqual(stereo["sample_count"], 3)
        self.assertEqual(stereo["accepted_for_mapping_count"], 2)
        self.assertEqual(stereo["mapping_rejected_count"], 1)
        self.assertEqual(stereo["mapping_rejected_reasons"], {"rtt_budget_exceeded": 1})
        self.assertLessEqual(stereo["rtt_ms_p95"], 20)
        self.assertGreater(stereo["raw_rtt_ms_p95"], 20)
        self.assertLessEqual(stereo["offset_span_ms"], 25)
        self.assertGreater(stereo["raw_offset_span_ms"], 25)


if __name__ == "__main__":
    unittest.main()
