from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from recompute_edge_session_metadata import local_quality_core_ids  # noqa: E402


class LocalQualityCoreIdsTests(unittest.TestCase):
    def test_uses_process_default_profile_when_session_profile_is_missing(self) -> None:
        with patch.dict(os.environ, {"EDGE_RUNTIME_PROFILE": "capture_plus_vlm"}, clear=False):
            core_ids = local_quality_core_ids({}, {})

        self.assertEqual(
            core_ids,
            {"capture_pose_present", "iphone_calibration_present", "time_sync_present"},
        )

    def test_explicit_ego_session_profile_does_not_require_teleop_artifacts(self) -> None:
        manifest = {
            "session_context": {
                "runtime_profile": "capture_plus_vlm",
            }
        }

        with patch.dict(os.environ, {"EDGE_RUNTIME_PROFILE": "teleop_fullstack"}, clear=False):
            core_ids = local_quality_core_ids(manifest, {})

        self.assertNotIn("human_demo_pose_present", core_ids)
        self.assertNotIn("teleop_frame_present", core_ids)
        self.assertIn("time_sync_present", core_ids)

    def test_teleop_profile_keeps_control_artifacts_strict(self) -> None:
        manifest = {
            "session_context": {
                "runtime_profile": "teleop_fullstack",
            }
        }

        core_ids = local_quality_core_ids(manifest, {})

        self.assertIn("human_demo_pose_present", core_ids)
        self.assertIn("teleop_frame_present", core_ids)


if __name__ == "__main__":
    unittest.main()
