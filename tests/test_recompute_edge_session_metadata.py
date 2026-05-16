from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from recompute_edge_session_metadata import (  # noqa: E402
    build_checks,
    build_line_counters,
    local_quality_core_ids,
    runtime_flags_for_session,
)
from materialize_session_media import (  # noqa: E402
    set_upload_manifest_media_shortcuts,
    upsert_upload_manifest_artifact,
)


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

    def test_raw_depth_accepts_chunk_media_index(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp)
            depth_media = session_root / "raw" / "iphone" / "depth" / "media_index.jsonl"
            depth_media.parent.mkdir(parents=True)
            depth_media.write_text('{"type":"media_chunk_index","media_track":"depth"}\n')

            checks = build_checks(session_root, build_line_counters(session_root))
            depth_check = next(item for item in checks if item["id"] == "raw_depth_present")

        self.assertTrue(depth_check["ok"])
        self.assertIn("raw/iphone/depth/media_index.jsonl 行数=1", depth_check["detail"])

    def test_ego_profile_does_not_warn_for_missing_robot_state(self) -> None:
        import tempfile

        manifest = {
            "session_context": {
                "runtime_profile": "capture_plus_facts",
                "runtime_flags": {
                    "phone_ingest_enabled": True,
                    "fusion_enabled": True,
                    "control_enabled": False,
                },
            }
        }
        runtime_flags = runtime_flags_for_session(manifest, {})

        with tempfile.TemporaryDirectory() as tmp:
            session_root = Path(tmp)
            checks = build_checks(session_root, build_line_counters(session_root), runtime_flags)
            robot_check = next(item for item in checks if item["id"] == "robot_state_present")

        self.assertTrue(robot_check["ok"])
        self.assertIn("control disabled by runtime profile", robot_check["detail"])

    def test_materialized_media_artifact_upsert_is_idempotent(self) -> None:
        upload_manifest = {"artifacts": []}

        upsert_upload_manifest_artifact(
            upload_manifest,
            artifact_id="iphone_main_merged_video",
            relpath="derived/media/iphone_main_merged.mp4",
            category="preview_derivative",
            exists=True,
            byte_size=123,
        )
        upsert_upload_manifest_artifact(
            upload_manifest,
            artifact_id="iphone_main_merged_video",
            relpath="derived/media/iphone_main_merged.mp4",
            category="preview_derivative",
            exists=True,
            byte_size=456,
        )

        artifacts = upload_manifest["artifacts"]
        self.assertEqual(len(artifacts), 1)
        self.assertEqual(artifacts[0]["byte_size"], 456)
        self.assertEqual(artifacts[0]["upload_state"], "ready")

    def test_materialized_media_shortcuts_follow_ready_artifacts(self) -> None:
        upload_manifest = {
            "artifacts": [
                {
                    "id": "derived_media_manifest",
                    "relpath": "derived/media/media_manifest.json",
                    "exists": True,
                },
                {
                    "id": "iphone_main_merged_video",
                    "relpath": "derived/media/iphone_main_merged.mp4",
                    "exists": True,
                },
                {
                    "id": "iphone_fisheye_merged_video",
                    "relpath": "derived/media/iphone_fisheye_merged.mp4",
                    "exists": False,
                },
            ],
            "iphone_fisheye_merged_video": "stale.mp4",
        }

        set_upload_manifest_media_shortcuts(upload_manifest)

        self.assertEqual(upload_manifest["derived_media_manifest"], "derived/media/media_manifest.json")
        self.assertEqual(upload_manifest["iphone_main_merged_video"], "derived/media/iphone_main_merged.mp4")
        self.assertNotIn("iphone_fisheye_merged_video", upload_manifest)


if __name__ == "__main__":
    unittest.main()
