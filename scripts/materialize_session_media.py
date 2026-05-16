#!/usr/bin/env python3
"""Materialize merged review videos from per-chunk iPhone media.

The live recorder stores raw phone media as short chunk files plus
`media_index.jsonl` because upload/ACK accounting is chunk-oriented. This
script creates human-reviewable merged MP4 derivatives under
`derived/media/` and records them back into the upload manifest.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


TRACKS = {
    "iphone_main": {
        "index": "raw/iphone/wide/media_index.jsonl",
        "media_track": "main",
        "output": "derived/media/iphone_main_merged.mp4",
    },
    "iphone_fisheye": {
        "index": "raw/iphone/fisheye/media_index.jsonl",
        "media_track": "fisheye",
        "output": "derived/media/iphone_fisheye_merged.mp4",
    },
}


def now_unix_ms() -> int:
    return int(time.time() * 1000)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_pretty(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def read_media_records(session_root: Path, relpath: str, media_track: str) -> list[dict[str, Any]]:
    index_path = session_root / relpath
    if not index_path.exists():
        return []
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    for line in index_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if item.get("media_track") != media_track or item.get("file_type") != "video":
            continue
        file_relpath = str(item.get("file_relpath") or "").strip()
        if not file_relpath or file_relpath in seen:
            continue
        file_path = safe_session_path(session_root, file_relpath)
        if file_path is None or not file_path.exists() or file_path.stat().st_size <= 0:
            continue
        seen.add(file_relpath)
        records.append(item)
    records.sort(key=lambda item: int(item.get("chunk_index") or 0))
    return records


def safe_session_path(session_root: Path, relpath: str) -> Path | None:
    candidate = (session_root / relpath).resolve()
    root = session_root.resolve()
    try:
        candidate.relative_to(root)
    except ValueError:
        return None
    return candidate


def ffmpeg_concat(records: list[dict[str, Any]], session_root: Path, output: Path, ffmpeg: str) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chek-media-concat-") as tmp_dir:
        concat_list = Path(tmp_dir) / "concat.txt"
        concat_list.write_text(
            "".join(
                f"file '{safe_session_path(session_root, str(item['file_relpath'])).as_posix()}'\n"
                for item in records
            ),
            encoding="utf-8",
        )
        copy_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-movflags",
            "+faststart",
            str(output),
        ]
        copied = subprocess.run(copy_cmd, capture_output=True, text=True, check=False)
        if copied.returncode == 0 and output.exists() and output.stat().st_size > 0:
            return
        transcode_cmd = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_list),
            "-map",
            "0:v:0",
            "-vf",
            "format=yuv420p",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "23",
            "-movflags",
            "+faststart",
            str(output),
        ]
        transcoded = subprocess.run(transcode_cmd, capture_output=True, text=True, check=False)
        if transcoded.returncode != 0:
            raise RuntimeError(
                "ffmpeg concat failed: "
                f"copy={copied.stderr.strip()} transcode={transcoded.stderr.strip()}"
            )


def upsert_upload_manifest_artifact(
    upload_manifest: dict[str, Any],
    *,
    artifact_id: str,
    relpath: str,
    category: str,
    exists: bool,
    byte_size: int,
) -> None:
    artifacts = upload_manifest.setdefault("artifacts", [])
    existing = next((item for item in artifacts if item.get("id") == artifact_id), None)
    item = existing if isinstance(existing, dict) else {}
    item.update(
        {
            "id": artifact_id,
            "relpath": relpath,
            "kind": "file",
            "category": category,
            "required": False,
            "exists": exists,
            "byte_size": byte_size,
            "line_count": None,
            "residency": "cloud_mirrored",
            "upload_state": "ready" if exists else "pending",
        }
    )
    if existing is None:
        artifacts.append(item)


def set_upload_manifest_media_shortcuts(upload_manifest: dict[str, Any]) -> None:
    artifacts = upload_manifest.get("artifacts") or []
    artifact_by_id = {
        str(item.get("id")): item
        for item in artifacts
        if isinstance(item, dict)
    }
    shortcut_keys = {
        "derived_media_manifest": "derived_media_manifest",
        "iphone_main_merged_video": "iphone_main_merged_video",
        "iphone_fisheye_merged_video": "iphone_fisheye_merged_video",
    }
    for key, artifact_id in shortcut_keys.items():
        artifact = artifact_by_id.get(artifact_id)
        if artifact and artifact.get("exists") and artifact.get("relpath"):
            upload_manifest[key] = artifact["relpath"]
        else:
            upload_manifest.pop(key, None)


def materialize(session_root: Path, ffmpeg: str, tracks: list[str]) -> dict[str, Any]:
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        raise RuntimeError(f"ffmpeg not found: {ffmpeg}")
    session_root = session_root.resolve()
    manifest_path = session_root / "derived" / "media" / "media_manifest.json"
    materialized: list[dict[str, Any]] = []
    for track_id in tracks:
        spec = TRACKS[track_id]
        records = read_media_records(session_root, spec["index"], spec["media_track"])
        output_relpath = spec["output"]
        output_path = session_root / output_relpath
        if not records:
            materialized.append(
                {
                    "id": track_id,
                    "status": "missing_source_chunks",
                    "source_index": spec["index"],
                    "output": output_relpath,
                    "chunk_count": 0,
                    "frame_count": 0,
                }
            )
            continue
        ffmpeg_concat(records, session_root, output_path, ffmpeg)
        materialized.append(
            {
                "id": track_id,
                "status": "ready",
                "source_index": spec["index"],
                "output": output_relpath,
                "chunk_count": len(records),
                "frame_count": sum(int(item.get("frame_count") or 0) for item in records),
                "byte_size": output_path.stat().st_size,
                "first_source_time_ns": records[0].get("source_start_time_ns"),
                "last_source_time_ns": records[-1].get("source_end_time_ns"),
            }
        )

    media_manifest = {
        "type": "derived_media_manifest",
        "schema_version": "1.0.0",
        "generated_unix_ms": now_unix_ms(),
        "session_id": session_root.name,
        "tracks": materialized,
    }
    write_json_pretty(manifest_path, media_manifest)

    upload_manifest_path = session_root / "upload" / "upload_manifest.json"
    upload_manifest = load_json(upload_manifest_path)
    if upload_manifest:
        upsert_upload_manifest_artifact(
            upload_manifest,
            artifact_id="derived_media_manifest",
            relpath="derived/media/media_manifest.json",
            category="preview_derivative",
            exists=True,
            byte_size=manifest_path.stat().st_size,
        )
        for item in materialized:
            output_relpath = str(item.get("output") or "")
            output_path = session_root / output_relpath
            upsert_upload_manifest_artifact(
                upload_manifest,
                artifact_id=f"{item['id']}_merged_video",
                relpath=output_relpath,
                category="preview_derivative",
                exists=output_path.exists() and output_path.stat().st_size > 0,
                byte_size=output_path.stat().st_size if output_path.exists() else 0,
            )
        artifacts = upload_manifest.get("artifacts") or []
        set_upload_manifest_media_shortcuts(upload_manifest)
        upload_manifest["generated_unix_ms"] = now_unix_ms()
        upload_manifest["artifact_count"] = len(artifacts)
        upload_manifest["ready_artifact_count"] = sum(1 for item in artifacts if item.get("exists"))
        write_json_pretty(upload_manifest_path, upload_manifest)
    return media_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--session-root", required=True, type=Path)
    parser.add_argument("--ffmpeg", default="ffmpeg")
    parser.add_argument(
        "--track",
        action="append",
        choices=sorted(TRACKS),
        help="Track to materialize; defaults to all known iPhone video tracks.",
    )
    args = parser.parse_args()
    result = materialize(args.session_root, args.ffmpeg, args.track or sorted(TRACKS))
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
