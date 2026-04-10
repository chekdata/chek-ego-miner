#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import urllib.parse
from pathlib import Path
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo_capture_policy_tool import export_redacted
from scripts.validate_capture_bundle import validate_bundle


SCRIPTS_DIR = REPO_ROOT / "scripts"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a basic synthetic capture -> download -> validation flow.")
    parser.add_argument("--edge-base-url", required=True)
    parser.add_argument("--edge-token", default="chek-ego-miner-local-token")
    parser.add_argument("--trip-id", required=True)
    parser.add_argument("--session-id", required=True)
    parser.add_argument("--duration-seconds", type=float, default=8.0)
    parser.add_argument("--interval-seconds", type=float, default=0.8)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--report-path")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def run_synthetic_feed(args: argparse.Namespace, report_path: Path) -> subprocess.CompletedProcess[str]:
    command = [
        sys.executable,
        str(SCRIPTS_DIR / "run_synthetic_phone_ingress_feed.py"),
        "--edge-base-url",
        args.edge_base_url,
        "--edge-token",
        args.edge_token,
        "--trip-id",
        args.trip_id,
        "--session-id",
        args.session_id,
        "--duration-seconds",
        str(args.duration_seconds),
        "--interval-seconds",
        str(args.interval_seconds),
        "--report-path",
        str(report_path),
    ]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def http_download(url: str, token: str, destination: Path) -> None:
    request = urllib_request.Request(url, headers={"Authorization": f"Bearer {token}"})
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib_request.urlopen(request, timeout=20) as response, destination.open("wb") as handle:
        handle.write(response.read())


def download_bundle_file(edge_base_url: str, edge_token: str, relpath: str, output_root: Path) -> str:
    normalized_relpath = relpath.strip().lstrip("/")
    destination = output_root / normalized_relpath
    encoded_relpath = urllib.parse.quote(normalized_relpath, safe="/")
    url = f"{edge_base_url.rstrip('/')}/live-preview/file/{encoded_relpath}"
    http_download(url, edge_token, destination)
    return normalized_relpath


def referenced_files(bundle: dict[str, object]) -> list[str]:
    relpaths = ["manifest.json", "demo_capture_bundle.json"]
    artifacts = bundle.get("artifacts") or {}
    if isinstance(artifacts, dict):
        for value in artifacts.values():
            if isinstance(value, str) and value.strip():
                relpath = value.strip()
                if Path(relpath).suffix:
                    relpaths.append(relpath)
    calibration_paths = bundle.get("calibration_snapshot_paths") or []
    if isinstance(calibration_paths, list):
        relpaths.extend(
            str(value).strip()
            for value in calibration_paths
            if isinstance(value, str) and str(value).strip()
        )
    return sorted(set(relpaths))


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    feed_report_path = output_dir / "synthetic_feed_report.json"
    bundle_path = output_dir / "demo_capture_bundle.json"

    feed_run = run_synthetic_feed(args, feed_report_path)
    feed_report = {}
    if feed_report_path.exists():
        feed_report = json.loads(feed_report_path.read_text(encoding="utf-8"))

    downloaded_files: list[str] = []
    download_errors: list[str] = []
    bundle = {}
    export_manifest: dict[str, object] = {}
    export_error = ""
    public_export_dir = output_dir / "public_download"
    public_bundle_path = public_export_dir / "demo_capture_bundle.json"
    try:
        download_bundle_file(args.edge_base_url, args.edge_token, "demo_capture_bundle.json", output_dir)
        bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
        for relpath in referenced_files(bundle):
            try:
                downloaded_files.append(
                    download_bundle_file(args.edge_base_url, args.edge_token, relpath, output_dir)
                )
            except Exception as exc:  # noqa: BLE001
                download_errors.append(f"{relpath}: {exc}")
    except Exception as exc:  # noqa: BLE001
        download_errors.append(f"demo_capture_bundle.json: {exc}")

    if bundle_path.is_file():
        try:
            export_manifest = export_redacted(
                bundle_path,
                REPO_ROOT / "configs" / "demo_capture_policy.json",
                public_export_dir,
                None,
            )
        except Exception as exc:  # noqa: BLE001
            export_error = str(exc)

    validation = (
        validate_bundle(public_bundle_path, tier="basic")
        if public_bundle_path.is_file()
        else {"ok": False, "missing": ["public_export_bundle_missing"]}
    )
    payload = {
        "ok": feed_run.returncode == 0 and not export_error and bool(validation.get("ok")),
        "edge_base_url": args.edge_base_url.rstrip("/"),
        "trip_id": args.trip_id,
        "session_id": args.session_id,
        "output_dir": str(output_dir),
        "public_download_dir": str(public_export_dir),
        "synthetic_feed": {
            "returncode": feed_run.returncode,
            "stdout": feed_run.stdout.strip(),
            "stderr": feed_run.stderr.strip(),
            "report": feed_report,
        },
        "downloaded_files": downloaded_files,
        "download_errors": download_errors,
        "public_export_manifest": export_manifest,
        "public_export_error": export_error,
        "validation": validation,
    }

    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print(encoded)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
