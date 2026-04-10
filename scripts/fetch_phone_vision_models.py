#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import urllib.request
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_MANIFEST = REPO_ROOT / "model-candidates" / "manifests" / "model_inventory.json"
REQUIRED_MODEL_PATHS = {
    "model-candidates/mediapipe/pose_landmarker_heavy.task",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the phone-vision model files required by CHEK EGO Miner.")
    parser.add_argument("--models-root", default=str(REPO_ROOT / "model-candidates"))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-path")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_required_models() -> list[dict[str, object]]:
    inventory = json.loads(MODEL_MANIFEST.read_text(encoding="utf-8"))
    downloaded = inventory.get("downloaded_models") or []
    return [
        item
        for item in downloaded
        if isinstance(item, dict) and str(item.get("path") or "") in REQUIRED_MODEL_PATHS
    ]


def download_model(source_url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(source_url, timeout=120) as response, destination.open("wb") as handle:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            handle.write(chunk)


def main() -> int:
    args = parse_args()
    models_root = Path(args.models_root).expanduser().resolve()
    repo_root = models_root.parent
    required_models = load_required_models()
    results: list[dict[str, object]] = []
    ok = True

    for item in required_models:
        relpath = str(item["path"])
        source_url = str(item["source_url"])
        expected_sha = str(item["sha256"])
        destination = repo_root / relpath
        status = "downloaded"

        if destination.is_file() and not args.force:
            if sha256_file(destination) == expected_sha:
                status = "already_present"
            else:
                status = "checksum_mismatch_replaced"
                download_model(source_url, destination)
        else:
            download_model(source_url, destination)

        actual_sha = sha256_file(destination)
        verified = actual_sha == expected_sha
        if not verified:
            ok = False
        results.append(
            {
                "path": relpath,
                "source_url": source_url,
                "destination": str(destination),
                "status": status,
                "sha256_expected": expected_sha,
                "sha256_actual": actual_sha,
                "verified": verified,
            }
        )

    payload = {
        "ok": ok,
        "models_root": str(models_root),
        "results": results,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print(encoded)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
