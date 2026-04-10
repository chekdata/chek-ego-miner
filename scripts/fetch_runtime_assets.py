#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import tarfile
import tempfile
from pathlib import Path
from urllib import request as urllib_request


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REPO = "hongzexin/chek-ego-miner"
ASSET_NAMES = {
    ("Linux", "x86_64"): "chek-ego-miner-runtime-linux-x86_64.tar.gz",
}


def detect_asset_name() -> str:
    key = (platform.system(), platform.machine())
    try:
        return ASSET_NAMES[key]
    except KeyError as exc:  # pragma: no cover - host dependent
        supported = ", ".join(f"{system}/{machine}" for system, machine in sorted(ASSET_NAMES))
        raise SystemExit(
            f"no published runtime asset mapping for {platform.system()}/{platform.machine()}; "
            f"supported mappings: {supported}"
        ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download published runtime assets for CHEK EGO Miner.")
    parser.add_argument("--repo", default=DEFAULT_REPO, help="GitHub repo in owner/name form.")
    parser.add_argument("--tag", default="", help="Release tag. Empty means latest release.")
    parser.add_argument("--asset-name", default="", help="Override the asset filename.")
    parser.add_argument(
        "--runtime-root",
        default=str(REPO_ROOT),
        help="Directory where the tarball should be extracted.",
    )
    return parser.parse_args()


def asset_url(repo: str, asset_name: str, tag: str) -> str:
    if tag.strip():
        return f"https://github.com/{repo}/releases/download/{tag.strip()}/{asset_name}"
    return f"https://github.com/{repo}/releases/latest/download/{asset_name}"


def download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib_request.urlopen(url, timeout=60) as response, destination.open("wb") as handle:
        handle.write(response.read())


def extract_tarball(archive_path: Path, runtime_root: Path) -> list[str]:
    extracted: list[str] = []
    with tarfile.open(archive_path, "r:gz") as tar:
        members = [member for member in tar.getmembers() if member.name and not member.name.startswith("/")]
        tar.extractall(runtime_root, members=members)
        extracted = sorted(member.name for member in members if member.isfile())
    return extracted


def main() -> int:
    args = parse_args()
    runtime_root = Path(args.runtime_root).expanduser().resolve()
    asset_name = args.asset_name.strip() or detect_asset_name()
    url = asset_url(args.repo, asset_name, args.tag)

    with tempfile.TemporaryDirectory(prefix="chek-ego-miner-runtime-") as temp_dir:
        archive_path = Path(temp_dir) / asset_name
        download_file(url, archive_path)
        extracted = extract_tarball(archive_path, runtime_root)

    report = {
        "ok": True,
        "repo": args.repo,
        "tag": args.tag.strip() or "latest",
        "asset_name": asset_name,
        "url": url,
        "runtime_root": str(runtime_root),
        "extracted_files": extracted,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
