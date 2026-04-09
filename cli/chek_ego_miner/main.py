#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"


def script_path(name: str) -> Path:
    return SCRIPTS_DIR / name


def run_python(script_name: str, args: list[str]) -> int:
    command = [sys.executable, str(script_path(script_name)), *args]
    return subprocess.run(command, check=False).returncode


def run_shell(script_name: str, args: list[str]) -> int:
    command = [str(script_path(script_name)), *args]
    return subprocess.run(command, check=False).returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Public-safe CLI for CHEK EGO Miner.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    doctor = subparsers.add_parser("doctor", help="Print a lightweight host report.")
    doctor.add_argument("--json", action="store_true")
    doctor.add_argument("--report-path")

    readiness = subparsers.add_parser("readiness", help="Run a public-safe readiness check.")
    readiness.add_argument("--tier", choices=["lite", "stereo", "pro"], required=True)
    readiness.add_argument("--json", action="store_true")
    readiness.add_argument("--report-path")

    charuco = subparsers.add_parser("charuco", help="Generate a Charuco board PDF and PNG.")
    charuco.add_argument("--output-dir", required=True)
    charuco.add_argument("--board-squares-x", type=int, default=8)
    charuco.add_argument("--board-squares-y", type=int, default=6)
    charuco.add_argument("--square-size-mm", type=float, default=24.0)
    charuco.add_argument("--marker-size-mm", type=float, default=18.0)
    charuco.add_argument("--dictionary-name", default="DICT_4X4_50")
    charuco.add_argument("--dpi", type=int, default=300)
    charuco.add_argument("--landscape", action="store_true")

    quickstart = subparsers.add_parser("quickstart", help="Print the next steps for a tier.")
    quickstart.add_argument("--tier", choices=["lite", "stereo", "pro"], required=True)

    safety = subparsers.add_parser("scan-public-safety", help="Scan the repo for internal-only patterns.")
    safety.add_argument("root", nargs="?", default=".")

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "doctor":
        extra = []
        if args.json:
            extra.append("--json")
        if args.report_path:
            extra.extend(["--report-path", args.report_path])
        return run_python("check_host_basics.py", extra)

    if args.command == "readiness":
        extra = ["--tier", args.tier]
        if args.json:
            extra.append("--json")
        if args.report_path:
            extra.extend(["--report-path", args.report_path])
        return run_python("readiness_public.py", extra)

    if args.command == "charuco":
        extra = [
            "--output-dir",
            args.output_dir,
            "--board-squares-x",
            str(args.board_squares_x),
            "--board-squares-y",
            str(args.board_squares_y),
            "--square-size-mm",
            str(args.square_size_mm),
            "--marker-size-mm",
            str(args.marker_size_mm),
            "--dictionary-name",
            args.dictionary_name,
            "--dpi",
            str(args.dpi),
        ]
        if args.landscape:
            extra.append("--landscape")
        return run_python("generate_charuco_a4_pdf.py", extra)

    if args.command == "quickstart":
        return run_python("quickstart_lane.py", ["--tier", args.tier])

    if args.command == "scan-public-safety":
        return run_shell("scan_public_safety.sh", [args.root])

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
