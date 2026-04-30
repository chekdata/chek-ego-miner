#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_DIR = REPO_ROOT / "scripts"
BOOTSTRAP_CLI = REPO_ROOT / "cli" / "bootstrap_cli.py"


def script_path(name: str) -> Path:
    return SCRIPTS_DIR / name


def run_python(script_name: str, args: list[str]) -> int:
    command = [sys.executable, str(script_path(script_name)), *args]
    return subprocess.run(command, check=False).returncode


def run_shell(script_name: str, args: list[str]) -> int:
    command = [str(script_path(script_name)), *args]
    return subprocess.run(command, check=False).returncode


def normalize_passthrough(args: list[str]) -> list[str]:
    if args[:1] == ["--"]:
        return args[1:]
    return args


def run_runtime_cli(args: list[str]) -> int:
    command = [sys.executable, str(BOOTSTRAP_CLI), *normalize_passthrough(args)]
    return subprocess.run(command, check=False).returncode


def add_passthrough_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
) -> None:
    parser = subparsers.add_parser(name, help=help_text)
    parser.add_argument("runtime_args", nargs=argparse.REMAINDER)


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

    camera_probe = subparsers.add_parser("camera-probe", help="Probe local camera devices.")
    camera_probe.add_argument("--json", action="store_true")
    camera_probe.add_argument("--report-path")
    camera_probe.add_argument("--capture-smoke", action="store_true")
    camera_probe.add_argument("--timeout", type=float, default=8)
    camera_probe.add_argument("--device-index", type=int, default=0)
    camera_probe.add_argument("--video-size", default="1280x720")
    camera_probe.add_argument("--framerate", default="30")

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

    safety = subparsers.add_parser(
        "scan-public-safety",
        help="Scan the repo for internal-only patterns.",
    )
    safety.add_argument("root", nargs="?", default=".")

    fetch_assets = subparsers.add_parser(
        "fetch-runtime-assets",
        help="Download published runtime binaries for the current host.",
    )
    fetch_assets.add_argument("--repo", default="hongzexin/chek-ego-miner")
    fetch_assets.add_argument("--tag", default="")
    fetch_assets.add_argument("--asset-name", default="")
    fetch_assets.add_argument("--runtime-root", default="")

    fetch_phone_vision_models = subparsers.add_parser(
        "fetch-phone-vision-models",
        help="Download the model files required by the local phone-vision sidecar.",
    )
    fetch_phone_vision_models.add_argument("--models-root", default="")
    fetch_phone_vision_models.add_argument("--force", action="store_true")
    fetch_phone_vision_models.add_argument("--json", action="store_true")
    fetch_phone_vision_models.add_argument("--report-path")

    fetch_vlm_models = subparsers.add_parser(
        "fetch-vlm-models",
        help="Download the VLM model files required by the Jetson Pro sidecar.",
    )
    fetch_vlm_models.add_argument("--models-root", default="")
    fetch_vlm_models.add_argument("--primary-model-id", default="SmolVLM2-500M")
    fetch_vlm_models.add_argument("--fallback-model-id", default="SmolVLM2-256M")
    fetch_vlm_models.add_argument("--skip-fallback", action="store_true")
    fetch_vlm_models.add_argument("--force", action="store_true")
    fetch_vlm_models.add_argument("--json", action="store_true")
    fetch_vlm_models.add_argument("--report-path")

    jetson_vlm_bootstrap = subparsers.add_parser(
        "jetson-vlm-bootstrap",
        help="Wire an existing Jetson VLM runtime into the public repo layout.",
    )
    jetson_vlm_bootstrap.add_argument("bootstrap_args", nargs=argparse.REMAINDER)

    jetson_professional_bootstrap = subparsers.add_parser(
        "jetson-professional-bootstrap",
        help="Wire existing Jetson professional runtime assets into the public repo layout.",
    )
    jetson_professional_bootstrap.add_argument("bootstrap_args", nargs=argparse.REMAINDER)

    validate = subparsers.add_parser(
        "validate-bundle",
        help="Validate a downloaded capture bundle against tier requirements.",
    )
    validate.add_argument("--bundle", required=True)
    validate.add_argument("--tier", choices=["basic", "stereo", "pro"], default="basic")
    validate.add_argument("--json", action="store_true")
    validate.add_argument("--report-path")

    basic_e2e = subparsers.add_parser(
        "basic-e2e",
        help="Run a synthetic basic capture -> bundle download -> validation flow.",
    )
    basic_e2e.add_argument("e2e_args", nargs=argparse.REMAINDER)

    synthetic = subparsers.add_parser(
        "synthetic-feed",
        help="Feed synthetic phone ingress packets into a running edge host.",
    )
    synthetic.add_argument("feed_args", nargs=argparse.REMAINDER)

    add_passthrough_parser(subparsers, "install", "Forward to the runtime install command.")
    add_passthrough_parser(subparsers, "status", "Forward to the runtime status command.")
    add_passthrough_parser(subparsers, "bind", "Forward to the runtime bind command.")
    add_passthrough_parser(subparsers, "runtime", "Forward arbitrary args to the runtime CLI.")
    add_passthrough_parser(
        subparsers,
        "capture-probe",
        "Forward to the runtime capture probe command.",
    )
    add_passthrough_parser(
        subparsers,
        "preview-test",
        "Forward to the runtime preview test command.",
    )
    add_passthrough_parser(
        subparsers,
        "upload-test",
        "Forward to the runtime upload test command.",
    )
    add_passthrough_parser(
        subparsers,
        "service-install",
        "Forward to the runtime service install command.",
    )
    phone_vision_start = subparsers.add_parser(
        "phone-vision-start",
        help="Start the local phone-vision sidecar in the foreground.",
    )
    phone_vision_start.add_argument("service_args", nargs=argparse.REMAINDER)

    vlm_start = subparsers.add_parser(
        "vlm-start",
        help="Start the local VLM sidecar in the foreground.",
    )
    vlm_start.add_argument("service_args", nargs=argparse.REMAINDER)

    return parser


def dispatch_passthrough(command: str, passthrough_args: list[str]) -> int:
    if command == "basic-e2e":
        return run_python("run_basic_e2e.py", normalize_passthrough(passthrough_args))

    if command == "synthetic-feed":
        return run_python(
            "run_synthetic_phone_ingress_feed.py",
            normalize_passthrough(passthrough_args),
        )

    if command == "runtime":
        return run_runtime_cli(passthrough_args)

    if command == "install":
        return run_runtime_cli(["install", *passthrough_args])

    if command == "status":
        return run_runtime_cli(["status", *passthrough_args])

    if command == "bind":
        return run_runtime_cli(["bind", *passthrough_args])

    if command == "capture-probe":
        return run_runtime_cli(["capture", "probe", *passthrough_args])

    if command == "preview-test":
        return run_runtime_cli(["preview", "test", *passthrough_args])

    if command == "upload-test":
        return run_runtime_cli(["upload", "test", *passthrough_args])

    if command == "service-install":
        return run_runtime_cli(["service", "install", *passthrough_args])

    raise ValueError(f"unsupported passthrough command: {command}")


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    passthrough_commands = {
        "basic-e2e",
        "synthetic-feed",
        "install",
        "status",
        "bind",
        "runtime",
        "capture-probe",
        "preview-test",
        "upload-test",
        "service-install",
    }
    if argv and argv[0] in passthrough_commands:
        return dispatch_passthrough(argv[0], argv[1:])

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

    if args.command == "camera-probe":
        extra = []
        if args.json:
            extra.append("--json")
        if args.report_path:
            extra.extend(["--report-path", args.report_path])
        if args.capture_smoke:
            extra.append("--capture-smoke")
        extra.extend(
            [
                "--timeout",
                str(args.timeout),
                "--device-index",
                str(args.device_index),
                "--video-size",
                args.video_size,
                "--framerate",
                args.framerate,
            ]
        )
        return run_python("camera_probe.py", extra)

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

    if args.command == "fetch-runtime-assets":
        extra = ["--repo", args.repo]
        if args.tag:
            extra.extend(["--tag", args.tag])
        if args.asset_name:
            extra.extend(["--asset-name", args.asset_name])
        if args.runtime_root:
            extra.extend(["--runtime-root", args.runtime_root])
        return run_python("fetch_runtime_assets.py", extra)

    if args.command == "fetch-phone-vision-models":
        extra = []
        if args.models_root:
            extra.extend(["--models-root", args.models_root])
        if args.force:
            extra.append("--force")
        if args.json:
            extra.append("--json")
        if args.report_path:
            extra.extend(["--report-path", args.report_path])
        return run_python("fetch_phone_vision_models.py", extra)

    if args.command == "fetch-vlm-models":
        extra = [
            "--primary-model-id",
            args.primary_model_id,
        ]
        if args.models_root:
            extra.extend(["--models-root", args.models_root])
        if args.fallback_model_id:
            extra.extend(["--fallback-model-id", args.fallback_model_id])
        if args.skip_fallback:
            extra.append("--skip-fallback")
        if args.force:
            extra.append("--force")
        if args.json:
            extra.append("--json")
        if args.report_path:
            extra.extend(["--report-path", args.report_path])
        return run_python("fetch_vlm_models.py", extra)

    if args.command == "jetson-vlm-bootstrap":
        return run_shell(
            "bootstrap_jetson_vlm_runtime.sh",
            normalize_passthrough(args.bootstrap_args),
        )

    if args.command == "jetson-professional-bootstrap":
        return run_shell(
            "bootstrap_jetson_professional_runtime.sh",
            normalize_passthrough(args.bootstrap_args),
        )

    if args.command == "validate-bundle":
        extra = ["--bundle", args.bundle, "--tier", args.tier]
        if args.json:
            extra.append("--json")
        if args.report_path:
            extra.extend(["--report-path", args.report_path])
        return run_python("validate_capture_bundle.py", extra)

    if args.command == "phone-vision-start":
        return run_shell(
            "start_edge_phone_vision_service.sh",
            normalize_passthrough(args.service_args),
        )

    if args.command == "vlm-start":
        return run_shell(
            "start_edge_vlm_sidecar.sh",
            normalize_passthrough(args.service_args),
        )

    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
