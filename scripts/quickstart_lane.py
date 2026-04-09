#!/usr/bin/env python3
from __future__ import annotations

import argparse
import platform


PROMPTS = {
    "lite": "prompts/install-lite.md",
    "stereo": "prompts/install-stereo.md",
    "pro": "prompts/install-pro-edge.md",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Print the next steps for a CHEK EGO Miner quickstart lane.")
    parser.add_argument("--tier", choices=["lite", "stereo", "pro"], required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    system_name = platform.system()
    print(f"CHEK EGO Miner quickstart: {args.tier}")
    print(f"- host OS: {system_name}")
    print("- next steps:")
    steps = [
        "Install the iOS app from TestFlight.",
        "Read the hardware guide and confirm your mount and camera plan.",
        "Run `python3 scripts/check_host_basics.py`.",
        f"Run `python3 scripts/readiness_public.py --tier {args.tier}`.",
    ]
    if args.tier in {"stereo", "pro"}:
        steps.append("If needed, generate and print a Charuco board for calibration.")
    steps.append(f"Copy `{PROMPTS[args.tier]}` into your preferred agent.")
    steps.append("Follow the agent one step at a time.")
    for index, step in enumerate(steps, start=1):
        print(f"  {index}. {step}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
