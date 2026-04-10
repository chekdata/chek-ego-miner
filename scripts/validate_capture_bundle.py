#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.demo_capture_policy_tool import build_check_result


REQUIRED_TOP_LEVEL_KEYS = ("schema_version", "session_id", "trip_id", "artifacts")
REQUIRED_ARTIFACT_KEYS = {
    "basic": ("capture_pose",),
    "stereo": ("capture_pose", "time_sync_samples", "stereo_pose"),
    "pro": ("capture_pose", "time_sync_samples", "stereo_pose", "wifi_pose"),
}
ADVISORY_ARTIFACT_KEYS = {
    "basic": ("time_sync_samples",),
    "stereo": (),
    "pro": (),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a CHEK capture bundle against tier requirements.")
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--tier", choices=["basic", "stereo", "pro"], default="basic")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--report-path")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_line_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def build_artifact_check(relpath: str, artifact_path: Path, artifact_key: str) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = [
        {
            "id": f"artifact_declared_{artifact_key}",
            "ok": bool(relpath),
            "detail": relpath or "not declared in bundle.artifacts",
        },
        {
            "id": f"artifact_exists_{artifact_key}",
            "ok": bool(relpath) and artifact_path.is_file(),
            "detail": relpath or "missing relpath",
        },
    ]
    if artifact_path.suffix == ".jsonl":
        line_count = jsonl_line_count(artifact_path)
        checks.append(
            {
                "id": f"artifact_nonempty_{artifact_key}",
                "ok": line_count > 0,
                "detail": f"{relpath} lines={line_count}",
            }
        )
    return checks


def validate_bundle(bundle_path: Path, *, tier: str) -> dict[str, object]:
    bundle = load_json(bundle_path)
    session_root = bundle_path.resolve().parent
    policy_result = build_check_result(bundle_path, REPO_ROOT / "configs" / "demo_capture_policy.json")
    checks: list[dict[str, object]] = []
    advisories: list[dict[str, object]] = []

    for key in REQUIRED_TOP_LEVEL_KEYS:
        checks.append(
            {
                "id": f"top_level_{key}",
                "ok": key in bundle,
                "detail": f"bundle contains key `{key}`" if key in bundle else f"bundle is missing key `{key}`",
            }
        )

    schema_ok = str(bundle.get("schema_version") or "").strip() == "1.0.0"
    checks.append(
        {
            "id": "schema_version_supported",
            "ok": schema_ok,
            "detail": f"schema_version={bundle.get('schema_version')!r}",
        }
    )

    artifacts = bundle.get("artifacts") or {}
    if not isinstance(artifacts, dict):
        artifacts = {}

    manifest_path = session_root / "manifest.json"
    checks.append(
        {
            "id": "manifest_present",
            "ok": manifest_path.is_file(),
            "detail": str(manifest_path.relative_to(session_root)),
        }
    )

    for artifact_key in REQUIRED_ARTIFACT_KEYS[tier]:
        relpath = str(artifacts.get(artifact_key) or "").strip()
        artifact_path = session_root / relpath if relpath else session_root / "__missing__"
        checks.extend(build_artifact_check(relpath, artifact_path, artifact_key))

    for artifact_key in ADVISORY_ARTIFACT_KEYS[tier]:
        relpath = str(artifacts.get(artifact_key) or "").strip()
        artifact_path = session_root / relpath if relpath else session_root / "__missing__"
        optional_checks = build_artifact_check(relpath, artifact_path, artifact_key)
        failing_optional = [item for item in optional_checks if not item["ok"]]
        if failing_optional:
            advisories.append(
                {
                    "id": f"advisory_{artifact_key}",
                    "detail": [item["detail"] for item in failing_optional],
                }
            )

    checks.append(
        {
            "id": "policy_boundary_passed",
            "ok": bool(policy_result.get("passed")),
            "detail": (
                "allowed prefixes only"
                if policy_result.get("passed")
                else f"blocked={policy_result.get('blocked_files')} out_of_scope={policy_result.get('out_of_scope_files')}"
            ),
        }
    )

    passed = sum(1 for check in checks if check["ok"])
    total = max(len(checks), 1)
    score_percent = round((passed / total) * 100.0, 2)
    missing = [check["id"] for check in checks if not check["ok"]]
    payload = {
        "ok": not missing,
        "tier": tier,
        "bundle": str(bundle_path),
        "session_root": str(session_root),
        "score_percent": score_percent,
        "missing": missing,
        "checks": checks,
        "advisories": advisories,
        "policy_result": policy_result,
    }
    return payload


def print_human(payload: dict[str, object]) -> None:
    print(f"CHEK EGO Miner bundle validation: {payload['tier']}")
    print(f"- ok: {'yes' if payload['ok'] else 'no'}")
    print(f"- score: {payload['score_percent']}%")
    if payload["missing"]:
        print("- missing or failed checks:")
        for item in payload["missing"]:
            print(f"  - {item}")


def main() -> int:
    args = parse_args()
    payload = validate_bundle(Path(args.bundle).expanduser().resolve(), tier=args.tier)
    encoded = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.report_path:
        Path(args.report_path).write_text(encoded + "\n", encoding="utf-8")
    if args.json:
        print(encoded)
    else:
        print_human(payload)
        print("- json:")
        print(encoded)
    return 0 if payload["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
