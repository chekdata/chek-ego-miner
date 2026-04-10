#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_POLICY_PATH = REPO_ROOT / "configs" / "demo_capture_policy.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="演示采集包上传边界、脱敏导出与审计工具")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser("check-bundle", help="检查 bundle 是否符合上传/脱敏策略")
    check.add_argument("--bundle", required=True, help="demo_capture_bundle.json 路径")
    check.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="策略文件路径",
    )
    check.add_argument("--output", help="检查结果输出路径")

    export = subparsers.add_parser("export-redacted", help="按策略导出脱敏后的 bundle 副本")
    export.add_argument("--bundle", required=True, help="demo_capture_bundle.json 路径")
    export.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="策略文件路径",
    )
    export.add_argument("--dest", required=True, help="导出目录")
    export.add_argument("--audit-log", help="审计日志路径，默认使用策略里的 audit_log_relpath")

    audit = subparsers.add_parser("log-access", help="记录一次访问/导出审计事件")
    audit.add_argument("--bundle", required=True, help="demo_capture_bundle.json 路径")
    audit.add_argument("--actor", required=True, help="访问者标识")
    audit.add_argument("--action", required=True, choices=["view", "export", "download"])
    audit.add_argument("--artifacts", nargs="*", default=[], help="涉及的 artifact 列表")
    audit.add_argument(
        "--policy",
        default=str(DEFAULT_POLICY_PATH),
        help="策略文件路径",
    )
    audit.add_argument("--audit-log", help="审计日志路径，默认使用策略里的 audit_log_relpath")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def session_root(bundle_path: Path) -> Path:
    return bundle_path.resolve().parent


def matches_any(path: str, prefixes: list[str]) -> bool:
    return any(path == prefix or path.startswith(prefix) for prefix in prefixes)


def redact_payload(value: Any, sensitive_fields: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            key: ("REDACTED" if key in sensitive_fields else redact_payload(val, sensitive_fields))
            for key, val in value.items()
        }
    if isinstance(value, list):
        return [redact_payload(item, sensitive_fields) for item in value]
    return value


def enumerate_bundle_files(bundle: dict[str, Any]) -> list[str]:
    artifacts = bundle.get("artifacts", {})
    paths = [str(value) for value in artifacts.values() if isinstance(value, str) and value]
    for relpath in bundle.get("calibration_snapshot_paths", []):
        if isinstance(relpath, str) and relpath:
            paths.append(relpath)
    for relpath in bundle.get("chunk_dirs", []):
        if isinstance(relpath, str) and relpath:
            paths.append(relpath)
    paths.append("manifest.json")
    paths.append("demo_capture_bundle.json")
    return sorted(set(paths))


def build_check_result(bundle_path: Path, policy_path: Path) -> dict[str, Any]:
    bundle = load_json(bundle_path)
    policy = load_json(policy_path)
    files = enumerate_bundle_files(bundle)
    allowed = policy.get("allowed_prefixes", [])
    blocked = policy.get("blocked_prefixes", [])
    sensitive_fields = set(policy.get("sensitive_fields", []))
    session_dir = session_root(bundle_path)

    blocked_files = [path for path in files if matches_any(path, blocked)]
    out_of_scope_files = [
        path for path in files if allowed and not matches_any(path, allowed) and not matches_any(path, blocked)
    ]

    json_artifacts = [
        session_dir / relpath
        for relpath in files
        if relpath.endswith(".json") or relpath.endswith(".jsonl")
    ]
    redaction_hits: dict[str, list[str]] = {}
    for artifact_path in json_artifacts:
        if not artifact_path.is_file():
            continue
        if artifact_path.suffix == ".jsonl":
            payloads = [
                json.loads(line)
                for line in artifact_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
        else:
            payloads = [load_json(artifact_path)]
        hits = sorted(
            {
                key
                for payload in payloads
                for key in collect_sensitive_fields(payload, sensitive_fields)
            }
        )
        if hits:
            redaction_hits[str(artifact_path.relative_to(session_dir))] = hits

    return {
        "schema_version": "1.0.0",
        "checked_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "bundle": str(bundle_path),
        "policy": str(policy_path),
        "edge_primary_truth_source": bool(policy.get("edge_primary_truth_source", True)),
        "allowed_prefixes": allowed,
        "blocked_prefixes": blocked,
        "blocked_files": blocked_files,
        "out_of_scope_files": out_of_scope_files,
        "redaction_hits": redaction_hits,
        "passed": not blocked_files and not out_of_scope_files,
    }


def collect_sensitive_fields(value: Any, sensitive_fields: set[str]) -> set[str]:
    hits: set[str] = set()
    if isinstance(value, dict):
        for key, inner in value.items():
            if key in sensitive_fields:
                hits.add(key)
            hits |= collect_sensitive_fields(inner, sensitive_fields)
    elif isinstance(value, list):
        for item in value:
            hits |= collect_sensitive_fields(item, sensitive_fields)
    return hits


def export_redacted(bundle_path: Path, policy_path: Path, dest: Path, audit_log: Path | None) -> dict[str, Any]:
    bundle = load_json(bundle_path)
    policy = load_json(policy_path)
    session_dir = session_root(bundle_path)
    files = enumerate_bundle_files(bundle)
    allowed = policy.get("allowed_prefixes", [])
    blocked = policy.get("blocked_prefixes", [])
    sensitive_fields = set(policy.get("sensitive_fields", []))

    copied: list[str] = []
    skipped: list[dict[str, str]] = []
    exported_bundle = redact_payload(bundle, sensitive_fields)
    for relpath in files:
        if matches_any(relpath, blocked):
            skipped.append({"path": relpath, "reason": "blocked_by_policy"})
            continue
        if allowed and not matches_any(relpath, allowed):
            skipped.append({"path": relpath, "reason": "out_of_scope"})
            continue
        source = session_dir / relpath
        target = dest / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.is_dir():
            shutil.copytree(source, target, dirs_exist_ok=True)
            copied.append(relpath)
            continue
        if not source.is_file():
            skipped.append({"path": relpath, "reason": "missing"})
            continue
        if relpath.endswith(".json"):
            payload = redact_payload(load_json(source), sensitive_fields)
            target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        elif relpath.endswith(".jsonl"):
            lines = []
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                payload = redact_payload(json.loads(line), sensitive_fields)
                lines.append(json.dumps(payload, ensure_ascii=False))
            target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        else:
            shutil.copy2(source, target)
        copied.append(relpath)

    copied_set = set(copied)
    artifacts = exported_bundle.get("artifacts")
    if isinstance(artifacts, dict):
        exported_bundle["artifacts"] = {
            key: value
            for key, value in artifacts.items()
            if isinstance(value, str) and value in copied_set
        }
    calibration_snapshot_paths = exported_bundle.get("calibration_snapshot_paths")
    if isinstance(calibration_snapshot_paths, list):
        exported_bundle["calibration_snapshot_paths"] = [
            value for value in calibration_snapshot_paths if isinstance(value, str) and value in copied_set
        ]
    chunk_dirs = exported_bundle.get("chunk_dirs")
    if isinstance(chunk_dirs, list):
        exported_bundle["chunk_dirs"] = [
            value for value in chunk_dirs if isinstance(value, str) and value in copied_set
        ]
    bundle_target = dest / "demo_capture_bundle.json"
    bundle_target.parent.mkdir(parents=True, exist_ok=True)
    bundle_target.write_text(
        json.dumps(exported_bundle, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    export_manifest = {
        "schema_version": "1.0.0",
        "exported_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_bundle": str(bundle_path),
        "policy": str(policy_path),
        "copied": copied,
        "skipped": skipped,
    }
    (dest / "export_manifest.json").write_text(
        json.dumps(export_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    audit_path = resolve_audit_log(dest, policy, audit_log)
    append_audit_event(
        audit_path,
        actor="policy_tool",
        action="export",
        bundle=str(bundle_path),
        artifacts=copied,
    )
    return export_manifest


def append_audit_event(
    audit_path: Path,
    actor: str,
    action: str,
    bundle: str,
    artifacts: list[str],
) -> None:
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    event = {
        "event_time": dt.datetime.now(dt.timezone.utc).isoformat(),
        "actor": actor,
        "action": action,
        "bundle": bundle,
        "artifacts": artifacts,
    }
    with audit_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


def resolve_audit_log(base: Path, policy: dict[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override
    relpath = str(policy.get("audit_log_relpath", "audit/access_audit.jsonl"))
    return base / relpath


def main() -> int:
    args = parse_args()
    if args.command == "check-bundle":
        result = build_check_result(Path(args.bundle), Path(args.policy))
        if args.output:
            output = Path(args.output)
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        else:
            print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0 if result["passed"] else 1

    if args.command == "export-redacted":
        dest = Path(args.dest)
        dest.mkdir(parents=True, exist_ok=True)
        manifest = export_redacted(
            Path(args.bundle),
            Path(args.policy),
            dest,
            Path(args.audit_log) if args.audit_log else None,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0

    if args.command == "log-access":
        policy = load_json(Path(args.policy))
        audit_path = resolve_audit_log(
            session_root(Path(args.bundle)),
            policy,
            Path(args.audit_log) if args.audit_log else None,
        )
        append_audit_event(audit_path, args.actor, args.action, args.bundle, args.artifacts)
        print(str(audit_path))
        return 0

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
