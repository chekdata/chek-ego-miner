#!/usr/bin/env python3
"""Check the CHEK EGO module boundary across the public and runtime repos.

This intentionally uses only the Python standard library. It is meant to run
from either repo without installing tooling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


EXPECTED_MODULES = [
    "device_health",
    "edge_auth_binding",
    "edge_capture_stereo",
    "edge_capture_usb",
    "edge_core",
    "edge_preview_packager",
    "edge_pro_extensions",
    "edge_upload_agent",
    "jetson_services",
    "local_ui",
]

REQUIRED_MODULE_KEYS = [
    "name",
    "display_name",
    "stage",
    "summary",
    "owners",
    "status",
]

MIRRORED_KEYS = [
    "name",
    "display_name",
    "stage",
    "summary",
    "status",
]

REQUIRED_CONTRACT_PHRASES = {
    "docs/repo-business-contract.md": [
        "Contract Source Of Truth",
        "PairingEnvelope",
        "ScopedUploadToken",
        "OwnerResolution",
        "Device Status Semantics",
        "Multi-Phone",
    ],
    "docs/cross-repo-module-boundary.md": [
        "Module Boundary Table",
        "Allowed Duplication",
        "Duplication To Avoid",
        "Device Status Vocabulary",
        "Multi-Phone Ownership",
        "Change Workflow",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check cross-repo EGO module names, summaries, and contract docs."
    )
    parser.add_argument(
        "--public-repo",
        type=Path,
        help="Path to chek-ego-miner. Defaults to sibling repo inference.",
    )
    parser.add_argument(
        "--runtime-repo",
        type=Path,
        help="Path to chek-edge-runtime. Defaults to sibling repo inference.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON.",
    )
    return parser.parse_args()


def find_current_repo() -> Path:
    for candidate in [Path.cwd().resolve(), *Path.cwd().resolve().parents]:
        if candidate.name in {"chek-ego-miner", "chek-edge-runtime"}:
            if (candidate / "modules").is_dir():
                return candidate
    raise SystemExit(
        "Cannot infer repo root. Run from chek-ego-miner or chek-edge-runtime, "
        "or pass --public-repo and --runtime-repo."
    )


def infer_roots(args: argparse.Namespace) -> tuple[Path, Path]:
    if args.public_repo and args.runtime_repo:
        return args.public_repo.resolve(), args.runtime_repo.resolve()

    current = find_current_repo()
    if args.public_repo:
        public_repo = args.public_repo
    elif current.name == "chek-ego-miner":
        public_repo = current
    else:
        public_repo = current.parent / "chek-ego-miner"

    if args.runtime_repo:
        runtime_repo = args.runtime_repo
    elif current.name == "chek-edge-runtime":
        runtime_repo = current
    else:
        runtime_repo = current.parent / "chek-edge-runtime"

    return public_repo.resolve(), runtime_repo.resolve()


def parse_simple_yaml(path: Path) -> dict[str, object]:
    data: dict[str, object] = {}
    current_list_key: str | None = None

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue

        if not line.startswith(" ") and ":" in line:
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip().strip("'\"")
            if value:
                data[key] = value
                current_list_key = None
            else:
                data[key] = []
                current_list_key = key
            continue

        if current_list_key and stripped.startswith("- "):
            item = stripped[2:].strip().strip("'\"")
            existing = data.setdefault(current_list_key, [])
            if not isinstance(existing, list):
                raise ValueError(f"{path}: key {current_list_key} is not a list")
            existing.append(item)

    return data


def load_modules(repo: Path) -> tuple[dict[str, dict[str, object]], list[str]]:
    errors: list[str] = []
    modules_dir = repo / "modules"
    modules: dict[str, dict[str, object]] = {}

    if not modules_dir.is_dir():
        return modules, [f"{repo}: missing modules/ directory"]

    for module_file in sorted(modules_dir.glob("*/module.yaml")):
        module_dir_name = module_file.parent.name
        try:
            parsed = parse_simple_yaml(module_file)
        except Exception as exc:  # noqa: BLE001 - report the exact file.
            errors.append(f"{module_file}: failed to parse simple module yaml: {exc}")
            continue

        module_name = str(parsed.get("name", ""))
        if module_name != module_dir_name:
            errors.append(
                f"{module_file}: name={module_name!r} does not match directory {module_dir_name!r}"
            )
        for key in REQUIRED_MODULE_KEYS:
            if key not in parsed:
                errors.append(f"{module_file}: missing required key {key!r}")
        modules[module_dir_name] = parsed

    return modules, errors


def check_module_sets(
    public_modules: dict[str, dict[str, object]],
    runtime_modules: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    expected = set(EXPECTED_MODULES)
    public_names = set(public_modules)
    runtime_names = set(runtime_modules)

    for repo_label, names in [
        ("chek-ego-miner", public_names),
        ("chek-edge-runtime", runtime_names),
    ]:
        missing = sorted(expected - names)
        extra = sorted(names - expected)
        if missing:
            errors.append(f"{repo_label}: missing expected modules: {', '.join(missing)}")
        if extra:
            errors.append(
                f"{repo_label}: unexpected modules need boundary-doc updates: {', '.join(extra)}"
            )

    if public_names != runtime_names:
        only_public = sorted(public_names - runtime_names)
        only_runtime = sorted(runtime_names - public_names)
        if only_public:
            errors.append(f"Only in chek-ego-miner modules/: {', '.join(only_public)}")
        if only_runtime:
            errors.append(f"Only in chek-edge-runtime modules/: {', '.join(only_runtime)}")

    return errors


def check_mirrored_module_fields(
    public_modules: dict[str, dict[str, object]],
    runtime_modules: dict[str, dict[str, object]],
) -> list[str]:
    errors: list[str] = []
    for name in sorted(set(public_modules) & set(runtime_modules)):
        public_module = public_modules[name]
        runtime_module = runtime_modules[name]
        for key in MIRRORED_KEYS:
            if public_module.get(key) != runtime_module.get(key):
                errors.append(
                    f"{name}: {key} drifted "
                    f"(public={public_module.get(key)!r}, runtime={runtime_module.get(key)!r})"
                )
    return errors


def check_docs(repo: Path, label: str) -> list[str]:
    errors: list[str] = []
    for relative_path, phrases in REQUIRED_CONTRACT_PHRASES.items():
        path = repo / relative_path
        if not path.is_file():
            errors.append(f"{label}: missing {relative_path}")
            continue
        text = path.read_text(encoding="utf-8")
        for phrase in phrases:
            if phrase not in text:
                errors.append(f"{label}: {relative_path} missing phrase {phrase!r}")
        if relative_path.endswith("cross-repo-module-boundary.md"):
            for module in EXPECTED_MODULES:
                if f"`{module}`" not in text:
                    errors.append(f"{label}: boundary doc missing module `{module}`")
    return errors


def main() -> int:
    args = parse_args()
    public_repo, runtime_repo = infer_roots(args)

    public_modules, public_errors = load_modules(public_repo)
    runtime_modules, runtime_errors = load_modules(runtime_repo)

    errors: list[str] = []
    errors.extend(public_errors)
    errors.extend(runtime_errors)
    errors.extend(check_module_sets(public_modules, runtime_modules))
    errors.extend(check_mirrored_module_fields(public_modules, runtime_modules))
    errors.extend(check_docs(public_repo, "chek-ego-miner"))
    errors.extend(check_docs(runtime_repo, "chek-edge-runtime"))

    result = {
        "ok": not errors,
        "public_repo": str(public_repo),
        "runtime_repo": str(runtime_repo),
        "checked_modules": sorted(set(public_modules) | set(runtime_modules)),
        "errors": errors,
    }

    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    elif errors:
        print("Cross-repo module contract check FAILED")
        for error in errors:
            print(f"- {error}")
    else:
        print(
            "Cross-repo module contract check OK: "
            f"{len(EXPECTED_MODULES)} modules and contract docs are aligned."
        )

    return 0 if not errors else 1


if __name__ == "__main__":
    sys.exit(main())
