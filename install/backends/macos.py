from __future__ import annotations

from pathlib import Path


def describe_backend() -> dict[str, str]:
    return {
        "name": "macos",
        "platform_hint": "macOS Apple Silicon / Intel laptops and workstations",
        "notes": "Use for basic profile and limited enhanced profile bring-up on developer machines.",
    }


def build_install_plan(*, profile_name: str, edge_root: Path) -> dict[str, object]:
    return {
        "backend": "macos",
        "profile": profile_name,
        "edge_root": str(edge_root),
        "steps": [
            "prepare python/node toolchains",
            "install runtime and capture adapters allowed on macOS",
            "stage repo-native launchd user agent artifacts",
            "optionally bootstrap launchd user agents via chek-edge service install",
        ],
        "systemd_templates": [],
        "launchd_templates": [
            f"services/launchd-user/{profile_name}",
        ],
    }
