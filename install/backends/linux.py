from __future__ import annotations

from pathlib import Path


def describe_backend() -> dict[str, str]:
    return {
        "name": "linux",
        "platform_hint": "Linux x86_64 / generic Linux hosts",
        "notes": "Use for basic and enhanced profiles on standard Linux machines.",
    }


def build_install_plan(*, profile_name: str, edge_root: Path) -> dict[str, object]:
    return {
        "backend": "linux",
        "profile": profile_name,
        "edge_root": str(edge_root),
        "steps": [
            "prepare python/node/rust toolchains",
            "install profile modules",
            "stage and optionally install systemd-user services for persistent auto-start",
            "write runtime config and local state",
        ],
        "systemd_templates": [],
    }
