from __future__ import annotations

from pathlib import Path


def describe_backend() -> dict[str, str]:
    return {
        "name": "jetson",
        "platform_hint": "Linux ARM64 / Jetson Orin family",
        "notes": "Use for professional profile and legacy install wrapper compatibility.",
    }


def build_install_plan(*, profile_name: str, edge_root: Path) -> dict[str, object]:
    return {
        "backend": "jetson",
        "profile": profile_name,
        "edge_root": str(edge_root),
        "steps": [
            "call legacy compatibility wrappers when needed",
            "install professional modules and local UI",
            "install systemd and systemd-user templates",
        ],
        "systemd_templates": [
            "services/systemd/professional",
            "services/systemd-user/professional",
        ],
    }

