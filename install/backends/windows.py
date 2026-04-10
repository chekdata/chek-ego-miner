from __future__ import annotations

from pathlib import Path


def describe_backend() -> dict[str, str]:
    return {
        "name": "windows",
        "platform_hint": "Windows 11 workstations / mini PCs",
        "notes": "Use for basic and enhanced profiles where stereo capture runs on a standard workstation.",
    }


def build_install_plan(*, profile_name: str, edge_root: Path) -> dict[str, object]:
    return {
        "backend": "windows",
        "profile": profile_name,
        "edge_root": str(edge_root),
        "steps": [
            "prepare python/node runtimes",
            "install profile modules",
            "stage Windows task-scheduler wrappers",
            "write Windows service/task wrapper state",
        ],
        "systemd_templates": [],
        "windows_task_templates": [
            f"services/windows-task/{profile_name}",
        ],
    }
