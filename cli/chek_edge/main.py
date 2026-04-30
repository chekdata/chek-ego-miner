from __future__ import annotations

import importlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from urllib import error as urllib_error
from urllib import parse as urllib_parse
from urllib import request as urllib_request
from datetime import datetime, timezone
from pathlib import Path
from pathlib import PureWindowsPath
from typing import Any

import typer
import yaml

app = typer.Typer(help="CHEK edge runtime CLI", no_args_is_help=True)
profile_app = typer.Typer(help="Profile manifests")
capture_app = typer.Typer(help="Capture helpers")
preview_app = typer.Typer(help="Preview helpers")
upload_app = typer.Typer(help="Upload helpers")
service_app = typer.Typer(help="Service helpers")
logs_app = typer.Typer(help="Logs helpers")

app.add_typer(profile_app, name="profile")
app.add_typer(capture_app, name="capture")
app.add_typer(preview_app, name="preview")
app.add_typer(upload_app, name="upload")
app.add_typer(service_app, name="service")
app.add_typer(logs_app, name="logs")

REPO_ROOT = Path(__file__).resolve().parents[2]
PROFILES_DIR = REPO_ROOT / "profiles"
MODULES_DIR = REPO_ROOT / "modules"
SERVICES_DIR = REPO_ROOT / "services"
STATE_DIR_NAME = ".chek-edge"
RUNTIME_SYNC_PATHS = [
    "cli",
    "install",
    "profiles",
    "modules",
    "services",
    "model-candidates",
    "edge-orchestrator",
    "ruview-leap-bridge",
    "ruview-unitree-bridge",
    "chek-edge-debug",
    "scripts",
    "sim/isaac_loader",
    "RuView/ui",
    "RuView/ui-react",
]
RUNTIME_SYNC_IGNORE_NAMES = (
    ".git",
    ".venv",
    "__pycache__",
    ".DS_Store",
    "node_modules",
)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _state_dir(edge_root: Path) -> Path:
    return edge_root / STATE_DIR_NAME


def _config_path(edge_root: Path) -> Path:
    return _state_dir(edge_root) / "config.yaml"


def _plan_path(edge_root: Path) -> Path:
    return _state_dir(edge_root) / "last-plan.json"


def _ensure_state_dir(edge_root: Path) -> Path:
    path = _state_dir(edge_root)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")


def _load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise typer.BadParameter(f"missing file: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise typer.BadParameter(f"expected mapping in {path}")
    return payload


def _load_config(edge_root: Path) -> dict[str, Any]:
    path = _config_path(edge_root)
    if not path.exists():
        return {}
    return _load_yaml(path)


def _save_config(edge_root: Path, config: dict[str, Any]) -> None:
    _ensure_state_dir(edge_root)
    _write_yaml(_config_path(edge_root), config)


def _join_url(base_url: str, path: str, params: dict[str, Any] | None = None) -> str:
    normalized_base = base_url.rstrip("/")
    normalized_path = path if path.startswith("/") else f"/{path}"
    url = f"{normalized_base}{normalized_path}"
    filtered_params = {
        key: value
        for key, value in (params or {}).items()
        if str(value or "").strip()
    }
    if filtered_params:
        url = f"{url}?{urllib_parse.urlencode(filtered_params)}"
    return url


def _auth_headers(*, bearer_token: str = "", user_id: str = "") -> dict[str, str]:
    headers = {"Content-Type": "application/json"}
    if bearer_token.strip():
        headers["Authorization"] = f"Bearer {bearer_token.strip()}"
    if user_id.strip():
        headers["X-User-One-Id"] = user_id.strip()
    return headers


def _http_json(
    method: str,
    url: str,
    *,
    headers: dict[str, str] | None = None,
    payload: dict[str, Any] | None = None,
) -> dict[str, Any]:
    body = None
    normalized_headers = dict(headers or {})
    if payload is not None:
        normalized_headers.setdefault("Content-Type", "application/json")
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib_request.Request(url, method=method.upper(), headers=normalized_headers, data=body)
    try:
        with urllib_request.urlopen(req, timeout=10) as response:
            raw = response.read().decode("utf-8")
            return {
                "ok": True,
                "status_code": response.status,
                "body": json.loads(raw) if raw else {},
            }
    except urllib_error.HTTPError as exc:
        raw = exc.read().decode("utf-8") if exc.fp is not None else ""
        try:
            parsed = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            parsed = {"detail": raw}
        return {
            "ok": False,
            "status_code": exc.code,
            "body": parsed,
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status_code": 0,
            "body": {"detail": repr(exc)},
        }


def _load_profile(profile: str) -> dict[str, Any]:
    return _load_yaml(PROFILES_DIR / f"{profile}.yaml")


def _available_profiles() -> list[dict[str, Any]]:
    profiles: list[dict[str, Any]] = []
    for path in sorted(PROFILES_DIR.glob("*.yaml")):
        payload = _load_yaml(path)
        payload.setdefault("name", path.stem)
        profiles.append(payload)
    return profiles


def _module_manifest_path(name: str) -> Path:
    return MODULES_DIR / name / "module.yaml"


def _load_module(name: str) -> dict[str, Any]:
    payload = _load_yaml(_module_manifest_path(name))
    payload.setdefault("name", name)
    payload.setdefault("path", str(MODULES_DIR / name))
    return payload


def _load_modules(names: list[str]) -> list[dict[str, Any]]:
    modules: list[dict[str, Any]] = []
    for name in names:
        path = _module_manifest_path(name)
        if not path.exists():
            modules.append(
                {
                    "name": name,
                    "path": str(MODULES_DIR / name),
                    "status": "missing_manifest",
                }
            )
            continue
        modules.append(_load_module(name))
    return modules


def _service_catalog_paths(profile_name: str) -> list[Path]:
    return sorted(path for path in SERVICES_DIR.glob(f"*/{profile_name}/manifest.yaml") if path.is_file())


def _all_service_catalog_paths() -> list[Path]:
    return sorted(path for path in SERVICES_DIR.glob("*/*/manifest.yaml") if path.is_file())


def _catalog_supports_backend(catalog: dict[str, Any], backend_name: str) -> bool:
    normalized_backend = backend_name.strip()
    if not normalized_backend:
        return True
    supported_backends = [str(value).strip() for value in list(catalog.get("supported_backends", [])) if str(value).strip()]
    if not supported_backends:
        return True
    return normalized_backend in supported_backends


def _load_service_catalogs(profile_name: str, backend_name: str = "") -> list[dict[str, Any]]:
    catalogs: list[dict[str, Any]] = []
    for path in _service_catalog_paths(profile_name):
        if not path.exists():
            continue
        payload = _load_yaml(path)
        if not _catalog_supports_backend(payload, backend_name):
            continue
        payload.setdefault("path", str(path.parent))
        catalogs.append(payload)
    return catalogs


def _collect_service_templates(profile_name: str, backend_name: str = "") -> list[str]:
    templates: list[str] = []
    for manifest_path in _service_catalog_paths(profile_name):
        if not manifest_path.exists():
            continue
        payload = _load_yaml(manifest_path)
        if not _catalog_supports_backend(payload, backend_name):
            continue
        root = manifest_path.parent
        for path in sorted(root.glob("*")):
            if path.name == "manifest.yaml":
                continue
            if path.is_file():
                templates.append(str(path.relative_to(REPO_ROOT)))
    return templates


def _detect_backend(profile_name: str) -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if profile_name == "professional" and system == "linux" and ("arm" in machine or "aarch64" in machine):
        return "jetson"
    if system == "darwin":
        return "macos"
    if system == "windows":
        return "windows"
    return "linux"


def _load_backend(name: str):
    return importlib.import_module(f"install.backends.{name}")


def _path_from_option(value: str) -> Path | None:
    if not value.strip():
        return None
    return Path(value).expanduser().resolve()


def _default_manager_root(manager: str) -> Path:
    if manager == "launchd-user":
        return Path.home() / "Library" / "LaunchAgents"
    if manager == "systemd-user":
        xdg_config_home = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
        return xdg_config_home / "systemd" / "user"
    if manager == "systemd":
        return Path("/etc/systemd/system")
    if manager == "windows-task":
        local_app_data = os.environ.get("LOCALAPPDATA", "").strip()
        if local_app_data:
            return Path(local_app_data) / "CHEK" / "Edge" / "Tasks"
        return Path.home() / "AppData" / "Local" / "CHEK" / "Edge" / "Tasks"
    raise typer.BadParameter(f"unsupported service manager: {manager}")


def _resolve_manager_root(
    manager: str,
    *,
    system_root: Path | None = None,
    systemd_user_root: Path | None = None,
    launchd_user_root: Path | None = None,
    windows_task_root: Path | None = None,
) -> Path:
    if manager == "systemd" and system_root is not None:
        return system_root
    if manager == "systemd-user" and systemd_user_root is not None:
        return systemd_user_root
    if manager == "launchd-user" and launchd_user_root is not None:
        return launchd_user_root
    if manager == "windows-task" and windows_task_root is not None:
        return windows_task_root
    return _default_manager_root(manager)


def _candidate_runtime_paths(edge_root: Path, relative_path: str) -> list[Path]:
    candidates = [
        edge_root / relative_path,
        REPO_ROOT / relative_path,
    ]
    seen: set[Path] = set()
    ordered: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        ordered.append(resolved)
        seen.add(resolved)
    return ordered


def _stack_script_path(edge_root: Path) -> Path:
    candidates = _candidate_runtime_paths(edge_root, "scripts/teleop_local_stack.sh")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _stack_ps_script_path(edge_root: Path) -> Path:
    candidates = _candidate_runtime_paths(edge_root, "scripts/teleop_local_stack.ps1")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _stack_cmd_script_path(edge_root: Path) -> Path:
    candidates = _candidate_runtime_paths(edge_root, "scripts/teleop_local_stack.cmd")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _launch_agent_script_path(edge_root: Path) -> Path:
    candidates = _candidate_runtime_paths(edge_root, "scripts/teleop_local_launch_agent.sh")
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _user_process_runtime_env(
    profile_name: str,
    *,
    edge_root: Path,
    runtime_edge_root: str = "",
    runtime_profile: str = "",
    bind_host: str = "127.0.0.1",
    public_host: str = "",
    control_enabled: str = "",
    sim_enabled: str = "",
) -> dict[str, str]:
    resolved_runtime_profile = runtime_profile.strip() or (
        "capture_plus_facts" if profile_name == "basic" else "teleop_fullstack"
    )
    resolved_control = control_enabled.strip() or ("0" if profile_name == "basic" else "1")
    resolved_sim = sim_enabled.strip() or "0"
    resolved_public_host = public_host.strip() or bind_host.strip() or "127.0.0.1"
    resolved_edge_root = runtime_edge_root.strip() or str(edge_root)
    return {
        "CHEK_EDGE_ROOT": resolved_edge_root,
        "EDGE_RUNTIME_PROFILE": resolved_runtime_profile,
        "EDGE_CONTROL_ENABLED": resolved_control,
        "EDGE_SIM_ENABLED": resolved_sim,
        "STACK_BIND_HOST": bind_host.strip() or "127.0.0.1",
        "STACK_CHECK_HOST": bind_host.strip() or "127.0.0.1",
        "STACK_PUBLIC_HOST": resolved_public_host,
    }


def _render_launchd_template(
    template_path: Path,
    *,
    edge_root: Path,
    service_env: dict[str, str],
) -> str:
    label = template_path.stem
    runtime_root = Path(service_env["CHEK_EDGE_ROOT"]).expanduser()
    log_dir = _state_dir(runtime_root) / "logs" / "launchd"
    log_dir.mkdir(parents=True, exist_ok=True)
    payload = template_path.read_text(encoding="utf-8")
    replacements = {
        "__SERVICE_LABEL__": label,
        "__EDGE_ROOT__": str(runtime_root),
        "__LOG_DIR__": str(log_dir),
        "__EDGE_RUNTIME_PROFILE__": service_env["EDGE_RUNTIME_PROFILE"],
        "__EDGE_CONTROL_ENABLED__": service_env["EDGE_CONTROL_ENABLED"],
        "__EDGE_SIM_ENABLED__": service_env["EDGE_SIM_ENABLED"],
        "__STACK_BIND_HOST__": service_env["STACK_BIND_HOST"],
        "__STACK_PUBLIC_HOST__": service_env["STACK_PUBLIC_HOST"],
    }
    for key, value in replacements.items():
        payload = payload.replace(key, value)
    return payload


def _render_systemd_template(
    template_path: Path,
    *,
    edge_root: Path,
    service_env: dict[str, str],
) -> str:
    payload = template_path.read_text(encoding="utf-8")
    runtime_root = Path(service_env["CHEK_EDGE_ROOT"]).expanduser()
    replacements = {
        "__EDGE_ROOT__": str(runtime_root),
        "__CHEK_EDGE_ROOT__": service_env["CHEK_EDGE_ROOT"],
        "__EDGE_RUNTIME_PROFILE__": service_env["EDGE_RUNTIME_PROFILE"],
        "__EDGE_CONTROL_ENABLED__": service_env["EDGE_CONTROL_ENABLED"],
        "__EDGE_SIM_ENABLED__": service_env["EDGE_SIM_ENABLED"],
        "__STACK_BIND_HOST__": service_env["STACK_BIND_HOST"],
        "__STACK_PUBLIC_HOST__": service_env["STACK_PUBLIC_HOST"],
    }
    for key, value in replacements.items():
        payload = payload.replace(key, value)
    return payload


def _render_windows_task_template(
    template_path: Path,
    *,
    service_env: dict[str, str],
) -> str:
    windows_edge_root = str(PureWindowsPath(service_env["CHEK_EDGE_ROOT"]))
    teleop_stack_cmd = str(PureWindowsPath(windows_edge_root) / "scripts" / "teleop_local_stack.cmd")
    payload = template_path.read_text(encoding="utf-8")
    replacements = {
        "__EDGE_ROOT__": windows_edge_root,
        "__TELEOP_STACK_CMD__": teleop_stack_cmd,
        "__EDGE_RUNTIME_PROFILE__": service_env["EDGE_RUNTIME_PROFILE"],
        "__EDGE_CONTROL_ENABLED__": service_env["EDGE_CONTROL_ENABLED"],
        "__EDGE_SIM_ENABLED__": service_env["EDGE_SIM_ENABLED"],
        "__STACK_BIND_HOST__": service_env["STACK_BIND_HOST"],
        "__STACK_PUBLIC_HOST__": service_env["STACK_PUBLIC_HOST"],
        "__NO_SIM_FLAG__": "--no-sim" if service_env["EDGE_SIM_ENABLED"] == "0" else "",
    }
    for key, value in replacements.items():
        payload = payload.replace(key, value)
    return payload


def _windows_task_run_command(destination_path: str) -> str:
    normalized_destination = str(PureWindowsPath(destination_path))
    return f'cmd.exe /c ""{normalized_destination}""'


def _stage_service_catalogs(
    profile_name: str,
    edge_root: Path,
    *,
    backend_name: str = "",
    service_env: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    rendered_catalogs: list[dict[str, Any]] = []
    resolved_env = service_env or _user_process_runtime_env(profile_name, edge_root=edge_root)
    rendered_root = _state_dir(edge_root) / "rendered-services"
    for manifest_path in _service_catalog_paths(profile_name):
        catalog = _load_yaml(manifest_path)
        if not _catalog_supports_backend(catalog, backend_name):
            continue
        manager = str(catalog.get("manager") or manifest_path.parent.parent.name)
        target_dir = rendered_root / manager / profile_name
        target_dir.mkdir(parents=True, exist_ok=True)

        rendered_templates: list[dict[str, Any]] = []
        for template_name in list(catalog.get("templates", [])):
            source_path = manifest_path.parent / template_name
            target_path = target_dir / template_name
            if not source_path.exists():
                rendered_templates.append(
                    {
                        "template": template_name,
                        "source_path": str(source_path),
                        "status": "missing_source",
                    }
                )
                continue
            if manager == "launchd-user" and source_path.suffix == ".plist":
                target_path.write_text(
                    _render_launchd_template(source_path, edge_root=edge_root, service_env=resolved_env),
                    encoding="utf-8",
                )
            elif manager in {"systemd", "systemd-user"} and source_path.suffix in {".service", ".socket", ".timer"}:
                target_path.write_text(
                    _render_systemd_template(source_path, edge_root=edge_root, service_env=resolved_env),
                    encoding="utf-8",
                )
            elif manager == "windows-task" and source_path.suffix in {".cmd", ".ps1"}:
                target_path.write_text(
                    _render_windows_task_template(source_path, service_env=resolved_env),
                    encoding="utf-8",
                )
            else:
                shutil.copy2(source_path, target_path)
            rendered_templates.append(
                {
                    "template": template_name,
                    "service_name": source_path.stem,
                    "source_path": str(source_path),
                    "rendered_path": str(target_path),
                    "status": "rendered",
                }
            )

        staged_manifest = {
            **catalog,
            "manager": manager,
            "source_manifest": str(manifest_path.relative_to(REPO_ROOT)),
            "rendered_at": _now_iso(),
        }
        staged_manifest_path = target_dir / "manifest.yaml"
        _write_yaml(staged_manifest_path, staged_manifest)
        rendered_catalogs.append(
            {
                "profile": profile_name,
                "manager": manager,
                "summary": catalog.get("summary", ""),
                "source_manifest": str(manifest_path),
                "rendered_manifest": str(staged_manifest_path),
                "templates": rendered_templates,
            }
        )
    return rendered_catalogs


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _install_receipt_path(edge_root: Path) -> Path:
    return _state_dir(edge_root) / "last-apply.json"


def _service_receipt_path(edge_root: Path) -> Path:
    return _state_dir(edge_root) / "last-service-install.json"


def _write_install_receipt(
    *,
    edge_root: Path,
    profile_name: str,
    backend_name: str,
    staged_catalogs: list[dict[str, Any]],
    installed_catalogs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    receipt = {
        "created_at": _now_iso(),
        "edge_root": str(edge_root),
        "profile": profile_name,
        "backend": backend_name,
        "config_path": str(_config_path(edge_root)),
        "plan_path": str(_plan_path(edge_root)),
        "staged_service_catalogs": staged_catalogs,
    }
    if installed_catalogs is not None:
        receipt["installed_catalogs"] = installed_catalogs
    _write_json(_install_receipt_path(edge_root), receipt)
    return receipt


def _remove_path(path: Path) -> None:
    if not path.exists():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
        return
    path.unlink()


def _copy_runtime_path(source: Path, destination: Path) -> None:
    if source.is_dir():
        shutil.copytree(
            source,
            destination,
            ignore=shutil.ignore_patterns(*RUNTIME_SYNC_IGNORE_NAMES),
            dirs_exist_ok=True,
        )
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _looks_like_windows_path(value: str) -> bool:
    normalized = value.strip()
    if len(normalized) >= 3 and normalized[1] == ":" and normalized[2] in {"\\", "/"}:
        return True
    return normalized.startswith("\\\\")


def _default_runtime_stage_root(*, profile_name: str, backend_name: str) -> Path:
    return Path.home() / STATE_DIR_NAME / "runtime" / backend_name / profile_name


def _resolve_runtime_edge_root(
    *,
    profile_name: str,
    edge_root: Path,
    backend_name: str,
    runtime_edge_root: str,
    installing_services: bool,
) -> tuple[str, str]:
    resolved_runtime_edge_root = runtime_edge_root.strip()
    if resolved_runtime_edge_root:
        return resolved_runtime_edge_root, "explicit"
    if backend_name == "macos" and installing_services:
        return (
            str(_default_runtime_stage_root(profile_name=profile_name, backend_name=backend_name)),
            "auto_staged_for_launchd",
        )
    return str(edge_root), "repo_root"


def _sync_runtime_root(source_root: Path, runtime_root: Path) -> dict[str, Any]:
    runtime_root.mkdir(parents=True, exist_ok=True)
    synced_paths: list[str] = []
    skipped_paths: list[str] = []
    for relative_path in RUNTIME_SYNC_PATHS:
        source_path = source_root / relative_path
        if not source_path.exists():
            skipped_paths.append(relative_path)
            continue
        destination_path = runtime_root / relative_path
        _remove_path(destination_path)
        _copy_runtime_path(source_path, destination_path)
        synced_paths.append(relative_path)
    return {
        "source_root": str(source_root),
        "runtime_root": str(runtime_root),
        "synced_paths": synced_paths,
        "skipped_paths": skipped_paths,
    }


def _run_command(
    command: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    use_sudo: bool = False,
) -> dict[str, Any]:
    resolved_command = (["sudo", "-n"] + command) if use_sudo else command
    try:
        completed = subprocess.run(
            resolved_command,
            cwd=str(cwd) if cwd is not None else None,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError as exc:
        return {
            "command": resolved_command,
            "cwd": str(cwd) if cwd is not None else "",
            "exit_code": 127,
            "ok": False,
            "stdout": "",
            "stderr": str(exc),
        }
    return {
        "command": resolved_command,
        "cwd": str(cwd) if cwd is not None else "",
        "exit_code": completed.returncode,
        "ok": completed.returncode == 0,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _matches_service_filter(service_name: str, template_name: str, service_filter: str) -> bool:
    normalized_filter = service_filter.strip()
    if not normalized_filter or normalized_filter == "all":
        return True
    return normalized_filter in service_name or normalized_filter in template_name


def _filter_staged_catalogs(
    catalogs: list[dict[str, Any]],
    service_filter: str,
) -> list[dict[str, Any]]:
    if not service_filter.strip():
        return catalogs
    filtered_catalogs: list[dict[str, Any]] = []
    for catalog in catalogs:
        templates = [
            template
            for template in list(catalog.get("templates", []))
            if _matches_service_filter(
                str(template.get("service_name") or ""),
                str(template.get("template") or ""),
                service_filter,
            )
        ]
        filtered_catalogs.append(
            {
                **catalog,
                "templates": templates,
            }
        )
    return filtered_catalogs


def _prepare_install_state(edge_root: Path, *, profile_name: str, backend_name: str) -> dict[str, Any]:
    config = _load_config(edge_root)
    if not config:
        config["created_at"] = _now_iso()
        config["control_plane_base_url"] = ""
        config["device_id"] = ""
        config["user_id"] = ""
    config.update(
        {
            "updated_at": _now_iso(),
            "repo_root": str(REPO_ROOT),
            "edge_root": str(edge_root),
            "profile": profile_name,
            "install_backend": backend_name,
        }
    )
    _save_config(edge_root, config)
    return config


def _apply_install_plan(
    *,
    edge_root: Path,
    profile_name: str,
    backend_name: str,
    service_env: dict[str, str],
) -> dict[str, Any]:
    config = _prepare_install_state(edge_root, profile_name=profile_name, backend_name=backend_name)
    staged_catalogs = _stage_service_catalogs(
        profile_name,
        edge_root,
        backend_name=backend_name,
        service_env=service_env,
    )
    _write_install_receipt(
        edge_root=edge_root,
        profile_name=profile_name,
        backend_name=backend_name,
        staged_catalogs=staged_catalogs,
    )
    return {
        "config": config,
        "receipt_path": str(_install_receipt_path(edge_root)),
        "staged_service_catalogs": staged_catalogs,
    }


def _install_catalogs_to_manager(
    catalogs: list[dict[str, Any]],
    *,
    system_root: Path | None = None,
    systemd_user_root: Path | None = None,
    launchd_user_root: Path | None = None,
    windows_task_root: Path | None = None,
    enable: bool = False,
    use_sudo: bool = False,
) -> list[dict[str, Any]]:
    installed_catalogs: list[dict[str, Any]] = []
    for catalog in catalogs:
        manager = str(catalog.get("manager") or "")
        catalog_templates = list(catalog.get("templates", []))
        if not catalog_templates:
            installed_catalogs.append(
                {
                    "manager": manager,
                    "target_root": "",
                    "installed_templates": [],
                    "activation_results": [],
                    "status": "skipped",
                    "reason": "catalog has no templates",
                }
            )
            continue
        try:
            target_root = _resolve_manager_root(
                manager,
                system_root=system_root,
                systemd_user_root=systemd_user_root,
                launchd_user_root=launchd_user_root,
                windows_task_root=windows_task_root,
            )
            if use_sudo and manager == "systemd":
                mkdir_result = _run_command(["mkdir", "-p", str(target_root)], use_sudo=True)
                if not mkdir_result["ok"]:
                    raise RuntimeError(mkdir_result["stderr"] or mkdir_result["stdout"] or "sudo mkdir failed")
            else:
                target_root.mkdir(parents=True, exist_ok=True)
        except Exception as exc:  # noqa: BLE001
            installed_catalogs.append(
                {
                    "manager": manager,
                    "target_root": "",
                    "installed_templates": [],
                    "activation_results": [],
                    "status": "error",
                    "error": repr(exc),
                }
            )
            continue

        installed_templates: list[dict[str, Any]] = []
        for template in catalog_templates:
            rendered_path = Path(str(template.get("rendered_path") or ""))
            if not rendered_path.exists():
                installed_templates.append(
                    {
                        "service_name": str(template.get("service_name") or ""),
                        "status": "missing_rendered_path",
                    }
                )
                continue
            destination = target_root / rendered_path.name
            copy_result: dict[str, Any] | None = None
            if use_sudo and manager == "systemd":
                copy_result = _run_command(
                    ["install", "-m", "644", str(rendered_path), str(destination)],
                    use_sudo=True,
                )
            else:
                try:
                    shutil.copy2(rendered_path, destination)
                except Exception as exc:  # noqa: BLE001
                    installed_templates.append(
                        {
                            "service_name": str(template.get("service_name") or rendered_path.stem),
                            "source_path": str(rendered_path),
                            "destination_path": str(destination),
                            "status": "error",
                            "error": repr(exc),
                        }
                    )
                    continue
            if copy_result is not None and not copy_result["ok"]:
                installed_templates.append(
                    {
                        "service_name": str(template.get("service_name") or rendered_path.stem),
                        "source_path": str(rendered_path),
                        "destination_path": str(destination),
                        "status": "error",
                        "error": copy_result["stderr"] or copy_result["stdout"] or "sudo install failed",
                    }
                )
                continue
            installed_templates.append(
                {
                    "service_name": str(template.get("service_name") or rendered_path.stem),
                    "source_path": str(rendered_path),
                    "destination_path": str(destination),
                    "status": "installed",
                }
            )

        activation_results: list[dict[str, Any]] = []
        if enable and installed_templates:
            if manager == "launchd-user":
                if shutil.which("launchctl") is None:
                    activation_results.append(
                        {
                            "manager": manager,
                            "status": "skipped",
                            "reason": "launchctl not available",
                        }
                    )
                else:
                    domain = f"gui/{os.getuid()}"
                    for template in installed_templates:
                        label = str(template.get("service_name") or "")
                        destination_path = str(template.get("destination_path") or "")
                        activation_results.append(
                            {
                                "service_name": label,
                                "bootout": _run_command(["launchctl", "bootout", f"{domain}/{label}"]),
                                "bootstrap": _run_command(["launchctl", "bootstrap", domain, destination_path]),
                                "kickstart": _run_command(["launchctl", "kickstart", "-k", f"{domain}/{label}"]),
                            }
                        )
            elif manager in {"systemd", "systemd-user"}:
                if shutil.which("systemctl") is None:
                    activation_results.append(
                        {
                            "manager": manager,
                            "status": "skipped",
                            "reason": "systemctl not available",
                        }
                    )
                else:
                    default_root = _default_manager_root(manager)
                    if target_root != default_root:
                        activation_results.append(
                            {
                                "manager": manager,
                                "status": "skipped",
                                "reason": "enable requires the active manager search path",
                                "expected_root": str(default_root),
                                "target_root": str(target_root),
                            }
                        )
                    elif manager == "systemd" and not use_sudo and hasattr(os, "geteuid") and os.geteuid() != 0:
                        activation_results.append(
                            {
                                "manager": manager,
                                "status": "skipped",
                                "reason": "system-level systemd install requires root",
                            }
                        )
                    else:
                        prefix = ["systemctl", "--user"] if manager == "systemd-user" else ["systemctl"]
                        activation_results.append(
                            {
                                "daemon_reload": _run_command(
                                    [*prefix, "daemon-reload"],
                                    use_sudo=use_sudo and manager == "systemd",
                                )
                            }
                        )
                        for template in installed_templates:
                            activation_results.append(
                                {
                                    "service_name": str(template.get("service_name") or ""),
                                    "enable_now": _run_command(
                                        [*prefix, "enable", "--now", str(template.get("service_name") or "")],
                                        use_sudo=use_sudo and manager == "systemd",
                                    ),
                                }
                            )
            elif manager == "windows-task":
                if shutil.which("schtasks") is None:
                    activation_results.append(
                        {
                            "manager": manager,
                            "status": "skipped",
                            "reason": "schtasks not available",
                        }
                    )
                else:
                    for template in installed_templates:
                        service_name = str(template.get("service_name") or "")
                        destination_path = str(template.get("destination_path") or "")
                        activation_results.append(
                            {
                                "service_name": service_name,
                                "create": _run_command(
                                    [
                                        "schtasks",
                                        "/Create",
                                        "/F",
                                        "/SC",
                                        "ONLOGON",
                                        "/TN",
                                        service_name,
                                        "/TR",
                                        _windows_task_run_command(destination_path),
                                    ]
                                ),
                                "run": _run_command(["schtasks", "/Run", "/TN", service_name]),
                            }
                        )

        installed_catalogs.append(
            {
                "manager": manager,
                "target_root": str(target_root),
                "installed_templates": installed_templates,
                "activation_results": activation_results,
            }
        )
    return installed_catalogs


def _restart_service_managers(
    catalogs: list[dict[str, Any]],
    *,
    use_sudo: bool = False,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    for catalog in catalogs:
        manager = str(catalog.get("manager") or "")
        templates = list(catalog.get("templates", []))
        if not templates:
            continue
        if manager == "launchd-user":
            if shutil.which("launchctl") is None:
                results.append(
                    {
                        "manager": manager,
                        "status": "skipped",
                        "reason": "launchctl not available",
                    }
                )
                continue
            domain = f"gui/{os.getuid()}"
            for template in templates:
                service_name = str(template.get("service_name") or "")
                results.append(
                    {
                        "manager": manager,
                        "service_name": service_name,
                        "restart": _run_command(["launchctl", "kickstart", "-k", f"{domain}/{service_name}"]),
                    }
                )
        elif manager in {"systemd", "systemd-user"}:
            if shutil.which("systemctl") is None:
                results.append(
                    {
                        "manager": manager,
                        "status": "skipped",
                        "reason": "systemctl not available",
                    }
                )
                continue
            if manager == "systemd" and not use_sudo and hasattr(os, "geteuid") and os.geteuid() != 0:
                results.append(
                    {
                        "manager": manager,
                        "status": "skipped",
                        "reason": "system-level systemd restart requires root",
                    }
                )
                continue
            prefix = ["systemctl", "--user"] if manager == "systemd-user" else ["systemctl"]
            for template in templates:
                service_name = str(template.get("service_name") or "")
                results.append(
                    {
                        "manager": manager,
                        "service_name": service_name,
                        "restart": _run_command(
                            [*prefix, "restart", service_name],
                            use_sudo=use_sudo and manager == "systemd",
                        ),
                    }
                )
        elif manager == "windows-task":
            if shutil.which("schtasks") is None:
                results.append(
                    {
                        "manager": manager,
                        "status": "skipped",
                        "reason": "schtasks not available",
                    }
                )
                continue
            for template in templates:
                service_name = str(template.get("service_name") or "")
                results.append(
                    {
                        "manager": manager,
                        "service_name": service_name,
                        "restart": _run_command(["schtasks", "/Run", "/TN", service_name]),
                    }
                )
    return results


def _restart_user_process_stack(
    *,
    edge_root: Path,
    profile_name: str,
    runtime_profile: str = "",
    bind_host: str = "127.0.0.1",
    public_host: str = "",
    control_enabled: str = "",
    sim_enabled: str = "",
) -> dict[str, Any]:
    env = os.environ.copy()
    env.update(
        _user_process_runtime_env(
            profile_name,
            edge_root=edge_root,
            runtime_profile=runtime_profile,
            bind_host=bind_host,
            public_host=public_host,
            control_enabled=control_enabled,
            sim_enabled=sim_enabled,
        )
    )
    command: list[str]
    stack_args = [
        "restart",
        "--bind-host",
        env["STACK_BIND_HOST"],
        "--public-host",
        env["STACK_PUBLIC_HOST"],
    ]
    if env["EDGE_SIM_ENABLED"] == "0":
        stack_args.append("--no-sim")
    if platform.system().lower() == "windows":
        stack_cmd_script = _stack_cmd_script_path(edge_root)
        cmd_bin = shutil.which("cmd.exe") or shutil.which("cmd")
        if cmd_bin and stack_cmd_script.exists():
            command = [
                cmd_bin,
                "/c",
                str(stack_cmd_script),
                *stack_args,
            ]
        else:
            stack_script = _stack_script_path(edge_root)
            if shutil.which("bash") is None or not stack_script.exists():
                return {
                    "command": [],
                    "cwd": str(edge_root),
                    "exit_code": 127,
                    "ok": False,
                    "stdout": "",
                    "stderr": "no usable Windows stack launcher found; expected cmd, PowerShell, or bash runtime wrapper",
                }
            command = ["bash", str(stack_script), *stack_args]
    else:
        stack_script = _stack_script_path(edge_root)
        command = ["bash", str(stack_script), *stack_args]
    return _run_command(command, cwd=edge_root, env=env)


def _tail_text(path: Path, lines: int) -> str:
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-lines:])


def _collect_log_candidates(edge_root: Path, service: str = "") -> list[Path]:
    filters = service.strip()
    roots = [
        _state_dir(edge_root) / "logs" / "launchd",
        edge_root / "edge-orchestrator" / "target" / "codex-local" / "teleop-stack",
        REPO_ROOT / "edge-orchestrator" / "target" / "codex-local" / "teleop-stack",
    ]
    seen: set[Path] = set()
    candidates: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.glob("*.log")):
            if path in seen:
                continue
            if filters and filters != "all":
                stem = path.stem
                if filters not in stem and not stem.startswith(filters):
                    continue
            candidates.append(path)
            seen.add(path)
    return candidates


def _collect_journal_logs(
    profile_name: str,
    *,
    backend_name: str = "",
    service_filter: str,
    lines: int,
    use_sudo: bool = False,
) -> list[dict[str, Any]]:
    journal_logs: list[dict[str, Any]] = []
    for manifest_path in _service_catalog_paths(profile_name):
        catalog = _load_yaml(manifest_path)
        if not _catalog_supports_backend(catalog, backend_name):
            continue
        manager = str(catalog.get("manager") or manifest_path.parent.parent.name)
        if manager not in {"systemd", "systemd-user"}:
            continue
        for template_name in list(catalog.get("templates", [])):
            service_name = Path(template_name).stem
            if template_name.endswith(".service"):
                service_name = template_name
            if not _matches_service_filter(service_name, template_name, service_filter):
                continue
            if manager == "systemd-user":
                command = ["journalctl", "--user", "-u", service_name, "-n", str(lines), "--no-pager"]
                result = _run_command(command)
            else:
                command = ["journalctl", "-u", service_name, "-n", str(lines), "--no-pager"]
                result = _run_command(command, use_sudo=use_sudo)
            journal_logs.append(
                {
                    "manager": manager,
                    "service_name": service_name,
                    "journal": result,
                }
            )
    return journal_logs


def _build_plan(profile_name: str, edge_root: Path, backend_name: str | None = None) -> dict[str, Any]:
    profile = _load_profile(profile_name)
    resolved_backend = backend_name or _detect_backend(profile_name)
    backend = _load_backend(resolved_backend)
    install_plan = backend.build_install_plan(profile_name=profile_name, edge_root=edge_root)
    module_names = list(
        dict.fromkeys(
            [
                *list(profile.get("required_modules", [])),
                *list(profile.get("capture_modules", [])),
                *list(profile.get("preview_modules", [])),
            ]
        )
    )
    return {
        "created_at": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "edge_root": str(edge_root),
        "profile": profile,
        "backend": backend.describe_backend(),
        "modules": _load_modules(module_names),
        "service_catalogs": _load_service_catalogs(profile_name, resolved_backend),
        "service_templates": _collect_service_templates(profile_name, resolved_backend),
        "install_plan": install_plan,
    }


def _command_probe(*candidates: str) -> dict[str, Any]:
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return {
                "candidates": list(candidates),
                "available": True,
                "path": resolved,
                "selected": candidate,
            }
    return {
        "candidates": list(candidates),
        "available": False,
        "path": "",
        "selected": "",
    }


def _runtime_python_probe() -> dict[str, Any]:
    if platform.system().lower() == "windows":
        for command in (["py", "-3"], ["python"]):
            if shutil.which(command[0]):
                return {
                    "available": True,
                    "command": command,
                    "display": " ".join(command),
                }
    else:
        for command in (["python3"], ["python"]):
            if shutil.which(command[0]):
                return {
                    "available": True,
                    "command": command,
                    "display": " ".join(command),
                }
    return {
        "available": False,
        "command": [],
        "display": "",
    }


def _run_python_inline(python_probe: dict[str, Any], code: str) -> dict[str, Any]:
    command = list(python_probe.get("command", []))
    if not command:
        return {
            "ok": False,
            "returncode": 127,
            "stdout": "",
            "stderr": "runtime python not available",
        }
    completed = subprocess.run(
        [*command, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    return {
        "ok": completed.returncode == 0,
        "returncode": completed.returncode,
        "stdout": completed.stdout.strip(),
        "stderr": completed.stderr.strip(),
    }


def _python_module_probe(python_probe: dict[str, Any], modules: list[str]) -> dict[str, Any]:
    if not modules:
        return {
            "runtime_python": str(python_probe.get("display", "")),
            "available": bool(python_probe.get("available")),
            "modules": {},
            "ok": True,
            "stderr": "",
        }
    result = _run_python_inline(
        python_probe,
        (
            "import importlib.util, json; "
            f"mods = {json.dumps(modules, ensure_ascii=True)}; "
            "print(json.dumps({name: bool(importlib.util.find_spec(name)) for name in mods}))"
        ),
    )
    parsed = {name: False for name in modules}
    if result["ok"] and result["stdout"]:
        try:
            payload = json.loads(result["stdout"])
        except json.JSONDecodeError:
            payload = {}
        if isinstance(payload, dict):
            parsed = {name: bool(payload.get(name)) for name in modules}
    return {
        "runtime_python": str(python_probe.get("display", "")),
        "available": bool(python_probe.get("available")),
        "modules": parsed,
        "ok": all(parsed.values()) if parsed else bool(python_probe.get("available")),
        "stderr": result["stderr"],
    }


def _binary_file_format(path: Path) -> str:
    try:
        magic = path.read_bytes()[:4]
    except OSError:
        return "unknown"
    if magic.startswith(b"\x7fELF"):
        return "elf"
    if magic.startswith(b"MZ"):
        return "pe"
    if magic in {
        b"\xfe\xed\xfa\xce",
        b"\xce\xfa\xed\xfe",
        b"\xfe\xed\xfa\xcf",
        b"\xcf\xfa\xed\xfe",
        b"\xca\xfe\xba\xbe",
        b"\xbe\xba\xfe\xca",
    }:
        return "mach-o"
    return "unknown"


def _artifact_probe(
    edge_root: Path,
    relative_paths: list[str],
    *,
    executable: bool = False,
    expected_format: str = "",
) -> dict[str, Any]:
    candidates: list[Path] = []
    seen: set[Path] = set()
    for relative_path in relative_paths:
        for candidate in _candidate_runtime_paths(edge_root, relative_path):
            resolved = candidate.resolve()
            if resolved in seen:
                continue
            candidates.append(resolved)
            seen.add(resolved)
    existing = [path for path in candidates if path.exists()]
    existing_details = [
        {
            "path": str(path),
            "format": _binary_file_format(path) if executable else "",
        }
        for path in existing
    ]
    executable_ready = any(
        (
            path.suffix.lower() == ".exe" or os.access(path, os.X_OK)
        ) and (
            not expected_format or _binary_file_format(path) == expected_format
        )
        for path in existing
    )
    return {
        "candidates": [str(path) for path in candidates],
        "existing": existing_details,
        "exists": bool(existing),
        "ready": bool(existing) and (executable_ready if executable else True),
        "expected_format": expected_format,
    }


def _expected_binary_format(backend_name: str) -> str:
    if backend_name == "windows":
        return "pe"
    if backend_name == "macos":
        return "mach-o"
    return "elf"


def _binary_probe(edge_root: Path, service_dir: str, binary_name: str, *, backend_name: str) -> dict[str, Any]:
    windows_binary = backend_name == "windows"
    suffix = ".exe" if windows_binary else ""
    return _artifact_probe(
        edge_root,
        [
            f"{service_dir}/target/debug/{binary_name}{suffix}",
            f"{service_dir}/target/release/{binary_name}{suffix}",
        ],
        executable=True,
        expected_format=_expected_binary_format(backend_name),
    )


def _backend_matches_current_host(backend_name: str, *, machine: str) -> bool:
    system = platform.system().lower()
    normalized_machine = machine.lower()
    if backend_name == "macos":
        return system == "darwin"
    if backend_name == "windows":
        return system == "windows"
    if backend_name == "jetson":
        return system == "linux" and ("arm" in normalized_machine or "aarch64" in normalized_machine)
    if backend_name == "linux":
        return system == "linux"
    return False


def _current_platform_tags(machine: str) -> set[str]:
    system = platform.system().lower()
    normalized_machine = machine.lower()
    tags = {system}
    if system == "darwin":
        tags.add("macos")
        if "arm" in normalized_machine or "aarch64" in normalized_machine:
            tags.add("macos-arm64")
        else:
            tags.add("macos-x86_64")
        return tags
    if system == "windows":
        tags.add("windows")
        if "arm" in normalized_machine or "aarch64" in normalized_machine:
            tags.add("windows-arm64")
        else:
            tags.add("windows-x86_64")
        return tags
    if system == "linux":
        tags.add("linux")
        if "arm" in normalized_machine or "aarch64" in normalized_machine:
            tags.update({"linux-arm64", "linux-aarch64"})
        else:
            tags.add("linux-x86_64")
        return tags
    return tags


def _service_manager_probe(manager: str) -> dict[str, Any]:
    if manager == "launchd-user":
        return _command_probe("launchctl")
    if manager in {"systemd", "systemd-user"}:
        return _command_probe("systemctl")
    if manager == "windows-task":
        return _command_probe("schtasks")
    return {
        "candidates": [],
        "available": False,
        "path": "",
        "selected": "",
    }


def _stereo_device_probe() -> dict[str, Any]:
    if platform.system().lower() == "windows":
        shell = _command_probe("pwsh", "powershell")
        if not shell["available"]:
            return {
                "device_count": 0,
                "devices": [],
                "ok": False,
                "reason": "powershell not available",
            }
        completed = subprocess.run(
            [
                str(shell["path"]),
                "-NoProfile",
                "-Command",
                (
                    "$devices = Get-CimInstance Win32_PnPEntity | "
                    "Where-Object { $_.Service -eq 'usbvideo' }; "
                    "$payload = @{ device_count = @($devices).Count; "
                    "devices = @($devices | ForEach-Object { $_.Name }) }; "
                    "$payload | ConvertTo-Json -Compress"
                ),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        devices: list[str] = []
        if completed.returncode == 0 and completed.stdout.strip():
            try:
                payload = json.loads(completed.stdout.strip())
            except json.JSONDecodeError:
                payload = {}
            if isinstance(payload, dict):
                devices = [str(item) for item in list(payload.get("devices", []))]
        return {
            "device_count": len(devices),
            "devices": devices,
            "ok": len(devices) >= 2,
            "reason": completed.stderr.strip(),
        }
    if platform.system().lower() == "darwin":
        ffmpeg = shutil.which("ffmpeg")
        devices: list[str] = []
        reason = ""
        if ffmpeg:
            try:
                completed = subprocess.run(
                    [
                        ffmpeg,
                        "-hide_banner",
                        "-f",
                        "avfoundation",
                        "-list_devices",
                        "true",
                        "-i",
                        "",
                    ],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=8,
                )
                in_video_section = False
                for raw_line in f"{completed.stdout}\n{completed.stderr}".splitlines():
                    line = raw_line.strip()
                    if "AVFoundation video devices:" in line:
                        in_video_section = True
                        continue
                    if "AVFoundation audio devices:" in line:
                        in_video_section = False
                        continue
                    if not in_video_section:
                        continue
                    match = re.search(r"\]\s+\[(\d+)\]\s+(.+)$", line)
                    if match and not match.group(2).strip().lower().startswith("capture screen"):
                        devices.append(f"avfoundation:{match.group(1)}:{match.group(2).strip()}")
                reason = completed.stderr.strip()
            except subprocess.TimeoutExpired:
                reason = "ffmpeg avfoundation device listing timed out"
        if not devices and shutil.which("system_profiler"):
            try:
                completed = subprocess.run(
                    ["system_profiler", "SPCameraDataType", "-json"],
                    capture_output=True,
                    text=True,
                    check=False,
                    timeout=8,
                )
                if completed.returncode == 0 and completed.stdout.strip():
                    try:
                        payload = json.loads(completed.stdout)
                    except json.JSONDecodeError:
                        payload = {}
                    for index, item in enumerate(payload.get("SPCameraDataType") or []):
                        if isinstance(item, dict):
                            name = str(item.get("_name") or item.get("spcamera_model-id") or "").strip()
                            if name:
                                devices.append(f"system_profiler:{index}:{name}")
                reason = reason or completed.stderr.strip()
            except subprocess.TimeoutExpired:
                reason = reason or "system_profiler camera listing timed out"
        return {
            "device_count": len(devices),
            "devices": devices,
            "ok": len(devices) >= 2,
            "reason": reason,
        }
    devices = [str(path) for path in sorted(Path("/dev").glob("video*"))]
    return {
        "device_count": len(devices),
        "devices": devices,
        "ok": len(devices) >= 2,
        "reason": "",
    }


def _probe_host_readiness(
    profile_name: str,
    edge_root: Path,
    backend_name: str | None = None,
) -> dict[str, Any]:
    plan = _build_plan(profile_name, edge_root, backend_name)
    resolved_backend = str(plan["backend"]["name"])
    machine = platform.machine()
    supported_platforms = {str(item).strip().lower() for item in list(plan["profile"].get("supported_platforms", [])) if str(item).strip()}
    current_platform_tags = _current_platform_tags(machine)
    runtime_env = _user_process_runtime_env(profile_name, edge_root=edge_root)
    control_enabled = runtime_env["EDGE_CONTROL_ENABLED"] == "1"
    managers = sorted({str(catalog.get("manager") or "") for catalog in list(plan["service_catalogs"]) if str(catalog.get("manager") or "")})
    python_probe = _runtime_python_probe()
    system = platform.system()
    commands = {
        "bash": _command_probe("bash"),
        "curl": _command_probe("curl"),
        "git": _command_probe("git"),
        "cargo": _command_probe("cargo"),
        "rustc": _command_probe("rustc"),
        "node": _command_probe("node"),
        "npm": _command_probe("npm"),
        "pnpm": _command_probe("pnpm"),
        "lsof": _command_probe("lsof"),
        "rsync": _command_probe("rsync"),
        "pwsh": _command_probe("pwsh"),
        "powershell": _command_probe("powershell"),
        "cmd": _command_probe("cmd.exe", "cmd"),
    }
    artifacts = {
        "stack_sh": _artifact_probe(edge_root, ["scripts/teleop_local_stack.sh"]),
        "stack_ps1": _artifact_probe(edge_root, ["scripts/teleop_local_stack.ps1"]),
        "stack_cmd": _artifact_probe(edge_root, ["scripts/teleop_local_stack.cmd"]),
        "launch_agent": _artifact_probe(edge_root, ["scripts/teleop_local_launch_agent.sh"]),
        "workstation_dist": _artifact_probe(edge_root, ["RuView/ui-react/dist/index.html"]),
        "edge_binary": _binary_probe(edge_root, "edge-orchestrator", "edge-orchestrator", backend_name=resolved_backend),
        "leap_binary": _binary_probe(edge_root, "ruview-leap-bridge", "ruview-leap-bridge", backend_name=resolved_backend),
        "unitree_binary": _binary_probe(edge_root, "ruview-unitree-bridge", "ruview-unitree-bridge", backend_name=resolved_backend),
    }
    stereo_modules = ["cv2", "numpy", "requests", "rtmlib"]
    stereo_python = _python_module_probe(python_probe, stereo_modules)
    stereo_devices = _stereo_device_probe()
    service_managers = {
        manager: _service_manager_probe(manager)
        for manager in managers
    }

    blockers: list[dict[str, str]] = []
    warnings: list[str] = []
    suggestions: list[str] = []

    def add_blocker(code: str, detail: str, suggestion: str = "") -> None:
        blockers.append({"code": code, "detail": detail})
        if suggestion and suggestion not in suggestions:
            suggestions.append(suggestion)

    if supported_platforms and supported_platforms.isdisjoint(current_platform_tags):
        add_blocker(
            "unsupported_platform",
            f"profile {profile_name} supports {sorted(supported_platforms)}, but current host matches {sorted(current_platform_tags)}",
            "Use a host that matches the profile supported_platforms before claiming live acceptance on this lane.",
        )

    if not _backend_matches_current_host(resolved_backend, machine=machine):
        add_blocker(
            "backend_host_mismatch",
            f"current host is {system}/{machine}, but profile {profile_name} resolves to backend={resolved_backend}",
            "Run this command on a host that matches the resolved backend, or keep using install/service staging only.",
        )

    if plan["profile"].get("service_mode") == "user-process":
        if not artifacts["workstation_dist"]["ready"] and not commands["npm"]["available"]:
            add_blocker(
                "node_toolchain_missing",
                "RuView/ui-react/dist is missing and npm is not available to build it.",
                "Install Node/npm on the host or sync a prebuilt RuView/ui-react/dist directory.",
            )
        if not artifacts["edge_binary"]["ready"] and not commands["cargo"]["available"]:
            add_blocker(
                "rust_toolchain_missing",
                "edge-orchestrator binary is missing and cargo is not available to build it.",
                "Install Rust/cargo on the host or sync a prebuilt edge-orchestrator binary.",
            )
        if control_enabled:
            if not artifacts["leap_binary"]["ready"] and not commands["cargo"]["available"]:
                add_blocker(
                    "rust_toolchain_missing",
                    "ruview-leap-bridge binary is missing and cargo is not available to build it.",
                    "Install Rust/cargo on the host or sync a prebuilt ruview-leap-bridge binary.",
                )
            if not artifacts["unitree_binary"]["ready"] and not commands["cargo"]["available"]:
                add_blocker(
                    "rust_toolchain_missing",
                    "ruview-unitree-bridge binary is missing and cargo is not available to build it.",
                    "Install Rust/cargo on the host or sync a prebuilt ruview-unitree-bridge binary.",
                )
        if resolved_backend == "windows":
            if not commands["bash"]["available"]:
                add_blocker(
                    "windows_bash_missing",
                    "Windows runtime wrappers still require bash in PATH.",
                    "Install Git Bash or another bash implementation and expose it in PATH.",
                )
            if not (commands["pwsh"]["available"] or commands["powershell"]["available"]):
                add_blocker(
                    "windows_shell_missing",
                    "Neither pwsh nor powershell is available for the Windows wrapper lane.",
                    "Install PowerShell or repair the host shell association before rerunning readiness.",
                )
        else:
            if not commands["bash"]["available"]:
                add_blocker(
                    "bash_missing",
                    "teleop_local_stack.sh requires bash.",
                    "Install bash on the host before attempting the user-process runtime lane.",
                )
        if not commands["curl"]["available"]:
            add_blocker(
                "curl_missing",
                "teleop_local_stack.sh health checks require curl.",
                "Install curl on the host before attempting runtime smoke.",
            )
        if not commands["lsof"]["available"]:
            add_blocker(
                "lsof_missing",
                "teleop_local_stack.sh port checks require lsof.",
                "Install lsof on the host before attempting runtime smoke.",
            )

    for manager, probe in service_managers.items():
        if not probe["available"]:
            add_blocker(
                f"{manager}_not_available",
                f"service manager {manager} is required by the resolved backend but its host command is unavailable.",
                f"Install or expose the {manager} manager command on the host, or use direct user-process restart where supported.",
            )

    if profile_name in {"enhanced", "professional"}:
        if not stereo_devices["ok"]:
            add_blocker(
                "stereo_device_not_detected",
                f"detected {stereo_devices['device_count']} stereo candidate devices on this host.",
                "Attach or expose at least two stereo-capable camera devices before claiming live stereo acceptance.",
            )
        if not stereo_python["ok"]:
            missing = [name for name, ready in stereo_python["modules"].items() if not ready]
            add_blocker(
                "stereo_python_deps_missing",
                f"runtime python is missing: {', '.join(missing)}",
                "Install stereo python deps with `python3 -m pip install --user numpy requests opencv-python rtmlib onnxruntime`.",
            )

    if not python_probe["available"]:
        add_blocker(
            "runtime_python_missing",
            "No runtime python executable was found for host probes or helper scripts.",
            "Install python3 on the host before rerunning readiness.",
        )

    if not commands["git"]["available"]:
        warnings.append("git_not_available")
    if not commands["rsync"]["available"]:
        warnings.append("rsync_not_available")

    return {
        "created_at": _now_iso(),
        "profile": profile_name,
        "backend": resolved_backend,
        "service_mode": plan["profile"].get("service_mode", ""),
        "runtime_env": runtime_env,
        "host": {
            "system": system,
            "machine": machine,
            "platform_tags": sorted(current_platform_tags),
            "python_version": platform.python_version(),
            "edge_root": str(edge_root),
        },
        "commands": commands,
        "artifacts": artifacts,
        "service_managers": service_managers,
        "stereo": {
            "devices": stereo_devices,
            "python": stereo_python,
        },
        "ready": not blockers,
        "blockers": blockers,
        "warnings": warnings,
        "suggested_next_steps": suggestions,
    }


def _save_plan(edge_root: Path, plan: dict[str, Any]) -> None:
    _ensure_state_dir(edge_root)
    _plan_path(edge_root).write_text(
        json.dumps(plan, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


@profile_app.command("list")
def profile_list() -> None:
    for payload in _available_profiles():
        typer.echo(
            f"{payload.get('name', '')}: {payload.get('display_name', '')} | {payload.get('summary', '')}"
        )


@profile_app.command("show")
def profile_show(
    profile: str = typer.Argument(..., help="basic / enhanced / professional"),
) -> None:
    typer.echo(
        yaml.safe_dump(_load_profile(profile), sort_keys=False, allow_unicode=True).strip()
    )


@app.command()
def init(
    profile: str = typer.Option("basic", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
) -> None:
    profile_payload = _load_profile(profile)
    state_dir = _ensure_state_dir(edge_root)
    config = {
        "created_at": _now_iso(),
        "repo_root": str(REPO_ROOT),
        "edge_root": str(edge_root),
        "profile": profile,
        "profile_summary": profile_payload.get("summary", ""),
        "device_id": "",
        "install_backend": _detect_backend(profile),
        "control_plane_base_url": "",
        "user_id": "",
    }
    _save_config(edge_root, config)
    typer.echo(f"Initialized {state_dir}")


@app.command()
def install(
    profile: str = typer.Option("basic", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    backend: str = typer.Option("", help="Override backend"),
    apply: bool = typer.Option(False, help="Persist install plan and stage backend assets"),
    system_install: bool = typer.Option(False, help="Copy rendered service artifacts into service-manager roots in the same command"),
    enable_services: bool = typer.Option(False, help="Enable or bootstrap installed services after copying them"),
    system_root: str = typer.Option("", help="Override /etc/systemd/system target root"),
    systemd_user_root: str = typer.Option("", help="Override ~/.config/systemd/user target root"),
    launchd_user_root: str = typer.Option("", help="Override ~/Library/LaunchAgents target root"),
    windows_task_root: str = typer.Option("", help="Override %LOCALAPPDATA%/CHEK/Edge/Tasks target root"),
    use_sudo: bool = typer.Option(False, "--sudo", help="Use sudo -n for privileged systemd install actions"),
    runtime_edge_root: str = typer.Option("", help="Override runtime root embedded in generated user-process service wrappers"),
    stage_runtime_root: bool = typer.Option(True, help="Sync a minimal runtime workspace when runtime_edge_root points to a local install root"),
    runtime_profile: str = typer.Option("", help="Override teleop runtime profile for generated user-process services"),
    bind_host: str = typer.Option("127.0.0.1", help="Bind host for generated user-process services"),
    public_host: str = typer.Option("", help="Public host for generated user-process services"),
    control_enabled: str = typer.Option("", help="Override EDGE_CONTROL_ENABLED for generated user-process services"),
    sim_enabled: str = typer.Option("", help="Override EDGE_SIM_ENABLED for generated user-process services"),
) -> None:
    plan = _build_plan(profile, edge_root, backend or None)
    apply_result: dict[str, Any] = {}
    should_apply = apply or system_install or enable_services
    if should_apply:
        _save_plan(edge_root, plan)
        runtime_root_sync: dict[str, Any] = {}
        backend_name = str(plan["backend"]["name"])
        resolved_runtime_edge_root, runtime_edge_root_source = _resolve_runtime_edge_root(
            profile_name=profile,
            edge_root=edge_root,
            backend_name=backend_name,
            runtime_edge_root=runtime_edge_root,
            installing_services=system_install or enable_services,
        )
        if (
            stage_runtime_root
            and resolved_runtime_edge_root
            and resolved_runtime_edge_root != str(edge_root)
            and not (
                backend_name == "windows"
                and platform.system().lower() != "windows"
                and _looks_like_windows_path(resolved_runtime_edge_root)
            )
        ):
            runtime_root_sync = _sync_runtime_root(
                edge_root,
                Path(resolved_runtime_edge_root).expanduser(),
            )
        service_env = _user_process_runtime_env(
            profile,
            edge_root=edge_root,
            runtime_edge_root=resolved_runtime_edge_root,
            runtime_profile=runtime_profile,
            bind_host=bind_host,
            public_host=public_host,
            control_enabled=control_enabled,
            sim_enabled=sim_enabled,
        )
        apply_result = _apply_install_plan(
            edge_root=edge_root,
            profile_name=profile,
            backend_name=backend_name,
            service_env=service_env,
        )
        apply_result["runtime_edge_root"] = resolved_runtime_edge_root
        apply_result["runtime_edge_root_source"] = runtime_edge_root_source
        if runtime_root_sync:
            apply_result["runtime_root_sync"] = runtime_root_sync
        if system_install or enable_services:
            installed_catalogs = _install_catalogs_to_manager(
                list(apply_result.get("staged_service_catalogs", [])),
                system_root=_path_from_option(system_root),
                systemd_user_root=_path_from_option(systemd_user_root),
                launchd_user_root=_path_from_option(launchd_user_root),
                windows_task_root=_path_from_option(windows_task_root),
                enable=enable_services,
                use_sudo=use_sudo,
            )
            apply_result["installed_catalogs"] = installed_catalogs
            _write_install_receipt(
                edge_root=edge_root,
                profile_name=profile,
                backend_name=backend_name,
                staged_catalogs=list(apply_result.get("staged_service_catalogs", [])),
                installed_catalogs=installed_catalogs,
            )
        typer.echo(f"Saved install plan to {_plan_path(edge_root)}")
    typer.echo(json.dumps({**plan, "apply_result": apply_result}, ensure_ascii=False, indent=2))


@app.command()
def doctor(
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
) -> None:
    config_exists = _config_path(edge_root).exists()
    checks = {
        "created_at": _now_iso(),
        "edge_root": str(edge_root),
        "config_present": config_exists,
        "profiles_present": sorted(path.stem for path in PROFILES_DIR.glob("*.yaml")),
        "modules_present": sorted(path.parent.name for path in MODULES_DIR.glob("*/module.yaml")),
        "service_catalogs_present": [
            str(path.relative_to(REPO_ROOT)) for path in _all_service_catalog_paths()
        ],
        "python_version": platform.python_version(),
        "platform": {
            "system": platform.system(),
            "machine": platform.machine(),
        },
    }
    typer.echo(json.dumps(checks, ensure_ascii=False, indent=2))


@app.command()
def readiness(
    profile: str = typer.Option("basic", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    backend: str = typer.Option("", help="Override backend"),
) -> None:
    typer.echo(
        json.dumps(
            _probe_host_readiness(profile, edge_root, backend or None),
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def status(
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    control_plane_base_url: str = typer.Option("", help="Override control-plane base url"),
    user_id: str = typer.Option("", help="Override X-User-One-Id"),
    bearer_token: str = typer.Option("", help="Override Authorization bearer"),
) -> None:
    config = _load_config(edge_root)
    plan = {}
    if _plan_path(edge_root).exists():
        plan = json.loads(_plan_path(edge_root).read_text(encoding="utf-8"))
    active_profile = str(config.get("profile") or plan.get("profile", {}).get("name") or "basic")
    active_profile_payload = _load_profile(active_profile) if (PROFILES_DIR / f"{active_profile}.yaml").exists() else {}
    active_backend = str(
        config.get("install_backend")
        or plan.get("backend", {}).get("name")
        or _detect_backend(active_profile)
    )
    active_modules = list(
        dict.fromkeys(
            [
                *list(active_profile_payload.get("required_modules", [])),
                *list(active_profile_payload.get("capture_modules", [])),
                *list(active_profile_payload.get("preview_modules", [])),
            ]
        )
    )
    resolved_base_url = control_plane_base_url.strip() or str(config.get("control_plane_base_url") or "").strip()
    resolved_user_id = user_id.strip() or str(config.get("user_id") or "").strip()
    remote_status: dict[str, Any] = {}
    if resolved_base_url and (resolved_user_id or bearer_token.strip()):
        device_id = str(config.get("device_id") or "").strip()
        headers = _auth_headers(bearer_token=bearer_token, user_id=resolved_user_id)
        remote_status = {
            "device_binding": _http_json(
                "GET",
                _join_url(
                    resolved_base_url,
                    "/v1/me/deviceBinding/current",
                    {"deviceId": device_id},
                ),
                headers=headers,
            ),
            "upload_scope": _http_json(
                "GET",
                _join_url(resolved_base_url, "/v1/me/uploadScope", {"deviceId": device_id}),
                headers=headers,
            ),
            "device_runtime_status": _http_json(
                "GET",
                _join_url(
                    resolved_base_url,
                    "/v1/me/deviceStatus/current",
                    {"deviceId": device_id},
                ),
                headers=headers,
            ),
        }
    typer.echo(
        json.dumps(
            {
                "config": config,
                "last_plan": plan,
                "modules": _load_modules(active_modules),
                "service_catalogs": _load_service_catalogs(active_profile, active_backend),
                "service_templates": _collect_service_templates(active_profile, active_backend),
                "remote_status": remote_status,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@app.command()
def bind(
    device_id: str = typer.Option(..., help="Device id from control-plane"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    device_name: str = typer.Option("", help="Display name for current edge node"),
    device_type: str = typer.Option("", help="Runtime device type / hardware profile"),
    claim_code: str = typer.Option("", help="Optional claim code"),
    control_plane_base_url: str = typer.Option("", help="Control-plane base url"),
    user_id: str = typer.Option("", help="Current user id for control-plane bind"),
    bearer_token: str = typer.Option("", help="Authorization bearer for control-plane bind"),
) -> None:
    config = _load_config(edge_root)
    config["device_id"] = device_id
    config["updated_at"] = _now_iso()
    if control_plane_base_url.strip():
        config["control_plane_base_url"] = control_plane_base_url.strip()
    if user_id.strip():
        config["user_id"] = user_id.strip()
    remote_bind = {}
    resolved_base_url = control_plane_base_url.strip() or str(config.get("control_plane_base_url") or "").strip()
    resolved_user_id = user_id.strip() or str(config.get("user_id") or "").strip()
    if resolved_base_url and (resolved_user_id or bearer_token.strip()):
        remote_bind = _http_json(
            "POST",
            _join_url(resolved_base_url, "/v1/me/deviceBinding/bind"),
            headers=_auth_headers(bearer_token=bearer_token, user_id=resolved_user_id),
            payload={
                "deviceId": device_id,
                "deviceName": device_name,
                "deviceType": device_type,
                "claimCode": claim_code,
                "metadata": {
                    "binding_source": "chek-edge-cli",
                    "edge_root": str(edge_root),
                },
            },
        )
        config["last_remote_bind"] = {
            "requested_at": _now_iso(),
            "response": remote_bind,
        }
    _save_config(edge_root, config)
    typer.echo(
        json.dumps(
            {
                "device_id": device_id,
                "config_path": str(_config_path(edge_root)),
                "remote_bind": remote_bind,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@capture_app.command("probe")
def capture_probe(
    profile: str = typer.Option("basic", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    control_plane_base_url: str = typer.Option("", help="Override control-plane base url"),
    user_id: str = typer.Option("", help="Override X-User-One-Id"),
    bearer_token: str = typer.Option("", help="Override Authorization bearer"),
    task_id: str = typer.Option("", help="Task id for readiness probe"),
) -> None:
    payload = _load_profile(profile)
    config = _load_config(edge_root)
    resolved_base_url = control_plane_base_url.strip() or str(config.get("control_plane_base_url") or "").strip()
    resolved_user_id = user_id.strip() or str(config.get("user_id") or "").strip()
    readiness = {}
    if resolved_base_url and (resolved_user_id or bearer_token.strip()):
        readiness = _http_json(
            "GET",
            _join_url(
                resolved_base_url,
                "/v1/me/capture-readiness",
                {
                    "deviceId": str(config.get("device_id") or "").strip(),
                    "taskId": task_id.strip(),
                },
            ),
            headers=_auth_headers(bearer_token=bearer_token, user_id=resolved_user_id),
        )
    typer.echo(
        json.dumps(
            {
                "profile": profile,
                "capture_modules": payload.get("capture_modules", []),
                "module_manifests": _load_modules(list(payload.get("capture_modules", []))),
                "capture_readiness": readiness,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@preview_app.command("test")
def preview_test(
    profile: str = typer.Option("basic", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    control_plane_base_url: str = typer.Option("", help="Override control-plane base url"),
    session_id: str = typer.Option("", help="Session id for preview contract probe"),
    user_id: str = typer.Option("", help="Override X-User-One-Id"),
    bearer_token: str = typer.Option("", help="Override Authorization bearer"),
) -> None:
    payload = _load_profile(profile)
    config = _load_config(edge_root)
    resolved_base_url = control_plane_base_url.strip() or str(config.get("control_plane_base_url") or "").strip()
    resolved_user_id = user_id.strip() or str(config.get("user_id") or "").strip()
    preview_probe = {}
    if resolved_base_url and session_id.strip():
        preview_probe = _http_json(
            "GET",
            _join_url(
                resolved_base_url,
                f"/v1/public/ego-dataset/sessions/{session_id.strip()}/preview",
            ),
            headers=_auth_headers(bearer_token=bearer_token, user_id=resolved_user_id),
        )
    typer.echo(
        json.dumps(
            {
                "profile": profile,
                "preview_modules": payload.get("preview_modules", []),
                "module_manifests": _load_modules(list(payload.get("preview_modules", []))),
                "preview_probe": preview_probe,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@upload_app.command("test")
def upload_test(
    profile: str = typer.Option("basic", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    control_plane_base_url: str = typer.Option("", help="Override control-plane base url"),
    edge_token: str = typer.Option("", help="Edge upload token"),
    session_id: str = typer.Option("cli-upload-smoke", help="Synthetic session id"),
    task_id: str = typer.Option("task-upload-smoke", help="Synthetic task id"),
) -> None:
    payload = _load_profile(profile)
    config = _load_config(edge_root)
    resolved_base_url = control_plane_base_url.strip() or str(config.get("control_plane_base_url") or "").strip()
    device_id = str(config.get("device_id") or "").strip()
    upload_probe: dict[str, Any] = {}
    if resolved_base_url and edge_token.strip() and device_id:
        headers = {
            "Authorization": f"Bearer {edge_token.strip()}",
            "Content-Type": "application/json",
        }
        upsert = _http_json(
            "POST",
            _join_url(resolved_base_url, "/v1/edge/sessions/upsert"),
            headers=headers,
            payload={
                "session_id": session_id,
                "capture_device_id": device_id,
                "task_id": task_id,
                "status": "ready_to_upload",
                "upload_policy": "metadata_plus_preview",
                "runtime_profile": profile,
                "raw_residency": "edge_only",
                "preview_residency": "cloud_preview_only",
                "ready_for_review": False,
                "metadata": {
                    "source": "chek-edge-cli",
                },
            },
        )
        artifact = _http_json(
            "POST",
            _join_url(resolved_base_url, f"/v1/edge/sessions/{session_id}/artifacts"),
            headers=headers,
            payload={
                "asset_id": "cli-preview-001",
                "relpath": "preview/frame-0001.jpg",
                "kind": "image",
                "category": "preview_derivative",
                "required": True,
                "status": "declared",
                "residency_status": "cloud_preview_only",
                "preview_status": "ready",
                "delivery_status": "not_requested",
                "declared_size_bytes": 1024,
                "metadata": {
                    "source": "chek-edge-cli",
                },
            },
        )
        receipt = _http_json(
            "POST",
            _join_url(resolved_base_url, f"/v1/edge/sessions/{session_id}/receipts"),
            headers=headers,
            payload={
                "asset_id": "cli-preview-001",
                "status": "uploaded",
                "stored_size_bytes": 1024,
                "storage_key": "crowd-data/dev/preview/cli-preview-001.jpg",
                "metadata": {
                    "source": "chek-edge-cli",
                },
            },
        )
        upload_probe = {
            "session_upsert": upsert,
            "artifact_upsert": artifact,
            "receipt": receipt,
        }
    typer.echo(
        json.dumps(
            {
                "profile": profile,
                "required_modules": payload.get("required_modules", []),
                "module_manifests": _load_modules(list(payload.get("required_modules", []))),
                "target_contract": "backend-crowd-data-saas session / artifact / receipt",
                "upload_probe": upload_probe,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@service_app.command("install")
def service_install(
    profile: str = typer.Option("professional", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    backend: str = typer.Option("", help="Override backend"),
    system_root: str = typer.Option("", help="Override /etc/systemd/system target root"),
    systemd_user_root: str = typer.Option("", help="Override ~/.config/systemd/user target root"),
    launchd_user_root: str = typer.Option("", help="Override ~/Library/LaunchAgents target root"),
    windows_task_root: str = typer.Option("", help="Override %LOCALAPPDATA%/CHEK/Edge/Tasks target root"),
    enable: bool = typer.Option(False, help="Enable or bootstrap installed services after copying"),
    service: str = typer.Option("", help="Only operate on matching service unit names"),
    use_sudo: bool = typer.Option(False, "--sudo", help="Use sudo -n for privileged systemd install actions"),
    runtime_edge_root: str = typer.Option("", help="Override runtime root embedded in generated user-process service wrappers"),
    stage_runtime_root: bool = typer.Option(True, help="Sync a minimal runtime workspace when runtime_edge_root points to a local install root"),
    runtime_profile: str = typer.Option("", help="Override teleop runtime profile for user-process services"),
    bind_host: str = typer.Option("127.0.0.1", help="Bind host for user-process services"),
    public_host: str = typer.Option("", help="Public host for user-process services"),
    control_enabled: str = typer.Option("", help="Override EDGE_CONTROL_ENABLED for user-process services"),
    sim_enabled: str = typer.Option("", help="Override EDGE_SIM_ENABLED for user-process services"),
) -> None:
    plan = _build_plan(profile, edge_root, backend or None)
    backend_name = str(plan["backend"]["name"])
    resolved_runtime_edge_root, runtime_edge_root_source = _resolve_runtime_edge_root(
        profile_name=profile,
        edge_root=edge_root,
        backend_name=backend_name,
        runtime_edge_root=runtime_edge_root,
        installing_services=True,
    )
    runtime_root_sync: dict[str, Any] = {}
    if (
        stage_runtime_root
        and resolved_runtime_edge_root
        and resolved_runtime_edge_root != str(edge_root)
        and not (
            backend_name == "windows"
            and platform.system().lower() != "windows"
            and _looks_like_windows_path(resolved_runtime_edge_root)
        )
    ):
        runtime_root_sync = _sync_runtime_root(
            edge_root,
            Path(resolved_runtime_edge_root).expanduser(),
        )
    staged_catalogs = _stage_service_catalogs(
        profile,
        edge_root,
        backend_name=backend_name,
        service_env=_user_process_runtime_env(
            profile,
            edge_root=edge_root,
            runtime_edge_root=resolved_runtime_edge_root,
            runtime_profile=runtime_profile,
            bind_host=bind_host,
            public_host=public_host,
            control_enabled=control_enabled,
            sim_enabled=sim_enabled,
        ),
    )
    filtered_catalogs = _filter_staged_catalogs(staged_catalogs, service)
    installed_catalogs = _install_catalogs_to_manager(
        filtered_catalogs,
        system_root=_path_from_option(system_root),
        systemd_user_root=_path_from_option(systemd_user_root),
        launchd_user_root=_path_from_option(launchd_user_root),
        windows_task_root=_path_from_option(windows_task_root),
        enable=enable,
        use_sudo=use_sudo,
    )
    receipt = {
        "created_at": _now_iso(),
        "profile": profile,
        "backend": backend_name,
        "runtime_edge_root": resolved_runtime_edge_root,
        "runtime_edge_root_source": runtime_edge_root_source,
        "staged_catalogs": filtered_catalogs,
        "installed_catalogs": installed_catalogs,
    }
    if runtime_root_sync:
        receipt["runtime_root_sync"] = runtime_root_sync
    _write_json(_service_receipt_path(edge_root), receipt)
    typer.echo(
        json.dumps(
            {
                "profile": profile,
                "backend": backend_name,
                "runtime_edge_root": resolved_runtime_edge_root,
                "runtime_edge_root_source": runtime_edge_root_source,
                "service_mode": plan["profile"].get("service_mode", ""),
                "service_catalogs": filtered_catalogs,
                "installed_catalogs": installed_catalogs,
                "runtime_root_sync": runtime_root_sync,
                "receipt_path": str(_service_receipt_path(edge_root)),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


@service_app.command("restart")
def service_restart(
    profile: str = typer.Option("professional", help="Deployment profile"),
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    backend: str = typer.Option("", help="Override backend"),
    service: str = typer.Option("", help="Only operate on matching service unit names"),
    use_sudo: bool = typer.Option(False, "--sudo", help="Use sudo -n for privileged systemd restart actions"),
    runtime_profile: str = typer.Option("", help="Override teleop runtime profile for direct user-process restart"),
    bind_host: str = typer.Option("127.0.0.1", help="Bind host for direct user-process restart"),
    public_host: str = typer.Option("", help="Public host for direct user-process restart"),
    control_enabled: str = typer.Option("", help="Override EDGE_CONTROL_ENABLED for direct user-process restart"),
    sim_enabled: str = typer.Option("", help="Override EDGE_SIM_ENABLED for direct user-process restart"),
    direct: bool = typer.Option(False, help="Bypass OS service managers and restart the local stack directly"),
) -> None:
    plan = _build_plan(profile, edge_root, backend or None)
    staged_catalogs = _stage_service_catalogs(profile, edge_root, backend_name=str(plan["backend"]["name"]))
    restart_result: dict[str, Any] = {
        "profile": profile,
        "backend": plan["backend"]["name"],
        "service_mode": plan["profile"].get("service_mode", ""),
    }
    filtered_catalogs = _filter_staged_catalogs(staged_catalogs, service)
    has_manager_catalogs = any(list(catalog.get("templates", [])) for catalog in filtered_catalogs)
    if direct or (str(plan["profile"].get("service_mode", "")) == "user-process" and not has_manager_catalogs):
        restart_result["direct_restart"] = _restart_user_process_stack(
            edge_root=edge_root,
            profile_name=profile,
            runtime_profile=runtime_profile,
            bind_host=bind_host,
            public_host=public_host,
            control_enabled=control_enabled,
            sim_enabled=sim_enabled,
        )
    else:
        restart_result["manager_restart"] = _restart_service_managers(filtered_catalogs, use_sudo=use_sudo)
    typer.echo(
        json.dumps(restart_result, ensure_ascii=False, indent=2)
    )


@logs_app.command("tail")
def logs_tail(
    edge_root: Path = typer.Option(Path.cwd(), help="Edge runtime root"),
    profile: str = typer.Option("", help="Override profile for service journal discovery"),
    service: str = typer.Option("all", help="Log file stem filter such as edge / viewer / com.chek.edge.basic"),
    lines: int = typer.Option(40, min=1, help="How many lines to include per log"),
    use_sudo: bool = typer.Option(False, "--sudo", help="Use sudo -n for privileged systemd journal reads"),
) -> None:
    config = _load_yaml(_config_path(edge_root)) if _config_path(edge_root).exists() else {}
    active_profile = profile.strip() or str(config.get("profile") or "professional")
    active_backend = str(config.get("install_backend") or _detect_backend(active_profile))
    log_paths = _collect_log_candidates(edge_root, service=service)
    typer.echo(
        json.dumps(
            {
                "edge_root": str(edge_root),
                "profile": active_profile,
                "journal_logs": _collect_journal_logs(
                    active_profile,
                    backend_name=active_backend,
                    service_filter=service,
                    lines=lines,
                    use_sudo=use_sudo,
                ),
                "logs": [
                    {
                        "path": str(path),
                        "tail": _tail_text(path, lines),
                    }
                    for path in log_paths
                ],
                "suggested_logs": _collect_service_templates(active_profile, active_backend),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    app()
