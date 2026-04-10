#!/usr/bin/env python3
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib import request as urllib_request

REPO_ROOT = Path(__file__).resolve().parents[1]
CLI_DIR = Path(__file__).resolve().parent
DEFAULT_VENV_DIR = REPO_ROOT / ".chek-ego-miner" / "tooling" / "runtime-cli-venv"


def _resolved_cli_venv_dir() -> Path:
    override = (
        os.environ.get("CHEK_EGO_MINER_RUNTIME_CLI_VENV", "").strip()
        or os.environ.get("CHEK_EDGE_CLI_VENV", "").strip()
    )
    if not override:
        return DEFAULT_VENV_DIR
    return Path(override).expanduser().resolve()


def _venv_python_path(venv_dir: Path) -> Path:
    candidates = [
        venv_dir / "bin" / "python",
        venv_dir / "Scripts" / "python.exe",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[1] if os.name == "nt" else candidates[0]


def _python_with_cli_imports(python_executable: Path) -> bool:
    if not python_executable.exists():
        return False
    try:
        result = subprocess.run(
            [str(python_executable), "-c", "import typer, yaml"],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        return False
    return result.returncode == 0


def _run_checked(command: list[str], *, description: str) -> None:
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode == 0:
        return
    message = completed.stderr.strip() or completed.stdout.strip() or "unknown bootstrap failure"
    raise RuntimeError(f"{description} failed: {message}")


def _python_has_pip(python_executable: Path) -> bool:
    completed = subprocess.run(
        [str(python_executable), "-m", "pip", "--version"],
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def _ensure_venv_pip(venv_python: Path) -> None:
    if _python_has_pip(venv_python):
        return
    ensurepip = subprocess.run(
        [str(venv_python), "-m", "ensurepip", "--upgrade"],
        capture_output=True,
        text=True,
        check=False,
    )
    if ensurepip.returncode == 0 and _python_has_pip(venv_python):
        return
    with tempfile.TemporaryDirectory(prefix="chek-ego-miner-get-pip-") as tmpdir:
        get_pip_path = Path(tmpdir) / "get-pip.py"
        urllib_request.urlretrieve("https://bootstrap.pypa.io/get-pip.py", get_pip_path)
        _run_checked(
            [str(venv_python), str(get_pip_path), "--disable-pip-version-check"],
            description="bootstrap pip into CLI venv",
        )


def _ensure_cli_venv(current_python: Path) -> Path:
    venv_dir = _resolved_cli_venv_dir()
    venv_python = _venv_python_path(venv_dir)
    if not venv_python.exists():
        if venv_dir.exists():
            shutil.rmtree(venv_dir, ignore_errors=True)
        venv_dir.parent.mkdir(parents=True, exist_ok=True)
        create_default = subprocess.run(
            [str(current_python), "-m", "venv", str(venv_dir)],
            capture_output=True,
            text=True,
            check=False,
        )
        if create_default.returncode != 0:
            create_without_pip = subprocess.run(
                [str(current_python), "-m", "venv", "--without-pip", str(venv_dir)],
                capture_output=True,
                text=True,
                check=False,
            )
            if create_without_pip.returncode != 0:
                message = create_default.stderr.strip() or create_default.stdout.strip() or "unknown venv failure"
                raise RuntimeError(f"create CLI venv at {venv_dir} failed: {message}")
    if not _python_with_cli_imports(venv_python):
        _ensure_venv_pip(venv_python)
        _run_checked(
            [
                str(venv_python),
                "-m",
                "pip",
                "install",
                "--disable-pip-version-check",
                "typer",
                "pyyaml",
            ],
            description="install typer and pyyaml into CLI venv",
        )
    return venv_python


def _run_cli(python_executable: Path, argv: list[str]) -> int:
    env = dict(os.environ)
    pythonpath_entries = [str(CLI_DIR), str(REPO_ROOT)]
    existing_pythonpath = env.get("PYTHONPATH", "").strip()
    if existing_pythonpath:
        pythonpath_entries.append(existing_pythonpath)
    env["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)
    completed = subprocess.run(
        [str(python_executable), "-m", "chek_edge.main", *argv],
        env=env,
        check=False,
    )
    return completed.returncode


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    current_python = Path(sys.executable).resolve()
    if _python_with_cli_imports(current_python):
        return _run_cli(current_python, args)
    cached_venv_python = _venv_python_path(_resolved_cli_venv_dir())
    if cached_venv_python.exists() and _python_with_cli_imports(cached_venv_python):
        return _run_cli(cached_venv_python, args)
    print(
        "chek-ego-miner runtime: current Python is missing typer/pyyaml, bootstrapping a repo-local CLI venv...",
        file=sys.stderr,
    )
    try:
        venv_python = _ensure_cli_venv(current_python)
    except RuntimeError as exc:
        print(f"chek-ego-miner runtime: {exc}", file=sys.stderr)
        return 1
    return _run_cli(venv_python, args)


if __name__ == "__main__":
    raise SystemExit(main())
